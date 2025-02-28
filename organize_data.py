import os
import json
import shutil
import numpy as np
import nibabel as nib
from pathlib import Path


def create_directory(directory):
    """Creates a directory if it does not exist."""
    os.makedirs(directory, exist_ok=True)


def create_dataset_json():
    """Returns a template for dataset.json."""
    return {
        "name": "Pancreas",
        "description": "Pancreas Segmentation Dataset",
        "tensorImageSize": "3D",
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "pancreas": 1, "lesion": 2},
        "class_types": {"subtype0": 0, "subtype1": 1, "subtype2": 2},
        "numTraining": 0,
        "numValidation": 0,
        "numTest": 0,
        "file_ending": ".nii.gz",
        "training": [],
        "validation": [],
        "test": []
    }


def get_split_names(split_name):
    """Returns names based on split."""
    folder_dictionary = {
        "train": "imagesTr",
        "validation": "imagesVal",
        "test": "imagesTs"
    }
    label_dictionary = {
        "train": "labelsTr",
        "validation": "labelsVal",
        "test": None  # Test does not have labels
    }
    split_key = "training" if split_name == "train" else split_name
    return split_key, folder_dictionary[split_name], label_dictionary[split_name]


def process_data_split(split_name, base_path, nnunet_dataset_dir, dataset_json, metadata):
    """
    Processes training, validation, and test splits.

    Args:
        split_name (str): One of ['train', 'validation', 'test'].
        base_path (str): Root directory of the dataset.
        nnunet_dataset_dir (str): Path to nnUNet formatted dataset.
        dataset_json (dict): Dataset configuration.
        metadata (dict): Stores subtype and split mapping.

    Returns:
        None
    """
    split_key, target_folder, label_folder = get_split_names(split_name)

    os.makedirs(os.path.join(nnunet_dataset_dir, target_folder), exist_ok=True)
    if label_folder:
        os.makedirs(os.path.join(nnunet_dataset_dir, label_folder), exist_ok=True)

    if split_name == "test":
        test_split_path = Path(os.path.join(base_path, split_name))
        for file in sorted(test_split_path.glob('*_0000.nii.gz')):
            shutil.copy2(file, os.path.join(nnunet_dataset_dir, target_folder, file.name))
            dataset_json[split_key].append({"image": f"./{target_folder}/{file.name}"})  # ✅ Add test case
    else:
        for subtype in ['subtype0', 'subtype1', 'subtype2']:
            subtype_path = Path(os.path.join(base_path, split_name, subtype))

            if not subtype_path.exists():
                print(f"Warning: {split_name} folder for {subtype} does not exist. Skipping...")
                continue

            for file in sorted(subtype_path.glob('*_0000.nii.gz')):
                case_id = file.stem.split('_0000')[0]
                mask_file = subtype_path / f"{case_id}.nii.gz"

                if not mask_file.exists():
                    print(f"Warning: Mask file missing for {split_name} case '{case_id}'. Skipping...")
                    continue

                # Copy both image and mask
                shutil.copy2(file, os.path.join(nnunet_dataset_dir, target_folder, file.name))
                # clip value so that mask can only have [0, 1, 2]
                mask = nib.load(mask_file)
                mask_data = np.round(mask.get_fdata()).astype(np.uint8)
                mask_data = np.clip(mask_data, 0, 2)
                new_mask = nib.Nifti1Image(mask_data, mask.affine)
                nib.save(new_mask, os.path.join(nnunet_dataset_dir, label_folder, f"{case_id}.nii.gz"))

                dataset_json[split_key].append({
                    "image": f"./{target_folder}/{file.name}",
                    "label": f"./{label_folder}/{case_id}.nii.gz"
                })
                metadata[case_id] = {"subtype": int(subtype[-1]), "split": split_name}


def organize_data_for_nnunet():
    """
    Organizes the dataset into nnUNet format.

    Returns:
        nnunet_dataset_dir (str): Path to the organized dataset.
        metadata (dict): Mapping of case IDs to subtypes and splits.
    """
    base_path = os.getenv("DATA_DIR")
    dataset_name = 'Dataset001'
    nnunet_dataset_dir = os.path.join(os.getenv("nnUNet_raw"), dataset_name)

    create_directory(nnunet_dataset_dir)

    dataset_json = create_dataset_json()
    metadata = {}

    # Process dataset splits
    for split in ['train', 'validation', 'test']:
        process_data_split(split, base_path, nnunet_dataset_dir, dataset_json, metadata)

    # Update dataset metadata
    dataset_json["numTraining"] = len(dataset_json["training"])
    dataset_json["numValidation"] = len(dataset_json["validation"])
    dataset_json["numTest"] = len(dataset_json["test"])  # ✅ Now correctly updated

    # Save dataset.json
    with open(os.path.join(nnunet_dataset_dir, 'dataset.json'), 'w') as f:
        json.dump(dataset_json, f, indent=4)

    # Save metadata (subtype mapping)
    with open(os.path.join(nnunet_dataset_dir, 'subtype_mapping.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    print_summary(dataset_json)
    return nnunet_dataset_dir, metadata


def print_summary(dataset_json):
    """
    Prints a summary of the dataset organization.

    Args:
        dataset_json (dict): Dataset metadata.

    Returns:
        None
    """
    print("\nNumber of cases in json:")
    print(f"Training: {dataset_json['numTraining']}")
    print(f"Validation: {dataset_json['numValidation']}")
    print(f"Test: {dataset_json['numTest']}")


if __name__ == "__main__":
    nnunet_dataset_path, metadata = organize_data_for_nnunet()

    # Verify the file structure
    print("\nNumber of .nii.gz file in folders:")
    for folder in ['imagesTr', 'labelsTr', 'imagesVal', 'labelsVal', 'imagesTs']:
        path = Path(os.path.join(nnunet_dataset_path, folder))
        print(f"{folder}: {len(list(path.glob('*.nii.gz')))} files")
