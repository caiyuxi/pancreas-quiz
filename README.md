# Deep Learning for Automatic Cancer Segmentation and Classification in 3D CT Scans
## Environments and Requirements

- Ubuntu 20.04.6LTS
- AMD Ryzen 9 5900X, 32GB RAM, NVIDIA GeForce RTX 3080
- CUDA version: 11.4
- python version: Python 3.11.11

To install requirements:

## Install the forked nnUNet
```setup
git clone https://github.com/caiyuxi/nnUNet-pancreas-CT
cd nnUNet
pip install -e .
cd ..
```

## Preprocessing

Putting the data into the data organization [specified in the nnUNet repo](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md):

```commandline
nnUNet_raw/Dataset001_BRATS/
├── dataset.json
├── imagesTr
│   ├── BRATS_001_0000.nii.gz
│   ├── BRATS_001_0001.nii.gz
│   ├── BRATS_002_0000.nii.gz
│   ├── ...
├── imagesTs
│   ├── BRATS_485_0000.nii.gz
│   ├── BRATS_485_0001.nii.gz
│   ├── BRATS_486_0000.nii.gz
│   ├── ...
└── labelsTr
    ├── BRATS_001.nii.gz
    ├── BRATS_002.nii.gz
    ├── ...
```

Running the data preprocessing code:

```bash
export nnUNet_raw="<your project path>/nnUNet_raw"
export DATA_DIR="<unzipped ML-Quiz-3DMedImg>"

python process_data.py
```
Then create a `dataset.json` under the folder nnUNetData/nnUNet_raw/Dataset101_Pancreas with the following content:
```
{
    "channel_names": {
        "0": "CT"
    }, 
    "labels": {
        "background": 0,
        "normal_pancreas": 1,
        "lesion": 2
    },
    "file_ending": ".nii.gz"
    "numTraining": 252,
}
```
Lastly preprocess the dataset:
```commandline
export nnUNet_preprocessed="<your project path>/nnUNet_preprocessed"
#nnUNetv2_plan_and_preprocess -d 001 -pl nnUNetPlannerResEncM
```
## Training

To train the model(s) in the paper, run this command to use our own `nnUNetTrainerWithClassification` class:

```bash
export nnUNet_results="<your project path>/nnUNet_results"
nnUNetv2_train 001 3d_fullres 0 -tr nnUNetTrainerWithClassification -p nnUNetResEncUNetMPlans --c
```
At the end of the training, the program also prints out out the quality of the model based on the validation set. 

## Test

1. To infer the testing cases, run this command:

```python
nnUNetv2_predict -i <your project path>/imagesTs -o <output folder> -d 001 -p nnUNetResEncUNetMPlans -chk checkpoint_best.pth -c 3d_fullres -f 0 -tr nnUNetTrainerWithClassification
```


## Results

Our method achieves the following performance

| Task                     |     Quality     | 
|--------------------------|:---------------:| 
| Segmentation (label 1+2) |  DSC = 80.75%   | 
| Segmentation (label 2)   |  DSC = 39.63%   | 
| Lesion Classification    | f1_score = 0.52 | 


