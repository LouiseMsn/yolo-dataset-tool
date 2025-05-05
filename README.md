# Yolo dataset creation tool
Creates a training dataset conforming to YOLO folder organisation and annotation format. Given a folder of images it will:
1. Generate N images of this image with random rotation, noise and brightness.
2. Dispatch these images into 3 folders: "train", "val" and "test".
3. (optional) It  annotates the images using SAM and saves the annoted version for humain verification.

The following folder arborescence is created :
```bash
source_folder # original images folder
└── yolo_dataset
    ├── images
    │   ├── annoted_images
    │   ├── test
    │   ├── train
    │   └── val
    └── labels
        ├── train
        └── val
```
The images given in input are not modified nor moved.

## Installation

```bash
# 1. Clone this repo
git clone git@github.com:LouiseMsn/yolo-dataset-creation-tool.git
cd yolo-dataset-creation-tool

# 2. Clone the dependencies
mkdir third-party && cd third-party
git clone git@github.com:luca-medeiros/lang-segment-anything.git
cd ..

# 3. Create the conda environment
conda env create -f environment.yml 
conda activate yolo-dataset
```
 
## Usage
### Dataset augmentation & annotation
Run
```bash
python augment_annotate_dataset.py -f <path/to/your/images> -a
```
Remove `-a` if you'd like to annotate the images yourself.

### Yolo training
In the terminal of your choice run :
```bash
yolo train data=<path/to/the/data.yaml>
```
See the YOLO documentation for more options of training
 
The results will be givenin the --- folder
### Yolo prediction
TODO : add command to train yolo
 TODO add command to predict with test folder
## Correct annotations
TODO with labelIMG copy paste .txt in the folders
##### Options:  
`-` : 


## Results

 