# Yolo dataset creation tool
Creates a training dataset conforming to YOLO folder organisation and annotation format. Given a folder of images it will:
1. Generate N images of this image with random rotation, noise and brightness.
2. Dispatch these images into 3 folders: "train", "val" and "test".
3. (optional) Annotate the images using SAM and save the annoted version for humain verification.

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

# 2. Create the conda environment
conda env create -n yolo-dataset python=3.12 -y
conda activate yolo-dataset

# 3.Install packages
pip install torch==2.4.1 torchvision==0.19.1 --extra-index-url https://download.pytorch.org/whl/cu124
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
```
 ---
## Usage
### Dataset augmentation & annotation
Run
```bash
python augment_annotate_dataset.py -f <path/to/your/images> -a
```
Options:  
`-a` : auto-annotation of the images using lang_sam, remove it if you'd like to annotate the images yourself.  
`-f` : [mandatory] folder where the input images are located.

## Correct annotations
It is very common for some annotations to be wrong so the dataset needs to be slightly corrected.  
The output dataset is formated to be corrected with [LabelImg](https://github.com/cloudy-sfu/labelimg).  

1. Install labelImg using the instruction in the repo, the pypi version kept crashing on my computer so used the build from source method.
2. Launch the programm using `python3 labelImg.py`
3. In LabelImg Click on `Open Dir` (CTRL+U) and open the /yolo-dataset/images/train or yolo-dataset/images/val directories depending on which one you want to correct.
4. Click on `Change Save Dir` (CTRL+R) and select the label directory matching the images
5. Manually correct the bounding boxes as you see fit.

### Yolo training
In the terminal of your choice run :
```bash
yolo train data=<path/to/the/data.yaml> model=<model.pt> 
```
>[!NOTE]  
> Usually model=yolo11n.pt. 
> See the YOLO documentation for additionnal parameters & options for training.  
 
The results will be given in the --- folder

### Yolo prediction
Test your training by running a prediction on the yolo-dataset/images/test folder:  
```bash
yolo predict model=</global/path/to/best.pt> source=<number of webcam or /path/to/dir> imgsz=<image_size>
```
