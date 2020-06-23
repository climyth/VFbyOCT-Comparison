# VFbyOCT : OCT device comparison (SD-OCT vs SS-OCT)

![](https://github.com/climyth/VFbyOCT-Comparison/blob/master/title.jpg?raw=true)

### Features
- InceptionResnet V2 backboned deep learning model
- Predicts Humphrey's visual field 24-2 total threshold values from Zeiss SD-OCT or Topcon SS-OCT
- Predicts entire picture of visual field
- Mean prediction error is 5.29 dB (Zeiss) and 4.51 dB (Topcon)

### Prerequisites
- python 3.6
- tensorflow >= 1.6.0
- keras 2.2.4

### How can I test OCT image?
1. Download all files from github
2. You can download weight file here: <br/>
   a. Zeiss model: https://drive.google.com/file/d/1kQquiJ18zkrzsWsfLyALsdNI3_D9N4EJ/view?usp=sharing <br/>
   b. Topcon model: https://drive.google.com/file/d/14NsTiurh30967hkJDfbROvtz3LP-E43N/view?usp=sharing <br/>
3. Open ShowTest.py
4. Modify "Setup"
```python
# Setup ====================================================================================
base_folder = "Z:/VFPredict"

zeiss_oct_image = "/TestImages/PT01_Zeiss.jpg"  # file name must starts with 'pid_' ex) "PT001_Zeiss.jpg"
topcon_oct_image = "/TestImages/PT01_Topcon.jpg"  # file name must starts with 'pid_' ex) "PT001_Topcon.jpg"

vf_file = "/TestImages/test_data_github.xlsm"
vf_sheet_name = "Sheet1"
pid_col = 0  # patients' ID column number (min = 0)
vf_thv_col = 115  # THV starting column number (min = 0)

weight_file_zeiss = "/Weights/InceptionResnet_24-2_SD_OCT.hdf5"
weight_file_topcon = "/Weights/InceptionResnet_SS_OCT.hdf5"
# ===========================================================================================
```
5. Run ShowTest.py;
6. You can see the popup window like below.
![](https://github.com/climyth/VFbyOCT-Comparison/blob/master/TestImages/test_example.JPG?raw=true)

### How can I make "combined OCT" image?
1. To make Zeiss combined OCT image, please visit <br/>
   https://github.com/climyth/VFbySD-OCT
   
2. To make Topcon combined OCT iamge, please visit <br/>
   https://github.com/climyth/VFbySS-OCT

### How can I train model with my own OCT images?
1. Prepare your own OCT images and visual field data (excel file)
2. Generate "combined OCT" images from your train set
3. In visual field excel file, <br/>
   (1) your data must be in "Train" data sheet <br/>
   (2) the first column must contain the list of image file names <br/>
   (3) visual field total threshold values must begin at 7th column by default (otherwise, you need to modify "LoadData") <br/>
   (4) There must be 54 columns of total threshold values (includes two physiologic scotoma point).
4. Modify 'Setup' in Train.py
```python
# Setup ====================================================================
vf_select = "24-2"  # 24-2 or 10-2
base_model_name = "InceptionResnet"   # ResNet, InceptionV3, VGG19, DenseNet, NASNet, InceptionResnet

oct_type = "zeiss"  # zeiss or topcon
base_folder = "Z:/VFPredict"
vf_file = "/SD_Train.xlsm"
weight_save_folder = "/Weights/" + base_model_name
graph_save_folder = ""
pretrained_weights = ""   # if not exist, leave ""
tensorboard_log_folder = "/Logs"
# ==========================================================================
```
5. Run the Train.py
6. You can monitor loss trend in tensorboard. This is our trend curve for example.<br/><br/>
7. To prevent overfitting, we used "repeated random sub-sampling cross validation method". To do this, just repeat to run Train.py. In each run, you can set "pretrained_weights" to continue the training from last weight file.


### For Research Use Only
The performance characteristics of this product have not been evaluated by the Food and Drug Administration and is not intended for commercial use or purposes beyond research use only.
