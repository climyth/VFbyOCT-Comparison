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
   a. Zeiss model: https://drive.google.com/open?id=1MEBzcT6MG9OfFdot_6mwhnIPsHdjVCnQ <br/>
   b. Topcon model: https://drive.google.com/open?id=1MEBzcT6MG9OfFdot_6mwhnIPsHdjVCnQ <br/>
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

weight_file_zeiss = "/Weights/InceptionResnet/InceptionResnet_24-2-improvement-324-3.86-60.94.hdf5"
weight_file_topcon = "/Weights/SSOCT_InceptionResnet_24-2-improvement-04-9.39-56.34.hdf5"
# ===========================================================================================
```
5. Run ShowTest.py;
6. You can see the popup window like below.
![](https://github.com/climyth/VFbyOCT-Comparison/blob/master/TestImages/test_example.JPG?raw=true)

### How can I make "combined OCT" image?
1. Download "panomaker.exe" in "utils" folder
2. In utils folder, there are sample OCT images to generate combined OCT image.<br/>
   You need 2 OCT images in pair like below. (1) macular OCT (2) ONH OCT<br/>
   ![](https://github.com/climyth/VFbySD-OCT/blob/master/example/oct_example.jpg?raw=true)
   <br/>
3. Image file name must follow the rule:<br/>
   (1) macular OCT: patientID_examdate_1.jpg  (ex. 012345678_20180403_1.jpg)<br/>
   (2) ONH OCT: patientID_examdate_2.jpg   (ex. 012345678_20180403_2.jpg)<br/>
   Note: Two images must have the same name (the only difference is last number _1 or _2)
4. Run "panomaker.exe"<br/><br/>
![](https://github.com/climyth/VFbySD-OCT/blob/master/example/panomaker.png?raw=true)
<br/><br/>
5. set source folder and output folder
6. press Start button. That's it!

### How can I train model with my own OCT images?
1. Prepare your own OCT images and visual field data (excel file)
2. Generate "combined OCT" images from your train set
3. In visual field excel file, <br/>
   (1) your data must be in "Train" data sheet <br/>
   (2) the first column must contain the list of image file names <br/>
   (3) visual field total threshold values must begin at 7th column by default (otherwise, you need to modify "LoadData") <br/>
   (4) There must be 54 columns of total threshold values (includes two physiologic scotoma point).
4. Modify 'Setup' in VFOCT_Train.py
```python
# Setup ====================================================================
image_folder = ""   # root image folder for train set (combined OCT images)
vf_file = "VFTrain.xlsm"   # visual field data excel file
weight_save_folder = "Weights"
graph_save_folder = ""   # model graph output folder
pretrained_weights = ""   # if no pretrained weight, just leave ""
tensorboard_log_folder = "logs"
# ==========================================================================
```
5. Run the VFOCT_Train.py
6. You can monitor loss trend in tensorboard. This is our trend curve for example.<br/><br/>
![](https://github.com/climyth/VFbySD-OCT/blob/master/example/train_log_1.jpg?raw=true)
<br/><br/>
7. To prevent overfitting, we used "repeated random sub-sampling cross validation method". To do this, just repeat to run VFOCT_Train.py. In each run, you can set "pretrained_weights" to continue the training from last weight file.


### For Research Use Only
The performance characteristics of this product have not been evaluated by the Food and Drug Administration and is not intended for commercial use or purposes beyond research use only.
