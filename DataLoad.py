from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import xlrd
import numpy as np
import os

# Setup ======================================
topcon_col = 0  # Topcon OCT filenames
zeiss_col = 1  # Zeiss OCT filenames
id_col_24 = 2  # patient's id column number (min = 0)
col_start_24 = 130  # THV starting column. (min = 0)
# =============================================


# Data loading function ===============================
def LoadData24(base_folder, vf_file, image_read_from_previous_numpy=False, oct_type="topcon", post_fix="train", sheet_name="24-2"):
    worksheet = xlrd.open_workbook(vf_file).sheet_by_name(sheet_name)
    nrows = worksheet.nrows - 1
    pids = np.empty([0, 2])
    y_data = np.empty([0, 52])
    if oct_type == "topcon":
        x_data = np.empty([0, 200, 480, 3])  # Topcon OCT image shape = 480 x 200 (w x h)
    if oct_type == "zeiss":
        x_data = np.empty([0, 161, 322, 3])  # Zeiss OCT image shape = 322 x 161 (w x h)
    print("reading %d rows" % (nrows))

    if image_read_from_previous_numpy == False:
        for r in range(1, nrows + 1):
            file_col = 0
            if oct_type == "topcon":
                file_col = topcon_col
            if oct_type == "zeiss":
                file_col = zeiss_col

            img_filename = base_folder + "/" + worksheet.cell_value(r, file_col)
            if os.path.isfile(img_filename) == False:
                print("Not found file: " + img_filename)
                continue

            pid_row = np.empty([1, 2])
            pid_row[0, 0] = worksheet.cell_value(r, id_col_24)   # pid
            if worksheet.cell_value(r, id_col_24 + 1) == "OD":   # Eye. OD=0, OS=1
                pid_row[0, 1] = 0
            else:
                pid_row[0, 1] = 1
            pids = np.concatenate((pids, pid_row), axis=0)

            img = load_img(img_filename)
            x_img = img_to_array(img)
            x_img = x_img.reshape((1,) + x_img.shape)
            x_data = np.concatenate((x_data, x_img), axis=0)
            print("[%d] concatenated: %s" % (r, img_filename))

            c1 = 0
            y_row = np.empty([1, 52])
            for c in range(0, 54):
                if c != 25 and c != 34:  # 암점은 제외한다.
                    y_row[0, c1] = worksheet.cell_value(r, c + col_start_24)
                    c1 = c1+1
            y_data = np.concatenate((y_data, y_row), axis=0)

    if image_read_from_previous_numpy:
        x_data = np.load(base_folder + "/img_data24_" + post_fix + "_" + oct_type + ".npy")
        y_data = np.load(base_folder + "/vf_data24_" + post_fix + "_" + oct_type + ".npy")
        pids = np.load(base_folder + "/pid_data24_" + post_fix + "_" + oct_type + ".npy")
    else:
        np.save(base_folder + "/img_data24_" + post_fix + "_" + oct_type, x_data)
        np.save(base_folder + "/vf_data24_" + post_fix + "_" + oct_type, y_data)
        np.save(base_folder + "/pid_data24_" + post_fix + "_" + oct_type, y_data)
        print("Image array saved")

    print("Loading completed. X data shape is %s and Y data shape is %s"%((x_data.shape), (y_data.shape)))
    print("Printing sample y_data:")
    print(y_data[0])
    return x_data, y_data, pids



