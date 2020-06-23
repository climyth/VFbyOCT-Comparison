from keras.preprocessing.image import array_to_img, img_to_array, load_img
from Model import *
import xlrd
from matplotlib import pyplot as plt
import os


# Setup ======================================
base_folder = "Z:/PaperResearch/VFbyOCT Device Comp"

zeiss_oct_image = "/Github/TestImages/PT01_Zeiss.jpg"  # file name must starts with 'pid_' ex) "PT001_Zeiss.jpg"
topcon_oct_image = "/Github/TestImages/PT01_Topcon.jpg"  # file name must starts with 'pid_' ex) "PT001_Topcon.jpg"

vf_file = "/Github/TestImages/test_data_github.xlsm"
vf_sheet_name = "Sheet1"
pid_col = 0  # patients' ID column number (min = 0)
vf_thv_col = 115  # THV starting column number (min = 0)

weight_file_zeiss = "/Weights//InceptionResnet/InceptionResnet_24-2-improvement-324-3.86-60.94.hdf5"
weight_file_topcon = "/Weights//SSOCT_InceptionResnet_24-2-improvement-04-9.39-56.34.hdf5"
# ============================================


def draw_visual_field(field_values, mplt, start_pos=(0, 0), rect_size=(35, 25), fontsize=9):
    #  ===== config =====================================================
    rect_pos = [                  [3,0],[4,0],[5,0],[6,0],
                            [2,1],[3,1],[4,1],[5,1],[6,1],[7,1],
                      [1,2],[2,2],[3,2],[4,2],[5,2],[6,2],[7,2],[8,2],
                [0,3],[1,3],[2,3],[3,3],[4,3],[5,3],[6,3],      [8,3],
                [0,4],[1,4],[2,4],[3,4],[4,4],[5,4],[6,4],      [8,4],
                      [1,5],[2,5],[3,5],[4,5],[5,5],[6,5],[7,5],[8,5],
                            [2,6],[3,6],[4,6],[5,6],[6,6],[7,6],
                                  [3,7],[4,7],[5,7],[6,7]]
    min_vf = 5   # for color range calculation
    max_vf = 35  # for color range calculation
    # ====================================================================

    for i in range(0, 52):
        x = start_pos[0] + rect_pos[i][0] * rect_size[0]
        y = start_pos[1] + (7 - rect_pos[i][1]) * rect_size[1]
        vf = field_values[i]
        bg_col = (vf - min_vf) / (max_vf - min_vf)   # max = 35, min = 5
        if bg_col < 0:
            bg_col = 0
        if bg_col > 1.0:
            bg_col = 1.0
        txt_color = 'black'
        if bg_col < 0.5:
            txt_color = 'white'
        rect = mplt.Rectangle((x, y), rect_size[0], rect_size[1], fill=True, fc=(bg_col, bg_col, bg_col))
        mplt.gca().add_patch(rect)
        mplt.gca().text(x+rect_size[0]/2, y+rect_size[1]/2, "{:.1f}".format(vf), fontsize=fontsize,
                        horizontalalignment='center', verticalalignment='center', color=txt_color)


# Data loading function ===============================
def LoadData():
    success = True
    worksheet = xlrd.open_workbook(base_folder + "/" + vf_file).sheet_by_name(vf_sheet_name)
    nrows = worksheet.nrows - 1
    y_data = np.empty([0, 52])
    x_data_zeiss = np.empty([0, 161, 322, 3])  # Zeiss OCT image shape = 322 x 161 (w x h)
    x_data_topcon = np.empty([0, 200, 480, 3])  # Topcon OCT image shape = 480 x 200 (w x h)

    zeiss_img_filename = base_folder + zeiss_oct_image
    topcon_img_filename = base_folder + topcon_oct_image
    pid1 = os.path.splitext(os.path.basename(zeiss_img_filename))[0].split('_')[0]
    pid2 = os.path.splitext(os.path.basename(topcon_img_filename))[0].split('_')[0]
    if pid1 != pid2:
        print("Patient's ID mismatch. Check image file name!")
        success = False
        return x_data_zeiss, x_data_topcon, y_data, success

    zeiss_img = load_img(zeiss_img_filename)
    x_img = img_to_array(zeiss_img)
    x_img = x_img.reshape((1,) + x_img.shape)
    x_data_zeiss = np.concatenate((x_data_zeiss, x_img), axis=0)

    topcon_img = load_img(topcon_img_filename)
    x_img = img_to_array(topcon_img)
    x_img = x_img.reshape((1,) + x_img.shape)
    x_data_topcon = np.concatenate((x_data_topcon, x_img), axis=0)

    for r in range(1, nrows + 1):
        pid_vf = worksheet.cell_value(r, pid_col)
        if pid1 == pid_vf:
            c1 = 0
            y_row = np.empty([1, 52])
            for c in range(0, 54):
                if c != 25 and c != 34:  # excludes physiologic scotomas
                    y_row[0, c1] = worksheet.cell_value(r, c + vf_thv_col)
                    c1 = c1+1
            y_data = np.concatenate((y_data, y_row), axis=0)
    if len(y_data) == 0:
        print("Could not find ground truth visual field data.")
        success = False
    else:
        print("Printing ground truth y_data:")
        print(y_data[0])
    return x_data_zeiss, x_data_topcon, y_data, success


# Data loading ===============================
print("Loading OCT images and visual field data...")
x_zeiss, x_topcon, y_test, success = LoadData()

if success is False:
    print("Error occured.")
else:
    # model build ================================
    print("Building AI model...")
    model_zeiss = GetModel24('zeiss')
    model_topcon = GetModel24('topcon')

    # compile the model (should be done *after* setting layers to non-trainable)
    model_zeiss.load_weights(base_folder + weight_file_zeiss)
    model_topcon.load_weights(base_folder + weight_file_topcon)

    # Predict ===========================================
    print("Making inferences...")

    images = [load_img(base_folder + zeiss_oct_image), load_img(base_folder + topcon_oct_image)]

    pred_zeiss = model_zeiss.predict(x_zeiss)
    pred_topcon = model_topcon.predict(x_topcon)

    # Draw prediction
    fig = plt.figure(figsize=(10, 8))
    rows = 3
    cols = 2
    fig.suptitle("Visual field prediction from OCT")

    # draw Zeiss OCT
    fig.add_subplot(rows, cols, 1)
    plt.imshow(images[0])

    # draw Topcon OCT
    fig.add_subplot(rows, cols, 2)
    plt.imshow(images[1])

    # draw predicted visual field (Zeiss)
    fig.add_subplot(rows, cols, 3)
    plt.gca().text(10, 10, "Zeiss", fontsize=9, color='black')
    draw_visual_field(pred_zeiss[0], plt, start_pos=(0, 10))
    plt.axis("scaled")

    # draw predicted visual field (Topcon)
    fig.add_subplot(rows, cols, 4)
    plt.gca().text(10, 10, "Topcon", fontsize=9, color='black')
    draw_visual_field(pred_topcon[0], plt, start_pos=(0, 10))
    plt.axis("scaled")

    # draw ground-truth visual field
    fig.add_subplot(rows, cols, 5)
    plt.gca().text(10, 10, "Truth", fontsize=9, color='black')
    draw_visual_field(y_test[0], plt, start_pos=(0, 10))
    plt.axis("scaled")

    plt.show()

