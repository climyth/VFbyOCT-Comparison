from keras.callbacks import ModelCheckpoint, TensorBoard
from DataLoad import *
from Model import *

# Setup ===========================================
vf_select = "24-2"  # 24-2 or 10-2
base_model_name = "InceptionResnet"   # ResNet, InceptionV3, VGG19, DenseNet, NASNet, InceptionResnet

oct_type = "zeiss"  # zeiss or topcon
base_folder = "Z:/PaperResearch/VFbySS+SD-OCT"
vf_file = "/SD_Train.xlsm"
weight_save_folder = "/Weights/" + base_model_name
graph_save_folder = ""
pretrained_weights = "/InceptionResnet_24-2-improvement-324-3.86-60.94.hdf5"   # if not exist, leave ""
tensorboard_log_folder = "/Logs"
# ==================================================

if vf_select == "24-2":
    # Data loading ===============================
    print("Data loading...")
    x_train, y_train, pids = LoadData24(base_folder, base_folder + vf_file, True, oct_type, "Train", "24-2")
    # model build ================================
    model = GetModel24(oct_type)
    if pretrained_weights != "":
        model.load_weights(base_folder + weight_save_folder + pretrained_weights)
    # plot_model(model, to_file=base_folder + graph_save_folder+"/graph24-2.png", show_shapes=True, show_layer_names=True)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='mean_squared_error')

    # checkpoint
    filepath = base_folder + weight_save_folder + "/" + base_model_name + "_24-2-improvement-{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(log_dir=tensorboard_log_folder, histogram_freq=1, write_graph=True, write_images=True)
    callbacks_list = [checkpoint, tensorboard]

    # Train ===========================================
    model.fit(x_train, y_train, batch_size=32, validation_split=0.1, epochs=10000, callbacks=callbacks_list)

