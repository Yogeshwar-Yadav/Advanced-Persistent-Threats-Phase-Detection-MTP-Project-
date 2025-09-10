# import datagenerator_v2 as datagenerator
# import modelgenerator
# import math
# from keras.models import Model, model_from_json
# from keras.callbacks import ModelCheckpoint, EarlyStopping
# from numpy.testing import assert_allclose
# from keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
# from keras.utils.vis_utils import plot_model
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import os

# # Specify which GPU(s) to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0

# # On CPU/GPU placement
# config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# config.gpu_options.allow_growth = True
# tf.compat.v1.Session(config=config)

# #dataset_train = datagenerator.loadDataset("/home/ubuntu/datasetEvaluation/attackFilesExtractor/CIC-IDS-2017/Monday-WorkingHours.pcap_ISCX_Cleaned.csv")

# dataset_train = datagenerator.loadDataset("/home/ubuntu/data/unb15_normal.csv")
# datasetType = 'unb15'
# _nTotal = dataset_train.shape[0]
# _nColumns = dataset_train.shape[1]
# #Using 75% of the data for training and remaining 25% for validation testing
# _nSamplesTrain = math.ceil(_nTotal * 0.75)
# _nSamplesValidation = _nTotal - _nSamplesTrain
# _nTimesteps = 3

# X_train  = datagenerator.getEncoderInput(datasetType, dataset_train, 0, _nSamplesTrain, _nColumns)
# #X_train = datagenerator.getEncoderInput(dataset_train, 0, _nSamplesTrain, _nColumns)
# #y = datagenerator.getEncoderLabelCoulmn(dataset_train, 0, _nSamplesTrain, _nColumns)
# y = datagenerator.getEncoderLabelCoulmn(datasetType, dataset_train, 0, _nSamplesTrain)

# Validation_X = datagenerator.getEncoderInput(datasetType, dataset_train, _nSamplesTrain, _nSamplesTrain+_nSamplesValidation, _nColumns)

# # Feature Scaling -Normalization recommended for RNN    
# sc = MinMaxScaler(feature_range = (0, 1))
# X_train = sc.fit_transform(X_train)
# Validation_X = sc.fit_transform(Validation_X)

# #Converting training inputs into LSTM training inputs
# _nOperatingColumns = len(X_train[0])
# X_train_sequence = datagenerator.getEncoderInputSequence(X_train, _nTimesteps, _nOperatingColumns)
# Validation_X_sequence = datagenerator.getEncoderInputSequence(Validation_X, _nTimesteps, _nOperatingColumns)

# sequence_autoencoder_semi = modelgenerator.getSAE_LSTM(_nTimesteps, _nOperatingColumns)
# sequence_autoencoder_semi.compile(optimizer='adam', loss='mean_squared_error')

# #modelName = "Round1_Monday_AllS"
# modelName = "unb15_model"

# #Adding checkpoints
# checkpointFile = "/home/ubuntu/aitd/" + modelName + ".h5"
# checkpoint = ModelCheckpoint(checkpointFile, monitor='loss', verbose=1, save_best_only=True, mode='min')
# earlyStopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
# callbacks_list = [checkpoint, earlyStopping]
# # callbacks_list = [checkpoint]

# #sequence_autoencoder_semi = load_model(checkpointFile)
# #sequence_autoencoder_semi.summary()
# plot_model(sequence_autoencoder_semi, to_file='sae_lstm_model_plot.png', show_shapes=True, show_layer_names=True)
# #Training autoencoder
# sequence_autoencoder_semi_history = sequence_autoencoder_semi.fit(X_train_sequence, X_train_sequence,
#                      epochs=32,
#                      batch_size=10,
#                      shuffle=False,
#                      validation_data=(Validation_X_sequence, 
#                      Validation_X_sequence), 
#                      callbacks=callbacks_list)



# # Save the model and serialize model to JSON and h5
# sequence_autoencoder_semi.save("/home/ubuntu/aitd/" + modelName + ".h5")
# print("Saved model to disk")


# loss = sequence_autoencoder_semi_history.history['loss']
# val_loss = sequence_autoencoder_semi_history.history['val_loss']
# epochs = range(len(loss))  #Auto-matches number of epochs
# plt.figure()
# plt.plot(epochs, loss, color='red', label='Training loss')
# plt.plot(epochs, val_loss, color='blue', label='Validation loss')
# plt.title('Training and Validation loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend()
# #plt.savefig('LossColored_' + modelName + '.png')
# plt.show()








































# import datagenerator_v2 as datagenerator
# import modelgenerator
# import math
# import os
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt

# from keras.models import Model
# from keras.callbacks import ModelCheckpoint, EarlyStopping
# from keras.utils.vis_utils import plot_model
# from sklearn.preprocessing import MinMaxScaler

# # ----------------------------
# # GPU Selection (Optional)
# # ----------------------------
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# config.gpu_options.allow_growth = True
# tf.compat.v1.Session(config=config)

# # ----------------------------
# # Dataset and Path Settings
# # ----------------------------
# datasetType = "custom"
# data_path = "/home/yogeshwar/Yogesh-MTP/csv/Cleaned"
# modelName = "custom_sae_lstm_model"
# checkpoint_dir = "/home/yogeshwar/Yogesh-MTP/Results/sae_lstm/"
# checkpointFile = os.path.join(checkpoint_dir, modelName + ".h5")

# os.makedirs(checkpoint_dir, exist_ok=True)

# # ----------------------------
# # Load and Prepare Dataset
# # ----------------------------
# dataset_train = datagenerator.loadDataset(os.path.join(data_path, datasetType + "_normal.csv"))
# _nTotal = dataset_train.shape[0]
# _nColumns = dataset_train.shape[1]
# _nSamplesTrain = math.ceil(_nTotal * 0.75)
# _nSamplesValidation = _nTotal - _nSamplesTrain
# _nTimesteps = 3

# X_train = datagenerator.getEncoderInput(datasetType, dataset_train, 0, _nSamplesTrain, _nColumns)
# y_train = datagenerator.getEncoderLabelCoulmn(datasetType, dataset_train, 0, _nSamplesTrain)
# Validation_X = datagenerator.getEncoderInput(datasetType, dataset_train, _nSamplesTrain, _nTotal, _nColumns)

# # ----------------------------
# # Normalize Features
# # ----------------------------
# sc = MinMaxScaler(feature_range=(0, 1))
# X_train = sc.fit_transform(X_train)
# Validation_X = sc.transform(Validation_X)

# # ----------------------------
# # Reshape into Sequences
# # ----------------------------
# _nOperatingColumns = X_train.shape[1]
# X_train_seq = datagenerator.getEncoderInputSequence(X_train, _nTimesteps, _nOperatingColumns)
# Validation_X_seq = datagenerator.getEncoderInputSequence(Validation_X, _nTimesteps, _nOperatingColumns)

# # ----------------------------
# # Build SAE-LSTM Model
# # ----------------------------
# model = modelgenerator.getSAE_LSTM(_nTimesteps, _nOperatingColumns)
# model.compile(optimizer='adam', loss='mean_squared_error')
# plot_model(model, to_file='sae_lstm_model_plot.png', show_shapes=True, show_layer_names=True)

# # ----------------------------
# # Callbacks
# # ----------------------------
# checkpoint = ModelCheckpoint(checkpointFile, monitor='loss', verbose=1, save_best_only=True, mode='min')
# earlyStopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
# callbacks_list = [checkpoint, earlyStopping]

# # ----------------------------
# # Train Model
# # ----------------------------
# history = model.fit(
#     X_train_seq, X_train_seq,
#     epochs=32,
#     batch_size=10,
#     shuffle=False,
#     validation_data=(Validation_X_seq, Validation_X_seq),
#     callbacks=callbacks_list
# )

# # ----------------------------
# # Save Final Model
# # ----------------------------
# model.save(checkpointFile)
# print(f"Saved model to disk at: {checkpointFile}")

# # ----------------------------
# # Plot Training Curve
# # ----------------------------
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(len(loss))

# plt.figure()
# plt.plot(epochs, loss, color='red', label='Training loss')
# plt.plot(epochs, val_loss, color='blue', label='Validation loss')
# plt.title('Training and Validation Loss (SAE-LSTM)')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig(os.path.join(checkpoint_dir, 'Loss_' + modelName + '.png'))
# plt.show()




import sys
sys.path.append("/home/yogeshwar/Yogesh-MTP/code") 

import datagenerator_v2 as datagenerator
import modelgenerator
import math
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# GPU: Enable usage of all available GPUs
# ----------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(f"Using {len(gpus)} GPUs")

# ----------------------------
# Input from User
# ----------------------------
data_path = input("Enter the full path to the folder containing `custom_normal.csv`:\n").strip()
datasetType = "custom"
modelName = "custom_sae_lstm_model"
code_folder = "/home/yogeshwar/Yogesh-MTP/code/"
checkpointFile = os.path.join(code_folder, modelName + ".h5")
plotFile = os.path.join(code_folder, 'Loss_' + modelName + '.png')

# ----------------------------
# Load and Prepare Dataset
# ----------------------------
dataset_train = datagenerator.loadDataset(os.path.join(data_path, datasetType + "_normal.csv"))
_nTotal = dataset_train.shape[0]
_nColumns = dataset_train.shape[1]
_nSamplesTrain = math.ceil(_nTotal * 0.75)
_nSamplesValidation = _nTotal - _nSamplesTrain
_nTimesteps = 3

X_train = datagenerator.getEncoderInput(datasetType, dataset_train, 0, _nSamplesTrain, _nColumns)
y_train = datagenerator.getEncoderLabelColumn(datasetType, dataset_train, 0, _nSamplesTrain)
Validation_X = datagenerator.getEncoderInput(datasetType, dataset_train, _nSamplesTrain, _nTotal, _nColumns)

# ----------------------------
# Normalize Features
# ----------------------------
sc = MinMaxScaler(feature_range=(0, 1))
X_train = sc.fit_transform(X_train)
Validation_X = sc.transform(Validation_X)

# ----------------------------
# Reshape into Sequences
# ----------------------------
_nOperatingColumns = X_train.shape[1]
X_train_seq = datagenerator.getEncoderInputSequence(X_train, _nTimesteps, _nOperatingColumns)
Validation_X_seq = datagenerator.getEncoderInputSequence(Validation_X, _nTimesteps, _nOperatingColumns)

# ----------------------------
# Build SAE-LSTM Model
# ----------------------------
model = modelgenerator.getSAE_LSTM(_nTimesteps, _nOperatingColumns)
model.compile(optimizer='adam', loss='mean_squared_error')
plot_model(model, to_file=os.path.join(code_folder, 'sae_lstm_model_plot.png'), show_shapes=True, show_layer_names=True)

# ----------------------------
# Callbacks
# ----------------------------
checkpoint = ModelCheckpoint(checkpointFile, monitor='loss', verbose=1, save_best_only=True, mode='min')
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
callbacks_list = [checkpoint, earlyStopping]

# ----------------------------
# Train Model
# ----------------------------
history = model.fit(
    X_train_seq, X_train_seq,
    epochs=32,
    batch_size=10,
    shuffle=False,
    validation_data=(Validation_X_seq, Validation_X_seq),
    callbacks=callbacks_list
)

# ----------------------------
# Save Final Model
# ----------------------------
model.save(checkpointFile)
print(f"\nModel saved at: {checkpointFile}")

# ----------------------------
# Plot Training Curve
# ----------------------------
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))

plt.figure()
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='blue', label='Validation loss')
plt.title('Training and Validation Loss (SAE-LSTM)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(plotFile)
plt.show()

print(f"\nLoss plot saved at: {plotFile}")
