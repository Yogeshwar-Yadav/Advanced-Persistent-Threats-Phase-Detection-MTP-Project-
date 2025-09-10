# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import ipaddress
# import time
# from datetime import datetime
# from keras.layers import Input, LSTM, RepeatVector, Dense 
# from sklearn.preprocessing import MinMaxScaler
# #regularizers
# from keras.models import Model
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import ipaddress
# import time
# from datetime import datetime
# from keras.layers import Input, Dense
# from keras.models import Model, model_from_json
# import math
# import datagenerator

# #Functions
# noise_factor = 0.5 #Used to add noise to training data
# modelName = "cicids2017"

# #DDoS Features were found to be 22, 23, 25, 27, 30, 32, 58, 79. So removed them. 
# #-------------------------------------------------------------------------------------------------------------------------------------------------------------
# #Part 2 - Importing training and test sets
# #-------------------------------------------------------------------------------------------------------------------------------------------------------------

# input_file = "/home/ubuntu/data/cicids2017_normal.csv"
# dataset_train = datagenerator.loadDataset(input_file)
# _nTotal = dataset_train.shape[0]
# _nSamplesTrain = math.ceil(_nTotal * 0.75)
# _nSamplesValidation = _nTotal - _nSamplesTrain
# _nColumns = dataset_train.shape[1]
# _nTimesteps = 3
# X_train = datagenerator.getEncoderInput(dataset_train, 0, _nSamplesTrain, _nColumns)
# y = datagenerator.getEncoderLabelCoulmn(dataset_train, 0, _nSamplesTrain)
# X_train_noisy = datagenerator.addNoise(X_train)

# Validation_X = datagenerator.getEncoderInput(dataset_train, _nSamplesTrain, _nSamplesTrain+_nSamplesValidation, _nColumns)
# Validation_y = datagenerator.getEncoderLabelCoulmn(dataset_train, _nSamplesTrain, _nSamplesTrain+_nSamplesValidation)
# #Validation_X_noisy = addNoise(Validation_X)

# # Feature Scaling -Normalization recommended for RNN    
# sc = MinMaxScaler(feature_range = (0, 1))
# X_train = sc.fit_transform(X_train)
# Validation_X = sc.fit_transform(Validation_X)
# X_train_noisy = sc.fit_transform(X_train_noisy)
# #Validation_X_noisy = sc.fit_transform(Validation_X_noisy)

# #Converting training inputs into LSTM training inputs
# #X_train_sequence = getEncoderInputSequence(X_train, _nTimesteps, _nOperatingColumns)
# #X_train_sequence_noisy = getEncoderInputSequence(X_train_noisy, _nTimesteps, _nOperatingColumns)

# #Validation_X_sequence = getEncoderInputSequence(Validation_X, _nTimesteps, _nOperatingColumns)
# #Validation_X_sequence_noisy = getEncoderInputSequence(Validation_X_noisy, _nTimesteps, _nOperatingColumns)

# #-------------------------------------------------------------------------------------------------------------------------------------------------------------
# #Part 2 - Building autoencoder
# #-------------------------------------------------------------------------------------------------------------------------------------------------------------
# # this is the size of our encoded representations
# encoding_dim1 = 60 
# encoding_dim2 = 35
# encoding_dim3 = 20
# _nOperatingColumns = 75

# # this is our input placeholder
# input = Input(shape=(_nOperatingColumns-1, ))
# # "encoded" is the encoded representation of the input
# encoded = Dense(encoding_dim1, activation = 'relu')(input)

# encoded = Dense(encoding_dim2, activation = 'relu')(encoded)

# encoded = Dense(encoding_dim3, activation = 'relu')(encoded)

# decoded = Dense(encoding_dim2, activation = 'relu')(encoded)

# decoded = Dense(encoding_dim1, activation = 'relu')(decoded)

# decoded = Dense(_nOperatingColumns-1, activation = 'sigmoid')(decoded)

# # this model maps an input to its reconstruction
# sequence_autoencoder = Model(input, decoded)

# # this model maps an input to its encoded representation
# encoder = Model(input,encoded)
# encoder.summary()
# sequence_autoencoder.summary()

# sequence_autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# #Part 3 - Training autoencoder
# sequence_autoencoder_history = sequence_autoencoder.fit(X_train, X_train,
#                      epochs=10,
#                      batch_size=10,
#                      shuffle=False,
#                      validation_data=(Validation_X, Validation_X))

# # Save the model
# # serialize model to JSON
# model_json = sequence_autoencoder.to_json()
# with open(modelName + ".json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# sequence_autoencoder.save_weights(modelName + ".weights.h5")
# print("Saved model to disk")

# loss = sequence_autoencoder_history.history['loss']
# val_loss = sequence_autoencoder_history.history['val_loss']
# epochs = range(10)
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and Validation loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend()
# plt.savefig('Loss_' + modelName + '.png')
# plt.show()

# #-------------------------------------------------------------------------------------------------------------------------------------------------------------
# #Part 3 - Predicting 
# #-------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Importing the libraries

# # load json and create model
# json_file = open(modelName + ".json", 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights(modelName + ".weights.h5")
# print("Loaded model from disk")

# #Prediction set
# _nSamplesPred = 100000
# _nColumns = 85
# _nOperatingColumns = 75

# #New dataset for prediction after training
# #fileType = "PortScan_"
# #dataset_test = loadDataset("..\\..\\Datasets\\iscxdownloads\\CIC-IDS-2017\\GeneratedLabelledFlows\\TrafficLabelling\\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX_Cleaned.csv")
# #fileType = "Ïnfilteration_"
# dataset_test = loadDataset("/home/ubuntu/data/cicids2017_portscan.csv")
# #fileType = "Tuesday_"
# #dataset_test = loadDataset("..\\..\\Datasets\\iscxdownloads\\CIC-IDS-2017\\GeneratedLabelledFlows\\TrafficLabelling\\Tuesday-WorkingHours.pcap_ISCX_Cleaned.csv")
# #fileType = "WebAttacks_"
# #dataset_test = loadDataset("..\\..\\Datasets\\iscxdownloads\\CIC-IDS-2017\\GeneratedLabelledFlows\\TrafficLabelling\\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX_Cleaned.csv")

# X_test = getEncoderInput(dataset_test, 0, _nSamplesPred, _nColumns)
# y_test = getEncoderLabelCoulmn(dataset_test, 0, _nSamplesPred)
# #X_test_noisy = addNoise(X_test)

# # Feature Scaling -Normalization recommended for RNN    
# from sklearn.preprocessing import MinMaxScaler
# sc_pred = MinMaxScaler(feature_range = (0, 1))
# X_test = sc_pred.fit_transform(X_test)

# #X_test_noisy = sc_pred.fit_transform(X_test_noisy)

# prediction = loaded_model.predict(X_test)
# """

# mse = np.mean(np.power(X_test - X_pred, 2), axis=1)
# rmse = np.sqrt(mse)
# validation_loss = 0.00079049  #00079049e       
# y_pred = rmse >  1.5

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)        
# print(cm)


# data = pd.DataFrame(X_test[0:_nSamplesPred, :])
# data_n = pd.DataFrame(X_test)
# data_n = data_n.astype('float32')

# dist = np.zeros(_nSamplesPred)
# for i, x in enumerate(data_n.iloc[0:_nSamplesPred, :].values):
#     dist[i] = np.linalg.norm(X_pred[i, :])
# """

# from sklearn.metrics import roc_curve, auc
# import matplotlib.pyplot as plt
# fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# roc_auc = auc(fpr, tpr)

# plt.figure()
# plt.plot(fpr, tpr, color='red', label='AUC = %0.2f)' % roc_auc)
# plt.xlim((0, 1))
# plt.ylim((0, 1))
# plt.plot([0, 1], [0, 1], color="navy", linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Autoencoder')
# plt.legend(loc='lower right')
# plt.savefig("ROC-curve_" + fileType + modelName + ".png")
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ipaddress
import time
from datetime import datetime
from keras.layers import Input, Dense
from keras.models import Model, model_from_json
import math
import datagenerator_v2 as datagenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
import os
import joblib

# Input prompts
datasetType = input("Please enter datasetType from below:\n\tcicids2017\n\tcicids2018\n\tcustom\n\tunb15\n")
dataPath = input("Please enter the folder path from where the files need to be picked up (without trailing slash)\n")
attackType = input("Please enter the attack type you want to predict from below:\n\treconnaissance\n\tfoothold_establishment\n\tlateral_movement\n\tdata_exfiltration\n")
shouldTrain = input("Do you want to do training? [y/n]\n")
modelName = f"{datasetType}_autoencoder"

# Load training data
input_file = f"{dataPath}/{datasetType}_normal.csv"
dataset_train = datagenerator.loadDataset(input_file)
_nTotal = dataset_train.shape[0]
_nSamplesTrain = math.ceil(_nTotal * 0.75)
_nSamplesValidation = _nTotal - _nSamplesTrain
_nColumns = dataset_train.shape[1]

X_train = datagenerator.getEncoderInput(datasetType, dataset_train, 0, _nSamplesTrain, _nColumns)
y = datagenerator.getEncoderLabelColumn(datasetType, dataset_train, 0, _nSamplesTrain)
Validation_X = datagenerator.getEncoderInput(datasetType, dataset_train, _nSamplesTrain, _nSamplesTrain+_nSamplesValidation, _nColumns)
Validation_y = datagenerator.getEncoderLabelColumn(datasetType, dataset_train, _nSamplesTrain, _nSamplesTrain+_nSamplesValidation)

# Dynamically set _nOperatingColumns
_nOperatingColumns = X_train.shape[1] + 1

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
if shouldTrain.lower() == 'y':
    X_train = sc.fit_transform(X_train)
    Validation_X = sc.transform(Validation_X)
else:
    if not os.path.exists(f"{datasetType}_scaler.sav"):
        raise FileNotFoundError(f"Scaler file {datasetType}_scaler.sav not found. Please train the model first.")
    sc = joblib.load(f"{datasetType}_scaler.sav")
    X_train = sc.transform(X_train)  # Needed for validation data in case of separate sessions
    Validation_X = sc.transform(Validation_X)

# Build autoencoder
encoding_dim1 = 60
encoding_dim2 = 35
encoding_dim3 = 20

input = Input(shape=(_nOperatingColumns-1,))
encoded = Dense(encoding_dim1, activation='relu')(input)
encoded = Dense(encoding_dim2, activation='relu')(encoded)
encoded = Dense(encoding_dim3, activation='relu')(encoded)
decoded = Dense(encoding_dim2, activation='relu')(encoded)
decoded = Dense(encoding_dim1, activation='relu')(decoded)
decoded = Dense(_nOperatingColumns-1, activation='sigmoid')(decoded)

sequence_autoencoder = Model(input, decoded)
encoder = Model(input, encoded)
encoder.summary()
sequence_autoencoder.summary()
sequence_autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train or load model
if shouldTrain.lower() == 'y':
    if os.path.exists(modelName + ".json") or os.path.exists(modelName + ".weights.h5"):
        print(f"Warning: Model files {modelName}.json and {modelName}.weights.h5 already exist and will be overwritten.")
    sequence_autoencoder_history = sequence_autoencoder.fit(X_train, X_train,
                                                            epochs=10,
                                                            batch_size=10,
                                                            shuffle=False,
                                                            validation_data=(Validation_X, Validation_X))
    model_json = sequence_autoencoder.to_json()
    with open(modelName + ".json", "w") as json_file:
        json_file.write(model_json)
    sequence_autoencoder.save_weights(modelName + ".weights.h5")
    joblib.dump(sc, f"{datasetType}_scaler.sav")  # Save scaler
    print("Saved model to disk")

    loss = sequence_autoencoder_history.history['loss']
    val_loss = sequence_autoencoder_history.history['val_loss']
    epochs = range(10)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('Loss_' + modelName + '.png')
    plt.show()
else:
    if not os.path.exists(modelName + ".json") or not os.path.exists(modelName + ".weights.h5"):
        raise FileNotFoundError(f"Model files {modelName}.json or {modelName}.weights.h5 not found. Please train the model first.")
    json_file = open(modelName + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(modelName + ".weights.h5")
    print("Loaded model from disk")
    loaded_model.compile(optimizer='adam', loss='mean_squared_error')

# Prediction
_nSamplesPred = 100000
_nColumns = 85

dataset_test = datagenerator.loadDataset(f"{dataPath}/{datasetType}_{attackType}.csv")
X_test = datagenerator.getEncoderInput(datasetType, dataset_test, 0, _nSamplesPred, _nColumns)
y_test = datagenerator.getEncoderLabelColumn(datasetType, dataset_test, 0, _nSamplesPred)
if X_test.shape[0] == 0:
    raise ValueError(f"Test dataset is empty! Check the file: {dataPath}/{datasetType}_{attackType}.csv")

X_test = sc.transform(X_test)

# Use the correct model for prediction based on shouldTrain
if shouldTrain.lower() == 'y':
    X_pred = sequence_autoencoder.predict(X_test)
else:
    X_pred = loaded_model.predict(X_test)
mse = np.mean(np.power(X_test - X_pred, 2), axis=1)
rmse = np.sqrt(mse)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, rmse)
roc_auc = auc(fpr, tpr)

from sklearn.metrics import precision_recall_curve, average_precision_score

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, rmse)
avg_precision = average_precision_score(y_test, rmse)

results_folder = "/home/yogeshwar/Yogesh-MTP/Results_New/SAE/exfilteration/"
os.makedirs(results_folder, exist_ok=True)  # ensure the folder exists

# Save Precision and Recall
np.savetxt(f"{results_folder}-{datasetType}-{attackType}-precision.csv", precision)
np.savetxt(f"{results_folder}-{datasetType}-{attackType}-recall.csv", recall)

# Plot PR Curve
plt.figure()
plt.plot(recall, precision, color='blue', label=f'Avg Precision = {avg_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve - Autoencoder ({attackType})')
plt.legend(loc='lower left')
plt.savefig(f"{results_folder}-PR-curve_{datasetType}_{attackType}.png")
plt.show()


# Save FPR/TPR
# os.makedirs("/home/yogeshwar/Yogesh-MTP/Results/autoencoder/exfilteration/", exist_ok=True)
np.savetxt(f"{results_folder}-{datasetType}-{attackType}-fpr.csv", fpr)
np.savetxt(f"{results_folder}-{datasetType}-{attackType}-tpr.csv", tpr)
print(f"Saved FPR/TPR to {results_folder}-{datasetType}-{attackType}-{{fpr,tpr}}.csv")

# Plot ROC
plt.figure()
plt.plot(fpr, tpr, color='red', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color="navy", linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Autoencoder - {attackType}')
plt.legend(loc='lower right')
plt.savefig(f"{results_folder}-ROC-curve_{datasetType}_{attackType}.png")
plt.show()