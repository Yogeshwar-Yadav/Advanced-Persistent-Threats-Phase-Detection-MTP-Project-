# import datagenerator_v2 as datagenerator
# import numpy as np
# import random
# #-------------------------------------------------------------------------------------------------------------------------------------------------------------
# #Part 3 - Predicting 
# #-------------------------------------------------------------------------------------------------------------------------------------------------------------

# from keras.models import load_model
# #modelName = "APT_SAELSTM_InitialModel_Monday_CICIDS2017"
# modelName = "unb15_model"
# #Loads a compiled and saved model
# sequence_autoencoder_semi = load_model("/home/ubuntu/aitd/" + modelName + ".h5")
# print("Loaded model from disk")


# testDatasetFile = input("Please enter the path to the test dataset: \n")
# dataset_test = datagenerator.loadDataset(testDatasetFile)

# _nSamplesPred = dataset_test.shape[0]
# _nColumns = dataset_test.shape[1]
# _nTimesteps = 3

# X_test = datagenerator.getEncoderInput(dataset_test, 0, _nSamplesPred, _nColumns)
# y_test = datagenerator.getEncoderLabelCoulmn(dataset_test, 0, _nSamplesPred-3)

# # Feature Scaling -Normalization recommended for RNN    
# from sklearn.preprocessing import MinMaxScaler
# sc_pred = MinMaxScaler(feature_range = (0, 1))
# X_test = sc_pred.fit_transform(X_test)

# #Converting prediction inputs into LSTM prediction inputs
# _nOperatingColumns = len(X_test[0])
# X_test_sequence = datagenerator.getEncoderInputSequence(X_test, _nTimesteps, _nOperatingColumns)
# #X_test_sequence_noisy = getEncoderInputSequence(X_test_noisy, _nTimesteps, _nOperatingColumns)

# prediction_sequence = sequence_autoencoder_semi.predict(X_test_sequence)

# #Removing timesteps in prediction result
# prediction_result = []
# prediction_input = []
# for i in range(len(prediction_sequence)):
#     prediction_input.append(X_test_sequence[i, 0, :])
#     prediction_result.append(prediction_sequence[i, 0, :])
# prediction_input, prediction_result = np.array(prediction_input), np.array(prediction_result)

# reconstruction_error = []
# for i in range(len(prediction_result)):
#     current_record = []
#     for j in range(len(prediction_result[0])):
#         current_record.append(np.mean(np.power(prediction_input[i, j] - prediction_result[i, j], 2)))
#     reconstruction_error.append(np.array(current_record))
# outputFileName = "ReconstructionError_" + modelName + ".csv"
# np.savetxt(outputFileName, np.array(reconstruction_error), delimiter=',', fmt="%s" )
# print("Your file has been saved in this same folder under the name : " + outputFileName)

                                      
# from sklearn.metrics import roc_curve, auc
# import matplotlib.pyplot as plt
# import pandas as pd

# data = pd.DataFrame(X_test[0:_nSamplesPred-_nTimesteps, :])
# data_n = pd.DataFrame(X_test)
# data_n = data_n.astype('float32')

# dist = np.zeros(_nSamplesPred-_nTimesteps)
# for i, x in enumerate(data_n.iloc[0:_nSamplesPred-_nTimesteps, :].values):
#     dist[i] = np.linalg.norm(prediction_result[i, :])
    
# fpr, tpr, thresholds = roc_curve(y_test, dist)
# roc_auc = auc(fpr, tpr)

# np.savetxt(modelName+'-fpr.csv', fpr, delimiter="\n")
# np.savetxt(modelName+'-tpr.csv', tpr, delimiter="\n")

# '''plt.figure()
# plt.plot(fpr, tpr, color='red', label='AUC = %0.2f)' % roc_auc)
# plt.xlim((0, 1))
# plt.ylim((0, 1))
# plt.plot([0, 1], [0, 1], color="navy", linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Autoencoder')
# plt.legend(loc='lower right')
# plt.savefig( testDatasetFile +"ROC-curve_" + "_" + modelName + ".png")
# plt.show()
# print(testDatasetFile)
# '''




import datagenerator_v2 as datagenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ---------------------------- Load model -----------------------------------
model_path = "/home/yogeshwar/Yogesh-MTP/code/custom_sae_lstm_model.h5"
modelName = "custom_sae_lstm_model"
datasetType = "custom"
sequence_autoencoder_semi = load_model(model_path)
print("Loaded model from:", model_path)

# -------------------------- Ask for dataset and results folder -------------------------
testDatasetFile = input("Enter the full path to the test CSV file (e.g., reconnaissance.csv):\n")
results_folder = "/home/yogeshwar/Yogesh-MTP/Results_New/SAE-LSTM/exfilteration/"
os.makedirs(results_folder, exist_ok=True)

# ---------------------------- Load test data --------------------------------
dataset_test = datagenerator.loadDataset(testDatasetFile)
_nSamplesPred = dataset_test.shape[0]
_nColumns = dataset_test.shape[1]
_nTimesteps = 3

X_test = datagenerator.getEncoderInput(datasetType,dataset_test, 0, _nSamplesPred, _nColumns)
y_test = datagenerator.getEncoderLabelColumn(datasetType,dataset_test, 0, _nSamplesPred - _nTimesteps)

# ---------------------------- Preprocess & Sequence --------------------------
sc_pred = MinMaxScaler(feature_range=(0, 1))
X_test = sc_pred.fit_transform(X_test)

_nOperatingColumns = len(X_test[0])
X_test_sequence = datagenerator.getEncoderInputSequence(X_test, _nTimesteps, _nOperatingColumns)

# ---------------------------- Make Predictions -------------------------------
prediction_sequence = sequence_autoencoder_semi.predict(X_test_sequence)

prediction_result = np.array([seq[0] for seq in prediction_sequence])
prediction_input = np.array([seq[0] for seq in X_test_sequence])

# ----------------------- Compute Reconstruction Error ------------------------
reconstruction_error = np.mean(np.power(prediction_input - prediction_result, 2), axis=1)

output_file_path = os.path.join(results_folder, f"ReconstructionError_{modelName}.csv")
np.savetxt(output_file_path, reconstruction_error, delimiter=",", fmt="%s")
print(f"Reconstruction error saved to: {output_file_path}")

# ---------------------------- ROC Curve -------------------------------------
dist = np.linalg.norm(prediction_result, axis=1)
fpr, tpr, thresholds = roc_curve(y_test, dist)
roc_auc = auc(fpr, tpr)

np.savetxt(os.path.join(results_folder, f"{modelName}_fpr.csv"), fpr, delimiter="\n")
np.savetxt(os.path.join(results_folder, f"{modelName}_tpr.csv"), tpr, delimiter="\n")

plt.figure()
plt.plot(fpr, tpr, color='red', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SAE-LSTM')
plt.legend(loc='lower right')
plt.savefig(os.path.join(results_folder, f"ROC_curve_{modelName}.png"))
plt.close()
print(f"ROC Curve saved to: ROC_curve_{modelName}.png")

# --------------------------- PR Curve ---------------------------------------
precision, recall, _ = precision_recall_curve(y_test, dist)
pr_auc = average_precision_score(y_test, dist)

plt.figure()
plt.plot(recall, precision, color='green', label=f'AP = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve - SAE-LSTM')
plt.legend(loc='lower left')
plt.savefig(os.path.join(results_folder, f"PR_curve_{modelName}.png"))
plt.close()
print(f"PR Curve saved to: PR_curve_{modelName}.png")
