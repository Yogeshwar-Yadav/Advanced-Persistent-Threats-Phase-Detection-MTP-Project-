# Importing the libraries
import numpy as np
import pandas as pd
import ipaddress

#Variables
noise_factor = 0.5
sample_size = 65000
#0 => Normal, 1 => Attack

#Load the dataset
def loadDataset(datasetFilePath):
    return pd.read_csv(datasetFilePath, header=0, encoding="ISO-8859-1")

#DDoS Features were found to be 22, 23, 25, 27, 30, 32, 58, 79. So removed them along with the flow ID and the timestamp
# def getEncoderInput(datasetType, dataset, start, nSamples, nColumns):
#     if 'unb15' in datasetType or 'custom' in datasetType:    
#         X = dataset.iloc[start:nSamples, 0:nColumns-2].values    
#     else:
#         X = dataset.iloc[start:nSamples, 0:nColumns-1].values   
 

#     if 'cicids2017' in datasetType:
#         X = np.delete(X, [0, 6, 22, 23, 25, 27, 30, 32, 58, 79], axis=1)
#         for i in range(len(X)):
#             X[i, 0] = int(ipaddress.ip_address(X[i, 0]))
#             X[i, 2] = int(ipaddress.ip_address(X[i, 2]))
#         #X[i, 5] = int(X[i, 5])
#     elif 'cicids2018' in datasetType:
#         X = np.delete(X, [2, 3, 18, 19, 21, 23, 26, 28, 54, 75], axis=1)

#     # Randomly sampling code - Sailik
#     #X = dt[np.random.choice(dt.shape[0], sample_size , replace=False), :] 
#     return X


def getEncoderInput(datasetType, dataset, start, nSamples, nColumns):
    if 'unb15' in datasetType or 'custom' in datasetType:    
        X = dataset.iloc[start:nSamples, 0:nColumns-2].values
        # Convert IP addresses to integers before dropping columns
        for i in range(len(X)):
            # Source IP (column 1 before dropping)
            X[i, 1] = int(ipaddress.ip_address(X[i, 1]))
            # Destination IP (column 3 before dropping)
            X[i, 3] = int(ipaddress.ip_address(X[i, 3]))
        # Drop columns
        X = np.delete(X, [0, 6, 22, 23, 25, 27, 30, 32, 58, 79], axis=1)
    else:
        X = dataset.iloc[start:nSamples, 0:nColumns-1].values   

    if 'cicids2017' in datasetType:
        X = np.delete(X, [0, 6, 22, 23, 25, 27, 30, 32, 58, 79], axis=1)
        for i in range(len(X)):
            X[i, 0] = int(ipaddress.ip_address(X[i, 0]))
            X[i, 2] = int(ipaddress.ip_address(X[i, 2]))
    elif 'cicids2018' in datasetType:
        X = np.delete(X, [2, 3, 18, 19, 21, 23, 26, 28, 54, 75], axis=1)

    return X


def getEncoderLabelColumn(datasetType, dataset, start, nSamples):
    if 'unb15' in datasetType or 'custom' in datasetType:    
        labelIndex = -2    
    else:
        labelIndex = -1
    y = dataset.iloc[start:nSamples, labelIndex].values 
    integerY = []
    for i in range(len(y)):
         integerY.append(int(str.lower(str(y[i])) != "benign" and str.lower(str(y[i])) != "normal"))
    y = np.array(integerY)    
    y = np.reshape(y, (y.shape[0], 1))
    return y

#Add noise in case you want to make your model to still be able to contruct the original input (a process known as denoising). Resulting model will fall under Denoising Stacked Autoencoders
def addNoise(X):
     return X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)    

#If you are using LSTM in your stacked encoder, you have to convert the input into sequences
def getEncoderInputSequence(X, nTimesteps, nColumns):
    X_sequence = []
    for i in range(nTimesteps, np.shape(X)[0]):
        X_sequence.append(X[i-nTimesteps:i, :])  
    X_sequence = np.array(X_sequence)
    X_sequence = np.reshape(X_sequence, (X_sequence.shape[0], X_sequence.shape[1], nColumns))
    return X_sequence
