# Importing the libraries
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ipaddress
import time
from datetime import datetime
from keras.layers import Input, LSTM, RepeatVector, Dense 
from sklearn.preprocessing import MinMaxScaler
#regularizers
from keras.models import Model
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
import tensorflow as tf
import os

modelName = "unb15"
# Specify which GPU(s) to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0

# On CPU/GPU placement
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

#Functions
noise_factor = 0.5
#DDoS Features were found to be 22, 23, 25, 27, 30, 32, 58, 79. So removed them. 
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#Part 2 - Importing training and test sets
#-------------------------------------------------------------------------------------------------------------------------------------------------------------

input_file = "/home/ubuntu/data/unb15_normal.csv"
dataset_train = datagenerator.loadDataset(input_file)
_nTotal = dataset_train.shape[0]
_nSamplesTrain = math.ceil(_nTotal * 0.75)
_nSamplesValidation = _nTotal - _nSamplesTrain
_nColumns = dataset_train.shape[1]
_nTimesteps = 3
X_train = datagenerator.getEncoderInput(dataset_train, 0, _nSamplesTrain, _nColumns)
y = datagenerator.getEncoderLabelColumn(dataset_train, 0, _nSamplesTrain)
X_train_noisy = datagenerator.addNoise(X_train)

Validation_X = datagenerator.getEncoderInput(dataset_train, _nSamplesTrain, _nSamplesTrain+_nSamplesValidation, _nColumns)
Validation_y = datagenerator.getEncoderLabelColumn(dataset_train, _nSamplesTrain, _nSamplesTrain+_nSamplesValidation)
#Validation_X_noisy = addNoise(Validation_X)

# Feature Scaling -Normalization recommended for RNN    
sc = MinMaxScaler(feature_range = (0, 1))
X_train = sc.fit_transform(X_train)
Validation_X = sc.fit_transform(Validation_X)
X_train_noisy = sc.fit_transform(X_train_noisy)

#Validation_X_noisy = sc.fit_transform(Validation_X_noisy)

#Converting training inputs into LSTM training inputs
#X_train_sequence = getEncoderInputSequence(X_train, _nTimesteps, _nOperatingColumns)
#X_train_sequence_noisy = getEncoderInputSequence(X_train_noisy, _nTimesteps, _nOperatingColumns)

#Validation_X_sequence = getEncoderInputSequence(Validation_X, _nTimesteps, _nOperatingColumns)
#Validation_X_sequence_noisy = getEncoderInputSequence(Validation_X_noisy, _nTimesteps, _nOperatingColumns)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#Part 2 - Building autoencoder
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
# this is the size of our encoded representations
encoding_dim1 = 60 
encoding_dim2 = 35
encoding_dim3 = 20
_nOperatingColumns = 44

# this is our input placeholder
input = Input(shape=(_nOperatingColumns-1, ))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim1, activation = 'relu')(input)

encoded = Dense(encoding_dim2, activation = 'relu')(encoded)

encoded = Dense(encoding_dim3, activation = 'relu')(encoded)

decoded = Dense(encoding_dim2, activation = 'relu')(encoded)

decoded = Dense(encoding_dim1, activation = 'relu')(decoded)

decoded = Dense(_nOperatingColumns-1, activation = 'sigmoid')(decoded)

# this model maps an input to its reconstruction
sequence_autoencoder = Model(input, decoded)

# this model maps an input to its encoded representation
encoder = Model(input,encoded)
encoder.summary()
sequence_autoencoder.summary()
sequence_autoencoder.compile(optimizer='adam', loss='mean_squared_error')

#Part 3 - Training autoencoder
sequence_autoencoder_history = sequence_autoencoder.fit(X_train, X_train,
                     epochs=10,
                     batch_size=10,
                     shuffle=False,
                     validation_data=(Validation_X, Validation_X))

# Save the model
# serialize model to JSON
model_json = sequence_autoencoder.to_json()
with open(modelName + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
sequence_autoencoder.save_weights(modelName + ".h5")
print("Saved model to disk")

loss = sequence_autoencoder_history.history['loss'] #Training loss
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


#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#Part 3 - Predicting 
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
# Importing the libraries

# load json and create model
json_file = open(modelName + ".json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(modelName + ".h5")
print("Loaded model from disk")

#Prediction set
_nSamplesPred = 100000

_nOperatingColumns = 45

#New dataset for prediction after training
#fileType = "PortScan_"
#dataset_test = loadDataset("..\\..\\Datasets\\iscxdownloads\\CIC-IDS-2017\\GeneratedLabelledFlows\\TrafficLabelling\\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX_Cleaned.csv")
fileType = "Ïnfilteration_"
dataset_test = datagenerator.loadDataset("/home/ubuntu/data/unb15_portscan.csv")
#fileType = "Tuesday_"
#dataset_test = loadDataset("..\\..\\Datasets\\iscxdownloads\\CIC-IDS-2017\\GeneratedLabelledFlows\\TrafficLabelling\\Tuesday-WorkingHours.pcap_ISCX_Cleaned.csv")
#fileType = "WebAttacks_"
#dataset_test = loadDataset("..\\..\\Datasets\\iscxdownloads\\CIC-IDS-2017\\GeneratedLabelledFlows\\TrafficLabelling\\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX_Cleaned.csv")
_nColumns = dataset_test.shape[1]
X_test = datagenerator.getEncoderInput(dataset_test, 0, _nSamplesPred, _nColumns)
y_test = datagenerator.getEncoderLabelColumn(dataset_test, 0, _nSamplesPred)
#X_test_noisy = addNoise(X_test)

# Feature Scaling -Normalization recommended for RNN    
from sklearn.preprocessing import MinMaxScaler
sc_pred = MinMaxScaler(feature_range = (0, 1))
X_test = sc_pred.fit_transform(X_test)
#X_test_noisy = sc_pred.fit_transform(X_test_noisy)

#prediction = loaded_model.predict(X_test)
X_pred = loaded_model.predict(X_test)

mse = np.mean(np.power(X_test - X_pred, 2), axis=1)
rmse = np.sqrt(mse)
validation_loss = 0.00079049  #00079049e       
y_pred = rmse >  0.05 #fixed threshold for portscan detection
#y_pred = mse > 0.00079049 #fixed threshold for portscan detection                                                          

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)        

data = pd.DataFrame(X_test[0:_nSamplesPred, :])
data_n = pd.DataFrame(X_test)
data_n = data_n.astype('float32')

dist = np.zeros(_nSamplesPred)
for i, x in enumerate(data_n.iloc[0:_nSamplesPred, :].values):
    dist[i] = np.linalg.norm(X_pred[i, :])


fpr, tpr, thresholds = roc_curve(y_test, y_pred)

np.savetxt(modelName+'-portscan-fpr.csv', fpr, delimiter="\n")
np.savetxt(modelName+'-portscan-tpr.csv', tpr, delimiter="\n")
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='red', label='AUC = %0.2f)' % roc_auc)
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.plot([0, 1], [0, 1], color="navy", linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Autoencoder')
plt.legend(loc='lower right')
plt.savefig("ROC-curve_" + fileType + modelName + ".png")
plt.show()
