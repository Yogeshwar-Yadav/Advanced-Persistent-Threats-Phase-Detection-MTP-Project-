import datagenerator
import modelgenerator
import math
from keras.models import Model, model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping
from numpy.testing import assert_allclose
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

#dataset_train = datagenerator.loadDataset("/home/ubuntu/datasetEvaluation/attackFilesExtractor/CIC-IDS-2017/Monday-WorkingHours.pcap_ISCX_Cleaned.csv")

dataset_train = datagenerator.loadDataset("/home/ubuntu/data/cicids2017_normal.csv")

_nTotal = dataset_train.shape[0]
_nColumns = dataset_train.shape[1]
#Using 75% of the data for training and remaining 25% for validation testing
_nSamplesTrain = math.ceil(_nTotal * 0.75)
_nSamplesValidation = _nTotal - _nSamplesTrain
_nTimesteps = 3

X_train = datagenerator.getEncoderInput(dataset_train, 0, _nSamplesTrain, _nColumns)
y = datagenerator.getEncoderLabelCoulmn(dataset_train, 0, _nSamplesTrain)

Validation_X = datagenerator.getEncoderInput(dataset_train, _nSamplesTrain, _nSamplesTrain+_nSamplesValidation, _nColumns)

# Feature Scaling -Normalization recommended for RNN    
sc = MinMaxScaler(feature_range = (0, 1))
X_train = sc.fit_transform(X_train)
Validation_X = sc.fit_transform(Validation_X)

#Converting training inputs into LSTM training inputs
_nOperatingColumns = len(X_train[0])
X_train_sequence = datagenerator.getEncoderInputSequence(X_train, _nTimesteps, _nOperatingColumns)
Validation_X_sequence = datagenerator.getEncoderInputSequence(Validation_X, _nTimesteps, _nOperatingColumns)

sequence_autoencoder_semi = modelgenerator.getSAE_LSTM(_nTimesteps, _nOperatingColumns)
sequence_autoencoder_semi.compile(optimizer='adam', loss='mean_squared_error')

#modelName = "Round1_Monday_AllS"
modelName = "cidids2018_model"

#Adding checkpoints
checkpointFile = "/home/ubuntu/aitd/" + modelName + ".h5"
checkpoint = ModelCheckpoint(checkpointFile, monitor='loss', verbose=1, save_best_only=True, mode='min')
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
callbacks_list = [checkpoint, earlyStopping]


#sequence_autoencoder_semi = load_model(checkpointFile)
#sequence_autoencoder_semi.summary()
plot_model(sequence_autoencoder_semi, to_file='sae_lstm_model_plot.png', show_shapes=True, show_layer_names=True)
#Training autoencoder
sequence_autoencoder_semi_history = sequence_autoencoder_semi.fit(X_train_sequence, X_train_sequence,
                     epochs=32,
                     batch_size=10,
                     shuffle=False,
                     validation_data=(Validation_X_sequence, 
                     Validation_X_sequence), 
                     callbacks=callbacks_list)



# Save the model and serialize model to JSON and h5
sequence_autoencoder_semi.save("/home/ubuntu/aitd/" + modelName + ".h5")
print("Saved model to disk")


loss = sequence_autoencoder_semi_history.history['loss']
val_loss = sequence_autoencoder_semi_history.history['val_loss']
epochs = range(32)
plt.figure()
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='blue', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
#plt.savefig('LossColored_' + modelName + '.png')
plt.show()
