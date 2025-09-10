from keras.layers import Input, LSTM, RepeatVector, Dense
#regularizers
from keras.models import Model

#Get the model 
def getSAE_LSTM(nTimesteps, nOperatingColumns):
    #This is the size of our encoded representations
    encoding_dim1 = 60 
    encoding_dim2 = 35
    encoding_dim3 = 20
    
    # this is our input placeholder
    input = Input(shape=(nTimesteps, nOperatingColumns))
    # "encoded" is the encoded representation of the input
    encoded = LSTM(encoding_dim1, return_sequences=True, dropout = 0.2)(input)
    
	#dropout will randomly make some cells void in generating the output. Makes the model better.
    encoded = LSTM(encoding_dim2, return_sequences=True, dropout = 0.2)(encoded)
    
	#return_sequences passes the sequences to the next layer. Since we have LSTM layers all the way, we need to pass the sequences to the next layers too. 
    encoded = LSTM(encoding_dim3, return_sequences=True, dropout = 0.2)(encoded)
    
    decoded = LSTM(encoding_dim2, return_sequences=True, dropout = 0.2)(encoded)
    
    decoded = LSTM(encoding_dim1, return_sequences=True, dropout = 0.2)(decoded)
    
    decoded = LSTM(nOperatingColumns, return_sequences=True)(decoded)
    
    # this model maps an input to its reconstruction
    sae_lstm = Model(input, decoded)
    
    return sae_lstm
