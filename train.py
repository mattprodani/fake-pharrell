import numpy
import re
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, CuDNNLSTM
# from keras.callbacks import ModelCheckpoint
# from keras.utils import np_utils, pad_sequences
# from sklearn.model_selection import train_test_split
# from keras.preprocessing.text import Tokenizer

def train_model(X, Y, config):
    """ Trains a model. 
    Args: 
        X: input vectors
        Y: output vectors
        config: config object
    """
    X = np.reshape(X , (X.shape[0], X.shape[1], 1))
    n_vocab = config.get("n_vocab")
    
    model = Sequential()
    model.add(CuDNNLSTM(255, input_shape =(X.shape[1], X.shape[2]), return_sequences = True))
    model.add(CuDNNLSTM(255, return_sequences = True))
    model.add(CuDNNLSTM(255, return_sequences = True))
    model.add(CuDNNLSTM(255, return_sequences = True))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(Y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    num_epochs = 2
    history = model.fit(X, Y, epochs=num_epochs, batch_size = 256, verbose=1, validation_split=0.2)


    return model
