#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 18:08:31 2017

@author: Ramón Casero <rcasero@gmail.com>
"""

from pkg_resources import parse_version

from keras import __version__ as keras_version
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
if (parse_version(keras_version) < parse_version('2.0.0')): # Keras 1
    from keras.layers import Convolution2D, MaxPooling2D
else: # Keras 2
    from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K

# DeepCell's feature net 61x61 with batch normalization
def bn_feature_net_61x61(n_features = 3, n_channels = 2, reg = 1e-5, init = 'he_normal'):

    if K.image_data_format() == 'channels_first':
        input_shape = (n_channels, 61, 61)
    else:
        input_shape = (61, 61, n_channels)
    
    model = Sequential()
    
    # Keras 1
    if (parse_version(keras_version) < parse_version('2.0.0')):
        
        model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', input_shape=(n_channels, 61, 61), W_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        
        model.add(Convolution2D(64, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        
        model.add(Convolution2D(64, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        
        model.add(Convolution2D(64, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Convolution2D(200,4,4, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        
        model.add(Flatten())
        
        model.add(Dense(200, init = init, W_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        
        model.add(Dense(n_features, init = init, W_regularizer = l2(reg)))
        model.add(Activation('softmax'))
        
        return model
    
    # Keras 2
    else:
        model.add(Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer=init, padding='valid', input_shape=input_shape, kernel_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        
        model.add(Conv2D(filters=64, kernel_size=(4, 4), kernel_initializer=init, padding='valid', kernel_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer=init, padding='valid', kernel_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        
        model.add(Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer=init, padding='valid', kernel_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer=init, padding='valid', kernel_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        
        model.add(Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer=init, padding='valid', kernel_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(filters=200, kernel_size=(4, 4), kernel_initializer=init, padding='valid', kernel_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        
        model.add(Flatten())
        
        model.add(Dense(units=200, kernel_initializer=init, kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        
        model.add(Dense(units=n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))
        model.add(Activation('softmax'))
        
        return model

