#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 17:59:44 2017

@author: rcasero
"""

from keras import __version__ as keras_version
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Permute
from cytometer.layers import DilatedMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
import numpy as np

# Reimplementation of DeepCell's sparse_feature_net_61x61 for Keras 2 without
# dilated max pooling
def sparse_feature_net_61x61_no_dilated_pooling(input_shape = (3,1080,1280), n_features = 3, reg = 0.001, init = 'he_normal', weights_path = None):

    model = Sequential()
    
    d = 1
    model.add(Conv2D(input_shape=input_shape, 
                     filters=64, kernel_size=(3, 3), dilation_rate=d, strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters=64, kernel_size=(4, 4), dilation_rate=d, strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=tuple(np.array((2,2))*d), strides=1,
                           padding='same'))
    
    d *= 2
    model.add(Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=d, strides=1, 
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=d, strides=1, 
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=tuple(np.array((2,2))*d), strides=1,
                           padding='same'))
    
    d *= 2
    model.add(Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=d, strides=1, 
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=d, strides=1, 
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=tuple(np.array((2,2))*d), strides=1,
                           padding='same'))
    
    d *= 2
    model.add(Conv2D(filters=200, kernel_size=(4, 4), dilation_rate=d, strides=1, 
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(Activation('relu'))
    
    if K.image_data_format() == 'channels_first':
        model.add(Permute((2, 3, 1)))
    
    model.add(Dense(units=200,
                    kernel_initializer=init, kernel_regularizer=l2(reg)))
    model.add(Activation('relu'))
    
    model.add(Dense(units=n_features, 
                    kernel_initializer=init, kernel_regularizer=l2(reg)))
    if K.image_data_format() == 'channels_first':
        model.add(Permute((3, 1, 2)))
    model.add(Activation('softmax'))
    
    return model


# Reimplementation of DeepCell's sparse_feature_net_61x61 for Keras 2
# Note: currently, dilated max pooling requires ad hoc code that we provide in
# this project
def sparse_feature_net_61x61(input_shape = (3,1080,1280), n_features = 3, reg = 0.001, init = 'he_normal', weights_path = None):

    model = Sequential()
    
    d = 1
    model.add(Conv2D(input_shape=input_shape, 
                     filters=64, kernel_size=(3, 3), dilation_rate=d, strides=1,
                     kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters=64, kernel_size=(4, 4), dilation_rate=d, strides=1,
                     kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
    model.add(Activation('relu'))
    model.add(DilatedMaxPooling2D(pool_size=(2, 2), dilation_rate=d, strides=1,
                           padding='valid'))
    
#    d *= 2
#    model.add(Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=d, strides=1, 
#                     kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
#    model.add(Activation('relu'))
#    
#    model.add(Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=d, strides=1, 
#                     kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
#    model.add(Activation('relu'))
#    model.add(DilatedMaxPooling2D(pool_size=(2, 2), dilation_rate=d, strides=1,
#                           padding='valid'))
#    
#    d *= 2
#    model.add(Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=d, strides=1, 
#                     kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
#    model.add(Activation('relu'))
#    
#    model.add(Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=d, strides=1, 
#                     kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
#    model.add(Activation('relu'))
#    model.add(DilatedMaxPooling2D(pool_size=(2, 2), dilation_rate=d, strides=1,
#                           padding='valid'))
#    
#    d *= 2
#    model.add(Conv2D(filters=200, kernel_size=(4, 4), dilation_rate=d, strides=1, 
#                     kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
#    model.add(Activation('relu'))
#    
#    model.add(Dense(units=200, 
#                    kernel_initializer=init, kernel_regularizer=l2(reg)))
#    model.add(Activation('relu'))
#    
#    model.add(Dense(units=n_features, 
#                    kernel_initializer=init, kernel_regularizer=l2(reg)))
#    model.add(Activation(K.softmax))
    
    # exit
    return model
