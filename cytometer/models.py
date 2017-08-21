#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 17:59:44 2017

@author: rcasero
"""

from keras import __version__ as keras_version
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from cytometer.layers import SlidingMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K

def sparse_feature_net_61x61(input_shape = (3,1080,1280), n_features = 3, reg = 0.001, init = 'he_normal', weights_path = None):

    model = Sequential()
    
    model.add(Conv2D(input_shape = input_shape, 
                     filters = 64, kernel_size = (3, 3), dilation_rate = 1, 
                     kernel_initializer = init, padding = 'valid', 
                     kernel_regularizer = l2(reg)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters = 64, kernel_size = (4, 4), dilation_rate = 1, 
                     kernel_initializer = init, padding = 'valid', 
                     kernel_regularizer = l2(reg)))
    model.add(Activation('relu'))
    model.add(SlidingMaxPooling2D(pool_size = (2, 2), strides = (2, 2), dilation_rate = (2, 2)))
#        d *= 2
#        
#        model.add(sparse_Convolution2D(64, 3, 3, d = d, init=init, border_mode='valid', W_regularizer = l2(reg)))
#        model.add(Activation('relu'))
#        
#        model.add(sparse_Convolution2D(64, 3, 3, d = d, init=init, border_mode = 'valid', W_regularizer = l2(reg)))
#        model.add(Activation('relu'))
#        model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
#        d *= 2
#        
#        model.add(sparse_Convolution2D(64, 3, 3, d = d, init=init, border_mode='valid', W_regularizer = l2(reg)))
#        model.add(Activation('relu'))
#        
#        model.add(sparse_Convolution2D(64, 3, 3, d = d, init=init, border_mode = 'valid', W_regularizer = l2(reg)))
#        model.add(Activation('relu'))
#        model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
#        d *= 2
#        
#        model.add(sparse_Convolution2D(200, 4, 4, d = d, init=init, border_mode = 'valid', W_regularizer = l2(reg)))
#        model.add(Activation('relu'))
#        
#        model.add(TensorProd2D(200, 200, kernel_initializer=init, W_regularizer = l2(reg)))
#        model.add(Activation('relu'))
#        
#        model.add(TensorProd2D(200, n_features, kernel_initializer=init, W_regularizer = l2(reg)))
#        model.add(Activation(tensorprod_softmax))
#        
    # exit
    return model