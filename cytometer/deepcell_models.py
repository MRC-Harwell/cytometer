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
from cytometer.deepcell import sparse_Convolution2D, sparse_MaxPooling2D, TensorProd2D, tensorprod_softmax, set_weights

# DeepCell is too hard-coded into theano data dimension ordering to fully rewrite it
if (parse_version(keras_version) < parse_version('2.0.0')) and (K.image_dim_ordering() != 'th'):
    raise ValueError('DeepCell requires keras.backend.image_dim_ordering()==\'th\'')
if (parse_version(keras_version) >= parse_version('2.0.0')) and (K.image_data_format() != 'channels_first'):
    raise ValueError('DeepCell requires keras.backend.image_data_format()==\'channel_first\'')

# DeepCell's CNN with 31x31 input images
def sparse_bn_feature_net_31x31(batch_input_shape = (1,1,1080,1280), n_features = 3, reg = 1e-5, init = 'he_normal', weights_path = None):

	model = Sequential()
	d = 1

	model.add(sparse_Convolution2D(32, 4, 4, d = d, init = init, batch_input_shape = batch_input_shape, border_mode='valid', W_regularizer = l2(reg)))	
	model.add(BatchNormalization(axis=1, mode=2))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis=1, mode=2))
	model.add(Activation('relu'))

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis=1, mode=2))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(128, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis=1, mode=2))
	model.add(Activation('relu'))
	
	model.add(sparse_Convolution2D(200, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis=1, mode=2))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, init = init, W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis=1, mode=2))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation(tensorprod_softmax))

	return model

# DeepCell's CNN with 61x61 input images
def sparse_bn_feature_net_61x61(batch_input_shape = (1,2,1080,1280), n_features = 3, reg = 1e-5, init = 'he_normal', weights_path = None):

	model = Sequential()
	d = 1
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, batch_input_shape = batch_input_shape, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis=1, mode=2))
	model.add(Activation('relu'))
	
	model.add(sparse_Convolution2D(64, 4, 4, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis=1, mode=2))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis=1, mode=2))
	model.add(Activation('relu'))
	
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis=1, mode=2))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis=1, mode=2))
	model.add(Activation('relu'))
	
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis=1, mode=2))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(200, 4, 4, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis=1, mode=2))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, init = init, W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis=1, mode=2))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation(tensorprod_softmax))

	return model













# DeepCell's feature net 31x31 with batch normalization
def bn_feature_net_31x31(n_channels = 1, n_features = 3, reg = 1e-5, init = 'he_normal'):

    model = Sequential()
    
    # Keras 1
    if (parse_version(keras_version) < parse_version('2.0.0')):
        
        if K.image_dim_ordering() == 'th':
            input_shape = (n_channels, 31, 31)
        else:
            input_shape = (31, 31, n_channels)
    
        model.add(Convolution2D(32, 4, 4, init = init, border_mode='valid', input_shape=input_shape, W_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(128, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))

        model.add(Convolution2D(200, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))

        model.add(Flatten())

        model.add(Dense(200, init = init, W_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))

        model.add(Dense(n_features, init = init, W_regularizer = l2(reg)))
        model.add(Activation('softmax'))
        
    # Keras 2
    else:

        if K.image_data_format() == 'channels_first':
            input_shape = (n_channels, 31, 31)
        else:
            input_shape = (31, 31, n_channels)
    
        model.add(Conv2D(filters=32, kernel_size=(4, 4), kernel_initializer=init, padding='valid', kernel_regularizer = l2(reg), 
                  input_shape=input_shape))
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

        model.add(Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer=init, padding='valid', kernel_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))

        model.add(Conv2D(filters=200, kernel_size=(3, 3), kernel_initializer=init, padding='valid', kernel_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))

        model.add(Flatten())

        model.add(Dense(units=200, kernel_initializer=init, kernel_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))

        model.add(Dense(n_features, kernel_initializer=init, kernel_regularizer = l2(reg)))
        model.add(Activation('softmax'))
        
    return model

# DeepCell's feature net 61x61 with batch normalization
def bn_feature_net_61x61(n_features = 3, n_channels = 2, reg = 1e-5, init = 'he_normal'):

    model = Sequential()
    
    # Keras 1
    if (parse_version(keras_version) < parse_version('2.0.0')):
        
        if K.image_dim_ordering() == 'th':
            input_shape = (n_channels, 61, 61)
        else:
            input_shape = (61, 61, n_channels)
    
        model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', input_shape=input_shape, W_regularizer = l2(reg)))
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
        
    # Keras 2
    else:
        
        if K.image_data_format() == 'channels_first':
            input_shape = (n_channels, 61, 61)
        else:
            input_shape = (61, 61, n_channels)
    
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
        
        model.add(Dense(units=200, kernel_initializer=init, kernel_regularizer = l2(reg)))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        
        model.add(Dense(units=n_features, kernel_initializer=init, kernel_regularizer = l2(reg)))
        model.add(Activation('softmax'))
        
    # exit
    return model
# DeepCell's feature net 61x61 with batch normalization
def sparse_feature_net_61x61(batch_input_shape = (1,2,1080,1280), n_features = 3, reg = 0.001, init = 'he_normal', weights_path = None):

    model = Sequential()
    
    # Keras 1
    if (parse_version(keras_version) < parse_version('2.0.0')):
        
        d = 1
        model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, batch_input_shape = batch_input_shape, border_mode='valid', W_regularizer = l2(reg)))
        model.add(Activation('relu'))
        
        model.add(sparse_Convolution2D(64, 4, 4, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
        model.add(Activation('relu'))
        model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
        d *= 2
        
        model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
        model.add(Activation('relu'))
        
        model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
        model.add(Activation('relu'))
        model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
        d *= 2
        
        model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
        model.add(Activation('relu'))
        
        model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
        model.add(Activation('relu'))
        model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
        d *= 2
        
        model.add(sparse_Convolution2D(200, 4, 4, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
        model.add(Activation('relu'))
        
        model.add(TensorProd2D(200, 200, init = init, W_regularizer = l2(reg)))
        model.add(Activation('relu'))
        
        model.add(TensorProd2D(200, n_features, init = init, W_regularizer = l2(reg)))
        model.add(Activation(tensorprod_softmax))
        
    # Keras 2
    else:
        
        d = 1
        model.add(sparse_Convolution2D(64, 3, 3, d = d, init=init, batch_input_shape = batch_input_shape, border_mode='valid', W_regularizer = l2(reg)))
        model.add(Activation('relu'))
        
        model.add(sparse_Convolution2D(64, 4, 4, d = d, init=init, border_mode = 'valid', W_regularizer = l2(reg)))
        model.add(Activation('relu'))
        model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
        d *= 2
        
        model.add(sparse_Convolution2D(64, 3, 3, d = d, init=init, border_mode='valid', W_regularizer = l2(reg)))
        model.add(Activation('relu'))
        
        model.add(sparse_Convolution2D(64, 3, 3, d = d, init=init, border_mode = 'valid', W_regularizer = l2(reg)))
        model.add(Activation('relu'))
        model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
        d *= 2
        
        model.add(sparse_Convolution2D(64, 3, 3, d = d, init=init, border_mode='valid', W_regularizer = l2(reg)))
        model.add(Activation('relu'))
        
        model.add(sparse_Convolution2D(64, 3, 3, d = d, init=init, border_mode = 'valid', W_regularizer = l2(reg)))
        model.add(Activation('relu'))
        model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
        d *= 2
        
        model.add(sparse_Convolution2D(200, 4, 4, d = d, init=init, border_mode = 'valid', W_regularizer = l2(reg)))
        model.add(Activation('relu'))
        
        model.add(TensorProd2D(200, 200, kernel_initializer=init, W_regularizer = l2(reg)))
        model.add(Activation('relu'))
        
        model.add(TensorProd2D(200, n_features, kernel_initializer=init, W_regularizer = l2(reg)))
        model.add(Activation(tensorprod_softmax))
        
    # read pre-computed weights if provided, and exit
    if weights_path is not None:
        model = set_weights(model, weights_path)
    return model
