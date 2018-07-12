#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 17:59:44 2017

@author: rcasero
"""

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation
from cytometer.layers import DilatedMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
import numpy as np


if K.image_data_format() == 'channels_first':
    default_input_shape = (3, None, None)
elif K.image_data_format() == 'channels_last':
    default_input_shape = (None, None, 3)


def basic_9_conv_8_bnorm_3_maxpool_binary_classifier(input_shape, for_receptive_field=False):
    """Deep binary classifier.
    Similar to basic_9c3mp (no dilation), but with default options, and using the
    x = layer(...)(x) notation
    :param input_shape: (tuple) size of inputs (W,H,C) without batch size, e.g. (200,200,3)
    :param for_receptive_field: (bool, def False) only set to True if you are going to estimate the size of the
    input receptive field using [`receptivefield`](https://github.com/kmkolasinski/receptivefield)
    :return: model: Keras model object
    """

    if K.image_data_format() != 'channels_last':
        raise ValueError('Expected Keras running with K.image_data_format()==channels_last')

    if for_receptive_field:
        activation = 'relu'
        pooling_func = MaxPooling2D
    else:
        activation = 'linear'
        pooling_func = AvgPool2D

    input = Input(shape=input_shape, dtype='float32')
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(input)
    x = BatchNormalization(axis=norm_axis)(x)
    x = Activation(activation)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    x = Activation(activation)(x)
    x = pooling_func(pool_size=(2, 2), strides=1, padding='same')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    x = Activation(activation)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    x = Activation(activation)(x)
    x = pooling_func(pool_size=(4, 4), strides=1, padding='same')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    x = Activation(activation)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    x = Activation(activation)(x)
    x = pooling_func(pool_size=(8, 8), strides=1, padding='same')(x)

    x = Conv2D(filters=200, kernel_size=(4, 4), strides=1, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    x = Activation(activation)(x)

    x = Conv2D(filters=200, kernel_size=(1, 1), strides=1, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    x = Activation(activation)(x)

    x = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same')(x)
    x = Activation('softmax')(x)

    main_output = Activation('softmax', name='main_output')(x)
    return Model(inputs=input, outputs=main_output)


def basic_9c3mp(input_shape=default_input_shape, reg=0.001, init='he_normal'):
    """Based on DeepCell's sparse_feature_net_61x61, but here we use no dilation
    """

    if K.image_data_format() == 'channels_first':
        norm_axis = 1
    elif K.image_data_format() == 'channels_last':
        norm_axis = 3


    model = Sequential()

    model.add(Conv2D(input_shape=input_shape,
                     filters=64, kernel_size=(3, 3), strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=norm_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=norm_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='same'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=norm_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=norm_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=1, padding='same'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=norm_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=norm_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(8, 8), strides=1, padding='same'))

    model.add(Conv2D(filters=200, kernel_size=(4, 4), strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=norm_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=200, kernel_size=(1, 1), strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=norm_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=4, kernel_size=(1, 1), strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(Activation('softmax'))

    return model


# Based on DeepCell's sparse_feature_net_61x61, but here the max pooling has no dilation
def sparse_feature_net_61x61_no_dilated_pooling(input_shape=(3, None, None), n_features=3, reg=0.001, init='he_normal',
                                                weights_path=None):
    model = Sequential()

    d = 1
    model.add(Conv2D(input_shape=input_shape,
                     filters=64, kernel_size=(3, 3), dilation_rate=d, strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=64, kernel_size=(4, 4), dilation_rate=d, strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=tuple(np.array((2, 2)) * d), strides=1, padding='same'))

    d *= 2
    model.add(Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=d, strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=d, strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2 * d, 2 * d), strides=1,
                           padding='same'))

    d *= 2
    model.add(Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=d, strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=d, strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2 * d, 2 * d), strides=1,
                           padding='same'))

    d *= 2
    model.add(Conv2D(filters=200, kernel_size=(4, 4), dilation_rate=d, strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=200, kernel_size=(1, 1), dilation_rate=1, strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=3, kernel_size=(1, 1), dilation_rate=1, strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(Activation('softmax'))

    return model


# Reimplementation of DeepCell's sparse_feature_net_61x61 for Keras 2
# Note: currently, dilated max pooling requires ad hoc code that we provide in
# this project
def sparse_feature_net_61x61(input_shape=(3,1080,1280), n_features=3, reg=0.001, init='he_normal', weights_path=None):

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
