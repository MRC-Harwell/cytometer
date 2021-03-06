#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 17:59:44 2017

@author: Ramon Casero <rcasero@gmail.com>
"""

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, AvgPool2D, Activation
from cytometer.layers import DilatedMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
import keras.engine
import numpy as np
import time


if K.image_data_format() == 'channels_first':
    default_input_shape = (3, None, None)
    norm_axis = 1
elif K.image_data_format() == 'channels_last':
    default_input_shape = (None, None, 3)
    norm_axis = -1


def load_model_with_retries(model, number_of_attempts=1, time_between_attempts=5):
    """
    Wrap keras.models.load_model in a loop so that if the loading fails due to some network filesystem errors, we wait a
    few seconds and then retry to load the model.
    :param model: string with filename containing a keras model.
    :param number_of_attempts: (def 1) Number of times we try to load the model before giving up.
    :param time_between_attempts: (def 5) Seconds between loading attempts.
    :return: keras model.
    """

    # if we are trying to load from a network filesystem, the server hosting the filesystem may return some network
    # error. Because we don't want the computation dying just because of that, we let the user try a couple of times,
    # with a little wait between tries
    for attempt in range(number_of_attempts):
        try:
            model = keras.models.load_model(model)
            break
        except ConnectionResetError:
            print(
                '# ======> ConnectionResetError. Attempt ' + str(attempt) + '/' + str(number_of_attempts) + ' ...')
            time.sleep(time_between_attempts)  # secs
        except RuntimeError as e:
            errno, strerror = e.args
            if strerror == 'Bad file descriptor':
                print(
                    '# ======> RuntimeError (' + strerror + '). Attempt ' + str(attempt) + '/' + str(number_of_attempts) + ' ...')
                time.sleep(time_between_attempts)  # secs
            else:
                # this may be a legitimate RuntimeError that has nothing to do with the network filesystem, so we
                # re-throw the exception
                raise
        except:
            raise

    return model


def change_input_size(model, batch_shape):
    """
    Change the expected shape of the model's input tensor.

    This function works by creating a new model with the same structure, and then copying the weights from the original
    model onto the new model. It follows the solution by Christos Kyrkou
    (https://medium.com/@ckyrkou/changing-input-size-of-pre-trained-models-in-keras-3dfbe3ca3091).

    :param model: Keras model.
    :param batch_shape: New input shape, e.g. (None, 500, 500, 3).
    :return: Keras model with modified input layer.
    """

    model._layers[0].batch_input_shape = batch_shape
    model_out = keras.models.model_from_json(model.to_json())

    for layer in model_out.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            pass

    return model_out


def check_model(model):
    """
    Check the layers with weights for NaNs.
    :param model: Keras model.
    :return: list with the names of layers with NaNs. If no weights contain NaNs, the list is empty.
    """

    # loop layers
    layers_with_nans = []
    for layer in model.layers:
        # get the weights in the layer
        weights_list = layer.get_weights()
        if not isinstance(weights_list, list):
            continue
        for weights in weights_list:
            if not isinstance(weights, np.ndarray):
                continue
            # print('Checking layer ' + layer.name)
            if np.any(np.isnan(weights)):
                # print('Layer ' + layer.name + ': contains NaNs')
                layers_with_nans.append(layer.name)
                continue

    return layers_with_nans


def fcn_sherrah2016_regression(input_shape, for_receptive_field=False):

    input = Input(shape=input_shape, dtype='float32', name='input_image')

    x = Conv2D(filters=32, kernel_size=(5, 5), strides=1, dilation_rate=1, padding='same')(input)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(3, 3), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)

    x = Conv2D(filters=96, kernel_size=(5, 5), strides=1, dilation_rate=2, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(5, 5), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, dilation_rate=4, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(9, 9), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(9, 9), strides=1, padding='same')(x)

    x = Conv2D(filters=196, kernel_size=(3, 3), strides=1, dilation_rate=8, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(17, 17), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(17, 17), strides=1, padding='same')(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), strides=1, dilation_rate=16, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
    else:
        x = Activation('relu')(x)

    # dimensionality reduction layer
    main_output = Conv2D(filters=1, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same',
                         name='main_output')(x)

    return Model(inputs=input, outputs=main_output)


def fcn_sherrah2016_contour(input_shape, for_receptive_field=False):

    input = Input(shape=input_shape, dtype='float32', name='input_image')

    x = Conv2D(filters=32, kernel_size=(5, 5), strides=1, dilation_rate=1, padding='same')(input)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(3, 3), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)

    x = Conv2D(filters=96, kernel_size=(5, 5), strides=1, dilation_rate=2, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(5, 5), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, dilation_rate=4, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(9, 9), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(9, 9), strides=1, padding='same')(x)

    x = Conv2D(filters=196, kernel_size=(3, 3), strides=1, dilation_rate=8, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(17, 17), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(17, 17), strides=1, padding='same')(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), strides=1, dilation_rate=16, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
    else:
        x = Activation('relu')(x)

    # classifier output
    x = Conv2D(filters=1, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)
    classification_output = Activation('sigmoid', name='classification_output')(x)

    return Model(inputs=input, outputs=classification_output)


def fcn_sherrah2016_regression_and_classifier(input_shape, for_receptive_field=False):

    input = Input(shape=input_shape, dtype='float32', name='input_image')

    x = Conv2D(filters=32, kernel_size=(5, 5), strides=1, dilation_rate=1, padding='same')(input)
    x = BatchNormalization(axis=norm_axis)(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(3, 3), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)

    x = Conv2D(filters=96, kernel_size=(5, 5), strides=1, dilation_rate=2, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(5, 5), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, dilation_rate=4, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(9, 9), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(9, 9), strides=1, padding='same')(x)

    x = Conv2D(filters=196, kernel_size=(3, 3), strides=1, dilation_rate=8, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(17, 17), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(17, 17), strides=1, padding='same')(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), strides=1, dilation_rate=16, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    if for_receptive_field:
        x = Activation('linear')(x)
    else:
        x = Activation('relu')(x)

    # regression output
    x = Conv2D(filters=1, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)
    regression_output = BatchNormalization(axis=norm_axis, name='regression_output')(x)

    # classification output
    x = Conv2D(filters=1, kernel_size=(32, 32), strides=1, dilation_rate=1, padding='same')(regression_output)
    x = BatchNormalization(axis=norm_axis)(x)
    classification_output = Activation('hard_sigmoid', name='classification_output')(x)

    return Model(inputs=input, outputs=[regression_output, classification_output])


def fcn_9_conv_8_bnorm_3_maxpool_binary_classifier(input_shape, for_receptive_field=False, dilation_rate=1):
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
    else:
        norm_axis = 3

    if for_receptive_field:
        activation = 'linear'
        pooling_func = AvgPool2D
    else:
        activation = 'relu'
        pooling_func = MaxPooling2D

    input = Input(shape=input_shape, dtype='float32', name='input_image')
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, dilation_rate=dilation_rate, padding='same')(input)
    x = BatchNormalization(axis=norm_axis)(x)
    x = Activation(activation)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, dilation_rate=dilation_rate, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    x = Activation(activation)(x)
    pool_size = dilation_rate * 3 * 2 - 1
    x = pooling_func(pool_size=(pool_size, pool_size), strides=1, padding='same')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, dilation_rate=dilation_rate**2, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    x = Activation(activation)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, dilation_rate=dilation_rate**2, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    x = Activation(activation)(x)
    pool_size += (dilation_rate**2) * 3 * 2 - 1
    x = pooling_func(pool_size=(pool_size, pool_size), strides=1, padding='same')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, dilation_rate=dilation_rate**3, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    x = Activation(activation)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, dilation_rate=dilation_rate**3, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    x = Activation(activation)(x)
    pool_size += (dilation_rate**3) * 3 * 2 - 1
    x = pooling_func(pool_size=(pool_size, pool_size), strides=1, padding='same')(x)

    x = Conv2D(filters=200, kernel_size=(4, 4), strides=1, dilation_rate=1, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    x = Activation(activation)(x)

    x = Conv2D(filters=3, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    x = Activation(activation)(x)

    x = Conv2D(filters=1, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)

    if for_receptive_field:
        main_output = Activation('linear', name='main_output')(x)
    else:
        main_output = Activation('softmax', name='main_output')(x)
    return Model(inputs=input, outputs=main_output)


def fcn_conv_bnorm_maxpool_regression(input_shape, for_receptive_field=False,
                                      nblocks=3, kernel_len=3, dilation_rate=1):
    """Deep CNN for regression.
    Similar to basic_9c3mp (no dilation), but with default options, and using the
    x = layer(...)(x) notation
    :param input_shape: (tuple) size of inputs (W,H,C) without batch size, e.g. (200,200,3)
    :param for_receptive_field: (bool, def False) only set to True if you are going to estimate the size of the
    input receptive field using [`receptivefield`](https://github.com/kmkolasinski/receptivefield)
    :return: model: Keras model object
    """

    if K.image_data_format() != 'channels_last':
        raise ValueError('Expected Keras running with K.image_data_format()==channels_last')
    else:
        norm_axis = 3

    # input layer of the network
    input = Input(shape=input_shape, dtype='float32', name='input_image')

    # convolutional blocks, with dilation=dilation_rate**i in each layer
    x = input
    for i in range(nblocks):
        # length of the pooling kernel so that it's as large as the dilated convolutional kernel
        pool_size = dilation_rate * (kernel_len - 1) + 1

        x = Conv2D(filters=64, kernel_size=(kernel_len, kernel_len), strides=1,
                   dilation_rate=dilation_rate, padding='same')(x)
        if not for_receptive_field:
            x = BatchNormalization(axis=norm_axis)(x)
        if for_receptive_field:
            x = Activation('linear')(x)
            x = AvgPool2D(pool_size=(pool_size, pool_size), strides=1, padding='same')(x)
        else:
            x = Activation('relu')(x)
            x = MaxPool2D(pool_size=(pool_size, pool_size), strides=1, padding='same')(x)

        # increase dilation rate
        dilation_rate *= dilation_rate

    # large-number-of-features block
    x = Conv2D(filters=100, kernel_size=(4, 4), strides=1, dilation_rate=1, padding='same')(x)
    if not for_receptive_field:
        x = BatchNormalization(axis=norm_axis)(x)
    if for_receptive_field:
        x = Activation('linear')(x)
    else:
        x = Activation('relu')(x)

    # feature reduction with 1x1 convolution
    x = Conv2D(filters=3, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)
    if not for_receptive_field:
        x = BatchNormalization(axis=norm_axis)(x)
    if for_receptive_field:
        x = Activation('linear')(x)
    else:
        x = Activation('relu')(x)

    # feature reduction with 1x1 convolution
    x = Conv2D(filters=3, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)
    if not for_receptive_field:
        x = BatchNormalization(axis=norm_axis)(x)
    if for_receptive_field:
        main_output = Activation('linear', name='main_output')(x)
    else:
        main_output = Activation('relu', name='main_output')(x)

    # create model object
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
