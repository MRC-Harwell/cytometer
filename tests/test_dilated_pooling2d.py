#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:36:31 2017

@author: rcasero
"""

import pytest

import os
os.environ['KERAS_BACKEND'] = 'theano'

if 'LIBRARY_PATH' in os.environ:
    os.environ['LIBRARY_PATH'] = os.environ['CONDA_PREFIX'] + '/lib:' + os.environ['LIBRARY_PATH']
else:
    os.environ['LIBRARY_PATH'] = os.environ['CONDA_PREFIX'] + '/lib'

import numpy as np

import keras.backend as K
data_format='channels_first'
K.set_image_data_format(data_format)

import cytometer.layers as layers


# compare Keras K.pool2d to dilated MaxPooling2D._pooling_function
def pooling_general_no_dilation(nbatch, nchan, nrows, ncols, pool_size=(2, 2), 
                    strides=(1, 1), padding='valid'):

    dilation_rate=1
    
    # generate test input
    inputs = K.variable(np.reshape(range(1,nbatch*nchan*nrows*ncols+1), 
                                   (nbatch, nchan, nrows, ncols)))
    
    # instantiate max pooling layer
    pool_layer = layers.MaxPooling2D(pool_size=pool_size, strides=strides, 
                                     padding=padding, data_format=data_format, 
                                     dilation_rate=dilation_rate)

    # compute output using MaxPooling2D    
    output = pool_layer._pooling_function(inputs, 
                                          pool_size=pool_layer.get_config()['pool_size'], 
                                          strides=pool_layer.get_config()['strides'],
                                          padding=pool_layer.get_config()['padding'], 
                                          data_format=pool_layer.get_config()['data_format'], 
                                          dilation_rate=pool_layer.get_config()['dilation_rate'])
    output = output.eval()

    # compute output using the Keras pooling function
    output_ref = K.pool2d(inputs, 
                          pool_size=pool_layer.get_config()['pool_size'], 
                          strides=pool_layer.get_config()['strides'],
                          padding=pool_layer.get_config()['padding'], 
                          data_format=pool_layer.get_config()['data_format'],
                          pool_mode='max')
    output_ref = output_ref.eval()
    
    # compare both methods
    np.testing.assert_array_equal(output, output_ref)

# simple (2,2) kernel, no dilation
def test_pooling_no_dilation_1_batch_1_channel_valid_padding():
    pooling_general_no_dilation(nbatch=1, nchan=1, nrows=5, ncols=8, padding='valid')

def test_pooling_no_dilation_1_batch_3_channels_valid_padding():
    pooling_general_no_dilation(nbatch=1, nchan=3, nrows=5, ncols=8, padding='valid')

def test_pooling_no_dilation_6_batch_3_channels_valid_padding():
    pooling_general_no_dilation(nbatch=6, nchan=3, nrows=5, ncols=8, padding='valid')

def test_pooling_no_dilation_1_batch_1_channel_same_padding():
    pooling_general_no_dilation(nbatch=1, nchan=1, nrows=5, ncols=8, padding='same')

def test_pooling_no_dilation_1_batch_3_channels_same_padding():
    pooling_general_no_dilation(nbatch=1, nchan=3, nrows=5, ncols=8, padding='same')

def test_pooling_no_dilation_6_batch_3_channels_same_padding():
    pooling_general_no_dilation(nbatch=6, nchan=3, nrows=5, ncols=8, padding='same')

# kernels of serveral sizes, without dilation
def test_pooling_no_dilation_kernel_3x3_valid_padding():
    pooling_general_no_dilation(nbatch=6, nchan=3, nrows=5, ncols=8, 
                                padding='valid', pool_size=(3, 3))
    
def test_pooling_no_dilation_kernel_3x4_valid_padding():
    pooling_general_no_dilation(nbatch=6, nchan=3, nrows=5, ncols=8, 
                                padding='valid', pool_size=(3, 4))
    
def test_pooling_no_dilation_kernel_4x7_valid_padding():
    pooling_general_no_dilation(nbatch=6, nchan=3, nrows=5, ncols=8, 
                                padding='valid', pool_size=(4, 7))
    
def test_pooling_no_dilation_kernel_3x3_same_padding():
    pooling_general_no_dilation(nbatch=6, nchan=3, nrows=5, ncols=8, 
                                padding='same', pool_size=(3, 3))
    
def test_pooling_no_dilation_kernel_3x4_same_padding():
    pooling_general_no_dilation(nbatch=6, nchan=3, nrows=5, ncols=8, 
                                padding='same', pool_size=(3, 4))
    
def test_pooling_no_dilation_kernel_4x7_same_padding():
    pooling_general_no_dilation(nbatch=6, nchan=3, nrows=5, ncols=8, 
                                padding='same', pool_size=(4, 7))
    
    

if __name__ == '__main__':
    pytest.main([__file__])
