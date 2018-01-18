#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:36:31 2017

@author: rcasero
"""

import pytest

import os
os.environ['KERAS_BACKEND'] = 'theano'
if (K.backend() == 'tensorflow'):
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

# different versions of conda keep the path in different variables
if 'CONDA_ENV_PATH' in os.environ:
    conda_env_path = os.environ['CONDA_ENV_PATH']
elif 'CONDA_PREFIX' in os.environ:
    conda_env_path = os.environ['CONDA_PREFIX']
else:
    conda_env_path = '.'

import numpy as np

import keras.backend as K
data_format='channels_first'
K.set_image_data_format(data_format)

import cytometer.layers as layers


# compare Keras K.pool2d to dilated MaxPooling2D._pooling_function. When 
# there's no pooling, they have to work the same
def pooling_general_no_dilation(nbatch, nchan, nrows, ncols, pool_size=(2, 2), 
                    strides=(1, 1), padding='valid'):

    dilation_rate=1
    
    # generate test input
    inputs = K.variable(np.reshape(range(1,nbatch*nchan*nrows*ncols+1), 
                                   (nbatch, nchan, nrows, ncols)))
    
    # instantiate max pooling layer
    pool_layer = layers.DilatedMaxPooling2D(pool_size=pool_size, strides=strides, 
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
    np.testing.assert_array_equal(output_ref, output)

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
    
# fails
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

# Test dilated pooling in MaxPooling2D._pooling_function
def test_dilated_pooling():

    inputs = K.variable(
        np.array([[[[ 47.,  68.,  25.,  67.,  83.,  23.,  92.,  57.,  14.,  23.,  72.],
                    [ 89.,  42.,  90.,   8.,  39.,  68.,  48.,   7.,  44.,   0.,  75.],
                    [ 55.,   6.,  19.,  60.,  44.,  63.,  69.,  56.,  24.,  55.,  53.],
                    [ 61.,  64.,  34.,  56.,  73.,  78.,  38.,   4.,   9.,  87.,  67.],
                    [ 72.,  83.,  48.,   1.,  64.,  16.,  31.,  93.,  44.,  92.,  71.],
                    [ 23.,  10.,  35.,  64.,  61.,   7.,  23.,  92.,  96.,  21.,  35.],
                    [ 97.,  67.,  91.,  97.,  19.,  27.,   4.,  19.,  33.,  30.,  57.],
                    [ 73.,  93.,  81.,  31.,   2.,  83.,  50.,  18.,  21.,  67.,  75.]]]]))

    # instantiate max pooling layer
    pool_layer = layers.DilatedMaxPooling2D(pool_size=(3,3), strides=(1,1), 
                                            padding='valid', data_format='channels_first', 
                                            dilation_rate=(2,3))
    
    # compute output using MaxPooling2D    
    outputs = pool_layer._pooling_function(inputs, 
                                          pool_size=pool_layer.get_config()['pool_size'], 
                                          strides=pool_layer.get_config()['strides'],
                                          padding=pool_layer.get_config()['padding'], 
                                          data_format=pool_layer.get_config()['data_format'], 
                                          dilation_rate=pool_layer.get_config()['dilation_rate'])
    outputs = outputs.eval()
    
    expected_outputs = np.array([[[
            [ 92.,  93.,  63.,  92.,  93.],
            [ 89.,  92.,  96.,  87.,  92.],
            [ 97.,  93.,  91.,  97.,  93.],
            [ 73.,  93.,  96.,  87.,  92.]]]])


    np.testing.assert_array_equal(expected_outputs, outputs)
    
#if __name__ == '__main__':
#    pytest.main([__file__])
