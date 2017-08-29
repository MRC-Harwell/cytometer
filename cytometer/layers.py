#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 18:13:57 2017

@author: rcasero
"""

from __future__ import absolute_import

import keras.backend as K
from keras.engine import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.legacy import interfaces

import numpy as np

class _Pooling2D(Layer):
    """Abstract class for different pooling 2D layers.
    """

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None, dilation_rate=1, **kwargs):
        super(_Pooling2D, self).__init__(**kwargs)
        data_format = conv_utils.normalize_data_format(data_format)
        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
        rows = conv_utils.conv_output_length(rows, self.pool_size[0],
                                             self.padding, self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.pool_size[1],
                                             self.padding, self.strides[1],
                                             self.dilation_rate[1])
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, input_shape[3])

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format, dilation_rate):
        raise NotImplementedError

    def call(self, inputs):
        output = self._pooling_function(inputs=inputs,
                                        pool_size=self.pool_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        data_format=self.data_format,
                                        dilation_rate=self.dilation_rate)
        return output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'padding': self.padding,
                  'strides': self.strides,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate}
        base_config = super(_Pooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MaxPooling2D(_Pooling2D):
    """Dilated/Atrous max pooling operation for spatial data.

    # Arguments
        pool_size: integer or tuple of 2 integers,
            factors by which to downscale (vertical, horizontal).
            (2, 2) will halve the input in both spatial dimension.
            If only one integer is specified, the same window length
            will be used for both dimensions.
        strides: Integer, tuple of 2 integers, or None.
            Strides values.
            If None, it will default to `pool_size`.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated pooling.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.

    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`

    # Output shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, pooled_rows, pooled_cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, pooled_rows, pooled_cols)`
    """

    @interfaces.legacy_pooling2d_support
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None, dilation_rate=1, **kwargs):
        super(MaxPooling2D, self).__init__(pool_size, strides, padding,
                                           data_format, dilation_rate, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format, dilation_rate):
        
        """We illustrate how this method works with an example. Let's have a 
        dilated kernel with pool_size=(3,2), dilation_rate=(2,4). The * 
        represent the non-zero elements
        
        * 0 0 0 *
        0 0 0 0 0
        * 0 0 0 *
        0 0 0 0 0
        * 0 0 0 *
        
        Let's have a large image with two different pixels (here we represent 
        only a corner of the image)
        
        X X X X X X X
        X X X X X X X
        X X X X X X X
        X X X X X X X
        X X X X X X X
        X X X X X X X
        + + + + + + +
        + + + + + + +
        + + + + + + +
        + + + + + + +
        + + + + + + +
        + + + + + + +
        
        The pooling kernel will slide over the image, similarly to convolution,
        computing a pooling result at each step. For example, for the vertical 
        slide
        
            dr = 0          dr = 1          dr = 2 
        
        * 0 0 0 * X X   X X X X X X X   X X X X X X X 
        0 0 0 0 0 X X   * 0 0 0 * X X   X X X X X X X 
        * 0 0 0 * X X   0 0 0 0 0 X X   * 0 0 0 * X X
        0 0 0 0 0 X X   * 0 0 0 * X X   0 0 0 0 0 X X
        * 0 0 0 * X X   0 0 0 0 0 X X   * 0 0 0 * X X
        X X X X X X X , * 0 0 0 * X X , 0 0 0 0 0 X X
        + + + + + + +   + + + + + + +   * 0 0 0 * + +
        + + + + + + +   + + + + + + +   + + + + + + +
        + + + + + + +   + + + + + + +   + + + + + + +
        + + + + + + +   + + + + + + +   + + + + + + +
        + + + + + + +   + + + + + + +   + + + + + + +
        + + + + + + +   + + + + + + +   + + + + + + +
        
        The last step, dr=2, is unnecesary. Instead, if we overlap dr=0 and 
        dr=2, we see that if we subsample every odd row, and every fourth 
        column, we can use the regular pool2d() function to compute the pooling
        
          dr = {0, 2}
        
        * 0 0 0 * X X                     * 0 0 0 * X X                     * *
        0 0 0 0 0 X X
        * 0 0 0 * X X                     * 0 0 0 * X X                     * *
        0 0 0 0 0 X X
        * 0 0 0 * X X                     * 0 0 0 * X X                     * *
        X X X X X X X  =subsample rows=>                 =subsample cols=>  
        * 0 0 0 * + +                     * 0 0 0 * + +                     * *
        0 0 0 0 0 + +
        * 0 0 0 * + +                     * 0 0 0 * + +                     * *
        0 0 0 0 0 + +
        * 0 0 0 * + +                     * 0 0 0 * + +                     * *
        + + + + + + +
        
        Thus, sliding pooling can be computed as a double loop
        
        for row in 0, 1, 2, 3, ..., pool_size[0]*dilation_rate[0]:
            for col in 0, 1, 2, 3, ..., pool_size[1]*dilation_rate[1]:
                pool2d(inputs[:, :, 0:end:dilation_rate[0], 0:end:dilation_rate[1]])
        
        """

        sz = K.shape(inputs)
        if data_format == 'channels_first': # (batch,chan,row,col)
            nbatch = sz[0]
            #nchan = sz[1]
            nrows = sz[2]
            ncols = sz[3]
        elif data_format == 'channels_last': # (batch,row,col,chan)
            nbatch = sz[0]
            #nchan = sz[1]
            nrows = sz[2]
            ncols = sz[3]
        else:
            raise ValueError('Expected data format to be channels_first or channels_last')

        # number of blocks to split the input into. Each dilation (row or 
        # column) goes into a separate block
        nblocks = dilation_rate
        
        # size of each block we are going to split the input images in
        block_sz = ((np.ceil(nrows / dilation_rate[0])), 
                    (np.ceil(ncols / dilation_rate[1])))
        block_sz = block_sz.eval()

        # pad inputs so that they can be split into equal blocks
        padded_size = np.multiply(block_sz, nblocks)
        padding_len = ((0, padded_size[0] - nrows), (0, padded_size[1] - ncols))
        inputs = K.spatial_2d_padding(inputs, padding=padding_len, data_format=data_format)
 
        # split the inputs into blocks
        split_inputs = []
        for row_offset in range(nblocks[0]):
            for col_offset in range(nblocks[1]):
                if data_format == 'channels_first': # (batch,chan,row,col)
                    split_inputs = split_inputs + [inputs[:, :, row_offset::dilation_rate[0], col_offset::dilation_rate[1]]]
                elif data_format == 'channels_last': # (batch,row,col,chan)
                    split_inputs = split_inputs + [inputs[:, row_offset::dilation_rate[0], col_offset::dilation_rate[1], :]]
        split_inputs = K.concatenate(split_inputs, axis=0)                        
    
        # pool each block without dilation
        split_inputs = K.pool2d(split_inputs, pool_size, strides=(1,1),
                                padding='same', data_format=data_format,
                                pool_mode='max')

        # reassemble blocks
        output = np.zeros(shape=list(inputs.shape.eval()), dtype=inputs.dtype)
        for idx in range(nbatch*nblocks[0]*nblocks[1]):
            multi_index = np.unravel_index(idx, dims=(nblocks[0], nblocks[1], nbatch))
            row_offset = multi_index[0]
            col_offset = multi_index[1]
            batch = multi_index[2]
            if data_format == 'channels_first': # (batch,chan,row,col)
                output[batch, :, row_offset::dilation_rate[0], col_offset::dilation_rate[1]] = split_inputs[idx, :, :, :].eval()
            elif data_format == 'channels_last': # (batch,row,col,chan)
                output[batch, row_offset::dilation_rate[0], col_offset::dilation_rate[1], :] = split_inputs[idx, :, :, :].eval()
        output = K.variable(output)
        
        # remove padding (following the same behaviour as K.pool2d)
        if padding == 'valid':
            
            if data_format == 'channels_first': # (batch,chan,row,col)
                output = output[:, :, (pool_size[0]-1)*dilation_rate[0]:nrows, 
                                (pool_size[1]-1)*dilation_rate[1]:ncols]
            elif data_format == 'channels_last': # (batch,row,col,chan)
                output = output[:, (pool_size[0]-1)*dilation_rate[0]:nrows, 
                                (pool_size[1]-1)*dilation_rate[1]:ncols, :]
                    
        elif padding == 'same':
        
            if data_format == 'channels_first': # (batch,chan,row,col)
                output = output[:, :, 0:nrows, 0:ncols]
            elif data_format == 'channels_last': # (batch,row,col,chan)
                output = output[:, 0:nrows, 0:ncols, :]
                
        else:
            
            raise NotImplementedError

        # return tensor
        return output

# Aliases

MaxPool2D = MaxPooling2D
