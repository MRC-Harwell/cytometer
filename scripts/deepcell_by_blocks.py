#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:01:09 2017

Process several synthetic images with DeepCell's HeLa model, to study how
changes in input intensities affect the output.

@author: Ramon Casero <rcasero@gmail.com>
"""

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import cytometer
import pysto.imgproc as pymproc

###############################################################################
## Keras configuration

# Keras backend
os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['KERAS_BACKEND'] = 'tensorflow'

# access to python libraries
if 'LIBRARY_PATH' in os.environ:
    os.environ['LIBRARY_PATH'] = os.environ['CONDA_PREFIX'] + '/lib:' + os.environ['LIBRARY_PATH']
else:
    os.environ['LIBRARY_PATH'] = os.environ['CONDA_PREFIX'] + '/lib'

# configure Theano global options
#os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cuda0,gpuarray.preallocate=0.5'
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cuda0,lib.cnmem=0.75'

# configure Theano
import theano
theano.config.enabled = True
theano.config.dnn.include_path = os.environ['CONDA_PREFIX'] + '/include'
theano.config.dnn.library_path = os.environ['CONDA_PREFIX'] + '/lib'
theano.config.blas.ldflags = '-lblas -lgfortran'
theano.config.nvcc.fastmath = True
theano.config.nvcc.flags = '-D_FORCE_INLINES'
theano.config.cxx = os.environ['CONDA_PREFIX'] + '/bin/g++'

# import and check Keras version
import keras
import keras.backend as K
from keras import __version__ as keras_version
from pkg_resources import parse_version
if (parse_version(keras_version) >= parse_version('2.0')):
    raise RuntimeError('DeepCell requires Keras 1 to run')

# configure Keras, to avoid using file ~/.keras/keras.json
K.set_image_dim_ordering('th') # theano's image format (required by DeepCell)
K.set_floatx('float32')
K.set_epsilon('1e-07')

##
###############################################################################

import cytometer.deepcell as deepcell
import cytometer.deepcell_models as deepcell_models

###############################################################################
## Check response changes with step size changes
###############################################################################

# create test images with a corner step
im = np.ones(shape=(1,2,200,200))
im[:,:,0:50,0:50]=2

im2 = np.ones(shape=(1,2,200,200))
im2[:,:,0:100,0:100]=2

im3 = np.ones(shape=(1,2,200,200))
im3[:,:,0:50,0:50]=2
im3[:,:,100:150,100:150]=2

im4 = np.ones(shape=(1,2,200,200)) + .5
im4[:,:,0:50,0:50]=2.5

im5 = np.ones(shape=(1,2,200,200)) + 1
im5[:,:,0:50,0:50]=3


# load one of the DeepCell models
basedatadir = os.path.normpath(os.path.join(cytometer.__path__[0], '../data/deepcell'))
netdir = os.path.join(basedatadir, 'trained_networks')
model_dir = os.path.join(netdir, 'HeLa')
model_weights_file = '2017-06-21_HeLa_all_61x61_bn_feature_net_61x61_'

# instantiate model (same for all validation data)
j = 0 # model index
model = deepcell_models.sparse_bn_feature_net_61x61(batch_input_shape = im.shape)
model = deepcell.set_weights(model, os.path.join(netdir, model_dir, model_weights_file + str(j) + '.h5'))
im_out = model.predict(im)
im2_out = model.predict(im2)
im3_out = model.predict(im3)
im4_out = model.predict(im4)
im5_out = model.predict(im5)

# add padding to the outputs
im_out = np.pad(im_out, pad_width=((0,0), (0,0), (30,30),(30,30)), mode='constant', constant_values=0)
im2_out = np.pad(im2_out, pad_width=((0,0), (0,0), (30,30),(30,30)), mode='constant', constant_values=0)
im3_out = np.pad(im3_out, pad_width=((0,0), (0,0), (30,30),(30,30)), mode='constant', constant_values=0)
im4_out = np.pad(im4_out, pad_width=((0,0), (0,0), (30,30),(30,30)), mode='constant', constant_values=0)
im5_out = np.pad(im5_out, pad_width=((0,0), (0,0), (30,30),(30,30)), mode='constant', constant_values=0)

# plot results
plt.subplot(5,2,1)
plt.imshow(im[:,0,:,:].reshape(200,200), vmin=1, vmax=3)
plt.subplot(5,2,2)
plt.imshow(im_out[:,0,:,:].reshape(200,200), vmin=0, vmax=1)
plt.subplot(5,2,3)
plt.imshow(im2[:,0,:,:].reshape(200,200), vmin=1, vmax=3)
plt.subplot(5,2,4)
plt.imshow(im2_out[:,0,:,:].reshape(200,200), vmin=0, vmax=1)
plt.subplot(5,2,5)
plt.imshow(im3[:,0,:,:].reshape(200,200), vmin=1, vmax=3)
plt.subplot(5,2,6)
plt.imshow(im3_out[:,0,:,:].reshape(200,200), vmin=0, vmax=1)
plt.subplot(5,2,7)
plt.imshow(im4[:,0,:,:].reshape(200,200), vmin=1, vmax=3)
plt.subplot(5,2,8)
plt.imshow(im4_out[:,0,:,:].reshape(200,200), vmin=0, vmax=1)
plt.subplot(5,2,9)
plt.imshow(im5[:,0,:,:].reshape(200,200), vmin=1, vmax=3)
plt.subplot(5,2,10)
plt.imshow(im5_out[:,0,:,:].reshape(200,200), vmin=0, vmax=1)

###############################################################################
## Compare processing whole image vs. processing by blocks
###############################################################################

pad_xy = 30

# create image
im = np.ones(shape=(1,2,200,200))
im[:,:,0:50,0:50]=2
im[:,:,100:150,100:150]=2

# apply DeepCell's preprocessing
im[0,0,:,:] = deepcell.process_image(im[0,0,:,:], 30, 30)
im[0,1,:,:] = deepcell.process_image(im[0,1,:,:], 30, 30)

# split into blocks
block_slices, blocks, im_padded = pymproc.block_split(im, nblocks=(1,1,1,2), pad_width=((0,0),(0,0),(pad_xy,pad_xy),(pad_xy,pad_xy)), mode='reflect')

# load a model
j = 0 # model index
model = deepcell_models.sparse_bn_feature_net_61x61(batch_input_shape = im.shape)
model = deepcell.set_weights(model, os.path.join(netdir, model_dir, model_weights_file + str(j) + '.h5'))

# process image and blocks
im_out = model.predict(im)
im_out = np.pad(im_out, pad_width=((0,0), (0,0), (pad_xy,pad_xy),(pad_xy,pad_xy)), mode='constant', constant_values=0)

blocks_out = []
for b in blocks:
    model = deepcell_models.sparse_bn_feature_net_61x61(batch_input_shape = b.shape)
    model = deepcell.set_weights(model, os.path.join(netdir, model_dir, model_weights_file + str(j) + '.h5'))
    b_out = model.predict(b)
    blocks_out += [b_out]
    
# correct slice indices
block_slices_out = []
for sl, b_out in zip(block_slices, blocks_out):
    block_slices_out += [[slice(0,1,1), slice(0,2,1), 
                          slice(sl[1].start, sl[1].start+b_out.shape[1],1), 
                          slice(sl[2].start, sl[2].start+b_out.shape[2], 1)]]

# restack blocks
im_out_by_blocks, _ = pymproc.block_stack(blocks_out, block_slices_out, 
                                          pad_width=((0,0),(0,0),(pad_xy-30,pad_xy-30),(pad_xy-30,pad_xy-30)))



plt.subplot(2,2,1)
plt.imshow(im[:,0,:,:].reshape(200,200), vmin=-0.5, vmax=0.9)
plt.subplot(2,2,2)
plt.imshow(im_out[:,0,:,:].reshape(200,200), vmin=0, vmax=0.9)
