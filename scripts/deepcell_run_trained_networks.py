#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:07:23 2017

@author: rcasero
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import cytometer

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
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cuda0,lib.cnmem=0.5'

# configure Theano
if os.environ['KERAS_BACKEND'] == 'theano':
    import theano
    theano.config.enabled = True
    theano.config.dnn.include_path = os.environ['CONDA_PREFIX'] + '/include'
    theano.config.dnn.library_path = os.environ['CONDA_PREFIX'] + '/lib'
    theano.config.blas.ldflags = '-lblas -lgfortran'
    theano.config.nvcc.fastmath = True
    theano.config.nvcc.flags = '-D_FORCE_INLINES'
    theano.config.cxx = os.environ['CONDA_PREFIX'] + '/bin/g++'
else :
    raise Exception('No configuration found when the backend is ' + os.environ['KERAS_BACKEND'])

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

# pop out window for plots
%matplotlib qt5

import cytometer.deepcell as deepcell
import cytometer.deepcell_models as deepcell_models

# data paths
# * datadir = ~/Software/cytometer/data/deepcell/validation_data
# * outdir = ~/Software/cytometer.wiki/deepcell/validation_data
basedatadir = os.path.normpath(os.path.join(cytometer.__path__[0], '../data/deepcell'))
netdir = os.path.join(basedatadir, 'trained_networks')
wikidir = os.path.normpath(os.path.join(cytometer.__path__[0], '../../cytometer.wiki'))
datadir = os.path.join(basedatadir, 'validation_data')

# list of validation datasets
ch0_file = [
        '3T3/RawImages/phase.tif',
        'HeLa/RawImages/phase.tif',
        'HeLa_plating/10000K/RawImages/img_channel000_position000_time000000000_z000.tif',
        'HeLa_plating/1250K/RawImages/img_channel000_position000_time000000000_z000.tif',
        'HeLa_plating/20000K/RawImages/img_channel000_position000_time000000000_z000.tif',
        'HeLa_plating/2500K/RawImages/img_channel000_position000_time000000000_z000.tif',
        'HeLa_plating/5000K/RawImages/img_channel000_position000_time000000000_z000.tif',
        'MCF10A/RawImages/phase.tif',
        ]

ch1_file = [
        '3T3/RawImages/dapi.tif',
        'HeLa/RawImages/farred.tif',
        'HeLa_plating/10000K/RawImages/img_channel001_position000_time000000000_z000.tif',
        'HeLa_plating/1250K/RawImages/img_channel001_position000_time000000000_z000.tif',
        'HeLa_plating/20000K/RawImages/img_channel001_position000_time000000000_z000.tif',
        'HeLa_plating/2500K/RawImages/img_channel001_position000_time000000000_z000.tif',
        'HeLa_plating/5000K/RawImages/img_channel001_position000_time000000000_z000.tif',
        'MCF10A/RawImages/dapi.tif',
        ]

seg_file = [
        '3T3/Validation/3T3_validation_interior.tif',
        'HeLa/Validation/hela_validation_interior.tif',
        'HeLa_plating/10000K/Validation/10000K_feature_1.png',
        'HeLa_plating/1250K/Validation/1250K_feature_1.png',
        'HeLa_plating/20000K/Validation/20000K_feature_1.png',
        'HeLa_plating/2500K/Validation/2500K_feature_1.png',
        'HeLa_plating/5000K/Validation/5000K_feature_1.png',
        'MCF10A/Validation/MCF10A_validation_interior.tif',
        ]

# corresponding deepcell model weights
model_dir = [
        os.path.join(netdir, '3T3'),
        os.path.join(netdir, 'HeLa'),
        os.path.join(netdir, 'HeLa'),
        os.path.join(netdir, 'HeLa'),
        os.path.join(netdir, 'HeLa'),
        os.path.join(netdir, 'HeLa'),
        os.path.join(netdir, 'HeLa'),
        os.path.join(netdir, 'MCF10A')
        ]

# instantiate model (same for all validation data)
model = deepcell_models.sparse_bn_feature_net_61x61(batch_input_shape = (1,2,500,500), weights_path = os.path.join(netdir, model_dir[0], '2016-07-12_3T3_all_61x61_bn_feature_net_61x61_0.h5'))

# load image
im = plt.imread(os.path.join(datadir, ch0_file[0]))
im = np.resize(im, (2,) + im.shape)
im[1,:,:] = plt.imread(os.path.join(datadir, ch1_file[0]))

# apply model
im_out = model.predict(im.reshape((1, 2, 500, 500)))

im_out = im_out.reshape(3, 440, 440)
im_out = im_out.astype(dtype='bool')
plt.imshow(im_out[2,:,:])



# create input for validation
im = np.zeros((1,2,500,500), dtype='float32')
im[:,0,:,:] = plt.imread('/home/rcasero/Software/cytometer/data/deepcell/validation_data/HeLa/RawImages/phase.tif')
im[:,1,:,:] = plt.imread('/home/rcasero/Software/cytometer/data/deepcell/validation_data/HeLa/RawImages/farred.tif')

out = model.predict(im)
out_plt = np.transpose(out, (2, 3, 1, 0)).reshape(440,440,3)

plt.imshow(im[:,0,:,:].reshape(500,500))
plt.imshow(im[:,1,:,:].reshape(500,500))
plt.imshow(out_plt)

