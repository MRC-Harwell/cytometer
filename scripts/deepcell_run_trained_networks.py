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
import sklearn.metrics

# data paths
# * datadir = ~/Software/cytometer/data/deepcell/validation_data
# * outdir = ~/Software/cytometer.wiki/deepcell/validation_data
basedatadir = os.path.normpath(os.path.join(cytometer.__path__[0], '../data/deepcell'))
netdir = os.path.join(basedatadir, 'trained_networks')
wikidir = os.path.normpath(os.path.join(cytometer.__path__[0], '../../cyto￼meter.wiki'))
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

model_weights_file = [
        '2016-07-12_3T3_all_61x61_bn_feature_net_61x61_',
        '2017-06-21_HeLa_all_61x61_bn_feature_net_61x61_',
        '2017-06-21_HeLa_all_61x61_bn_feature_net_61x61_',
        '2017-06-21_HeLa_all_61x61_bn_feature_net_61x61_',
        '2017-06-21_HeLa_all_61x61_bn_feature_net_61x61_',
        '2017-06-21_HeLa_all_61x61_bn_feature_net_61x61_',
        '2017-06-21_HeLa_all_61x61_bn_feature_net_61x61_',
        '2016-07-11_MCF10A_61x61_bn_feature_net_61x61_'
        ]

# index of validation dataset
i = 1

# amount of padding in the X,Y coordinates
pad_xy = 30

# load image
im = plt.imread(os.path.join(datadir, ch0_file[i]))
im = np.resize(im, (2,) + im.shape)
im[1,:,:] = plt.imread(os.path.join(datadir, ch1_file[i]))

# image preprocessing
im = im.astype(dtype='float32')
im[0,:,:] = deepcell.process_image(im[0,:,:], 30, 30)
im[1,:,:] = deepcell.process_image(im[1,:,:], 30, 30)

# split into chunks, so that the GPU doesn't crash the system
block_slices, blocks, im_padded = pymproc.block_split(im, nblocks=(1,1,1), pad_width=((0,0),(pad_xy,pad_xy),(pad_xy,pad_xy)), mode='reflect')

# apply models and compute average result
blocks_out = []
for b in blocks:
    b = b.reshape((1,) + b.shape)
    for j in range(5):
        # instantiate model (same for all validation data)
        model = deepcell_models.sparse_bn_feature_net_61x61(batch_input_shape = b.shape)
        model = deepcell.set_weights(model, os.path.join(netdir, model_dir[i], model_weights_file[i] + str(j) + '.h5'))
        if j == 0:
            b_out = model.predict(b)
        else:
            b_out += model.predict(b)
    b_out /= 5
    b_out = b_out.reshape(b_out.shape[1:])
    blocks_out += [b_out]

# correct slice indices
block_slices_out = []
for sl, b_out in zip(block_slices, blocks_out):
    block_slices_out += [[slice(0,3,1), slice(sl[1].start, sl[1].start+b_out.shape[1],1), slice(sl[2].start, sl[2].start+b_out.shape[2], 1)]]

# reassemble images
im_out, _ = pymproc.block_stack(blocks_out, block_slices_out, pad_width=((0,0),(pad_xy-30,pad_xy-30),(pad_xy-30,pad_xy-30)))

# plot output
plt.subplot(2,2,1)
plt.imshow(im[0,:,:])
plt.subplot(2,2,2)
plt.imshow(im_out[0,:,:])
plt.subplot(2,2,3)
plt.imshow(im_out[1,:,:])
plt.subplot(2,2,4)
plt.imshow(im_out[2,:,:])

# load hand segmentation
seg = plt.imread(os.path.join(datadir, seg_file[i]))
seg = seg > 0

# compute ROC and area under the curve
fpr, tpr, _ = sklearn.metrics.roc_curve(seg.flatten(), im_out[1,:,:].flatten(), drop_intermediate=True)
roc_auc = sklearn.metrics.auc(fpr, tpr)

# plot ROC
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
