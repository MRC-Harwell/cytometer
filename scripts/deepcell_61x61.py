#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:07:23 2017

@author: rcasero
"""

import os
os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['KERAS_BACKEND'] = 'tensorflow'

if 'LIBRARY_PATH' in os.environ:
    os.environ['LIBRARY_PATH'] = os.environ['CONDA_PREFIX'] + '/lib:' + os.environ['LIBRARY_PATH']
else:
    os.environ['LIBRARY_PATH'] = os.environ['CONDA_PREFIX'] + '/lib'

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

# fix "RuntimeError: Invalid DISPLAY variable" in cluster runs
#import matplotlib
#matplotlib.use('agg')

# load module dependencies
#import datetime
#import matplotlib.pyplot as plt
#import numpy as np

import cytometer.deepcell as deepcell
import cytometer.deepcell_models as deepcell_models
#import cytometer.models as models
#import cytometer.layers as layers
#reload(deepcell)
#reload(deepcell_models)
#reload(layers)
#reload(models)

model = deepcell_models.sparse_bn_feature_net_31x31()

# load pre-computed weights
weights_path='/home/rcasero/Software/cytometer/data/deepcell/trained_networks/slip/2017-07-14_slip_31x31_bn_feature_net_31x31_0.h5'
deepcell.set_weights(model, weights_path)

import h5py
foo = np.load('/home/rcasero/Software/DeepCell/trained_networks/slip/2017-06-06_slip_61x61_bn_feature_net_61x61_0.npz')
weights_path='/home/rcasero/Software/DeepCell/trained_networks/slip/2017-06-06_slip_61x61_bn_feature_net_61x61_0.h5'
foo = h5py.File(weights_path ,'r')
