#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 17:59:37 2017

@author: rcasero
"""

# Keras backend
import os
os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['KERAS_BACKEND'] = 'tensorflow'

# access to python libraries
if 'LIBRARY_PATH' in os.environ:
    os.environ['LIBRARY_PATH'] = os.environ['CONDA_PREFIX'] + '/lib:' + os.environ['LIBRARY_PATH']
else:
    os.environ['LIBRARY_PATH'] = os.environ['CONDA_PREFIX'] + '/lib'

# configure Theano global options
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cuda0'

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

# load dependencies
import matplotlib.pyplot as plt
import numpy as np
