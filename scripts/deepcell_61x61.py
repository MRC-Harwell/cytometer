#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:07:23 2017

@author: rcasero
"""

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
    theano.config.gcc.cxxflags = '-D_hypot=hypot' # fix "error: narrowing conversion"
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

# fix "RuntimeError: Invalid DISPLAY variable" in cluster runs
#import matplotlib
#matplotlib.use('agg')

import cytometer.deepcell as deepcell
import cytometer.deepcell_models as deepcell_models

# instantiate model
model = deepcell_models.sparse_bn_feature_net_61x61(batch_input_shape = (1,2,500,500))

optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
lr_sched = deepcell.rate_scheduler(lr = 0.01, decay = 0.95)
class_weight = {0:1, 1:1, 2:1}

model.compile(loss='categorical_crossentropy',
			  optimizer=optimizer,
			  metrics=['accuracy'])

# load pre-computed weights
weights_path='/home/rcasero/Software/cytometer/data/deepcell/trained_networks/HeLa/2017-06-21_HeLa_all_61x61_bn_feature_net_61x61_0.h5'
model = deepcell.set_weights(model, weights_path)

# create input for validation
im = np.zeros((1,2,500,500), dtype='float32')
im[:,0,:,:] = plt.imread('/home/rcasero/Software/DeepCell/validation_data/HeLa/RawImages/phase.tif')
im[:,1,:,:] = plt.imread('/home/rcasero/Software/DeepCell/validation_data/HeLa/RawImages/farred.tif')

out = model.predict(im)
out_plt = np.transpose(out, (2, 3, 1, 0)).reshape(440,440,3)

plt.imshow(im[:,0,:,:].reshape(500,500))
plt.imshow(im[:,1,:,:].reshape(500,500))
plt.imshow(out_plt)

