#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 22:15:54 2017

@author: rcasero
"""

import os
os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['KERAS_BACKEND'] = 'tensorflow'

if 'LIBRARY_PATH' in os.environ:
    os.environ['LIBRARY_PATH'] = os.environ['CONDA_PREFIX'] + '/lib:' + os.environ['LIBRARY_PATH']
else:
    os.environ['LIBRARY_PATH'] = os.environ['CONDA_PREFIX'] + '/lib'

from importlib import reload
import keras

# configure Keras, to avoid using file ~/.keras/keras.json
keras.backend.set_image_data_format('channels_first') # theano's image format (required by DeepCell)

# fix "RuntimeError: Invalid DISPLAY variable" in cluster runs
#import matplotlib
#matplotlib.use('agg')

# load module dependencies
import datetime
import matplotlib.pyplot as plt
import numpy as np

import cytometer.deepcell as deepcell
import cytometer.deepcell_models as deepcell_models
import cytometer.models as models
import cytometer.layers as layers
#reload(deepcell)
#reload(deepcell_models)
reload(layers)
reload(models)

model = models.sparse_feature_net_61x61_no_dilated_pooling()

optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
lr_sched = deepcell.rate_scheduler(lr = 0.01, decay = 0.95)
class_weight = {0:1, 1:1, 2:1}

model.compile(loss='categorical_crossentropy',
			  optimizer=optimizer,
			  metrics=['accuracy'])

