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
#reload(layers)
#reload(models)


# instantiate CNN
model = models.sparse_feature_net_61x61_no_dilated_pooling()

optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
lr_sched = deepcell.rate_scheduler(lr = 0.01, decay = 0.95)
class_weight = {0:1, 1:1, 2:1}

model.compile(loss='categorical_crossentropy',
			  optimizer=optimizer,
			  metrics=['accuracy'])

# load data
datadir = "/home/rcasero/Software/cytometer/data/deepcell/training_data_npz/3T3"
datafile = "3T3_all_61x61.npz"
outdir = "/tmp"
outfile = "3T3_all_61x61"

train_dict, (X_test, Y_test) = deepcell.get_data_sample(os.path.join(datadir, datafile))


it = 0 # iteration
batch_size = 256
n_epoch = 25

training_data_file_name = os.path.join(direc_data, dataset + ".npz")
todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

file_name_save = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it)  + ".h5")

file_name_save_loss = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it) + ".npz")

train_dict, (X_test, Y_test) = deepcell.get_data_sample(training_data_file_name)

# the data, shuffled and split between train and test sets
print('X_train shape:', train_dict["channels"].shape)
print(train_dict["pixels_x"].shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
