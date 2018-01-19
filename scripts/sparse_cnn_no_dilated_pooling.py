#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 22:15:54 2017

@author: rcasero
"""

import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

import sys

cytometer_dir = os.path.expanduser("~/Software/cytometer")
if cytometer_dir not in sys.path:
    sys.path.append(cytometer_dir)

# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# different versions of conda keep the path in different variables
if 'CONDA_ENV_PATH' in os.environ:
    conda_env_path = os.environ['CONDA_ENV_PATH']
elif 'CONDA_PREFIX' in os.environ:
    conda_env_path = os.environ['CONDA_PREFIX']
else:
    conda_env_path = '.'

if os.environ['KERAS_BACKEND'] == 'theano':
    # configure Theano
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cuda0,' \
                                 + 'dnn.include_path=' + conda_env_path + '/include,' \
                                 + 'dnn.library_path=' + conda_env_path + '/lib,' \
                                 + 'gcc.cxxflags=-I/usr/local/cuda-9.1/targets/x86_64-linux/include'
    import theano
elif os.environ['KERAS_BACKEND'] == 'tensorflow':
    # configure tensorflow
    import tensorflow
else:
    raise Exception('No configuration found when the backend is ' + os.environ['KERAS_BACKEND'])

import keras.backend as K
#from keras import __version__ as keras_version
#from pkg_resources import parse_version

# configure Keras, to avoid using file ~/.keras/keras.json
K.set_image_dim_ordering('tf')
K.set_floatx('float32')
K.set_epsilon('1e-07')
# fix "RuntimeError: Invalid DISPLAY variable" in cluster runs
# import matplotlib
# matplotlib.use('agg')

# load module dependencies
#import datetime
import matplotlib.pyplot as plt
import numpy as np

#import cytometer.deepcell as deepcell
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
