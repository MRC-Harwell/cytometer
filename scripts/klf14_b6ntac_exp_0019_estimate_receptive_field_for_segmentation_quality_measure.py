"""
Training of CNN for regression of Dice coefficients.

Only one value per cell, located at the center of mass of the segmentation.

To avoid interference from adjacent cells during training, we do a colouring of the cell adjacency graph. The colouring
ensures that if we mask one colour, the receptive field of the network doesn't see more than one cell at a time.
"""

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import pickle
import inspect

# other imports
import glob
import shutil
import datetime
import numpy as np
import matplotlib.pyplot as plt

# use CPU for testing on laptop
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AvgPool2D, Activation, UpSampling2D, Dropout, concatenate

# for data parallelism in keras models
from keras.utils import multi_gpu_model

from receptivefield.keras import KerasReceptiveField

import cytometer.data
import cytometer.model_checkpoint_parallel
import cytometer.utils
import tensorflow as tf
from skimage.measure import regionprops

# # limit GPU memory used
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1.0
# set_session(tf.Session(config=config))

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False

# number of blocks to split each image into so that training fits into GPU memory
nblocks = 2

# number of folds for k-fold cross validation
n_folds = 11

# number of epochs for training
epochs = 20

'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

# script name to identify this experiment
experiment_id = inspect.getfile(inspect.currentframe())
if experiment_id == '<input>':
    experiment_id = 'unknownscript'
else:
    experiment_id = os.path.splitext(os.path.basename(experiment_id))[0]

'''Load labels to evaluate what input size we need 
'''

saved_contour_model_basename = 'klf14_b6ntac_exp_0006_cnn_contour'  # contour

# load list of images, and indices for training vs. testing indices
contour_model_kfold_filename = os.path.join(saved_models_dir, saved_contour_model_basename + '_info.pickle')
aux = pickle.load(open(contour_model_kfold_filename, 'rb'))
im_orig_file_list = aux['file_list']

# correct home directory if we are in a different system than what was used to train the models
im_orig_file_list = cytometer.data.change_home_directory(im_orig_file_list, '/users/rittscher/rcasero', home,
                                                         check_isfile=True)

# load the train and test data: im, seg, dmap and mask data
dataset, _, _ = cytometer.data.load_datasets(im_orig_file_list, prefix_from='im', prefix_to=['lab'], nblocks=nblocks)

# get the bounding box size of each cell
bbox_len = []
for i in range(dataset['lab'].shape[0]):
    props = regionprops(dataset['lab'][i, :, :, 0])
    for j in range(len(props)):
        if props[j]['label'] > 1:
            bbox = props[j]['bbox']
            bbox_len.append(bbox[3] - bbox[1])
            bbox_len.append(bbox[2] - bbox[0])

print('Maximum box size: ' + str(np.max(bbox_len)))

# plot box size values
plt.clf()
plt.hist(bbox_len)
plt.xlabel('Box size', fontsize=18)
plt.ylabel('Count', fontsize=18)

'''CNN classifier to estimate Dice coefficient
'''


def unet(input_shape=(256, 256, 1), for_receptive_field=False):

    if for_receptive_field:
        activation = 'linear'
        pool_f = AvgPool2D
    else:
        activation = 'relu'
        pool_f = MaxPooling2D

    inputs = Input(input_shape, name='input_image')
    conv1 = Conv2D(64, 3, activation=activation, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation=activation, padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = pool_f(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation=activation, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation=activation, padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = pool_f(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation=activation, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation=activation, padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = pool_f(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation=activation, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation=activation, padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = pool_f(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation=activation, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation=activation, padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation=activation, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 3, activation=activation, padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation=activation, padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation=activation, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation=activation, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation=activation, padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation=activation, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation=activation, padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation=activation, padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation=activation, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation=activation, padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation=activation, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation=activation, padding='same', kernel_initializer='he_normal')(conv9)
    if for_receptive_field:
        regression_output = Conv2D(1, 1, activation='linear', name='regression_output')(conv9)
    else:
        regression_output = Conv2D(1, 1, activation='sigmoid', name='regression_output')(conv9)

    model = Model(inputs=inputs, outputs=regression_output)

    return model


def fcn_sherrah2016_regression(input_shape, for_receptive_field=False):

    input = Input(shape=input_shape, dtype='float32', name='input_image')

    x = Conv2D(filters=32, kernel_size=(5, 5), strides=1, dilation_rate=1, padding='same')(input)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(3, 3), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)

    x = Conv2D(filters=int(96/2), kernel_size=(5, 5), strides=1, dilation_rate=2, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(5, 5), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(x)

    x = Conv2D(filters=int(128/2), kernel_size=(3, 3), strides=1, dilation_rate=4, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(9, 9), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(9, 9), strides=1, padding='same')(x)

    x = Conv2D(filters=int(196/2), kernel_size=(3, 3), strides=1, dilation_rate=8, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(17, 17), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(17, 17), strides=1, padding='same')(x)

    x = Conv2D(filters=int(512/2), kernel_size=(3, 3), strides=1, dilation_rate=16, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
    else:
        x = Activation('relu')(x)

    # regression output
    x = Conv2D(filters=1, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)
    if for_receptive_field:
        regression_output = Activation('linear', name='regression_output')(x)
    else:
        regression_output = Activation('hard_sigmoid', name='regression_output')(x)

    return Model(inputs=input, outputs=[regression_output])

'''Estimate the effective receptive field pre-training
'''

# estimate receptive field of the model
def model_build_func(input_shape):
    model = fcn_sherrah2016_regression(input_shape=input_shape, for_receptive_field=True)
    return model

rf = KerasReceptiveField(model_build_func, init_weights=True)

rf_params = rf.compute(
    input_shape=(512, 512, 3),
    input_layer='input_image',
    output_layers=['regression_output'])
print(rf_params)

print('Sherrah 2016 effective receptive field: ' + str(rf._rf_params[0].size))

# estimate receptive field of the model
def model_build_func(input_shape):
    # model = fcn_sherrah2016_regression(input_shape=input_shape, for_receptive_field=True)
    model = unet(input_shape=input_shape, for_receptive_field=True)
    return model

rf = KerasReceptiveField(model_build_func, init_weights=True)

rf_params = rf.compute(
    input_shape=(512, 512, 3),
    input_layer='input_image',
    output_layers=['regression_output'])
print(rf_params)

print('U-net effective receptive field: ' + str(rf._rf_params[0].size))
