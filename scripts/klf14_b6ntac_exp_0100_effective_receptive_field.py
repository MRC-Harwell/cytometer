"""
Estimate effective receptive field for the CNNs used in pipeline v7.
"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0100_effective_receptive_field'

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

# # limit number of GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AvgPool2D, Activation, BatchNormalization

# for data parallelism in keras models
from keras.utils import multi_gpu_model

from receptivefield.keras import KerasReceptiveField

import cytometer.data
import cytometer.model_checkpoint_parallel
import cytometer.utils
# import cytometer.resnet
import keras.applications
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

# number of folds for k-fold cross validation
n_folds = 10

'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

'''CNNs
'''

##########################################################################################
# klf14_b6ntac_exp_0086_cnn_dmap.py
##########################################################################################

def fcn_sherrah2016_regression(input_shape, for_receptive_field=False):

    def activation_pooling_if(for_receptive_field, pool_size, x):
        if for_receptive_field:
            # for estimation of receptive field, we need to use linear operators
            x = Activation('linear')(x)
            x = AvgPool2D(pool_size=pool_size, strides=1, padding='same')(x)
        else:
            # for regular training and inference, we use non-linear operators
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=pool_size, strides=1, padding='same')(x)
        return x

    input = Input(shape=input_shape, dtype='float32', name='input_image')

    x = Conv2D(filters=32, kernel_size=(5, 5), strides=1, dilation_rate=1, padding='same',
               kernel_initializer='he_uniform')(input)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(3, 3), x=x)

    x = Conv2D(filters=48, kernel_size=(5, 5), strides=1, dilation_rate=2, padding='same',
               kernel_initializer='he_uniform')(x)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(5, 5), x=x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, dilation_rate=4, padding='same',
               kernel_initializer='he_uniform')(x)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(9, 9), x=x)

    x = Conv2D(filters=98, kernel_size=(3, 3), strides=1, dilation_rate=8, padding='same',
               kernel_initializer='he_uniform')(x)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(17, 17), x=x)

    x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, dilation_rate=16, padding='same',
               kernel_initializer='he_uniform')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
    else:
        x = Activation('relu')(x)

    # regression output
    regression_output = Conv2D(filters=1, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same',
                               kernel_initializer='he_uniform', name='regression_output')(x)

    return Model(inputs=input, outputs=[regression_output])

## Estimate ERF

# output receptive field sizes (height, width)
erf = np.zeros(shape=(n_folds, 2), dtype=np.float32)

for i_fold in range(n_folds):
    model_filename = os.path.join(saved_models_dir, 'klf14_b6ntac_exp_0086_cnn_dmap_model_fold_' + str(i_fold) + '.h5')

    # function to pass to KerasReceptiveField
    def model_build_func(input_shape):
        model_erf = fcn_sherrah2016_regression(input_shape=input_shape, for_receptive_field=True)

        # load model with trained weights
        model = keras.models.load_model(model_filename)

        # copy weights from trained model to EFC model
        for layer_target, layer_source in zip(model_erf.layers, model.layers):

            weights_source = layer_source.get_weights()
            if len(weights_source) > 0:
                # copy weights over
                layer_target.set_weights(weights_source)

        return model_erf

    rf = KerasReceptiveField(model_build_func, init_weights=False)

    rf_params = rf.compute(
        input_shape=(512, 512, 3),
        input_layer='input_image',
        output_layers=['regression_output'])
    print(rf_params)

    print('Effective receptive field: ' + str(rf._rf_params[0].size))

    # save output for later (height, width)
    erf[i_fold, :] = np.array(rf._rf_params[0].size)[::-1]


##########################################################################################
# klf14_b6ntac_exp_0091_cnn_contour_after_dmap.py
##########################################################################################

def fcn_sherrah2016_classifier(input_shape, for_receptive_field=False):


    def activation_if(for_receptive_field, x):
        if for_receptive_field:
            # for estimation of receptive field, we need to use linear operators
            x = Activation('linear')(x)
            x = BatchNormalization(axis=3)(x)
        else:
            # for regular training and inference, we use non-linear operators
            x = Activation('relu')(x)
            x = BatchNormalization(axis=3)(x)
        return x

    def activation_pooling_if(for_receptive_field, pool_size, x):
        if for_receptive_field:
            # for estimation of receptive field, we need to use linear operators
            x = Activation('linear')(x)
            x = AvgPool2D(pool_size=pool_size, strides=1, padding='same')(x)
            x = BatchNormalization(axis=3)(x)
        else:
            # for regular training and inference, we use non-linear operators
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=pool_size, strides=1, padding='same')(x)
            x = BatchNormalization(axis=3)(x)
        return x

    input = Input(shape=input_shape, dtype='float32', name='input_image')

    x = Conv2D(filters=32, kernel_size=(5, 5), strides=1, dilation_rate=1, padding='same')(input)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(3, 3), x=x)

    x = Conv2D(filters=48, kernel_size=(5, 5), strides=1, dilation_rate=2, padding='same')(x)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(5, 5), x=x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, dilation_rate=4, padding='same')(x)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(9, 9), x=x)

    x = Conv2D(filters=98, kernel_size=(3, 3), strides=1, dilation_rate=8, padding='same')(x)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(17, 17), x=x)

    x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, dilation_rate=16, padding='same')(x)
    x = activation_if(for_receptive_field=for_receptive_field, x=x)

    # dimensionality reduction to 1 feature map
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)
    x = activation_if(for_receptive_field=for_receptive_field, x=x)

    x = Conv2D(filters=8, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)
    x = activation_if(for_receptive_field=for_receptive_field, x=x)

    x = Conv2D(filters=1, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)

    # classification output
    # replace hard_sigmoid by linear for ERF
    classification_output = Activation('linear', name='classification_output')(x)

    return Model(inputs=input, outputs=[classification_output])


## Estimate ERF

# output receptive field sizes (height, width)
erf = np.zeros(shape=(n_folds, 2), dtype=np.float32)

for i_fold in range(n_folds):
    model_filename = os.path.join(saved_models_dir,
                                  'klf14_b6ntac_exp_0091_cnn_contour_after_dmap_model_fold_' + str(i_fold) + '.h5')

    # function to pass to KerasReceptiveField
    def model_build_func(input_shape):
        model_erf = fcn_sherrah2016_classifier(input_shape=input_shape, for_receptive_field=True)

        # load model with trained weights
        model = keras.models.load_model(model_filename)

        # copy weights from trained model to EFC model
        for layer_target, layer_source in zip(model_erf.layers, model.layers):

            weights_source = layer_source.get_weights()
            if len(weights_source) > 0:
                # copy weights over
                layer_target.set_weights(weights_source)

        return model_erf

    rf = KerasReceptiveField(model_build_func, init_weights=False)

    rf_params = rf.compute(
        input_shape=(512, 512, 1),
        input_layer='input_image',
        output_layers=['classification_output'])
    print(rf_params)

    print('Effective receptive field: ' + str(rf._rf_params[0].size))

    # save output for later (height, width)
    erf[i_fold, :] = np.array(rf._rf_params[0].size)[::-1]


##########################################################################################
# klf14_b6ntac_exp_0095_cnn_tissue_classifier_fcn.py
##########################################################################################

def fcn_sherrah2016_classifier(input_shape, for_receptive_field=False):


    def activation_if(for_receptive_field, x):
        if for_receptive_field:
            # for estimation of receptive field, we need to use linear operators
            x = Activation('linear')(x)
            x = BatchNormalization(axis=3)(x)
        else:
            # for regular training and inference, we use non-linear operators
            x = Activation('relu')(x)
            x = BatchNormalization(axis=3)(x)
        return x

    def activation_pooling_if(for_receptive_field, pool_size, x):
        if for_receptive_field:
            # for estimation of receptive field, we need to use linear operators
            x = Activation('linear')(x)
            x = AvgPool2D(pool_size=pool_size, strides=1, padding='same')(x)
            x = BatchNormalization(axis=3)(x)
        else:
            # for regular training and inference, we use non-linear operators
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=pool_size, strides=1, padding='same')(x)
            x = BatchNormalization(axis=3)(x)
        return x

    input = Input(shape=input_shape, dtype='float32', name='input_image')

    x = Conv2D(filters=32, kernel_size=(5, 5), strides=1, dilation_rate=1, padding='same')(input)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(3, 3), x=x)

    x = Conv2D(filters=48, kernel_size=(5, 5), strides=1, dilation_rate=2, padding='same')(x)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(5, 5), x=x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, dilation_rate=4, padding='same')(x)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(9, 9), x=x)

    x = Conv2D(filters=98, kernel_size=(3, 3), strides=1, dilation_rate=8, padding='same')(x)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(17, 17), x=x)

    x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, dilation_rate=16, padding='same')(x)
    x = activation_if(for_receptive_field=for_receptive_field, x=x)

    # dimensionality reduction to 1 feature map
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)
    x = activation_if(for_receptive_field=for_receptive_field, x=x)

    x = Conv2D(filters=8, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)
    x = activation_if(for_receptive_field=for_receptive_field, x=x)

    x = Conv2D(filters=1, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)

    # classification output
    # replace hard_sigmoid by linear for ERF
    classification_output = Activation('linear', name='classification_output')(x)

    return Model(inputs=input, outputs=[classification_output])


## Estimate ERF

# output receptive field sizes (height, width)
erf = np.zeros(shape=(n_folds, 2), dtype=np.float32)

for i_fold in range(n_folds):
    model_filename = os.path.join(saved_models_dir,
                                  'klf14_b6ntac_exp_0095_cnn_tissue_classifier_fcn_model_fold_' + str(i_fold) + '.h5')

    # function to pass to KerasReceptiveField
    def model_build_func(input_shape):
        model_erf = fcn_sherrah2016_classifier(input_shape=input_shape, for_receptive_field=True)

        # load model with trained weights
        model = keras.models.load_model(model_filename)

        # copy weights from trained model to EFC model
        for layer_target, layer_source in zip(model_erf.layers, model.layers):

            weights_source = layer_source.get_weights()
            if len(weights_source) > 0:
                # copy weights over
                layer_target.set_weights(weights_source)

        return model_erf

    rf = KerasReceptiveField(model_build_func, init_weights=False)

    rf_params = rf.compute(
        input_shape=(512, 512, 3),
        input_layer='input_image',
        output_layers=['classification_output'])
    print(rf_params)

    print('Effective receptive field: ' + str(rf._rf_params[0].size))

    # save output for later (height, width)
    erf[i_fold, :] = np.array(rf._rf_params[0].size)[::-1]


##########################################################################################
# klf14_b6ntac_exp_0089_cnn_segmentation_correction_overlapping_scaled_contours.py
##########################################################################################

def fcn_sherrah2016_regression(input_shape, for_receptive_field=False):

    def activation_if(for_receptive_field, x):
        if for_receptive_field:
            # for estimation of receptive field, we need to use linear operators
            x = Activation('linear')(x)
            x = BatchNormalization(axis=3)(x)
        else:
            # for regular training and inference, we use non-linear operators
            x = Activation('relu')(x)
            x = BatchNormalization(axis=3)(x)
        return x

    def activation_pooling_if(for_receptive_field, pool_size, x):
        if for_receptive_field:
            # for estimation of receptive field, we need to use linear operators
            x = Activation('linear')(x)
            x = AvgPool2D(pool_size=pool_size, strides=1, padding='same')(x)
            x = BatchNormalization(axis=3)(x)
        else:
            # for regular training and inference, we use non-linear operators
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=pool_size, strides=1, padding='same')(x)
            x = BatchNormalization(axis=3)(x)
        return x

    cnn_input = Input(shape=input_shape, dtype='float32', name='input_image')

    x = Conv2D(filters=32, kernel_size=(5, 5), strides=1, dilation_rate=1, padding='same',
               kernel_initializer='he_uniform')(cnn_input)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(3, 3), x=x)

    x = Conv2D(filters=48, kernel_size=(5, 5), strides=1, dilation_rate=2, padding='same',
               kernel_initializer='he_uniform')(x)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(5, 5), x=x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, dilation_rate=4, padding='same',
               kernel_initializer='he_uniform')(x)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(9, 9), x=x)

    x = Conv2D(filters=98, kernel_size=(3, 3), strides=1, dilation_rate=8, padding='same',
               kernel_initializer='he_uniform')(x)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(17, 17), x=x)

    x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, dilation_rate=16, padding='same',
               kernel_initializer='he_uniform')(x)
    x = activation_if(for_receptive_field=for_receptive_field, x=x)

    # dimensionality reduction to 1 feature map
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same',
               kernel_initializer='he_uniform')(x)
    x = activation_if(for_receptive_field=for_receptive_field, x=x)

    x = Conv2D(filters=8, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same',
               kernel_initializer='he_uniform')(x)
    x = activation_if(for_receptive_field=for_receptive_field, x=x)

    regression_output = Conv2D(filters=1, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same',
                               kernel_initializer='he_uniform', name='regression_output')(x)

    return Model(inputs=cnn_input, outputs=[regression_output])


## Estimate ERF

# output receptive field sizes (height, width)
erf = np.zeros(shape=(n_folds, 2), dtype=np.float32)

for i_fold in range(n_folds):
    model_filename = os.path.join(saved_models_dir,
                                  'klf14_b6ntac_exp_0089_cnn_segmentation_correction_overlapping_scaled_contours_model_fold_' + str(i_fold) + '.h5')

    # function to pass to KerasReceptiveField
    def model_build_func(input_shape):
        model_erf = fcn_sherrah2016_regression(input_shape=input_shape, for_receptive_field=True)

        # load model with trained weights
        model = keras.models.load_model(model_filename)

        # copy weights from trained model to EFC model
        for layer_target, layer_source in zip(model_erf.layers, model.layers):

            weights_source = layer_source.get_weights()
            if len(weights_source) > 0:
                # copy weights over
                layer_target.set_weights(weights_source)

        return model_erf

    rf = KerasReceptiveField(model_build_func, init_weights=False)

    rf_params = rf.compute(
        input_shape=(512, 512, 3),
        input_layer='input_image',
        output_layers=['regression_output'])
    print(rf_params)

    print('Effective receptive field: ' + str(rf._rf_params[0].size))

    # save output for later (height, width)
    erf[i_fold, :] = np.array(rf._rf_params[0].size)[::-1]

