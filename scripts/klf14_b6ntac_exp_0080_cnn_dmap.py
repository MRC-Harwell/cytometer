'''
Dmap regression for all folds.

Training vs testing is done at the histology slide level, not at the window level. This way, we really guarantee that
the network has not been trained with data sampled from the same image as the test data.

Like 0056, but:
    * k-folds = 10.
    * Adding Gianluca's data.
    * Fill small gaps in the training mask with a (3,3) dilation.
    * Save training history variable, instead of relying on text output.
'''

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0080_cnn_dmap'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import json
import pickle
import warnings

# other imports
import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt

# # limit number of GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AvgPool2D, Activation

# for data parallelism in keras models
from keras.utils import multi_gpu_model

import cytometer.data
import cytometer.model_checkpoint_parallel
import tensorflow as tf

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

# training parameters
epochs = 100
batch_size = 10

'''Directories and filenames'''

# data paths
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
klf14_training_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training')
klf14_training_non_overlap_data_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training_non_overlap')
klf14_training_augmented_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training_augmented')

c3h_root_data_dir = os.path.join(home, 'Data/cytometer_data/c3h')
c3h_training_dir = os.path.join(c3h_root_data_dir, 'c3h_hfd_training')
c3h_training_non_overlap_data_dir = os.path.join(c3h_root_data_dir, 'c3h_hfd_training_non_overlap')
c3h_training_augmented_dir = os.path.join(c3h_root_data_dir, 'c3h_hfd_training_augmented')

saved_models_dir = os.path.join(klf14_root_data_dir, 'saved_models')

saved_kfolds_filename = 'klf14_b6ntac_exp_0077_generate_kfolds.pickle'

'''CNN Model'''


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

    x = Conv2D(filters=int(48), kernel_size=(5, 5), strides=1, dilation_rate=2, padding='same',
               kernel_initializer='he_uniform')(x)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(5, 5), x=x)

    x = Conv2D(filters=int(64), kernel_size=(3, 3), strides=1, dilation_rate=4, padding='same',
               kernel_initializer='he_uniform')(x)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(9, 9), x=x)

    x = Conv2D(filters=int(98), kernel_size=(3, 3), strides=1, dilation_rate=8, padding='same',
               kernel_initializer='he_uniform')(x)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(17, 17), x=x)

    x = Conv2D(filters=int(256), kernel_size=(3, 3), strides=1, dilation_rate=16, padding='same',
               kernel_initializer='he_uniform')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
    else:
        x = Activation('relu')(x)

    # regression output
    regression_output = Conv2D(filters=1, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same',
                               kernel_initializer='he_uniform', name='regression_output')(x)

    return Model(inputs=input, outputs=[regression_output])


'''Load folds'''

# load list of images, and indices for training vs. testing indices
contour_model_kfold_filename = os.path.join(saved_models_dir, saved_kfolds_filename)
with open(contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
svg_file_list = aux['file_list']
idx_test_all = aux['idx_test']
idx_train_all = aux['idx_train']

'''Model training'''

# TIF files that correspond to the SVG files (without augmentation)
im_orig_file_list = []
for i, file in enumerate(svg_file_list):
    im_orig_file_list.append(file.replace('.svg', '.tif'))
    im_orig_file_list[i] = os.path.join(os.path.dirname(im_orig_file_list[i]) + '_augmented',
                                        'im_seed_nan_' + os.path.basename(im_orig_file_list[i]))

    # check that files exist
    if not os.path.isfile(file):
        warnings.warn('i = ' + str(i) + ': File does not exist: ' + os.path.basename(file))
    if not os.path.isfile(im_orig_file_list[i]):
        warnings.warn('i = ' + str(i) + ': File does not exist: ' + os.path.basename(im_orig_file_list[i]))

# loop each fold: we split the data into train vs test, train a model, and compute errors with the
# test data. In each fold, the test data is different
# for i_fold, idx_test in enumerate(idx_test_all):
history = []
for i_fold, idx_test in enumerate(idx_orig_test_all):

    '''Load data'''

    # split the data list into training and testing lists
    im_test_file_list, im_train_file_list = cytometer.data.split_list(im_orig_file_list, idx_test)

    # add the augmented image files
    im_test_file_list = cytometer.data.augment_file_list(im_test_file_list, '_nan_', '_*_')
    im_train_file_list = cytometer.data.augment_file_list(im_train_file_list, '_nan_', '_*_')

    # load the train and test data: im, seg, dmap and mask data
    train_dataset, train_file_list, train_shuffle_idx = \
        cytometer.data.load_datasets(im_train_file_list, prefix_from='im', prefix_to=['im', 'dmap', 'mask'],
                                     nblocks=nblocks, shuffle_seed=i_fold)
    test_dataset, test_file_list, test_shuffle_idx = \
        cytometer.data.load_datasets(im_test_file_list, prefix_from='im', prefix_to=['im', 'dmap', 'mask'],
                                     nblocks=nblocks, shuffle_seed=i_fold)

    # remove training data where the mask has very few valid pixels
    train_dataset = cytometer.data.remove_poor_data(train_dataset, prefix='mask', threshold=1000)
    test_dataset = cytometer.data.remove_poor_data(test_dataset, prefix='mask', threshold=1000)

    # fill in the little gaps in the mask
    kernel = np.ones((3, 3), np.uint8)
    for i in range(test_dataset['mask'].shape[0]):
        test_dataset['mask'][i, :, :, 0] = cv2.dilate(test_dataset['mask'][i, :, :, 0].astype(np.uint8),
                                                      kernel=kernel, iterations=1)
    for i in range(train_dataset['mask'].shape[0]):
        train_dataset['mask'][i, :, :, 0] = cv2.dilate(train_dataset['mask'][i, :, :, 0].astype(np.uint8),
                                                       kernel=kernel, iterations=1)

    if DEBUG:
        i = 150
        plt.clf()
        for pi, prefix in enumerate(train_dataset.keys()):
            plt.subplot(1, len(train_dataset.keys()), pi + 1)
            if train_dataset[prefix].shape[-1] < 3:
                plt.imshow(train_dataset[prefix][i, :, :, 0])
            else:
                plt.imshow(train_dataset[prefix][i, :, :, :])
            plt.title('out[' + prefix + ']')

        i = 22
        plt.clf()
        for pi, prefix in enumerate(test_dataset.keys()):
            plt.subplot(1, len(test_dataset.keys()), pi + 1)
            if test_dataset[prefix].shape[-1] < 3:
                plt.imshow(test_dataset[prefix][i, :, :, 0])
            else:
                plt.imshow(test_dataset[prefix][i, :, :, :])
            plt.title('out[' + prefix + ']')

    '''Convolutional neural network training
    
    Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
    function
    '''

    # list all CPUs and GPUs
    device_list = K.get_session().list_devices()

    # number of GPUs
    gpu_number = np.count_nonzero([':GPU:' in str(x) for x in device_list])

    # instantiate model
    with tf.device('/cpu:0'):
        model = fcn_sherrah2016_regression(input_shape=train_dataset['im'].shape[1:])

    saved_model_filename = os.path.join(saved_models_dir, experiment_id + '_model_fold_' + str(i_fold) + '.h5')

    if gpu_number > 1:  # compile and train model: Multiple GPUs

        # checkpoint to save model after each epoch
        checkpointer = cytometer.model_checkpoint_parallel.ModelCheckpoint(filepath=saved_model_filename,
                                                                           verbose=1, save_best_only=True)
        # compile model
        parallel_model = multi_gpu_model(model, gpus=gpu_number)
        parallel_model.compile(loss={'regression_output': 'mean_absolute_error'},
                               optimizer='Adadelta',
                               metrics={'regression_output': ['mse', 'mae']},
                               sample_weight_mode='element')

        # train model
        tic = datetime.datetime.now()
        hist = parallel_model.fit(train_dataset['im'],
                                  {'regression_output': train_dataset['dmap']},
                                  sample_weight={'regression_output': train_dataset['mask'][..., 0]},
                                  validation_data=(test_dataset['im'],
                                                   {'regression_output': test_dataset['dmap']},
                                                   {'regression_output': test_dataset['mask'][..., 0]}),
                                  batch_size=batch_size, epochs=epochs, initial_epoch=0,
                                  callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))
        history.append(hist.history)

    else:  # compile and train model: One GPU

        # checkpoint to save model after each epoch
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=saved_model_filename,
                                                       verbose=1, save_best_only=True)

        # compile model
        model.compile(loss={'regression_output': 'mean_absolute_error'},
                      optimizer='Adadelta',
                      metrics={'regression_output': ['mse', 'mae']},
                      sample_weight_mode='element')

        # train model
        tic = datetime.datetime.now()
        hist = model.fit(train_dataset['im'],
                         {'regression_output': train_dataset['dmap']},
                         sample_weight={'regression_output': train_dataset['mask'][..., 0]},
                         validation_data=(test_dataset['im'],
                                          {'regression_output': test_dataset['dmap']},
                                          {'regression_output': test_dataset['mask'][..., 0]}),
                         batch_size=batch_size, epochs=epochs, initial_epoch=0,
                         callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))
        history.append(hist.history)

# save training history
history_filename = os.path.join(saved_models_dir, experiment_id + '_history.npz')
with open(history_filename, 'w') as f:
    json.dump(history, f)

if DEBUG:
    with open(history_filename, 'r') as f:
        history = json.load(f)

    plt.clf()
    plt.plot(history[i_fold]['mean_absolute_error'], label='mean_absolute_error')
    # plt.plot(history[i_fold]['mean_squared_error'], label='mean_squared_error')
    plt.plot(history[i_fold]['val_mean_absolute_error'], label='val_mean_absolute_error')
    # plt.plot(history[i_fold]['val_mean_squared_error'], label='val_mean_squared_error')
    plt.plot(history[i_fold]['loss'], label='loss')
    plt.plot(history[i_fold]['val_loss'], label='val_loss')
    plt.legend()
