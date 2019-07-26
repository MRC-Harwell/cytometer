'''
Inspect 0076, Dmap regression for all folds.

Training vs testing is done at the histology slide level, not at the window level. This way, we really guarantee that
the network has not been trained with data sampled from the same image as the test data.

Like 0056, but:
    * k-folds = 10.
    * Adding Gianluca's data.
    * Fill small gaps in the training mask with a (3,3) dilation.
    * Save training history variable, instead of relying on text output.
'''

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_inspect_exp_0076_cnn_dmap'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import json

# other imports
import glob
import pickle
import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt

# limit number of GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

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

'''Directories and filenames'''

# data paths
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
# klf14_training_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training')
# klf14_training_non_overlap_data_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training_non_overlap')
klf14_training_augmented_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training_augmented')
#
# c3h_root_data_dir = os.path.join(home, 'Data/cytometer_data/c3h')
# c3h_training_dir = os.path.join(c3h_root_data_dir, 'c3h_hfd_training')
# c3h_training_non_overlap_data_dir = os.path.join(c3h_root_data_dir, 'c3h_hfd_training_non_overlap')
# c3h_training_augmented_dir = os.path.join(c3h_root_data_dir, 'c3h_hfd_training_augmented')

saved_models_dir = os.path.join(klf14_root_data_dir, 'saved_models')

'''Load folds'''

# load k-folds info
kfold_info_filename = os.path.join(saved_models_dir, 'klf14_b6ntac_exp_0076_cnn_dmap_kfold_info.pickle')
with open(kfold_info_filename, 'rb') as f:
    aux = pickle.load(f)
im_svg_file_list = aux['file_list']
idx_test_all = aux['idx_test']
idx_train_all = aux['idx_train']

'''Model testing'''

# TIF files that correspond to the SVG files
im_orig_file_list = []
for i, file in enumerate(im_svg_file_list):
    im_orig_file_list.append(file.replace('.svg', '.tif'))
    im_orig_file_list[i] = os.path.join(klf14_training_augmented_dir, 'im_seed_nan_' + os.path.basename(im_orig_file_list[i]))

# loop each fold: we split the data into train vs test, train a model, and compute errors with the
# test data. In each fold, the test data is different
# for i_fold, idx_test in enumerate(idx_test_all):
for i_fold, idx_test in enumerate(idx_test_all):

    '''Load data'''

    # list of testing images
    im_test_file_list = np.array(im_orig_file_list)[idx_test]

    # load the test data: im, seg, dmap and mask data
    test_dataset, test_file_list, test_shuffle_idx = \
        cytometer.data.load_datasets(im_test_file_list, prefix_from='im', prefix_to=['im', 'dmap', 'mask'])

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
        parallel_model.compile(loss={'regression_output': 'mse'},
                               optimizer='Adadelta',
                               metrics={'regression_output': ['mse', 'mae']},
                               sample_weight_mode='element')

        # train model
        tic = datetime.datetime.now()
        history = parallel_model.fit(train_dataset['im'],
                                     {'regression_output': train_dataset['dmap']},
                                     sample_weight={'regression_output': train_dataset['mask'][..., 0]},
                                     validation_data=(test_dataset['im'],
                                                      {'regression_output': test_dataset['dmap']},
                                                      {'regression_output': test_dataset['mask'][..., 0]}),
                                     batch_size=10, epochs=epochs, initial_epoch=0,
                                     callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))

    else:  # compile and train model: One GPU

        # checkpoint to save model after each epoch
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=saved_model_filename,
                                                       verbose=1, save_best_only=True)

        # compile model
        model.compile(loss={'regression_output': 'mse'},
                      optimizer='Adadelta',
                      metrics={'regression_output': ['mse', 'mae']},
                      sample_weight_mode='element')

        # train model
        tic = datetime.datetime.now()
        history = model.fit(train_dataset['im'],
                            {'regression_output': train_dataset['dmap']},
                            sample_weight={'regression_output': train_dataset['mask'][..., 0]},
                            validation_data=(test_dataset['im'],
                                             {'regression_output': test_dataset['dmap']},
                                             {'regression_output': test_dataset['mask'][..., 0]}),
                            batch_size=10, epochs=epochs, initial_epoch=0,
                            callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))

# save training history
history_filename = os.path.join(saved_models_dir, experiment_id + '_history.npz')
with open(history_filename, 'w') as f:
    json.dump(history.history, f)

if DEBUG:
    with open(history_filename, 'r') as f:
        history = json.load(f)

    plt.clf()
    plt.plot(history['mean_squared_error'])
    plt.plot(history['val_mean_squared_error'])
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
