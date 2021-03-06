"""
Training of CNN for regression of Dice coefficients.

We mask the histology with segmentation mask as: +1 (foreground) / -1 (background)

Following QualityNet^1 [1], but using the DenseNet as CNN.
Inputs are the RGB histology masked by the segmentation mask. The output is the Dice coefficient.

[1] Huang et al. "QualityNet: Segmentation quality evaluation with deep convolutional networks", 2016 Visual
Communications and Image Processing (VCIP).
"""

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
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
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

from keras_applications.densenet import DenseNet
from keras.models import Model
from keras.layers import Dense, Input

# for data parallelism in keras models
from keras.utils import multi_gpu_model

import cytometer.data
import cytometer.model_checkpoint_parallel
import cytometer.utils
import tensorflow as tf
from skimage.morphology import watershed

# limit GPU memory used
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False

# number of folds for k-fold cross validation
n_folds = 11

# number of epochs for training
epochs = 40

# area (pixel**2) of the smallest object we accept as a cell (pi * (16 pixel)**2 = 804.2 pixel**2)
smallest_cell_area = 804

# training window length
training_window_len = 401

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

'''Loop folds (in this script, we actually only work with fold 0)
'''

# loop each fold: we split the data into train vs test, train a model, and compute errors with the
# test data. In each fold, the test data is different
# for i_fold, idx_test in enumerate(idx_test_all):
for i_fold in [0]:  # this is a clunky way of doing i_fold = 0, but that can be easily extended for more folds

    '''Load data
    '''

    # one-cell windows from the segmentations obtained with our pipeline
    # ['train_cell_im', 'train_cell_reflab', 'train_cell_testlab', 'train_cell_dice', 'test_cell_im',
    # 'test_cell_reflab', 'test_cell_testlab', 'test_cell_dice']
    onecell_filename = os.path.join(training_augmented_dir, 'onecell_kfold_' + str(i_fold).zfill(2) + '_worsen_000.npz')
    dataset_orig = np.load(onecell_filename)

    # extra data created by worsening the segmentations above
    # ['train_cell_testlab', 'train_cell_dice', 'test_cell_testlab', 'test_cell_dice']
    onecell_filename = os.path.join(training_augmented_dir, 'onecell_kfold_' + str(i_fold).zfill(2) + '_worsen_030.npz')
    dataset_worsen = np.load(onecell_filename)

    # read the data into memory
    train_cell_im = dataset_orig['train_cell_im']
    train_cell_reflab = dataset_orig['train_cell_reflab']
    train_cell_testlab = dataset_orig['train_cell_testlab']
    train_cell_dice = dataset_orig['train_cell_dice']
    test_cell_im = dataset_orig['test_cell_im']
    test_cell_reflab = dataset_orig['test_cell_reflab']
    test_cell_testlab = dataset_orig['test_cell_testlab']
    test_cell_dice = dataset_orig['test_cell_dice']
    worsen_train_cell_testlab = dataset_worsen['train_cell_testlab']
    worsen_train_cell_dice = dataset_worsen['train_cell_dice']
    worsen_test_cell_testlab = dataset_worsen['test_cell_testlab']
    worsen_test_cell_dice = dataset_worsen['test_cell_dice']

    # free up memory
    del dataset_orig
    del dataset_worsen

    if DEBUG:
        i = 1006
        plt.clf()
        plt.imshow(train_cell_im[i, :, :, :])
        plt.contour(train_cell_testlab[i, :, :, 0], levels=1, colors='green')
        plt.contour(worsen_train_cell_testlab[i, :, :, 0], levels=1, colors='red')
        plt.title('Dice = ' + str("{:.2f}".format(train_cell_dice[i])))

    # concatenate training and test data, the original one-cell windows and the worsened ones
    train_cell_im = np.concatenate((train_cell_im, train_cell_im), axis=0)
    train_cell_testlab = np.concatenate((train_cell_testlab, worsen_train_cell_testlab), axis=0)
    train_cell_dice = np.concatenate((train_cell_dice, worsen_train_cell_dice), axis=0)
    test_cell_im = np.concatenate((test_cell_im, test_cell_im), axis=0)
    test_cell_testlab = np.concatenate((test_cell_testlab, worsen_test_cell_testlab), axis=0)
    test_cell_dice = np.concatenate((test_cell_dice, worsen_test_cell_dice), axis=0)

    if DEBUG:
        plt.clf()
        plt.subplot(121)
        i = 1006
        plt.imshow(train_cell_im[i, :, :, :])
        plt.contour(train_cell_testlab[i, :, :, 0], levels=1, colors='green')
        plt.title('Dice = ' + str("{:.2f}".format(train_cell_dice[i])))
        plt.subplot(122)
        i += int(train_cell_im.shape[0]/2)
        plt.imshow(train_cell_im[i, :, :, :])
        plt.contour(train_cell_testlab[i, :, :, 0], levels=1, colors='red')
        plt.title('Dice = ' + str("{:.2f}".format(train_cell_dice[i])))

    # convert the {0, 1} segmentation mask to {-1, 1} segmentation mask
    train_cell_testlab = train_cell_testlab.astype(np.float32)
    train_cell_testlab[train_cell_testlab == 0] = -1.0

    # multiply histology by segmentations to have a single input tensor
    train_cell_im *= np.repeat(train_cell_testlab, repeats=train_cell_im.shape[3], axis=3)
    test_cell_im *= np.repeat(test_cell_testlab, repeats=test_cell_im.shape[3], axis=3)

    if DEBUG:
        plt.clf()
        plt.subplot(121)
        i = 1006
        plt.imshow(np.abs(train_cell_im[i, :, :, :]))
        plt.contour(train_cell_testlab[i, :, :, 0], levels=1, colors='green')
        plt.title('Dice = ' + str("{:.2f}".format(train_cell_dice[i])))
        plt.text(175, 180, '+1', fontsize=14, verticalalignment='top')
        plt.text(100, 75, '-1', fontsize=14, verticalalignment='top')
        plt.subplot(122)
        i += int(train_cell_im.shape[0]/2)
        plt.imshow(np.abs(train_cell_im[i, :, :, :]))
        plt.contour(train_cell_testlab[i, :, :, 0], levels=1, colors='red')
        plt.title('Dice = ' + str("{:.2f}".format(train_cell_dice[i])))
        plt.text(175, 180, '+1', fontsize=14, verticalalignment='top')
        plt.text(100, 75, '-1', fontsize=14, verticalalignment='top')

    '''Neural network training
    '''

    # list all CPUs and GPUs
    device_list = K.get_session().list_devices()

    # number of GPUs
    gpu_number = np.count_nonzero(['GPU' in str(x) for x in device_list])

    # instantiate model
    with tf.device('/cpu:0'):
        # we start with the DenseNet without the final Dense layer, because it has a softmax activation, and we only
        # want to classify 1 class. So then we manually add an extra Dense layer with a sigmoid activation as final
        # output
        base_model = DenseNet(blocks=[6, 12, 24, 16], include_top=False, weights=None, input_shape=(401, 401, 3),
                              pooling='avg')
        x = Dense(units=1, activation='sigmoid', name='fc1')(base_model.output)
        model = Model(inputs=base_model.input, outputs=x)

    saved_model_filename = os.path.join(saved_models_dir, experiment_id + '_model_fold_' + str(i_fold) + '.h5')

    if gpu_number > 1:  # compile and train model: Multiple GPUs

        # checkpoint to save model after each epoch
        checkpointer = cytometer.model_checkpoint_parallel.ModelCheckpoint(filepath=saved_model_filename,
                                                                           verbose=1, save_best_only=True)
        # compile model
        parallel_model = multi_gpu_model(model, gpus=gpu_number)
        parallel_model.compile(loss={'fc1': 'mse'},
                               optimizer='Adadelta',
                               metrics={'fc1': ['mse', 'mae']})

        # train model
        tic = datetime.datetime.now()
        parallel_model.fit(train_cell_im,
                           {'fc1': train_cell_dice},
                           validation_data=(test_cell_im,
                                            {'fc1': test_cell_dice}),
                           batch_size=16, epochs=epochs, initial_epoch=0,
                           callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))

    else:  # compile and train model: One GPU

        # checkpoint to save model after each epoch
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=saved_model_filename,
                                                       verbose=1, save_best_only=True)

        # compile model
        model.compile(loss={'fc1': 'mse'},
                      optimizer='Adadelta',
                      metrics={'fc1': ['mse', 'mae']})

        # train model
        tic = datetime.datetime.now()
        model.fit(train_cell_im,
                  {'fc1': train_cell_dice},
                  validation_data=(test_cell_im,
                                   {'fc1': test_cell_dice}),
                  batch_size=16, epochs=epochs, initial_epoch=0,
                  callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))

# if we run the script with qsub on the cluster, the standard output is in file
# klf14_b6ntac_exp_0001_cnn_dmap_contour.sge.sh.oPID where PID is the process ID
# Save it to saved_models directory
log_filename = os.path.join(saved_models_dir, experiment_id + '.log')
stdout_filename = os.path.join(home, 'Software', 'cytometer', 'scripts', experiment_id + '.sge.sh.o*')
stdout_filename = glob.glob(stdout_filename)[0]
if stdout_filename and os.path.isfile(stdout_filename):
    shutil.copy2(stdout_filename, log_filename)
else:
    # if we ran the script with nohup in linux, the standard output is in file nohup.out.
    # Save it to saved_models directory
    log_filename = os.path.join(saved_models_dir, experiment_id + '.log')
    nohup_filename = os.path.join(home, 'Software', 'cytometer', 'scripts', 'nohup.out')
    if os.path.isfile(nohup_filename):
        shutil.copy2(nohup_filename, log_filename)
