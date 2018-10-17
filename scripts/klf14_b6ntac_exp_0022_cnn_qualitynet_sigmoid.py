"""
Training of CNN for regression of Dice coefficients.

Following QualityNet^1 [1], but using the DenseNet as CNN.
Inputs are the RGB histology masked by the segmentation mask. The output is the Dice coefficient.

[1] Huang et al. "QualityNet: Segmentation quality evaluation with deep convolutional networks", 2016 Visual
Communications and Image Processing (VCIP).
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

# # limit GPU memory used
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# set_session(tf.Session(config=config))

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False

# number of folds for k-fold cross validation
n_folds = 11

# number of epochs for training
epochs = 20

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

'''CNN classifier to estimate Dice coefficient
'''



'''Models that were used to generate the segmentations
'''

saved_contour_model_basename = 'klf14_b6ntac_exp_0006_cnn_contour'  # contour
saved_dmap_model_basename = 'klf14_b6ntac_exp_0015_cnn_dmap'  # dmap

contour_model_name = saved_contour_model_basename + '*.h5'
dmap_model_name = saved_dmap_model_basename + '*.h5'

'''Filenames of whole dataset, and indices of train vs. test subsets
'''

# load list of images, and indices for training vs. testing indices
contour_model_kfold_filename = os.path.join(saved_models_dir, saved_contour_model_basename + '_info.pickle')
with open(contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
im_orig_file_list = aux['file_list']
idx_orig_test_all = aux['idx_test_all']

# number of original training images
n_im = len(im_orig_file_list)

# correct home directory if we are in a different system than what was used to train the models
im_orig_file_list = cytometer.data.change_home_directory(im_orig_file_list, '/users/rittscher/rcasero', home,
                                                         check_isfile=True)

'''Loop folds (in this script, we actually only work with fold 0)
'''

# loop each fold: we split the data into train vs test, train a model, and compute errors with the
# test data. In each fold, the test data is different
# for i_fold, idx_test in enumerate(idx_test_all):
for i_fold, idx_test in enumerate([idx_orig_test_all[0]]):

    '''Load data
    '''

    # split the data into training and testing datasets
    im_orig_test_file_list, im_orig_train_file_list = cytometer.data.split_list(im_orig_file_list, idx_test)

    # add the augmented image files
    im_train_file_list = cytometer.data.augment_file_list(im_orig_train_file_list, '_nan_', '_*_')
    im_test_file_list = cytometer.data.augment_file_list(im_orig_test_file_list, '_nan_', '_*_')

    # load the train and test data: im, mask, preddice_kfold_00, predlab_kfold_00 data
    # Note: we need to load whole images, to avoid splitting cells, that then become unusable
    preddice_str = 'preddice_kfold_' + str(i_fold).zfill(2)
    predlab_str = 'predlab_kfold_' + str(i_fold).zfill(2)
    train_dataset, train_file_list, train_shuffle_idx = \
        cytometer.data.load_datasets(im_train_file_list, prefix_from='im',
                                     prefix_to=['im', 'mask', preddice_str, predlab_str],
                                     nblocks=1, shuffle_seed=i_fold)
    test_dataset, test_file_list, test_shuffle_idx = \
        cytometer.data.load_datasets(im_test_file_list, prefix_from='im',
                                     prefix_to=['im', 'mask', preddice_str, predlab_str],
                                     nblocks=1, shuffle_seed=i_fold)

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

    '''Extract one window for each individual cell, with the corresponding Dice coefficient
    '''

    train_cell_im, train_cell_lab, \
    train_cell_dice = cytometer.utils.one_image_and_dice_per_cell(dataset_im=train_dataset['im'],
                                                                  dataset_lab=train_dataset[predlab_str],
                                                                  dataset_dice=train_dataset[preddice_str],
                                                                  training_window_len=training_window_len,
                                                                  smallest_cell_area=smallest_cell_area)
    test_cell_im, test_cell_lab, \
    test_cell_dice = cytometer.utils.one_image_and_dice_per_cell(dataset_im=test_dataset['im'],
                                                                 dataset_lab=test_dataset[predlab_str],
                                                                 dataset_dice=test_dataset[preddice_str],
                                                                 training_window_len=training_window_len,
                                                                 smallest_cell_area=smallest_cell_area)

    if DEBUG:
        i = 150
        plt.clf()
        plt.imshow(train_cell_im[i, :, :, :])
        plt.contour(train_cell_lab[i, :, :, 0], levels=1)
        plt.title('Dice = ' + str(train_cell_dice[i]))

    # multiply histology by segmentations to have a single input tensor
    train_cell_im *= np.repeat(train_cell_lab.astype(np.float32), repeats=train_cell_im.shape[3], axis=3)
    test_cell_im *= np.repeat(test_cell_lab.astype(np.float32), repeats=test_cell_im.shape[3], axis=3)

    if DEBUG:
        i = 150
        plt.clf()
        plt.imshow(train_cell_im[i, :, :, :])
        plt.contour(train_cell_lab[i, :, :, 0], levels=1)
        plt.title('Dice = ' + str(train_cell_dice[i]))

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
