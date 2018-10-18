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

# for data parallelism in keras models
from keras.utils import multi_gpu_model

import cytometer.data
import cytometer.model_checkpoint_parallel
import cytometer.utils
import cytometer.models

# limit GPU memory used
import tensorflow as tf
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
experiment_id = 'klf14_b6ntac_exp_0023_cnn_qualitynet_sigmoid_transfer_learning_imagenet'

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

i_fold = 0
idx_test = idx_orig_test_all[i_fold]

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

'''Load neural network for predictions
'''

fold_i = 0

model_name = experiment_id + '_model_fold_' + str(i_fold) + '.h5'

saved_model_filename = os.path.join(saved_models_dir, model_name)

# list of model files to inspect
model_files = glob.glob(os.path.join(saved_models_dir, model_name))

model_file = model_files[fold_i]

# load model
model = keras.models.load_model(model_file)

test_cell_preddice = np.zeros(shape=test_cell_dice.shape, dtype=test_cell_dice.dtype)
for i in range(test_cell_im.shape[0]):

    if not i % 10:
        print('Test image: ' + str(i) + '/' + str(test_cell_im.shape[0]-1))

    # predict Dice coefficient for test segmentation
    test_cell_preddice[i] = model.predict(np.expand_dims(test_cell_im[i, :, :, :], axis=0))


# plot all predicted vs. ground truth Dice values
if DEBUG:
    plt.clf()
    plt.scatter(test_cell_dice, test_cell_preddice)
    plt.plot([0.2, 1.0], [0.2, 1.0])
    plt.plot([.9, .9], [.1, 1.0], 'r')
    plt.plot([.1, 1.0], [.9, .9], 'r')
    plt.xlabel('Ground truth')
    plt.ylabel('Estimated')
    plt.title('Dice coefficient')

# plot prediction
if DEBUG:
    i = 150
    plt.clf()
    plt.imshow(test_cell_im[i, :, :, :])
    plt.title('Dice: ' + str(test_cell_dice[i]) + ' (ground truth)\n' + str(test_cell_preddice[i]) + ' (estimated)')

    i = 1001
    plt.clf()
    plt.imshow(test_cell_im[i, :, :, :])
    plt.title('Dice: ' + str(test_cell_dice[i]) + ' (ground truth)\n' + str(test_cell_preddice[i]) + ' (estimated)')

'''Look at distribution of Dice coefficients
'''

plt.clf()
plt.hist(np.concatenate((test_cell_dice, train_cell_dice)))
plt.title('Distribution of Dice coeff values')
plt.xlabel('Dice coeff')

'''Plot metrics and convergence
'''

fold_i = 0
model_file = model_files[fold_i]

log_filename = os.path.join(saved_models_dir, experiment_id + '.log')

if os.path.isfile(log_filename):

    # read Keras output
    df_list = cytometer.data.read_keras_training_output(log_filename)

    # plot metrics with every iteration
    plt.clf()
    for df in df_list:
        plt.subplot(211)
        loss_plot, = plt.semilogy(df.index, df.loss, label='loss')
        epoch_ends = np.concatenate((np.where(np.diff(df.epoch))[0], [len(df.epoch)-1, ]))
        epoch_ends_plot1, = plt.semilogy(epoch_ends, df.loss[epoch_ends], 'ro', label='end of epoch')
        plt.legend(handles=[loss_plot, epoch_ends_plot1])
        plt.subplot(212)
        regr_mae_plot, = plt.plot(df.index, df.mean_absolute_error, label='dmap mae')
        regr_mse_plot, = plt.plot(df.index, np.sqrt(df.mean_squared_error), label='sqrt(dmap mse)')
        regr_mae_epoch_ends_plot2, = plt.plot(epoch_ends, df.mean_absolute_error[epoch_ends], 'ro', label='end of epoch')
        regr_mse_epoch_ends_plot2, = plt.plot(epoch_ends, np.sqrt(df.mean_squared_error[epoch_ends]), 'ro', label='end of epoch')
        plt.legend(handles=[regr_mae_plot, regr_mse_plot, regr_mae_epoch_ends_plot2])
