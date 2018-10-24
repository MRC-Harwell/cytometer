"""
Extract a window centered on each cell from test data (histology, ground truth segmentation, algorithm segmentation and
Dice coefficient.
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
import re

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
from skimage.measure import regionprops
import cv2

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

    # load the train and test data: im, mask, predlab_kfold_00 data
    # Note: we need to load whole images, to avoid splitting cells, that then become unusable
    predlab_str = 'predlab_kfold_' + str(i_fold).zfill(2)
    train_dataset, train_file_list, train_shuffle_idx = \
        cytometer.data.load_datasets(im_train_file_list, prefix_from='im',
                                     prefix_to=['im', 'mask', 'lab', predlab_str],
                                     nblocks=1, shuffle_seed=i_fold)
    test_dataset, test_file_list, test_shuffle_idx = \
        cytometer.data.load_datasets(im_test_file_list, prefix_from='im',
                                     prefix_to=['im', 'mask', 'lab', predlab_str],
                                     nblocks=1, shuffle_seed=i_fold)

    # remove borders between cells in the lab_train data. For this experiment, we want labels touching each other
    for i in range(train_dataset['lab'].shape[0]):
        train_dataset['lab'][i, :, :, 0] = watershed(image=np.zeros(shape=train_dataset['lab'].shape[1:3],
                                                                    dtype=train_dataset['lab'].dtype),
                                                     markers=train_dataset['lab'][i, :, :, 0],
                                                     watershed_line=False)

    # change the background label from 1 to 0
    train_dataset['lab'][train_dataset['lab'] == 1] = 0

    # remove borders between cells in the lab_test data. For this experiment, we want labels touching each other
    for i in range(test_dataset['lab'].shape[0]):
        test_dataset['lab'][i, :, :, 0] = watershed(image=np.zeros(shape=test_dataset['lab'].shape[1:3],
                                                                   dtype=test_dataset['lab'].dtype),
                                                    markers=test_dataset['lab'][i, :, :, 0],
                                                    watershed_line=False)

    # change the background label from 1 to 0
    test_dataset['lab'][test_dataset['lab'] == 1] = 0

    # load the mapping between predicted and ground truth segmentations
    # Note: the correspondence between labels doesn't change with agumentation, so we load the same labcorr file for all
    # the _seed_???_ variations
    labcorr_str = 'labcorr_kfold_' + str(i_fold).zfill(2)
    train_labcorr = []
    for i in range(len(im_train_file_list)):
        labcorr_file = im_train_file_list[i].replace('im_', labcorr_str + '_')
        labcorr_file = re.sub(r'_seed_..._', '_seed_nan_', labcorr_file)
        labcorr_file = os.path.join(training_augmented_dir,
                                    labcorr_file.replace('.tif', '.npy'))
        train_labcorr.append(np.load(labcorr_file))

    test_labcorr = []
    for i in range(len(im_test_file_list)):
        labcorr_file = im_test_file_list[i].replace('im_', labcorr_str + '_')
        labcorr_file = re.sub(r'_seed_..._', '_seed_nan_', labcorr_file)
        labcorr_file = os.path.join(training_augmented_dir,
                                    labcorr_file.replace('.tif', '.npy'))
        test_labcorr.append(np.load(labcorr_file))


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

    train_cell_im, train_cell_reflab, train_cell_testlab, train_cell_dice = \
        cytometer.utils.one_image_per_label(dataset_im=train_dataset['im'],
                                            dataset_lab_ref=train_dataset['lab'],
                                            dataset_lab_test=train_dataset[predlab_str],
                                            training_window_len=training_window_len,
                                            smallest_cell_area=smallest_cell_area)
    test_cell_im, test_cell_reflab, test_cell_testlab, test_cell_dice = \
        cytometer.utils.one_image_per_label(dataset_im=test_dataset['im'],
                                            dataset_lab_ref=test_dataset['lab'],
                                            dataset_lab_test=test_dataset[predlab_str],
                                            training_window_len=training_window_len,
                                            smallest_cell_area=smallest_cell_area)

    if DEBUG:
        i = 150
        plt.clf()
        plt.imshow(train_cell_im[i, :, :, :])
        plt.contour(train_cell_testlab[i, :, :, 0], levels=1)
        plt.title('Dice = ' + str(train_cell_dice[i]))

    if DEBUG:
        i = 1000
        plt.clf()
        plt.imshow(test_cell_im[i, :, :, :])
        plt.contour(test_cell_testlab[i, :, :, 0], levels=1)
        plt.title('Dice = ' + str(test_cell_dice[i]))

    # save one-cell data for later use
    onecell_filename = os.path.join(training_augmented_dir, 'onecell_kfold_' + str(i_fold).zfill(2) + '_worsen_000.npz')
    np.savez(onecell_filename, train_cell_im=train_cell_im, train_cell_reflab=train_cell_reflab,
                        train_cell_testlab=train_cell_testlab, train_cell_dice=train_cell_dice,
                        test_cell_im=test_cell_im, test_cell_reflab=test_cell_reflab,
                        test_cell_testlab=test_cell_testlab, test_cell_dice=test_cell_dice)

    # make copy of training data to worsen it
    worsen_train_cell_testlab = train_cell_testlab.copy()
    worsen_train_cell_dice = train_cell_dice.copy()
    worsen_test_cell_testlab = test_cell_testlab.copy()
    worsen_test_cell_dice = test_cell_dice.copy()

    # train images: apply random changes to the segmentations, to worsen the Dice values
    for i in range(train_cell_im.shape[0]):

        # get bounding box around segmentation
        props = regionprops(train_cell_testlab[i, :, :, 0])

        # diameter of a circle with the same area as the cell segmentation
        diameter_test = np.sqrt(4 / np.pi * np.count_nonzero(train_cell_testlab[i, :, :, 0]))

        # dilation or erosion kernel. Its size is a random percentage of diameter_test
        np.random.seed(i)
        diameter = int(np.random.uniform(low=0, high=0.3) * diameter_test)
        kernel = np.ones(shape=(diameter, diameter))

        # random dilation or erosion
        if np.random.random_integers(low=0, high=1) == 0:
            worsen_train_cell_testlab[i, :, :, 0] = cv2.erode(train_cell_testlab[i, :, :, 0], kernel=kernel)
        else:
            worsen_train_cell_testlab[i, :, :, 0] = cv2.dilate(train_cell_testlab[i, :, :, 0], kernel=kernel)
        # recompute Dice coefficient
        a_intersect_b = np.count_nonzero(np.logical_and(worsen_train_cell_testlab[i, :, :, 0],
                                                        train_cell_reflab[i, :, :, 0]))
        a = np.count_nonzero(worsen_train_cell_testlab[i, :, :, 0])
        b = np.count_nonzero(train_cell_reflab[i, :, :, 0])
        worsen_train_cell_dice[i] = 2 * a_intersect_b / (a + b)

    # test images: apply random changes to the segmentations, to worsen the Dice values
    for i in range(test_cell_im.shape[0]):

        # get bounding box around segmentation
        props = regionprops(test_cell_testlab[i, :, :, 0])

        # diameter of a circle with the same area as the cell segmentation
        diameter_test = np.sqrt(4 / np.pi * np.count_nonzero(test_cell_testlab[i, :, :, 0]))

        # dilation or erosion kernel. Its size is a random percentage of diameter_test
        np.random.seed(i)
        diameter = int(np.random.uniform(low=0, high=0.3) * diameter_test)
        kernel = np.ones(shape=(diameter, diameter))

        # random dilation or erosion
        if np.random.random_integers(low=0, high=1) == 0:
            worsen_test_cell_testlab[i, :, :, 0] = cv2.erode(test_cell_testlab[i, :, :, 0], kernel=kernel)
        else:
            worsen_test_cell_testlab[i, :, :, 0] = cv2.dilate(test_cell_testlab[i, :, :, 0], kernel=kernel)
        # recompute Dice coefficient
        a_intersect_b = np.count_nonzero(np.logical_and(worsen_test_cell_testlab[i, :, :, 0],
                                                        test_cell_reflab[i, :, :, 0]))
        a = np.count_nonzero(worsen_test_cell_testlab[i, :, :, 0])
        b = np.count_nonzero(test_cell_reflab[i, :, :, 0])
        worsen_test_cell_dice[i] = 2 * a_intersect_b / (a + b)

    # save worsened one-cell data for later use
    onecell_filename = os.path.join(training_augmented_dir, 'onecell_kfold_' + str(i_fold).zfill(2) + '_worsen_030.npz')
    np.savez(onecell_filename, train_cell_testlab=worsen_train_cell_testlab,
                        train_cell_dice=worsen_train_cell_dice, test_cell_testlab=worsen_test_cell_testlab,
                        test_cell_dice=worsen_test_cell_dice)

    if DEBUG:
        i = 1006
        plt.clf()
        plt.imshow(train_cell_im[i, :, :, :])
        plt.contour(train_cell_testlab[i, :, :, 0], levels=1, colors='green')
        plt.contour(worsen_train_cell_testlab[i, :, :, 0], levels=1, colors='red')
        plt.title('Dice = ' + "{:.2f}".format(worsen_train_cell_dice[i]) + ' (from '
                  + "{:.2f}".format(train_cell_dice[i]) + ')')

    if DEBUG:
        # histograms of Dice values
        plt.clf()
        plt.subplot(221)
        plt.hist(train_cell_dice, histtype='step')
        plt.hist(worsen_train_cell_dice, histtype='step')
        plt.legend(['original', 'worsened'], loc='upper left')
        plt.title('Training set')
        plt.subplot(222)
        plt.hist(test_cell_dice, histtype='step')
        plt.hist(worsen_test_cell_dice, histtype='step')
        plt.legend(['original', 'worsened'], loc='upper left')
        plt.title('Testing set')
        plt.subplot(223)
        plt.hist(np.concatenate((train_cell_dice, worsen_train_cell_dice)), histtype='step')
        plt.legend(['aggregate'], loc='upper left')
        plt.xlabel('Dice coeff.')
        plt.subplot(224)
        plt.hist(np.concatenate((test_cell_dice, worsen_test_cell_dice)), histtype='step')
        plt.legend(['aggregate'], loc='upper left')
        plt.xlabel('Dice coeff.')


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
