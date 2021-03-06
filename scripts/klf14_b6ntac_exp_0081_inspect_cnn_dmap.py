"""
Inspect Dmap regression for all folds (KLF14 dataset).

Training vs testing is done at the histology slide level, not at the window level. This way, we really guarantee that
the network has not been trained with data sampled from the same image as the test data.

Like 0056, but:
    * k-folds = 10.
    * Fill small gaps in the training mask with a (3,3) dilation.
    * Save training history variable, instead of relying on text output.
"""

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_inspect_exp_0081_cnn_dmap'

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
import numpy as np
import cv2
import matplotlib.pyplot as plt

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

import cytometer.data
import cytometer.utils
import cytometer.model_checkpoint_parallel

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False

'''Directories and filenames'''

# data paths
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
klf14_training_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training')
klf14_training_non_overlap_data_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training_non_overlap')
klf14_training_augmented_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training_augmented')

saved_models_dir = os.path.join(klf14_root_data_dir, 'saved_models')

saved_kfolds_filename = 'klf14_b6ntac_exp_0079_generate_kfolds.pickle'


'''Load folds'''

# load list of images, and indices for training vs. testing indices
contour_model_kfold_filename = os.path.join(saved_models_dir, saved_kfolds_filename)
with open(contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
svg_file_list = aux['file_list']
idx_test_all = aux['idx_test']
idx_train_all = aux['idx_train']

# correct home directory
svg_file_list = [x.replace('/home/rcasero', home) for x in svg_file_list]

# TIFF files that correspond to the SVG files (without augmentation)
im_orig_file_list = []
for i, file in enumerate(svg_file_list):
    im_orig_file_list.append(file.replace('.svg', '.tif'))
    im_orig_file_list[i] = os.path.join(os.path.dirname(im_orig_file_list[i]) + '_augmented',
                                        'im_seed_nan_' + os.path.basename(im_orig_file_list[i]))

    # check that files exist
    if not os.path.isfile(file):
        # warnings.warn('i = ' + str(i) + ': File does not exist: ' + os.path.basename(file))
        warnings.warn('i = ' + str(i) + ': File does not exist: ' + file)
    if not os.path.isfile(im_orig_file_list[i]):
        # warnings.warn('i = ' + str(i) + ': File does not exist: ' + os.path.basename(im_orig_file_list[i]))
        warnings.warn('i = ' + str(i) + ': File does not exist: ' + im_orig_file_list[i])

'''Inspect training convergence'''

# load training history
history_filename = os.path.join(saved_models_dir, 'klf14_b6ntac_exp_0081_cnn_dmap_history.npz')
with open(history_filename, 'r') as f:
    history = json.load(f)

if DEBUG:

    plt.clf()
    for i_fold in range(len(history)):
        plt.plot(history[i_fold]['mean_absolute_error'], label='fold = ' + str(i_fold))
    plt.tick_params(axis="both", labelsize=14)
    plt.ylabel('Mean absolute error', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.legend()

    plt.clf()
    for i_fold in range(len(history)):
        plt.plot(history[i_fold]['val_mean_absolute_error'], label='fold = ' + str(i_fold))
    plt.tick_params(axis="both", labelsize=14)
    plt.ylabel('Validation mean_absolute_error', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.legend()

    plt.clf()
    for i_fold in range(len(history)):
        plt.plot(history[i_fold]['loss'], label='fold = ' + str(i_fold))
    plt.tick_params(axis="both", labelsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.legend()

    plt.clf()
    for i_fold in range(len(history)):
        plt.plot(history[i_fold]['val_loss'], label='fold = ' + str(i_fold))
    plt.tick_params(axis="both", labelsize=14)
    plt.ylabel('Validation loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.legend()


'''Inspect model results'''

for i_fold, idx_test in enumerate(idx_test_all):

    print('Fold ' + str(i_fold) + '/' + str(len(idx_test_all)-1))


    '''Load data'''

    # split the data list into training and testing lists
    im_test_file_list, im_train_file_list = cytometer.data.split_list(im_orig_file_list, idx_test)

    # add the augmented image files
    im_test_file_list = cytometer.data.augment_file_list(im_test_file_list, '_nan_', '_*_')
    im_train_file_list = cytometer.data.augment_file_list(im_train_file_list, '_nan_', '_*_')

    # load the test data (im, dmap, mask)
    test_dataset, test_file_list, test_shuffle_idx = \
        cytometer.data.load_datasets(im_test_file_list, prefix_from='im', prefix_to=['im', 'dmap', 'mask'],
                                     nblocks=1, shuffle_seed=None)

    # remove training data where the mask has very few valid pixels (note: this will discard all the images without
    # cells)
    test_dataset = cytometer.data.remove_poor_data(test_dataset, prefix='mask', threshold=1000)

    # fill in the little gaps in the mask
    kernel = np.ones((3, 3), np.uint8)
    for i in range(test_dataset['mask'].shape[0]):
        test_dataset['mask'][i, :, :, 0] = cv2.dilate(test_dataset['mask'][i, :, :, 0].astype(np.uint8),
                                                      kernel=kernel, iterations=1)

    # load dmap model, and adjust input size
    saved_model_filename = os.path.join(saved_models_dir,
                                        'klf14_b6ntac_exp_0081_cnn_dmap_model_fold_' + str(i_fold) + '.h5')
    dmap_model = keras.models.load_model(saved_model_filename)
    if dmap_model.input_shape[1:3] != test_dataset['im'].shape[1:3]:
        dmap_model = cytometer.utils.change_input_size(dmap_model, batch_shape=test_dataset['im'].shape)

    # estimate dmaps
    pred_dmap = dmap_model.predict(test_dataset['im'], batch_size=4)

    if DEBUG:
        for i in range(test_dataset['im'].shape[0]):
            plt.clf()
            plt.subplot(221)
            plt.imshow(test_dataset['im'][i, :, :, :])
            plt.axis('off')
            plt.subplot(222)
            plt.imshow(test_dataset['dmap'][i, :, :, 0])
            plt.axis('off')
            plt.subplot(223)
            plt.imshow(test_dataset['mask'][i, :, :, 0])
            plt.axis('off')
            plt.subplot(224)
            plt.imshow(pred_dmap[i, :, :, 0])
            plt.axis('off')
