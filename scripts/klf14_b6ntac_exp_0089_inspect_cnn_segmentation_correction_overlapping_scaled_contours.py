"""
Segmentation Correction Network.

Use original hand traced cells (with overlaps) for training.

We create a ground truth segmentation, and then a series of eroded and dilated segmentations for training.

We create a bounding box to crop the images, and the scale everything to the same training window size, to remove
differences between cell sizes.

Training/test datasets created by random selection of individual cell images.

CHANGE: Here we assign cells to train or test sets grouped by image. This way, we guarantee that at testing time, the
network has not seen neighbour cells to the ones used for training.

Training for the CNN:
* Input: histology multiplied by segmentation.
* Output: mask = segmentation - ground truth.
* Other: mask for the loss function, to avoid looking too far from the cell.
"""

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

# script name to identify this experiment
original_experiment_id = 'klf14_b6ntac_exp_0089_cnn_segmentation_correction_overlapping_scaled_contours'
experiment_id = 'klf14_b6ntac_inspect_exp_0089_cnn_segmentation_correction_overlapping_scaled_contours'
print('Experiment ID: ' + experiment_id)

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import pickle
import json

# other imports
import numpy as np
import matplotlib.pyplot as plt

# # limit number of GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

import cytometer.utils
import cytometer.data
import tensorflow as tf

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

LIMIT_GPU_MEMORY = False

if LIMIT_GPU_MEMORY:
    # limit GPU memory used
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    set_session(tf.Session(config=config))

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False

# area (pixel**2) of the smallest object we accept as a cell (pi * (16 pixel)**2 = 804.2 pixel**2)
smallest_cell_area = 804

# training window length
training_window_len = 201

# remove from training cells that don't have a good enough overlap with a reference label
smallest_dice = 0.5

# segmentations with Dice >= threshold are accepted
dice_threshold = 0.9

# batch size for training
batch_size = 12

# number of epochs for training
epochs = 100


'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

saved_kfolds_filename = 'klf14_b6ntac_exp_0079_generate_kfolds.pickle'


'''Load folds'''

# load list of images, and indices for training vs. testing indices
contour_model_kfold_filename = os.path.join(saved_models_dir, saved_kfolds_filename)
with open(contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
file_svg_list = aux['file_list']
idx_test_all = aux['idx_test']
idx_train_all = aux['idx_train']

# correct home directory
file_svg_list = [x.replace('/home/rcasero', home) for x in file_svg_list]

# number of images
n_im = len(file_svg_list)

# number of folds
n_folds = len(idx_test_all)

'''TensorBoard logs'''

# list of metrics in the logs (we assume all folds have the same)
size_guidance = {  # limit number of elements that can be loaded so that memory is not overfilled
    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
    event_accumulator.IMAGES: 4,
    event_accumulator.AUDIO: 4,
    event_accumulator.SCALARS: 0,
    event_accumulator.HISTOGRAMS: 1,
}
i_fold = 0
saved_logs_dir = os.path.join(saved_models_dir, original_experiment_id + '_logs_fold_' + str(i_fold))
ea = event_accumulator.EventAccumulator(saved_logs_dir, size_guidance=size_guidance)
ea.Reload()  # loads events from file
#tags = ea.Tags()['scalars']
tags = ['loss', 'mean_squared_error', 'val_loss', 'val_mean_squared_error']

if DEBUG:

    plt.clf()

    for i_fold in range(10):

        saved_logs_dir = os.path.join(saved_models_dir, original_experiment_id + '_logs_fold_' + str(i_fold))
        if not os.path.isdir(saved_logs_dir):
            continue

        # load log data for current fold
        ea = event_accumulator.EventAccumulator(saved_logs_dir, size_guidance=size_guidance)
        ea.Reload()

        # plot curves for each metric
        for i, tag in enumerate(tags):
            df = pd.DataFrame(ea.Scalars(tag))

            # plot
            plt.subplot(2, 2, i + 1)
            plt.plot(df['step'], df['value'], label='fold ' + str(i_fold))

    for i, tag in enumerate(tags):
        plt.subplot(2, 2, i + 1)
        plt.tick_params(axis="both", labelsize=14)
        plt.ylabel(tag, fontsize=14)
        plt.xlabel('Epoch', fontsize=14)
        # plt.legend(fontsize=12)
        plt.tight_layout()


'''Inspect the data
'''

# load training/test data. Note that there's an image per cell, not per histology window
result = np.load(os.path.join(saved_models_dir, original_experiment_id + '_data.npz'))
window_im_all = result['window_im_all']
window_out_all = result['window_out_all']
window_mask_loss_all = result['window_mask_loss_all']
window_idx_all = result['window_idx_all']
del result

for i_fold in range(n_folds):

    print('# Fold ' + str(i_fold) + '/' + str(n_folds - 1))

    # get test image indices
    idx_test = idx_test_all[i_fold]

    # convert image indices to cell indices
    idx_test = np.where([x in idx_test for x in window_idx_all[:, 0]])[0]

    print('## len(idx_test) = ' + str(len(idx_test)))

    # memory-map the precomputed data
    result = np.load(os.path.join(saved_models_dir, original_experiment_id + '_data.npz'), mmap_mode='r')
    window_idx_all = result['window_idx_all']

    # get testing data
    window_im_test = window_im_all[idx_test, :, :, :]
    window_out_test = window_out_all[idx_test, :, :]
    window_mask_loss_test = window_mask_loss_all[idx_test, :]

    # load segmentation correction model
    saved_model_filename = os.path.join(saved_models_dir, original_experiment_id + '_model_fold_' + str(i_fold) + '.h5')
    correction_model = keras.models.load_model(saved_model_filename)
    if correction_model.input_shape[1:3] != window_im_test.shape[1:3]:
        correction_model = cytometer.utils.change_input_size(correction_model, batch_shape=window_im_test.shape)

    # apply correction model to histology masked with -1/+1 mask
    window_correction_test = correction_model.predict(window_im_test)

    # threshold correction to -1/+1
    window_correction_thres_test = (window_correction_test >= 0.5).astype(np.int8)
    window_correction_thres_test[window_correction_test <= -0.5] = -1

    if DEBUG:
        j = 10
        plt.clf()
        plt.subplot(221)
        plt.imshow(np.abs(window_im_test[j, :, :, :]))
        plt.subplot(222)
        plt.imshow(window_out_test[j, :, :, 0], cmap='Accent', clim=(-1, 1))
        plt.subplot(223)
        plt.imshow(window_correction_test[j, :, :, 0])
        plt.subplot(224)
        plt.imshow(window_correction_thres_test[j, :, :, 0], cmap='Accent', clim=(-1, 1))
