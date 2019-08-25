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
import datetime
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import time

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K
import keras_contrib

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AvgPool2D, Activation, BatchNormalization

# for data parallelism in keras models
from keras.utils import multi_gpu_model

import cytometer.model_checkpoint_parallel
import cytometer.utils
import cytometer.data
import tensorflow as tf

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

'''Inspect the data
'''

result = np.load(os.path.join(saved_models_dir, experiment_id + '_data.npz'))
window_im_all = result['window_im_all']
window_out_all = result['window_out_all']
window_mask_loss_all = result['window_mask_loss_all']
window_idx_all = result['window_idx_all']
del result

for i_fold in range(n_folds):

    print('# Fold ' + str(i_fold) + '/' + str(n_folds - 1))

    # test and training image indices
    idx_test = idx_test_all[i_fold]

    # memory-map the precomputed data
    result = np.load(os.path.join(saved_models_dir, experiment_id + '_data.npz'), mmap_mode='r')
    window_idx_all = result['window_idx_all']

    # get cell indices for test and training, based on the image indices
    idx_test = np.where([x in idx_test for x in window_idx_all[:, 0]])[0]

    print('## len(idx_test) = ' + str(len(idx_test)))

    # split data into training and testing
    window_im_test = window_im_all[idx_test, :, :, :]

    window_out_test = window_out_all[idx_test, :, :]

    window_mask_loss_test = window_mask_loss_all[idx_test, :]

    # model and logs filenames
    saved_model_filename = os.path.join(saved_models_dir, original_experiment_id + '_model_fold_' + str(i_fold) + '.h5')
    saved_logs_dir = os.path.join(saved_models_dir, original_experiment_id + '_logs_fold_' + str(i_fold))
    history_filename = os.path.join(saved_models_dir, experiment_id + '_history_fold_' + str(i_fold) + '.json')

    # load model
    model = keras.models.load_model(saved_model_filename)
    if model.input_shape[1:3] != window_im_test.shape[1:3]:
        model = cytometer.utils.change_input_size(model, batch_shape=window_im_test.shape)

    # segmentation correction
    window_out_all = model.predict(window_im_test, batch_size=4)

    # threshold for segmentation correction
    window_seg_corrected = window_seg_all[j, :, :]
    window_seg_corrected[window_out_all[j, :, :, 0] >= 0.5] = 0  # the segmentation went too far
    window_seg_corrected[window_out_all[j, :, :, 0] <= -0.5] = 1  # the segmentation fell short

    if DEBUG:
        j = 670
        plt.clf()

        plt.subplot(221)
        plt.cla()
        aux = 0.2989 * window_im_all[j, :, :, 0] \
              + 0.5870 * window_im_all[j, :, :, 1] \
              + 0.114 * window_im_all[j, :, :, 2]
        plt.imshow(aux.astype(np.uint8), cmap='gray')
        plt.contour(window_out_gtruth_all[j, :, :], linewidths=1, colors='green')

        plt.subplot(222)
        plt.cla()
        plt.imshow(window_out_gtruth_all[j, :, :])

        plt.subplot(223)
        plt.cla()
        plt.imshow(window_out_all[j, :, :, 0])

        plt.subplot(224)
        plt.cla()
        plt.imshow(window_seg_corrected)
        plt.contour(window_seg_gtruth_all[j, :, :], levels=0.5, linewidths=1, colors='green')
        plt.contour(window_seg_all[j, :, :], levels=0.5, linewidths=1, colors='red')
        plt.contour(window_seg_corrected, linewidths=1, colors='blue')

