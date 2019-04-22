"""
New approach to Quality Network, using sherrah2016 CNN.

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
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import time

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AvgPool2D, Activation

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

# number of folds for k-fold cross validation
n_folds = 10

# number of epochs for training
epochs = 25

# area (pixel**2) of the smallest object we accept as a cell (pi * (16 pixel)**2 = 804.2 pixel**2)
smallest_cell_area = 804

# training window length
training_window_len = 401

# remove from training cells that don't have a good enough overlap with a reference label
smallest_dice = 0.5

# segmentations with Dice >= threshold are accepted
dice_threshold = 0.9

# batch size for training
batch_size = 16


'''CNN Model
'''


def fcn_sherrah2016_regression(input_shape, for_receptive_field=False):

    cnn_input = Input(shape=input_shape, dtype='float32', name='input_image')

    x = Conv2D(filters=32, kernel_size=(5, 5), strides=1, dilation_rate=1, padding='same')(cnn_input)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(3, 3), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)

    x = Conv2D(filters=int(96/2), kernel_size=(5, 5), strides=1, dilation_rate=2, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(5, 5), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(x)

    x = Conv2D(filters=int(128/2), kernel_size=(3, 3), strides=1, dilation_rate=4, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(9, 9), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(9, 9), strides=1, padding='same')(x)

    x = Conv2D(filters=int(196/2), kernel_size=(3, 3), strides=1, dilation_rate=8, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(17, 17), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(17, 17), strides=1, padding='same')(x)

    x = Conv2D(filters=int(512/2), kernel_size=(3, 3), strides=1, dilation_rate=16, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
    else:
        x = Activation('relu')(x)

    # regression output
    regression_output = Conv2D(filters=1, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same',
                               name='regression_output')(x)

    return Model(inputs=cnn_input, outputs=[regression_output])


'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0053_cnn_quality_network_fcn_overlapping_scaled_contours'

# load k-folds training and testing data
kfold_info_filename = os.path.join(saved_models_dir, experiment_id + '_kfold_info.pickle')
with open(kfold_info_filename, 'rb') as f:
    kfold_info = pickle.load(f)
file_list = kfold_info['file_list']
idx_test_all = kfold_info['idx_test']
idx_train_all = kfold_info['idx_train']
del kfold_info

# number of images
n_im = len(file_list)

'''Load the test data of each fold
'''

for i_fold in range(n_folds):

    # list of test files in this fold
    file_list_test = np.array(file_list)[idx_test_all[i_fold]]

    # init output
    window_im_all = []
    window_seg_gtruth_all = []
    window_seg_all = []
    window_out_gtruth_all = []
    window_idx_all = []
    window_pixel_size_all = []
    for i, file_svg in enumerate(file_list_test):

        print('file ' + str(i) + '/' + str(len(idx_test_all[i_fold]) - 1))

        # change file extension from .svg to .tif
        file_tif = file_svg.replace('.svg', '.tif')

        # open histology training image
        im = Image.open(file_tif)

        # read pixel size information
        xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
        yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

        # make array copy
        im_array = np.array(im)

        if DEBUG:
            plt.clf()
            plt.imshow(im)

        # read the ground truth cell contours in the SVG file. This produces a list [contour_0, ..., contour_N-1]
        # where each contour_i = [(X_0, Y_0), ..., (X_P-1, X_P-1)]
        contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False)

        # loop ground truth cell contours
        for j, contour in enumerate(contours):

            if DEBUG:
                # centre of current cell
                xy_c = (np.mean([p[0] for p in contour]), np.mean([p[1] for p in contour]))

                plt.clf()
                plt.subplot(221)
                plt.imshow(im)
                plt.plot([p[0] for p in contour], [p[1] for p in contour])
                plt.scatter(xy_c[0], xy_c[1])

            # rasterise current ground truth segmentation
            cell_seg_gtruth = Image.new("1", im.size, "black")  # I = 32-bit signed integer pixels
            draw = ImageDraw.Draw(cell_seg_gtruth)
            draw.polygon(contour, outline="white", fill="white")
            cell_seg_gtruth = np.array(cell_seg_gtruth, dtype=np.uint8)

            if DEBUG:
                plt.subplot(222)
                plt.imshow(cell_seg_gtruth)

            # loop different perturbations in the mask to have a collection of better and worse
            # segmentations
            for inc in [-0.20, -0.15, -0.10, -.07, -0.03, 0.0, 0.03, 0.07, 0.10, 0.15, 0.20]:

                # erode or dilate the ground truth mask to create the segmentation mask
                cell_seg = cytometer.utils.quality_model_mask(cell_seg_gtruth, quality_model_type='0_1_prop_band',
                                                              quality_model_type_param=inc)[0, :, :, 0].astype(np.uint8)

                # compute bounding box that contains the mask, and leaves some margin
                bbox_x0, bbox_y0, bbox_xend, bbox_yend = \
                    cytometer.utils.bounding_box_with_margin(cell_seg, coordinates='xy', inc=1.00)
                bbox_r0, bbox_c0, bbox_rend, bbox_cend = \
                    cytometer.utils.bounding_box_with_margin(cell_seg, coordinates='rc', inc=1.00)

                if DEBUG:
                    plt.subplot(223)
                    plt.cla()
                    plt.imshow(cell_seg)
                    plt.plot((bbox_x0, bbox_xend, bbox_xend, bbox_x0, bbox_x0),
                             (bbox_y0, bbox_y0, bbox_yend, bbox_yend, bbox_y0))

                # crop image and masks according to bounding box
                window_im = cytometer.utils.extract_bbox(im_array, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))
                window_seg_gtruth = cytometer.utils.extract_bbox(cell_seg_gtruth, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))
                window_seg = cytometer.utils.extract_bbox(cell_seg, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))

                if DEBUG:
                    plt.clf()
                    plt.subplot(221)
                    plt.cla()
                    plt.imshow(im_array)
                    plt.contour(cell_seg_gtruth, linewidths=1, levels=0.5, colors='green')
                    plt.contour(cell_seg, linewidths=1, levels=0.5, colors='blue')
                    plt.plot((bbox_x0, bbox_xend, bbox_xend, bbox_x0, bbox_x0),
                             (bbox_y0, bbox_y0, bbox_yend, bbox_yend, bbox_y0), 'black')

                    plt.subplot(222)
                    plt.cla()
                    plt.imshow(window_im)
                    plt.contour(window_seg_gtruth, linewidths=1, levels=0.5, colors='green')
                    plt.contour(window_seg, linewidths=1, levels=0.5, colors='blue')

                # input to the CNN: multiply histology by +1/-1 segmentation mask
                window_im = \
                    cytometer.utils.quality_model_mask(window_seg.astype(np.float32), im=window_im.astype(np.float32),
                                                       quality_model_type='-1_1')[0, :, :, :]

                # expected CNN output: segmentation - ground truth
                window_out_gtruth = window_seg.astype(np.float32) - window_seg_gtruth.astype(np.float32)

                if DEBUG:
                    plt.subplot(223)
                    plt.cla()
                    aux = 0.2989 * window_im[:, :, 0] + 0.5870 * window_im[:, :, 1] + 0.1140 * window_im[:, :, 2]
                    plt.imshow(aux)
                    plt.title('CNN input: histology * +1/-1 segmentation mask')

                    plt.subplot(224)
                    plt.cla()
                    plt.imshow(window_out_gtruth)

                # scaling factors for the training image
                training_size = (training_window_len, training_window_len)
                scaling_factor = np.array(training_size) / np.array(window_im.shape[0:2])
                window_pixel_size = np.array([xres, yres]) / scaling_factor  # (um, um)

                # resize the images to training window size
                window_im = cytometer.utils.resize(window_im, size=training_size, resample=Image.LINEAR)
                window_seg_gtruth = cytometer.utils.resize(window_seg_gtruth, size=training_size, resample=Image.NEAREST)
                window_seg = cytometer.utils.resize(window_seg, size=training_size, resample=Image.NEAREST)
                window_out_gtruth = cytometer.utils.resize(window_out_gtruth, size=training_size, resample=Image.NEAREST)

                if DEBUG:
                    plt.subplot(224)
                    plt.cla()
                    aux = 0.2989 * window_im[:, :, 0] + 0.5870 * window_im[:, :, 1] + 0.1140 * window_im[:, :, 2]
                    plt.imshow(aux)
                    plt.contour(window_out_gtruth, linewidths=1, levels=(-0.5, 0.5), colors='white')

                # add dummy dimensions for keras
                window_im = np.expand_dims(window_im, axis=0)
                window_seg_gtruth = np.expand_dims(window_seg_gtruth, axis=0)
                window_seg = np.expand_dims(window_seg, axis=0)
                window_out_gtruth = np.expand_dims(window_out_gtruth, axis=0)

                # check sizes and types
                assert(window_im.ndim == 4 and window_im.dtype == np.float32)
                assert(window_out_gtruth.ndim == 3 and window_out_gtruth.dtype == np.float32)

                # append images to use for training
                window_im_all.append(window_im)
                window_seg_gtruth_all.append(window_seg_gtruth)
                window_seg_all.append(window_seg)
                window_out_gtruth_all.append(window_out_gtruth)
                window_idx_all.append(np.array([i, j]))
                window_pixel_size_all.append(window_pixel_size)

    # end of loop: for i, file_svg in enumerate(np.array(file_list)[idx_test_all[i_fold]])
    #   this has now computed the test data for one fold

    # collapse lists into arrays
    window_im_all = np.concatenate(window_im_all)
    window_seg_gtruth_all = np.concatenate(window_seg_gtruth_all)
    window_seg_all = np.concatenate(window_seg_all)
    window_out_gtruth_all = np.concatenate(window_out_gtruth_all)
    window_idx_all = np.vstack(window_idx_all)
    window_pixel_size_all = np.vstack(window_pixel_size_all)

    # load quality model
    quality_model_filename = os.path.join(saved_models_dir, experiment_id + '_model_fold_' + str(i_fold) + '.h5')
    quality_model = keras.models.load_model(quality_model_filename)

    # process histology * mask
    window_out_all = quality_model.predict(window_im_all, batch_size=batch_size)

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
              + 0.114  * window_im_all[j, :, :, 2]
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
