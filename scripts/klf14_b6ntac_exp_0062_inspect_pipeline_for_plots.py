"""
Produce plots from pipeline using classifier (0059) and segmentation quality (0053) networks.

Here, we segment the images and apply the classifier and quality networks to those segmentations. For validation, we
need to use the ground truth contours as the basis, but I want to preserve this script in case I need to generate those
kind of figures.

I'll start another script (0063) for proper pipeline validation.
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
import pickle

# other imports
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

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
experiment_id = 'klf14_b6ntac_exp_0062_validate_pipeline'

# load k-folds training and testing data
kfold_info_filename = os.path.join(saved_models_dir, 'klf14_b6ntac_exp_0053_cnn_quality_network_fcn_overlapping_scaled_contours_kfold_info.pickle')
with open(kfold_info_filename, 'rb') as f:
    kfold_info = pickle.load(f)
file_list = kfold_info['file_list']
idx_test_all = kfold_info['idx_test']
del kfold_info

# model names
contour_model_basename = 'klf14_b6ntac_exp_0055_cnn_contour_model'
dmap_model_basename = 'klf14_b6ntac_exp_0056_cnn_dmap_model'
classifier_model_basename = 'klf14_b6ntac_exp_0059_cnn_tissue_classifier_fcn_overlapping_scaled_contours_model'
quality_model_basename = 'klf14_b6ntac_exp_0053_cnn_quality_network_fcn_overlapping_scaled_contours_model'

# number of images
n_im = len(file_list)

'''Load the test data of each fold
'''

for i_fold in range(n_folds):

    print('## Fold ' + str(i_fold) + '/' + str(n_folds - 1))

    '''Load data
    '''

    # list of test files in this fold
    file_list_test = np.array(file_list)[idx_test_all[i_fold]]

    # load quality model
    quality_model_filename = os.path.join(saved_models_dir, quality_model_basename + '_fold_' + str(i_fold) + '.h5')
    quality_model = keras.models.load_model(quality_model_filename)

    # load classifier model
    classifier_model_filename = os.path.join(saved_models_dir, classifier_model_basename + '_fold_' + str(i_fold) + '.h5')
    classifier_model = keras.models.load_model(classifier_model_filename)

    for i, file_svg in enumerate(file_list_test):

        print('file ' + str(i) + '/' + str(len(idx_test_all[i_fold]) - 1))

        # change file extension from .svg to .tif
        file_tif = file_svg.replace('.svg', '.tif')

        # open histology training image
        im = Image.open(file_tif)

        # read pixel size information
        xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
        yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

        # read the ground truth cell contours in the SVG file. This produces a list [contour_0, ..., contour_N-1]
        # where each contour_i = [(X_0, Y_0), ..., (X_P-1, X_P-1)]
        cell_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False,
                                                                minimum_npoints=3)
        other_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Other', add_offset_from_filename=False,
                                                                 minimum_npoints=3)
        brown_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Brown', add_offset_from_filename=False,
                                                                 minimum_npoints=3)
        contours = cell_contours + other_contours + brown_contours

        # make a list with the type of cell each contour is classified as
        contour_type_all = [np.zeros(shape=(len(cell_contours),), dtype=np.uint8),  # 0: white-adipocyte
                            np.ones(shape=(len(other_contours),), dtype=np.uint8),  # 1: other types of tissue
                            np.ones(shape=(len(brown_contours),),
                                dtype=np.uint8)]  # 1: brown cells (treated as "other" tissue)
        contour_type_all = np.concatenate(contour_type_all)

        print('Cells: ' + str(len(cell_contours)))
        print('Other: ' + str(len(other_contours)))
        print('Brown: ' + str(len(brown_contours)))
        print('')

        if DEBUG:
            plt.clf()
            plt.imshow(im)

        '''Segment image
        '''

        # make array copy
        im_array = np.array(im, dtype=np.float32)
        im_array /= 255  # NOTE: quality network 0053 expects intensity values in [-255, 255], but contour and dmap do

        # load contour and dmap models
        contour_model_filename = os.path.join(saved_models_dir, contour_model_basename + '_fold_' + str(i_fold) + '.h5')
        dmap_model_filename = os.path.join(saved_models_dir, dmap_model_basename + '_fold_' + str(i_fold) + '.h5')

        contour_model = keras.models.load_model(contour_model_filename)
        dmap_model = keras.models.load_model(dmap_model_filename)

        # set input layer to size of images
        contour_model = cytometer.models.change_input_size(contour_model, batch_shape=(None,) + im_array.shape)
        dmap_model = cytometer.models.change_input_size(dmap_model, batch_shape=(None,) + im_array.shape)

        # run histology image through network
        contour_pred = contour_model.predict(np.expand_dims(im_array, axis=0))
        dmap_pred = dmap_model.predict(np.expand_dims(im_array, axis=0))

        # cell segmentation
        labels, labels_borders \
            = cytometer.utils.segment_dmap_contour(dmap_pred[0, :, :, 0],
                                                   contour=contour_pred[0, :, :, 0],
                                                   border_dilation=0)

        if DEBUG:
            plt.clf()

            plt.subplot(121)
            plt.imshow(im)

            plt.subplot(122)
            plt.imshow(labels)

        # loop labels
        for lab in np.unique(labels):

            # isolate segmentation mask
            cell_seg = (labels == lab).astype(np.uint8)

            # compute bounding box that contains the mask, and leaves some margin
            bbox_x0, bbox_y0, bbox_xend, bbox_yend = \
                cytometer.utils.bounding_box_with_margin(cell_seg, coordinates='xy', inc=1.00)
            bbox_r0, bbox_c0, bbox_rend, bbox_cend = \
                cytometer.utils.bounding_box_with_margin(cell_seg, coordinates='rc', inc=1.00)

            if DEBUG:
                plt.clf()
                plt.subplot(221)
                plt.imshow(im)

                plt.subplot(222)
                plt.imshow(labels)

                plt.subplot(223)
                plt.cla()
                plt.imshow(cell_seg)
                plt.plot((bbox_x0, bbox_xend, bbox_xend, bbox_x0, bbox_x0),
                         (bbox_y0, bbox_y0, bbox_yend, bbox_yend, bbox_y0))

            # crop image and masks according to bounding box
            window_im = cytometer.utils.extract_bbox(im_array, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))
            window_seg = cytometer.utils.extract_bbox(cell_seg, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))

            if DEBUG:
                plt.clf()
                plt.subplot(121)
                plt.cla()
                plt.imshow(im_array)
                plt.contour(cell_seg, linewidths=1, levels=0.5, colors='blue')
                plt.plot((bbox_x0, bbox_xend, bbox_xend, bbox_x0, bbox_x0),
                         (bbox_y0, bbox_y0, bbox_yend, bbox_yend, bbox_y0), 'black')

                plt.subplot(122)
                plt.cla()
                plt.imshow(window_im)
                plt.contour(window_seg, linewidths=1, levels=0.5, colors='blue')

            # input to segmentation quality CNN: multiply histology by +1/-1 segmentation mask
            window_masked_im = \
                cytometer.utils.quality_model_mask(window_seg.astype(np.float32), im=window_im.astype(np.float32),
                                                   quality_model_type='-1_1')[0, :, :, :]
            # NOTE: quality network 0053 expects values in [-255, 255], float32
            window_masked_im *= 255

            # scaling factors for the training image
            training_size = (training_window_len, training_window_len)
            scaling_factor = np.array(training_size) / np.array(window_masked_im.shape[0:2])
            window_pixel_size = np.array([xres, yres]) / scaling_factor  # (um, um)

            # resize the images to training window size
            window_im = cytometer.utils.resize(window_im, size=training_size, resample=Image.LINEAR)
            window_masked_im = cytometer.utils.resize(window_masked_im, size=training_size, resample=Image.LINEAR)
            window_seg = cytometer.utils.resize(window_seg, size=training_size, resample=Image.NEAREST)

            # add dummy dimensions for keras
            window_im = np.expand_dims(window_im, axis=0)
            window_masked_im = np.expand_dims(window_masked_im, axis=0)
            window_seg = np.expand_dims(window_seg, axis=0)

            # check sizes and types
            assert(window_im.ndim == 4 and window_im.dtype == np.float32)
            assert(window_masked_im.ndim == 4 and window_masked_im.dtype == np.float32)

            # process histology * mask for quality
            window_quality_out = quality_model.predict(window_masked_im, batch_size=batch_size)

            # correction for segmentation
            window_seg_correction = (window_seg[0, :, :].copy() * 0).astype(np.int8)
            window_seg_correction[window_quality_out[0, :, :, 0] >= 0.5] = 1  # the segmentation went too far
            window_seg_correction[window_quality_out[0, :, :, 0] <= -0.5] = -1  # the segmentation fell short

            # corrected segmentation
            window_seg_corrected = window_seg[0, :, :].copy()
            window_seg_corrected[window_seg_correction == 1] = 0
            window_seg_corrected[window_seg_correction == -1] = 1

            if DEBUG:
                # plot segmentation correction
                plt.clf()

                plt.subplot(221)
                plt.cla()
                aux = 0.2989 * window_masked_im[0, :, :, 0] \
                      + 0.5870 * window_masked_im[0, :, :, 1] \
                      + 0.1140 * window_masked_im[0, :, :, 2]
                plt.imshow(aux)

                plt.subplot(222)
                plt.cla()
                plt.imshow(window_quality_out[0, :, :, 0])

                plt.subplot(223)
                plt.cla()
                plt.imshow(window_seg_correction)

                plt.subplot(224)
                plt.cla()
                plt.imshow(window_im[0, :, :, :])
                plt.contour(window_seg[0, :, :], linewidths=1, levels=0.5, colors='red')
                plt.contour(window_seg_corrected, linewidths=1, levels=0.5, colors='green')

            # process histology for classification
            window_classifier_out = classifier_model.predict(window_im, batch_size=batch_size)

            # get classification label for each pixel
            window_classifier_class = np.argmax(window_classifier_out, axis=3)

            # proportion of "Other" pixels in the mask
            window_other_prop = np.count_nonzero(window_seg * window_classifier_class) \
                                / np.count_nonzero(window_seg)

            if DEBUG:
                # plot classification
                plt.clf()

                plt.subplot(221)
                plt.cla()
                plt.imshow(window_im[0, :, :, :])
                plt.contour(window_seg[0, :, :], linewidths=1, colors='blue', linestyles='dotted')
                plt.title('Histology')
                plt.axis('off')

                plt.subplot(222)
                plt.cla()
                aux = window_classifier_class[0, :, :]
                plt.imshow(aux)
                plt.contour(window_seg[0, :, :], linewidths=1, colors='white', linestyles='solid')
                plt.title('"Other" prop = ' + str("{:.0f}".format(window_other_prop * 100)) + '%')
                plt.axis('off')

                plt.subplot(223)
                plt.cla()
                plt.imshow(window_classifier_out[0, :, :, 0])
                plt.title('Cell')
                plt.axis('off')

                plt.subplot(224)
                plt.cla()
                plt.imshow(window_classifier_out[0, :, :, 1])
                plt.title('Other')
                plt.axis('off')

