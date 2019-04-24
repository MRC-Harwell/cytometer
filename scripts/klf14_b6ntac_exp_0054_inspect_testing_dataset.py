"""
Load testing data from each fold according to exp 0053, apply segmentation and quality network.
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
experiment_id = 'klf14_b6ntac_exp_0054_inspect_testing_dataset'

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
quality_model_basename = 'klf14_b6ntac_exp_0053_cnn_quality_network_fcn_overlapping_scaled_contours_model'
classifier_model_basename = 'klf14_b6ntac_exp_0057_cnn_tissue_classifier_fcn_overlapping_scaled_contours_model'

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
        contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False)

        if DEBUG:
            plt.clf()
            plt.imshow(im)

        '''Segment image
        '''

        # make array copy
        im_array = np.array(im, dtype=np.float32)
        im_array /= 255

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

            plt.subplot(221)
            plt.imshow(im)

            plt.subplot(222)
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
                plt.subplot(223)
                plt.cla()
                plt.imshow(cell_seg)
                plt.plot((bbox_x0, bbox_xend, bbox_xend, bbox_x0, bbox_x0),
                         (bbox_y0, bbox_y0, bbox_yend, bbox_yend, bbox_y0))

            # crop image and masks according to bounding box
            window_im = cytometer.utils.extract_bbox(im_array, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))
            # window_seg_gtruth = cytometer.utils.extract_bbox(cell_seg_gtruth, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))
            window_seg = cytometer.utils.extract_bbox(cell_seg, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))

            if DEBUG:
                plt.clf()
                plt.subplot(221)
                plt.cla()
                plt.imshow(im_array)
                # plt.contour(cell_seg_gtruth, linewidths=1, levels=0.5, colors='green')
                plt.contour(cell_seg, linewidths=1, levels=0.5, colors='blue')
                plt.plot((bbox_x0, bbox_xend, bbox_xend, bbox_x0, bbox_x0),
                         (bbox_y0, bbox_y0, bbox_yend, bbox_yend, bbox_y0), 'black')

                plt.subplot(222)
                plt.cla()
                plt.imshow(window_im)
                # plt.contour(window_seg_gtruth, linewidths=1, levels=0.5, colors='green')
                plt.contour(window_seg, linewidths=1, levels=0.5, colors='blue')

            # input to the CNN: multiply histology by +1/-1 segmentation mask
            window_masked_im = \
                cytometer.utils.quality_model_mask(window_seg.astype(np.float32), im=window_im.astype(np.float32),
                                                   quality_model_type='-1_1')[0, :, :, :]

            # scaling factors for the training image
            training_size = (training_window_len, training_window_len)
            scaling_factor = np.array(training_size) / np.array(window_masked_im.shape[0:2])
            window_pixel_size = np.array([xres, yres]) / scaling_factor  # (um, um)

            # resize the images to training window size
            window_im = cytometer.utils.resize(window_im, size=training_size, resample=Image.LINEAR)
            window_masked_im = cytometer.utils.resize(window_masked_im, size=training_size, resample=Image.LINEAR)
            # window_seg_gtruth = cytometer.utils.resize(window_seg_gtruth, size=training_size, resample=Image.NEAREST)
            window_seg = cytometer.utils.resize(window_seg, size=training_size, resample=Image.NEAREST)
            # window_out_gtruth = cytometer.utils.resize(window_out_gtruth, size=training_size, resample=Image.NEAREST)

            # add dummy dimensions for keras
            window_im = np.expand_dims(window_im, axis=0)
            window_masked_im = np.expand_dims(window_masked_im, axis=0)
            # window_seg_gtruth = np.expand_dims(window_seg_gtruth, axis=0)
            window_seg = np.expand_dims(window_seg, axis=0)
            # window_out_gtruth = np.expand_dims(window_out_gtruth, axis=0)

            # check sizes and types
            assert(window_im.ndim == 4 and window_im.dtype == np.float32)
            assert(window_masked_im.ndim == 4 and window_masked_im.dtype == np.float32)
            # assert(window_out_gtruth.ndim == 3 and window_out_gtruth.dtype == np.float32)

            # process histology * mask for quality
            window_classifier_out = quality_model.predict(window_im, batch_size=batch_size)

            # process histology for classification
            window_quality_out = quality_model.predict(window_masked_im, batch_size=batch_size)

            # correction for segmentation
            window_seg_correction = window_seg[0, :, :] * 0
            window_seg_correction[window_quality_out[0, :, :, 0] >= 0.5] = 1  # the segmentation went too far
            window_seg_correction[window_quality_out[0, :, :, 0] <= -0.5] = -1  # the segmentation fell short

            # corrected segmentation
            window_seg_corrected = window_seg[0, :, :]
            window_seg_corrected[window_quality_out[0, :, :, 0] >= 0.5] = 0  # the segmentation went too far
            window_seg_corrected[window_quality_out[0, :, :, 0] <= -0.5] = 1  # the segmentation fell short

            if DEBUG:
                plt.clf()

                plt.subplot(221)
                plt.cla()
                aux = 0.2989 * window_masked_im[0, :, :, 0] \
                      + 0.5870 * window_masked_im[0, :, :, 1] \
                      + 0.1140 * window_masked_im[0, :, :, 2]
                plt.imshow(aux)
                # plt.contour(window_out_gtruth_all[j, :, :], linewidths=1, colors='green')

                plt.subplot(222)
                plt.cla()
                plt.imshow(window_quality_out[0, :, :, 0])

                plt.subplot(223)
                plt.cla()
                plt.imshow(window_seg_correction)

                plt.subplot(224)
                plt.cla()
                plt.imshow(window_seg_corrected)
                plt.contour(window_seg[0, :, :], linewidths=1, colors='white')
