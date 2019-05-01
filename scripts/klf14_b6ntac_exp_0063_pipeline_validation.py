"""
Validate pipeline:
 * segmentation
   * contour (0055)
   * dmap (0056)
 * classifier (0059)
 * segmentation quality (0053) networks.

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
import pandas as pd
import time

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
experiment_id = 'klf14_b6ntac_exp_0063_pipeline_validation'

# model names
contour_model_basename = 'klf14_b6ntac_exp_0055_cnn_contour_model'
dmap_model_basename = 'klf14_b6ntac_exp_0056_cnn_dmap_model'
classifier_model_basename = 'klf14_b6ntac_exp_0059_cnn_tissue_classifier_fcn_overlapping_scaled_contours_model'
quality_model_basename = 'klf14_b6ntac_exp_0053_cnn_quality_network_fcn_overlapping_scaled_contours_model'

# load k-folds training and testing data
kfold_info_filename = os.path.join(saved_models_dir, 'klf14_b6ntac_exp_0053_cnn_quality_network_fcn_overlapping_scaled_contours_kfold_info.pickle')
with open(kfold_info_filename, 'rb') as f:
    kfold_info = pickle.load(f)
file_list = kfold_info['file_list']
idx_test_all = kfold_info['idx_test']
del kfold_info

# number of images
n_im = len(file_list)

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(root_data_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

'''Process the test data of each fold with the corresponding trained networks
'''

time0 = time.time()
df_all = pd.DataFrame()

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

        '''Segment image using the first part of the pipeline. We segment whole test windows
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

        # initialise dataframe to keep results: one cell per row, tagged with mouse metainformation
        df_im = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_tif),
                                                          values=[i_fold], values_tag='fold',
                                                          tags_to_keep=['id', 'ko', 'sex'])

        '''Cell by cell processing
        '''

        # loop contours
        for j, contour in enumerate(contours):

            # start dataframe row for this contour
            df = df_im.copy()
            df['im'] = i
            df['contour'] = j

            if DEBUG:
                # centre of current cell
                xy_c = (np.mean([p[0] for p in contour]), np.mean([p[1] for p in contour]))

                plt.clf()
                plt.subplot(221)
                plt.imshow(im_array)
                plt.plot([p[0] for p in contour], [p[1] for p in contour], color='green')
                plt.scatter(xy_c[0], xy_c[1])

            # rasterise current ground truth segmentation
            cell_seg_gtruth = Image.new("1", im.size, "black")  # I = 32-bit signed integer pixels
            draw = ImageDraw.Draw(cell_seg_gtruth)
            draw.polygon(contour, outline="white", fill="white")
            cell_seg_gtruth = np.array(cell_seg_gtruth, dtype=np.uint8)

            # compute bounding box that contains the mask, and leaves some margin
            bbox_x0, bbox_y0, bbox_xend, bbox_yend = \
                cytometer.utils.bounding_box_with_margin(cell_seg_gtruth, coordinates='xy', inc=1.00)
            bbox_r0, bbox_c0, bbox_rend, bbox_cend = \
                cytometer.utils.bounding_box_with_margin(cell_seg_gtruth, coordinates='rc', inc=1.00)

            # find the segmented label that best overlaps with the ground truth label
            match_out = \
                cytometer.utils.match_overlapping_labels(labels_ref=labels, labels_test=cell_seg_gtruth,
                                                         allow_repeat_ref=False)
            if len(match_out) != 1:
                raise Warning('No correspondence found for contour: ' + str(j))
                continue

            # isolate said segmented label
            cell_seg = (labels == match_out[0]['lab_ref']).astype(np.uint8)

            # add to dataframe: segmentation areas and Dice coefficient
            df['area_gtruth'] = match_out[0]['area_test'] * xres * yres  # um^2
            df['area_seg'] = match_out[0]['area_ref'] * xres * yres  # um^2
            df['dice'] = match_out[0]['dice']

            if DEBUG:
                plt.subplot(222)
                plt.imshow(im_array)
                plt.contour(cell_seg_gtruth, linewidths=1, levels=0.5, colors='green')
                plt.contour(cell_seg, linewidths=1, levels=0.5, colors='red')
                plt.xlim(bbox_x0, bbox_xend)
                plt.ylim(bbox_yend, bbox_y0)

            # crop image and masks according to bounding box
            window_im = cytometer.utils.extract_bbox(im_array, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))
            window_seg_gtruth = cytometer.utils.extract_bbox(cell_seg_gtruth, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))
            window_seg = cytometer.utils.extract_bbox(cell_seg, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))

            if DEBUG:
                plt.subplot(223)
                plt.imshow(window_im)
                plt.contour(window_seg_gtruth, linewidths=1, levels=0.5, colors='green')
                plt.contour(window_seg, linewidths=1, levels=0.5, colors='red')

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
            window_seg_gtruth = cytometer.utils.resize(window_seg_gtruth, size=training_size, resample=Image.NEAREST)
            window_seg = cytometer.utils.resize(window_seg, size=training_size, resample=Image.NEAREST)

            # add dummy dimensions for keras
            window_im = np.expand_dims(window_im, axis=0)
            window_masked_im = np.expand_dims(window_masked_im, axis=0)
            window_seg = np.expand_dims(window_seg, axis=0)

            # correct types
            window_seg = window_seg.astype(np.float32)
            window_seg_gtruth = window_seg_gtruth.astype(np.float32)

            # check sizes and types
            assert(window_im.ndim == 4 and window_im.dtype == np.float32)
            assert(window_masked_im.ndim == 4 and window_masked_im.dtype == np.float32)
            assert(window_seg.ndim == 3 and window_seg.dtype == np.float32)
            assert(window_seg_gtruth.ndim == 2 and window_seg_gtruth.dtype == np.float32)

            # process (histology * mask) for quality
            window_quality_out = quality_model.predict(window_masked_im, batch_size=batch_size)

            # correction for segmentation
            window_seg_correction = (window_seg[0, :, :].copy() * 0).astype(np.int8)
            window_seg_correction[window_quality_out[0, :, :, 0] >= 0.5] = 1  # the segmentation went too far
            window_seg_correction[window_quality_out[0, :, :, 0] <= -0.5] = -1  # the segmentation fell short

            # corrected segmentation
            window_seg_corrected = window_seg[0, :, :].copy()
            window_seg_corrected[window_seg_correction == 1] = 0
            window_seg_corrected[window_seg_correction == -1] = 1

            # corrected segmentation area
            area_seg_corrected = np.count_nonzero(window_seg_corrected) * window_pixel_size[0] * window_pixel_size[1]
            df['area_seg_corrected'] = area_seg_corrected

            if DEBUG:
                # plot segmentation correction
                plt.subplot(223)
                plt.cla()
                plt.imshow(window_im[0, :, :, :])
                plt.contour(window_seg_gtruth, linewidths=1, levels=0.5, colors='green')
                plt.contour(window_seg[0, :, :], linewidths=1, levels=0.5, colors='red')
                plt.contour(window_seg_corrected, linewidths=1, levels=0.5, colors='black')

            # process histology for classification
            window_classifier_out = classifier_model.predict(window_im, batch_size=batch_size)

            # get classification label for each pixel
            window_classifier_class = np.argmax(window_classifier_out, axis=3)

            # proportion of "Other" pixels in the mask
            window_other_prop = np.count_nonzero(window_seg * window_classifier_class) \
                                / np.count_nonzero(window_seg)

            # add to dataframe row
            df['other_gtruth'] = contour_type_all[j]
            df['other_prop'] = window_other_prop

            if DEBUG:
                # plot classification
                plt.subplot(224)
                plt.cla()
                plt.imshow(window_classifier_class[0, :, :])
                plt.contour(window_seg_gtruth, linewidths=1, levels=0.5, colors='green')
                plt.contour(window_seg[0, :, :], linewidths=1, levels=0.5, colors='red')
                plt.title('"Other" prop = ' + str("{:.0f}".format(window_other_prop * 100)) + '%')

            # append current results to global dataframe
            df_all = pd.concat([df_all, df])

        print('Time so far: ' + str(time.time() - time0) + ' s')

# save results
dataframe_filename = os.path.join(saved_models_dir, experiment_id + '_dataframe.pkl')
df_all.to_pickle(dataframe_filename)
