"""
Validate pipeline v3:
 * segmentation
   * dmap (0056)
   * contour (0070)
 * classifier (0061)
 * segmentation quality (0053) networks

"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0072_pipeline_validation'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import warnings
import pickle
import pandas as pd
import time

# other imports
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import linregress
from skimage.morphology import remove_small_holes, binary_closing, binary_dilation
from scipy.ndimage.morphology import binary_fill_holes
import cv2

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
smallest_cell_area = 1500

# training window length
training_window_len = 401

# remove from training cells that don't have a good enough overlap with a reference label
smallest_dice = 0.5

# segmentations with Dice >= threshold are accepted
dice_threshold = 0.9

# segmentation parameters
min_cell_area = 1500
median_size = 0
closing_size = 11
contour_seed_threshold = 0.005
batch_size = 16

# rough_foreground_mask() parameters
downsample_factor = 8.0
dilation_size = 25
component_size_threshold = 0
hole_size_treshold = 8000


'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

# model names
dmap_model_basename = 'klf14_b6ntac_exp_0056_cnn_dmap_model'
contour_model_basename = 'klf14_b6ntac_exp_0070_cnn_contour_after_dmap_model'
classifier_model_basename = 'klf14_b6ntac_exp_0061_cnn_tissue_classifier_fcn_overlapping_scaled_contours_model'
quality_model_basename = 'klf14_b6ntac_exp_0053_cnn_quality_network_fcn_overlapping_scaled_contours_model'

# load k-folds training and testing data
kfold_info_filename = os.path.join(saved_models_dir, 'klf14_b6ntac_exp_0055_cnn_contour_kfold_info.pickle')
with open(kfold_info_filename, 'rb') as f:
    kfold_info = pickle.load(f)
file_list = kfold_info['file_list']
idx_test_all = kfold_info['idx_test']
del kfold_info
n_folds = len(idx_test_all)

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
    # file_list_train = np.array(file_list)[idx_train_all[i_fold]]

    # correct home directory in file paths
    file_list_test = cytometer.data.change_home_directory(list(file_list_test),
                                                          '/users/rittscher/rcasero', home, check_isfile=True)

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

        # open histology testing image
        im = Image.open(file_tif)

        # read pixel size information
        xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
        yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(im)
            plt.axis('off')
            plt.title('Histology', fontsize=14)

        # # rough segmentation of the tissue in the image
        # rough_mask, im_downsampled = \
        #     cytometer.utils.rough_foreground_mask(im, downsample_factor=downsample_factor,
        #                                           dilation_size=dilation_size,
        #                                           component_size_threshold=component_size_threshold,
        #                                           hole_size_treshold=hole_size_treshold,
        #                                           return_im=True)
        #
        # if DEBUG:
        #     plt.subplot(222)
        #     plt.imshow(im_downsampled)
        #     plt.imshow(rough_mask, alpha=0.5)
        #     plt.axis('off')
        #     plt.title('Downsampled rough mask', fontsize=14)

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

        '''Segmentation part of the pipeline. We segment whole test windows
        '''

        # NOTE: quality network 0053 expects intensity values in [-255, 255], but contour and dmap expect [0, 1]

        # filenames of contour and dmap models
        contour_model_filename = os.path.join(saved_models_dir, contour_model_basename + '_fold_' + str(i_fold) + '.h5')
        dmap_model_filename = os.path.join(saved_models_dir, dmap_model_basename + '_fold_' + str(i_fold) + '.h5')

        # segment histology
        labels, _ = cytometer.utils.segment_dmap_contour_v3(im, contour_model=contour_model_filename,
                                                            dmap_model=dmap_model_filename)

        if DEBUG:
            plt.subplot(223)
            plt.cla()
            plt.imshow(im)
            plt.contour(labels, levels=np.unique(labels), colors='k')
            plt.axis('off')
            plt.title('Segmentation', fontsize=14)

        # remove labels that touch the edges, that are too small or that don't overlap enough with the rough foreground
        # mask
        labels \
            = cytometer.utils.clean_segmentation(labels, remove_edge_labels=True, min_cell_area=min_cell_area)

        if DEBUG:
            plt.subplot(224)
            plt.cla()
            plt.imshow(im)
            plt.contour(labels, levels=np.unique(labels), colors='k')
            plt.axis('off')
            plt.title('Cleaned segmentation', fontsize=14)

        # list of unique segmentation labels (remove background)
        labels_seg = list(np.unique(labels))
        if 0 in labels_seg:
            labels_seg.remove(0)

        if len(labels_seg) == 0:
            warnings.warn('No labels produced!')

        # initialise dataframe to keep results: one cell per row, tagged with mouse metainformation
        df_im = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_tif),
                                                          values=[i_fold], values_tag='fold',
                                                          tags_to_keep=['id', 'ko', 'sex'])

        '''Ground truth cell by cell processing
        '''

        # loop contours
        for j, contour in enumerate(contours):

            # start dataframe row for this contour
            df = df_im.copy()
            df['im'] = i
            df['contour'] = j

            if DEBUG:
                # # centre of current cell
                # xy_c = (np.mean([p[0] for p in contour]), np.mean([p[1] for p in contour]))
                # plt.scatter(xy_c[0], xy_c[1])

                # close the contour
                contour_aux = contour.copy()
                contour_aux.append(contour[0])

                plt.clf()
                plt.imshow(im)
                plt.contour(labels, levels=np.unique(labels), colors='k')
                plt.plot([p[0] for p in contour_aux], [p[1] for p in contour_aux], color='green')
                plt.axis('off')

            '''Match hand-traced segmentation to automatic segmentation'''

            # rasterise current ground truth segmentation
            cell_seg_gtruth = Image.new("1", im.size, "black")  # I = 32-bit signed integer pixels
            draw = ImageDraw.Draw(cell_seg_gtruth)
            draw.polygon(contour, outline="white", fill="white")
            cell_seg_gtruth = np.array(cell_seg_gtruth, dtype=np.uint8)

            # find the segmented label that best overlaps with the ground truth label
            match_out = \
                cytometer.utils.match_overlapping_labels(labels_ref=labels, labels_test=cell_seg_gtruth,
                                                         allow_repeat_ref=False)

            if len(match_out) != 1:
                warnings.warn('No correspondence found for contour: ' + str(j))
                continue

            if DEBUG:
                plt.contour(labels == match_out['lab_ref'], levels=np.unique(labels), colors='r')

            # remove matched segmentation from list of remaining segmentations
            if match_out[0]['lab_ref'] in labels_seg:
                labels_seg.remove(match_out[0]['lab_ref'])

            # isolate said segmented label
            cell_seg = (labels == match_out[0]['lab_ref']).astype(np.uint8)

            # compute bounding box for the ground truth cell and the corresponding segmentation
            bbox_seg_gtruth_x0, bbox_seg_gtruth_y0, bbox_seg_gtruth_xend, bbox_seg_gtruth_yend = \
                cytometer.utils.bounding_box_with_margin(cell_seg_gtruth, coordinates='xy', inc=1.00)
            bbox_seg_gtruth_r0, bbox_seg_gtruth_c0, bbox_seg_gtruth_rend, bbox_seg_gtruth_cend = \
                cytometer.utils.bounding_box_with_margin(cell_seg_gtruth, coordinates='rc', inc=1.00)

            bbox_seg_x0, bbox_seg_y0, bbox_seg_xend, bbox_seg_yend = \
                cytometer.utils.bounding_box_with_margin(cell_seg, coordinates='xy', inc=1.00)
            bbox_seg_r0, bbox_seg_c0, bbox_seg_rend, bbox_seg_cend = \
                cytometer.utils.bounding_box_with_margin(cell_seg, coordinates='rc', inc=1.00)

            # add to dataframe: segmentation areas and Dice coefficient
            df['area_gtruth'] = match_out[0]['area_test'] * xres * yres  # um^2
            df['area_seg'] = match_out[0]['area_ref'] * xres * yres  # um^2
            df['dice'] = match_out[0]['dice']

            if DEBUG:
                plt.clf()
                plt.imshow(im)
                plt.contour(cell_seg_gtruth, linewidths=1, colors='green')
                plt.contour(cell_seg, linewidths=1, colors='red')
                plt.plot((bbox_seg_gtruth_x0, bbox_seg_gtruth_xend, bbox_seg_gtruth_xend, bbox_seg_gtruth_x0, bbox_seg_gtruth_x0),
                         (bbox_seg_gtruth_y0, bbox_seg_gtruth_y0, bbox_seg_gtruth_yend, bbox_seg_gtruth_yend, bbox_seg_gtruth_y0),
                         color='green')
                plt.plot((bbox_seg_x0, bbox_seg_xend, bbox_seg_xend, bbox_seg_x0, bbox_seg_x0),
                         (bbox_seg_y0, bbox_seg_y0, bbox_seg_yend, bbox_seg_yend, bbox_seg_y0),
                         color='red')
                bbox_both_x0 = np.min((bbox_seg_gtruth_x0, bbox_seg_x0))
                bbox_both_y0 = np.min((bbox_seg_gtruth_y0, bbox_seg_y0))
                bbox_both_xend = np.max((bbox_seg_gtruth_xend, bbox_seg_xend))
                bbox_both_yend = np.max((bbox_seg_gtruth_yend, bbox_seg_yend))
                plt.xlim(bbox_both_x0 - (bbox_both_xend - bbox_both_x0) * 0.1,
                         bbox_both_xend + (bbox_both_xend - bbox_both_x0) * 0.1)
                plt.ylim(bbox_both_yend + (bbox_both_yend - bbox_both_y0) * 0.1,
                         bbox_both_y0 - (bbox_both_yend - bbox_both_y0) * 0.1)
                plt.axis('off')

            # crop image and masks according to bounding box of automatic segmentation
            window_im = cytometer.utils.extract_bbox(np.array(im), (bbox_seg_r0, bbox_seg_c0, bbox_seg_rend, bbox_seg_cend))
            window_seg_gtruth = cytometer.utils.extract_bbox(cell_seg_gtruth, (bbox_seg_r0, bbox_seg_c0, bbox_seg_rend, bbox_seg_cend))
            window_seg = cytometer.utils.extract_bbox(cell_seg, (bbox_seg_r0, bbox_seg_c0, bbox_seg_rend, bbox_seg_cend))

            if DEBUG:
                plt.clf()
                plt.imshow(window_im)
                plt.contour(window_seg_gtruth, linewidths=1, colors='green')
                plt.contour(window_seg, linewidths=1, colors='red')
                plt.axis('off')

            '''Cropping and resizing of individual cell crop-outs'''

            # input to segmentation quality CNN: multiply histology by +1/-1 segmentation mask
            # NOTE: quality network 0053 expects values in [-255, 255], float32
            window_masked_im = \
                cytometer.utils.quality_model_mask(window_seg.astype(np.float32), im=window_im.astype(np.float32),
                                                   quality_model_type='-1_1')[0, :, :, :]

            # scaling factors for the training image
            training_size = (training_window_len, training_window_len)
            scaling_factor = np.array(training_size) / np.array(window_seg.shape[0:2])
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
            window_im = window_im.astype(np.float32) / 255.0
            window_seg = window_seg.astype(np.float32)
            window_seg_gtruth = window_seg_gtruth.astype(np.float32)

            # check sizes and types
            assert(window_im.ndim == 4 and window_im.dtype == np.float32
                   and np.min(window_im) >= 0 and np.max(window_im) <= 1.0)
            assert(window_masked_im.ndim == 4 and window_masked_im.dtype == np.float32
                   and np.min(window_masked_im) >= -255.0 and np.max(window_masked_im) <= 255.0)
            assert(window_seg.ndim == 3 and window_seg.dtype == np.float32)
            assert(window_seg_gtruth.ndim == 2 and window_seg_gtruth.dtype == np.float32)

            '''Object classification as "white adipocyte" or "other"'''

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
                plt.clf()
                plt.imshow(window_classifier_class[0, :, :])
                # plt.contour(window_seg_gtruth, linewidths=1, colors='green')
                plt.contour(window_seg[0, :, :], linewidths=1, colors='red')
                plt.title('"Other" prop = ' + str("{:.0f}".format(window_other_prop * 100)) + '%', fontsize=14)
                plt.axis('off')

            '''Segmentation correction'''

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

            # change type
            window_seg_corrected = window_seg_corrected.astype(np.uint8)

            # fill holes
            window_seg_corrected = binary_fill_holes(window_seg_corrected > 0).astype(window_seg_corrected.dtype)

            # keep only the largest segmentation
            _, labels_aux, stats_aux, _ = cv2.connectedComponentsWithStats(window_seg_corrected)
            stats_aux = stats_aux[1:, :]  # remove 0 label stats
            lab = np.argmax(stats_aux[:, cv2.CC_STAT_AREA]) + 1
            window_seg_corrected = (labels_aux == lab).astype(np.uint8)

            # smooth segmentation
            window_seg_corrected = binary_closing(window_seg_corrected, selem=np.ones((11, 11)))

            # corrected segmentation area
            area_seg_corrected = np.count_nonzero(window_seg_corrected) * window_pixel_size[0] * window_pixel_size[1]
            df['area_seg_corrected'] = area_seg_corrected

            if DEBUG:
                # plot segmentation correction
                plt.clf()
                plt.imshow(window_im[0, :, :, :])
                plt.contour(window_seg_gtruth, linewidths=1, colors='green')
                plt.contour(window_seg[0, :, :], linewidths=1, colors='red')
                plt.contour(window_seg_corrected, linewidths=1, colors='black')
                plt.axis('off')

            # append current results to global dataframe
            df_all = pd.concat([df_all, df])

        print('Time so far: ' + str(time.time() - time0) + ' s')

        '''Automatic segmentation cell by cell processing
        '''

        # segmentations have no corresponding contours here
        contour = -1

        for lab in labels_seg:

            # isolate segmented label
            cell_seg = (labels == lab).astype(np.uint8)

            # compute bounding box for the segmentation
            bbox_seg_x0, bbox_seg_y0, bbox_seg_xend, bbox_seg_yend = \
                cytometer.utils.bounding_box_with_margin(cell_seg, coordinates='xy', inc=1.00)
            bbox_seg_r0, bbox_seg_c0, bbox_seg_rend, bbox_seg_cend = \
                cytometer.utils.bounding_box_with_margin(cell_seg, coordinates='rc', inc=1.00)

            # add to dataframe: segmentation areas and Dice coefficient
            df['area_gtruth'] = np.nan
            df['area_seg'] = match_out[0]['area_ref'] * xres * yres  # um^2
            df['dice'] = np.nan

            if DEBUG:
                plt.clf()
                plt.imshow(im)
                plt.contour(cell_seg, linewidths=1, colors='red')
                plt.plot((bbox_seg_x0, bbox_seg_xend, bbox_seg_xend, bbox_seg_x0, bbox_seg_x0),
                         (bbox_seg_y0, bbox_seg_y0, bbox_seg_yend, bbox_seg_yend, bbox_seg_y0),
                         color='red')
                plt.xlim(bbox_seg_x0 - (bbox_seg_xend - bbox_seg_x0) * 0.1,
                         bbox_seg_xend + (bbox_seg_xend - bbox_seg_x0) * 0.1)
                plt.ylim(bbox_seg_yend + (bbox_seg_yend - bbox_seg_y0) * 0.1,
                         bbox_seg_y0 - (bbox_seg_yend - bbox_seg_y0) * 0.1)
                plt.axis('off')

            # crop image and masks according to bounding box of automatic segmentation
            window_im = cytometer.utils.extract_bbox(np.array(im),
                                                     (bbox_seg_r0, bbox_seg_c0, bbox_seg_rend, bbox_seg_cend))
            window_seg = cytometer.utils.extract_bbox(cell_seg,
                                                      (bbox_seg_r0, bbox_seg_c0, bbox_seg_rend, bbox_seg_cend))

            if DEBUG:
                plt.clf()
                plt.imshow(window_im)
                plt.contour(window_seg, linewidths=1, colors='red')

            '''Cropping and resizing of individual cell crop-outs'''

            # input to segmentation quality CNN: multiply histology by +1/-1 segmentation mask
            # NOTE: quality network 0053 expects values in [-255, 255], float32
            window_masked_im = \
                cytometer.utils.quality_model_mask(window_seg.astype(np.float32), im=window_im.astype(np.float32),
                                                   quality_model_type='-1_1')[0, :, :, :]

            # scaling factors for the training image
            training_size = (training_window_len, training_window_len)
            scaling_factor = np.array(training_size) / np.array(window_seg.shape[0:2])
            window_pixel_size = np.array([xres, yres]) / scaling_factor  # (um, um)

            # resize the images to training window size
            window_im = cytometer.utils.resize(window_im, size=training_size, resample=Image.LINEAR)
            window_masked_im = cytometer.utils.resize(window_masked_im, size=training_size, resample=Image.LINEAR)
            window_seg = cytometer.utils.resize(window_seg, size=training_size, resample=Image.NEAREST)

            # add dummy dimensions for keras
            window_im = np.expand_dims(window_im, axis=0)
            window_masked_im = np.expand_dims(window_masked_im, axis=0)
            window_seg = np.expand_dims(window_seg, axis=0)

            # correct types
            window_im = window_im.astype(np.float32) / 255.0
            window_seg = window_seg.astype(np.float32)
            window_seg_gtruth = window_seg_gtruth.astype(np.float32)

            # check sizes and types
            assert(window_im.ndim == 4 and window_im.dtype == np.float32
                   and np.min(window_im) >= 0 and np.max(window_im) <= 1.0)
            assert(window_masked_im.ndim == 4 and window_masked_im.dtype == np.float32
                   and np.min(window_masked_im) >= -255.0 and np.max(window_masked_im) <= 255.0)
            assert(window_seg.ndim == 3 and window_seg.dtype == np.float32)

            '''Object classification as "white adipocyte" or "other"'''

            # process histology for classification
            window_classifier_out = classifier_model.predict(window_im, batch_size=batch_size)

            # get classification label for each pixel
            window_classifier_class = np.argmax(window_classifier_out, axis=3)

            # proportion of "Other" pixels in the mask
            window_other_prop = np.count_nonzero(window_seg * window_classifier_class) \
                                / np.count_nonzero(window_seg)

            # add to dataframe row
            df['other_gtruth'] = -1
            df['other_prop'] = window_other_prop

            if DEBUG:
                # plot classification
                plt.clf()
                plt.imshow(window_classifier_class[0, :, :])
                plt.contour(window_seg[0, :, :], linewidths=1, colors='red')
                plt.title('"Other" prop = ' + str("{:.0f}".format(window_other_prop * 100)) + '%', fontsize=14)
                plt.axis('off')

            '''Segmentation correction'''

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

            # change type
            window_seg_corrected = window_seg_corrected.astype(np.uint8)

            # fill holes
            window_seg_corrected = binary_fill_holes(window_seg_corrected > 0).astype(window_seg_corrected.dtype)

            # keep only the largest segmentation
            _, labels_aux, stats_aux, _ = cv2.connectedComponentsWithStats(window_seg_corrected)
            stats_aux = stats_aux[1:, :]  # remove 0 label stats
            lab = np.argmax(stats_aux[:, cv2.CC_STAT_AREA]) + 1
            window_seg_corrected = (labels_aux == lab).astype(np.uint8)

            # smooth segmentation
            window_seg_corrected = binary_closing(window_seg_corrected, selem=np.ones((11, 11)))

            # corrected segmentation area
            area_seg_corrected = np.count_nonzero(window_seg_corrected) * window_pixel_size[0] * window_pixel_size[1]
            df['area_seg_corrected'] = area_seg_corrected

            if DEBUG:
                # plot segmentation correction
                plt.clf()
                plt.imshow(window_im[0, :, :, :])
                plt.contour(window_seg[0, :, :], linewidths=1, colors='red')
                plt.contour(window_seg_corrected, linewidths=1, colors='black')
                plt.axis('off')

            # append current results to global dataframe
            df_all = pd.concat([df_all, df])

        print('Time so far: ' + str(time.time() - time0) + ' s')

# reset indices
df_all.reset_index(drop=True, inplace=True)

# save results
dataframe_filename = os.path.join(saved_models_dir, experiment_id + '_dataframe.pkl')
df_all.to_pickle(dataframe_filename)

'''Dataframe analysis of segmentations with corresponding ground truth
'''

# dataframe:
#  ['index', 'id', 'ko', 'sex', 'fold', 'im', 'contour', 'area_gtruth',
#   'area_seg', 'dice', 'area_seg_corrected']

## previous experiment: classifier without data augmentation (only segmentations with corresponding ground truth)

# load results
dataframe_filename_0063 = os.path.join(saved_models_dir, 'klf14_b6ntac_exp_0063_pipeline_validation_dataframe.pkl')
df_0063 = pd.read_pickle(dataframe_filename_0063)

# reject rows with very low Dice values, because that means that the ground truth contours doesn't really overlap with
# an automatic segmentation. Instead, it's probably just touching a nearby one
idx = df_0063['dice'] >= 0.5
df_0063 = df_0063.loc[idx, :]

# remove rows with segmentations that have no corresponding ground truth
idx = df_0063['other_gtruth'] != -1
df_0063 = df_0063.loc[idx, :]

# add columns for cell estimates. This could be computed on the fly in the plots, but this way makes the code below
# a bit easier to read, and less likely to have bugs if we forget the 1-x operation
df_0063['cell_gtruth'] = 1 - df_0063['other_gtruth']
df_0063['cell_prop'] = 1 - df_0063['other_prop']

# classifier ROC (we make cell=1, other=0 for clarity of the results)
fpr_0063, tpr_0063, thr_0063 = roc_curve(y_true=df_0063['cell_gtruth'],
                                         y_score=df_0063['cell_prop'])
roc_auc_0063 = auc(fpr_0063, tpr_0063)

# find point in the curve for False Positive Rate ~ 10%
idx_0063 = np.where(fpr_0063 >= 0.1)[0][0]

## this experiment: classifier with data augmentation (only segmentations with corresponding ground truth)

# load results
dataframe_filename_0064 = os.path.join(saved_models_dir, 'klf14_b6ntac_exp_0064_pipeline_validation_dataframe.pkl')
df_0064 = pd.read_pickle(dataframe_filename_0064)

# remove rows with very low Dice values, because that means that the ground truth contours doesn't really overlap with
# an automatic segmentation. Instead, it's probably just touching a nearby one
idx = df_0064['dice'] >= 0.5
df_0064 = df_0064.loc[idx, :]

# remove rows with segmentations that have no corresponding ground truth
idx = df_0064['other_gtruth'] != -1
df_0064 = df_0064.loc[idx, :]

# add columns for cell estimates. This could be computed on the fly in the plots, but this way makes the code below
# a bit easier to read, and less likely to have bugs if we forget the 1-x operation
df_0064['cell_gtruth'] = 1 - df_0064['other_gtruth']
df_0064['cell_prop'] = 1 - df_0064['other_prop']

# classifier ROC (we make cell=1, other=0 for clarity of the results)
fpr_0064, tpr_0064, thr_0064 = roc_curve(y_true=df_0064['cell_gtruth'],
                                         y_score=df_0064['cell_prop'])
roc_auc_0064 = auc(fpr_0064, tpr_0064)

# find point in the curve for False Positive Rate = 10%
idx_0064 = np.where(fpr_0064 <= 0.1)[0][-1]

## show imbalance between classes
n_cell = np.count_nonzero(df_0064['cell_gtruth'] == 1)
n_other = np.count_nonzero(df_0064['other_gtruth'] == 1)
print('Number of cell objects: ' + str(n_cell) + ' (%0.1f' % (n_cell / (n_cell + n_other) * 100) + '%)')
print('Number of other objects: ' + str(n_other) + ' (%0.1f' % (n_other / (n_cell + n_other) * 100) + '%)')

## plots for both classifiers

if DEBUG:
    # ROC curve before and after data augmentation
    plt.clf()
    plt.plot(fpr_0063, tpr_0063, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_0063)
    plt.plot(fpr_0064, tpr_0064, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_0064)
    plt.scatter(fpr_0063[idx_0063], tpr_0063[idx_0063],
                label='Thr =  %0.3f, FPR = %0.3f, TPR = %0.3f'
                      % (thr_0063[idx_0063], fpr_0063[idx_0063], tpr_0063[idx_0063]),
                color='darkorange')
    plt.scatter(fpr_0064[idx_0064], tpr_0064[idx_0064],
                label='Thr =  %0.3f, FPR = %0.3f, TPR = %0.3f'
                      % (thr_0064[idx_0064], fpr_0064[idx_0064], tpr_0064[idx_0064]),
                color='blue')
    plt.tick_params(labelsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="lower right")


# classifier confusion matrix
cytometer.utils.plot_confusion_matrix(y_true=df_0063['cell_gtruth'],
                                      y_pred=df_0063['cell_prop'] >= thr_0063[idx_0063],
                                      normalize=True,
                                      title='Without data agumentation',
                                      xlabel='"Cell" predicted',
                                      ylabel='"Cell" is ground truth',
                                      cmap=plt.cm.Blues,
                                      colorbar=False)

cytometer.utils.plot_confusion_matrix(y_true=df_0064['cell_gtruth'],
                                      y_pred=df_0064['cell_prop'] >= thr_0064[idx_0064],
                                      normalize=True,
                                      title='With data augmentation',
                                      xlabel='"Cell" predicted',
                                      ylabel='"Cell" is ground truth',
                                      cmap=plt.cm.Blues,
                                      colorbar=False)

## segmentation correction

# indices of "Cell" objects
idx_cell = df_0064['cell_gtruth'] == 1

# linear regression
slope_0064_seg, intercept_0064_seg, \
r_value_0064_seg, p_value_0064_seg, std_err_0064_seg = \
    linregress(df_0064.loc[idx_cell, 'area_gtruth'], df_0064.loc[idx_cell, 'area_seg'])

slope_0064_seg_corrected, intercept_0064_seg_corrected, \
r_value_0064_seg_corrected, p_value_0064_seg_corrected, std_err_0064_seg_corrected = \
    linregress(df_0064.loc[idx_cell, 'area_gtruth'], df_0064.loc[idx_cell, 'area_seg_corrected'])

if DEBUG:
    # linear regression: ground truth area vs. best matching segmentation area
    # No area correction
    plt.clf()
    plt.scatter(df_0064.loc[idx_cell, 'area_gtruth'], df_0064.loc[idx_cell, 'area_seg'], label='')
    plt.plot([0, 20e3], [0, 20e3], color='darkorange', label='Identity')
    plt.plot([0, 20e3],
             [intercept_0064_seg, intercept_0064_seg + 20e3 * slope_0064_seg],
             color='red', label='Linear regression')
    plt.xlabel('Ground truth area ($\mu$m$^2$)', fontsize=14)
    plt.ylabel('Segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()

    # linear regression: ground truth area vs. best matching segmentation area
    # Area correction
    plt.clf()
    plt.scatter(df_0064.loc[idx_cell, 'area_gtruth'], df_0064.loc[idx_cell, 'area_seg_corrected'], label='')
    plt.plot([0, 20e3], [0, 20e3], color='darkorange', label='Identity')
    plt.plot([0, 20e3],
             [intercept_0064_seg_corrected, intercept_0064_seg_corrected + 20e3 * slope_0064_seg_corrected],
             color='red', label='Linear regression')
    plt.xlabel('Ground truth area ($\mu$m$^2$)', fontsize=14)
    plt.ylabel('Segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()

## population area profiles

if DEBUG:

    # hand vs. automatic vs. corrected segmentation
    plt.clf()
    plt.hist(df_0064.loc[idx_cell, 'area_gtruth'], density=True, bins=50, histtype='step',
             label='Hand segmentation')
    plt.hist(df_0064.loc[idx_cell, 'area_seg'], density=True, bins=50, histtype='step',
             label='Automatic segmentation')
    plt.hist(df_0064.loc[idx_cell, 'area_seg_corrected'], density=True, bins=50, histtype='step',
             label='Corrected segmentation')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.legend()

## reload this experiment, including segmentations without corresponding ground truth

# load results
dataframe_filename_0064 = os.path.join(saved_models_dir, 'klf14_b6ntac_exp_0064_pipeline_validation_dataframe.pkl')
df_0064_all = pd.read_pickle(dataframe_filename_0064)

# new column with proportion of cell pixels in segmentation, for convenience
df_0064_all['cell_prop'] = 1 - df_0064_all['other_prop']

# population profiles for automatic segmentations and automatic classification
idx_cell_auto = df_0064_all['cell_prop'] >= thr_0064[idx_0064]

# median values
print('Ground truth median area: %0.0f ' % np.median(df_0064.loc[idx_cell, 'area_gtruth']) + r'(um^2)')
print('Pipeline median area: %0.0f ' % np.median(df_0064_all.loc[idx_cell_auto, 'area_seg_corrected']) + r'(um^2)')

if DEBUG:

    # histograms: hand vs. full test windows
    plt.clf()
    plt.hist(df_0064.loc[idx_cell, 'area_gtruth'], density=True, bins=50, histtype='step',
             label='Hand segmentations')
    plt.hist(df_0064_all.loc[idx_cell_auto, 'area_seg_corrected'], density=True, bins=50, histtype='step',
             label='Classifier segmentations')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.legend()

    # boxplots: hand vs. full test windows
    plt.clf()
    plt.boxplot((df_0064.loc[idx_cell, 'area_gtruth'],
                 df_0064_all.loc[idx_cell_auto, 'area_seg_corrected']), notch=True,
                labels=('Hand segmentations', 'Classifier segmentations'))
    plt.ylabel('Segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(-840, 6800)
