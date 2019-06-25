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
import re

# other imports
from enum import IntEnum
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import linregress, mode
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
min_cell_area = 75
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
histology_dir = os.path.join(root_data_dir, 'Maz Yon')
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

'''
************************************************************************************************************************
CLASSIFIER SANITY CHECK:

  Apply classifier trained with each 10 folds to the other fold testing objects. 
  
  Note that here the "other" and "BAT" objects are going to be quite different from the carefully and more or less
  consistent WAT cell.
************************************************************************************************************************
'''

## Create dataframe with validation measures

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

    # # load quality model
    # quality_model_filename = os.path.join(saved_models_dir, quality_model_basename + '_fold_' + str(i_fold) + '.h5')
    # quality_model = keras.models.load_model(quality_model_filename)

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
            plt.imshow(im)
            plt.axis('off')
            plt.title('Histology', fontsize=14)

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
        contour_type_all = ['wat', ] * len(cell_contours) \
                           + ['other', ] * len(other_contours) \
                           + ['bat', ] * len(brown_contours)

        print('Cells: ' + str(len(cell_contours)))
        print('Other: ' + str(len(other_contours)))
        print('Brown: ' + str(len(brown_contours)))
        print('')

        '''Segmentation part of the pipeline. We segment whole test windows
        '''

        # initialise dataframe to keep results: one cell per row, tagged with mouse metainformation
        df_im = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_tif),
                                                          values=[i_fold], values_tag='fold',
                                                          tags_to_keep=['id', 'ko', 'sex'])

        # loop contours
        for j, contour in enumerate(contours):

            # start dataframe row for this contour
            df = df_im.copy()
            df['im'] = i
            df['contour'] = j

            if DEBUG:
                # centre of current cell
                xy_c = (np.mean([p[0] for p in contour]), np.mean([p[1] for p in contour]))
                plt.text(xy_c[0], xy_c[1], 'j=' + str(j))
                plt.scatter(xy_c[0], xy_c[1])

                # close the contour for the plot
                contour_aux = contour.copy()
                contour_aux.append(contour[0])

                plt.clf()
                plt.imshow(im)
                plt.plot([p[0] for p in contour_aux], [p[1] for p in contour_aux], color='green')
                plt.axis('off')

            # rasterise object described by contour
            cell_seg_contour = Image.new("1", im.size, "black")  # I = 32-bit signed integer pixels
            draw = ImageDraw.Draw(cell_seg_contour)
            draw.polygon(contour, outline="white", fill="white")
            cell_seg_contour = np.array(cell_seg_contour, dtype=np.uint8)

            '''Bounding boxes'''

            # compute bounding box for the contour
            bbox_seg_contour_x0, bbox_seg_contour_y0, bbox_seg_contour_xend, bbox_seg_contour_yend = \
                cytometer.utils.bounding_box_with_margin(cell_seg_contour, coordinates='xy', inc=1.00)
            bbox_seg_contour_r0, bbox_seg_contour_c0, bbox_seg_contour_rend, bbox_seg_contour_cend = \
                cytometer.utils.bounding_box_with_margin(cell_seg_contour, coordinates='rc', inc=1.00)

            # this renaming here is redundant, but it'll be useful in the validation of automatic segmentation
            bbox_total_x0 = bbox_seg_contour_x0
            bbox_total_y0 = bbox_seg_contour_y0
            bbox_total_xend = bbox_seg_contour_xend
            bbox_total_yend = bbox_seg_contour_yend

            bbox_total_r0 = bbox_seg_contour_r0
            bbox_total_c0 = bbox_seg_contour_c0
            bbox_total_rend = bbox_seg_contour_rend
            bbox_total_cend = bbox_seg_contour_cend
            bbox_total = (bbox_total_r0, bbox_total_c0, bbox_total_rend, bbox_total_cend)

            # add to dataframe: segmentation areas and Dice coefficient
            df['area_contour'] = np.count_nonzero(cell_seg_contour) * xres * yres  # um^2

            if DEBUG:
                plt.clf()
                plt.imshow(im)
                plt.contour(cell_seg_contour, linewidths=1, colors='green')

                plt.plot((bbox_total_x0, bbox_total_xend, bbox_total_xend, bbox_total_x0, bbox_total_x0),
                         (bbox_total_y0, bbox_total_y0, bbox_total_yend, bbox_total_yend, bbox_total_y0),
                         color='green')

                plt.xlim(bbox_total_x0 - (bbox_total_xend - bbox_total_x0) * 0.1,
                         bbox_total_xend + (bbox_total_xend - bbox_total_x0) * 0.1)
                plt.ylim(bbox_total_yend + (bbox_total_yend - bbox_total_y0) * 0.1,
                         bbox_total_y0 - (bbox_total_yend - bbox_total_y0) * 0.1)
                plt.axis('off')

            # crop image and masks according to bounding box of automatic segmentation
            window_im = cytometer.utils.extract_bbox(np.array(im), bbox_total)
            window_seg_contour = cytometer.utils.extract_bbox(cell_seg_contour, bbox_total)

            if DEBUG:
                plt.clf()
                plt.imshow(window_im)
                plt.contour(window_seg_contour, linewidths=1, colors='green')
                plt.axis('off')

            '''Cropping and resizing of individual contour'''

            # scaling factors for the training image
            training_size = (training_window_len, training_window_len)
            scaling_factor = np.array(training_size) / np.array(window_seg_contour.shape[0:2])
            window_pixel_size = np.array([xres, yres]) / scaling_factor  # (um, um)

            # resize the images to training window size
            window_im = cytometer.utils.resize(window_im, size=training_size, resample=Image.LINEAR)
            window_seg_contour = cytometer.utils.resize(window_seg_contour, size=training_size, resample=Image.NEAREST)

            # add dummy dimensions for keras
            window_im = np.expand_dims(window_im, axis=0)

            # correct types
            window_im = window_im.astype(np.float32) / 255.0
            window_seg_contour = window_seg_contour.astype(np.float32)

            # check sizes and types
            assert(window_im.ndim == 4 and window_im.dtype == np.float32
                   and np.min(window_im) >= 0 and np.max(window_im) <= 1.0)
            assert(window_seg_contour.ndim == 2 and window_seg_contour.dtype == np.float32)

            '''Object classification as "white adipocyte" or "other"'''

            # process histology for classification
            window_classifier_out = classifier_model.predict(window_im, batch_size=batch_size)

            # get classification label for each pixel
            window_classifier_class = np.argmax(window_classifier_out, axis=3)

            # proportion of "Other" pixels in the mask
            window_other_prop = np.count_nonzero(window_seg_contour * window_classifier_class) \
                                / np.count_nonzero(window_seg_contour)

            # add to dataframe row
            df['contour_type'] = contour_type_all[j]
            df['contour_type_prop'] = window_other_prop

            if DEBUG:
                # plot classification
                plt.clf()
                plt.imshow(window_classifier_class[0, :, :])
                plt.contour(window_seg_contour, linewidths=1, colors='green')
                plt.title('"Other" prop = ' + str("{:.0f}".format(window_other_prop * 100)) + '%', fontsize=14)
                plt.axis('off')

            # append current results to global dataframe
            df_all = pd.concat([df_all, df])

        print('Time so far: ' + str(time.time() - time0) + ' s')

# reset indices
df_all.reset_index(drop=True, inplace=True)

# save results
dataframe_filename = os.path.join(saved_models_dir, experiment_id + '_dataframe_classifier.pkl')
df_all.to_pickle(dataframe_filename)

## Results analysis

# load results
dataframe_filename_0072 = os.path.join(saved_models_dir, experiment_id + '_dataframe_classifier.pkl')
df_0072 = pd.read_pickle(dataframe_filename_0072)

## show imbalance between classes
n_tot = len(df_0072['contour_type'])
n_wat = np.count_nonzero(df_0072['contour_type'] == 'wat')
n_other = np.count_nonzero(df_0072['contour_type'] == 'other')
n_bat = np.count_nonzero(df_0072['contour_type'] == 'bat')
n_non_wat = n_tot - n_wat
print('Number of WAT cells: ' + str(n_wat) + ' (%0.1f' % (n_wat / n_tot * 100) + '%)')
print('Number of Other objects: ' + str(n_wat) + ' (%0.1f' % (n_other / n_tot * 100) + '%)')
print('Number of BAT objects: ' + str(n_bat) + ' (%0.1f' % (n_bat / n_tot * 100) + '%)')
print('Number of non-WAT objects: ' + str(n_non_wat) + ' (%0.1f' % (n_non_wat / n_tot * 100) + '%)')

## ROC

# classifier ROC (we make cell=1, other/brown=0 for clarity of the results)
fpr_0072, tpr_0072, thr_0072 = roc_curve(y_true=df_0072['contour_type'] == 'wat',
                                         y_score=1 - df_0072['contour_type_prop'])
roc_auc_0072 = auc(fpr_0072, tpr_0072)

# find point in the curve for False Positive Rate close to 10%
idx_0072 = np.argmin(np.abs(fpr_0072 - 0.1))

# plots for both classifiers

if DEBUG:
    # ROC curve before and after data augmentation
    plt.clf()
    plt.plot(fpr_0072, tpr_0072, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_0072)
    plt.scatter(fpr_0072[idx_0072], tpr_0072[idx_0072],
                label='Thr =  %0.3f, FPR = %0.3f, TPR = %0.3f'
                      % (thr_0072[idx_0072], fpr_0072[idx_0072], tpr_0072[idx_0072]),
                color='r')
    plt.tick_params(labelsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="lower right")


# classifier confusion matrix
cytometer.utils.plot_confusion_matrix(y_true=df_0072['contour_type'] == 'wat',
                                      y_pred=1 - df_0072['contour_type_prop'] >= thr_0072[idx_0072],
                                      normalize=True,
                                      title='With data augmentation',
                                      xlabel='"WAT" predicted',
                                      ylabel='"WAT" is ground truth',
                                      cmap=plt.cm.Blues,
                                      colorbar=False)

## Boxplots

if DEBUG:
    plt.clf()
    idx_wat = df_0072['contour_type'] == 'wat'
    plt.boxplot([1 - df_0072['contour_type_prop'][np.logical_not(idx_wat)],
                1 - df_0072['contour_type_prop'][idx_wat]], labels=['Not WAT', 'WAT'])
    plt.plot([0.75, 2.25], [thr_0072[idx_0072], thr_0072[idx_0072]], 'r', linewidth=2)
    plt.tick_params(axis='both', labelsize=14)
    plt.ylabel('WAT pixels / Segmentation pixels', fontsize=14)
    plt.tight_layout()

'''
************************************************************************************************************************
CLASSIFIER VALIDATION BASED ON AUTOMATIC SEGMENTATIONS:

  Apply segmentation trained with each 10 folds to the images in the other fold. This produces a lot of automatic
  segmentations.
  
  We then have to label each automatic segmentation as being a "WAT" / "Other" / "BAT" according to hand traced
  contours.
  
  We then apply the classifier to each automatic segmentation.

  Note that here the "other" and "BAT" objects are going to be quite different from the carefully and more or less
  consistent WAT cell.
************************************************************************************************************************
'''

# types of pixels
class PixelType(IntEnum):
    UNDETERMINED = 0
    WAT = 1
    NON_WAT = 2

## Create dataframe with validation measures

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

    # load classifier model
    classifier_model_filename = os.path.join(saved_models_dir,
                                             classifier_model_basename + '_fold_' + str(i_fold) + '.h5')
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
            plt.imshow(im)
            plt.axis('off')
            plt.title('Histology', fontsize=14)

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
        contour_type_all = ['wat', ] * len(cell_contours) \
                           + ['other', ] * len(other_contours) \
                           + ['bat', ] * len(brown_contours)

        print('Cells: ' + str(len(cell_contours)))
        print('Other: ' + str(len(other_contours)))
        print('Brown: ' + str(len(brown_contours)))
        print('')

        '''Label pixels of image as either WAT/non-WAT'''

        # initialise arrays to keep track of which pixels are WAT/Other/BAT
        pixel_type_wat = np.zeros(shape=im.size, dtype=np.uint16)
        pixel_type_non_wat = np.zeros(shape=im.size, dtype=np.uint16)

        # loop contours
        for j, contour in enumerate(contours):

            if DEBUG:
                # centre of current cell
                xy_c = (np.mean([p[0] for p in contour]), np.mean([p[1] for p in contour]))
                plt.text(xy_c[0], xy_c[1], 'j=' + str(j))
                plt.scatter(xy_c[0], xy_c[1])

                # close the contour for the plot
                contour_aux = contour.copy()
                contour_aux.append(contour[0])

                plt.clf()
                plt.imshow(im)
                plt.plot([p[0] for p in contour_aux], [p[1] for p in contour_aux], color='green')
                plt.axis('off')

            # rasterise object described by contour
            cell_seg_contour = Image.new("1", im.size, "black")  # I = 32-bit signed integer pixels
            draw = ImageDraw.Draw(cell_seg_contour)
            draw.polygon(contour, outline="white", fill="white")
            cell_seg_contour = np.array(cell_seg_contour, dtype=np.uint8)

            # add the object to the pixel label arrays
            if contour_type_all[j] == 'wat':
                pixel_type_wat += cell_seg_contour
            else:
                pixel_type_non_wat += cell_seg_contour

        # each pixel now can be labelled as WAT/non-WAT/undetermined
        pixel_type = np.full(shape=pixel_type_wat.shape, fill_value=PixelType.UNDETERMINED, dtype=PixelType)
        pixel_type[pixel_type_wat > pixel_type_non_wat] = PixelType.WAT
        pixel_type[pixel_type_wat < pixel_type_non_wat] = PixelType.NON_WAT
        del pixel_type_wat
        del pixel_type_non_wat

        if DEBUG:
            plt.clf()
            plt.imshow(pixel_type.astype(np.uint8))

        '''Automatic segmentation of image'''

        # filenames of contour and dmap models
        contour_model_filename = os.path.join(saved_models_dir, contour_model_basename + '_fold_' + str(i_fold) + '.h5')
        dmap_model_filename = os.path.join(saved_models_dir, dmap_model_basename + '_fold_' + str(i_fold) + '.h5')

        # segment histology
        labels, _ = cytometer.utils.segment_dmap_contour_v3(im, contour_model=contour_model_filename,
                                                            dmap_model=dmap_model_filename)

        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(im)
            plt.axis('off')
            plt.title('Histology', fontsize=14)

            plt.subplot(222)
            plt.cla()
            plt.imshow(labels)
            plt.axis('off')
            plt.title('Segmentation', fontsize=14)

            plt.subplot(223)
            plt.cla()
            plt.imshow(im)
            plt.contour(labels, levels=np.unique(labels), colors='k')
            plt.axis('off')
            plt.title('Segmentation on histology', fontsize=14)

        # remove labels that touch the edges or that are too small
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
            raise ValueError('No labels produced!')

        # initialise dataframe to keep results: one cell per row, tagged with mouse metainformation
        df_im = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_tif),
                                                          values=[i_fold], values_tag='fold',
                                                          tags_to_keep=['id', 'ko', 'sex'])

        # loop automatic segmentations
        for j, lab in enumerate(labels_seg):

            # start dataframe row for this contour
            df = df_im.copy()
            df['im'] = i
            df['lab'] = lab

            # boolean indices of current segmentation
            cell_seg = labels == lab

            # segmentation label is the most common pixel label
            seg_type = mode(pixel_type[cell_seg])[0][0]

            # add to dataframe if WAT/non-WAT, or skip if undetermined
            if seg_type == PixelType.WAT:
                df['seg_type'] = 'wat'
            elif seg_type == PixelType.NON_WAT:
                df['seg_type'] = 'non_wat'
            elif seg_type == PixelType.UNDETERMINED:
                continue
            else:
                raise ValueError('Unrecognised segmentation type')

            '''Bounding boxes'''

            # compute bounding box for the automatic segmentation
            bbox_seg_x0, bbox_seg_y0, bbox_seg_xend, bbox_seg_yend = \
                cytometer.utils.bounding_box_with_margin(cell_seg, coordinates='xy', inc=1.00)
            bbox_seg_r0, bbox_seg_c0, bbox_seg_rend, bbox_seg_cend = \
                cytometer.utils.bounding_box_with_margin(cell_seg, coordinates='rc', inc=1.00)

            # this renaming here is redundant, but it'll be useful in later validation
            bbox_total_x0 = bbox_seg_x0
            bbox_total_y0 = bbox_seg_y0
            bbox_total_xend = bbox_seg_xend
            bbox_total_yend = bbox_seg_yend

            bbox_total_r0 = bbox_seg_r0
            bbox_total_c0 = bbox_seg_c0
            bbox_total_rend = bbox_seg_rend
            bbox_total_cend = bbox_seg_cend
            bbox_total = (bbox_total_r0, bbox_total_c0, bbox_total_rend, bbox_total_cend)

            if DEBUG:
                plt.clf()
                plt.imshow(im)
                plt.contour(cell_seg, linewidths=1, colors='green')

                plt.plot((bbox_total_x0, bbox_total_xend, bbox_total_xend, bbox_total_x0, bbox_total_x0),
                         (bbox_total_y0, bbox_total_y0, bbox_total_yend, bbox_total_yend, bbox_total_y0),
                         color='green')

                plt.xlim(bbox_total_x0 - (bbox_total_xend - bbox_total_x0) * 0.1,
                         bbox_total_xend + (bbox_total_xend - bbox_total_x0) * 0.1)
                plt.ylim(bbox_total_yend + (bbox_total_yend - bbox_total_y0) * 0.1,
                         bbox_total_y0 - (bbox_total_yend - bbox_total_y0) * 0.1)
                plt.axis('off')

            '''Cropping and resizing of individual contour'''

            # crop image and masks according to bounding box of automatic segmentation
            window_im = cytometer.utils.extract_bbox(np.array(im), bbox_total)
            window_seg = cytometer.utils.extract_bbox(cell_seg.astype(np.uint8), bbox_total)

            if DEBUG:
                plt.clf()
                plt.imshow(window_im)
                plt.contour(window_seg, linewidths=1, colors='green')
                plt.axis('off')

            # scaling factors for the training image
            training_size = (training_window_len, training_window_len)
            scaling_factor = np.array(training_size) / np.array(window_seg.shape[0:2])
            window_pixel_size = np.array([xres, yres]) / scaling_factor  # (um, um)

            # resize the images to training window size
            window_im = cytometer.utils.resize(window_im, size=training_size, resample=Image.LINEAR)
            window_seg = cytometer.utils.resize(window_seg, size=training_size, resample=Image.NEAREST)

            # add dummy dimensions for keras
            window_im = np.expand_dims(window_im, axis=0)

            # correct types
            window_im = window_im.astype(np.float32) / 255.0
            # window_seg = window_seg.astype(np.float32)

            # check sizes and types
            assert (window_im.ndim == 4 and window_im.dtype == np.float32
                    and np.min(window_im) >= 0 and np.max(window_im) <= 1.0)
            # assert (window_seg.ndim == 2 and window_seg.dtype == np.float32)

            '''Object classification as "WAT" or "other"'''

            # process histology for classification
            window_classifier_out = classifier_model.predict(window_im, batch_size=batch_size)

            # get classification label for each pixel
            window_classifier_class = np.argmax(window_classifier_out, axis=3)

            # proportion of "Other" pixels in the mask
            window_other_prop = np.count_nonzero(window_seg * window_classifier_class) \
                                / np.count_nonzero(window_seg)

            # add to dataframe row
            df['seg_type_prop'] = window_other_prop

            if DEBUG:
                # plot classification
                plt.clf()
                plt.imshow(window_classifier_class[0, :, :])
                plt.contour(window_seg, linewidths=1, colors='green')
                plt.title('"Other" prop = ' + str("{:.0f}".format(window_other_prop * 100)) + '%', fontsize=14)
                plt.axis('off')

            # append current results to global dataframe
            df_all = pd.concat([df_all, df])

        print('Time so far: ' + str(time.time() - time0) + ' s')

# reset indices
df_all.reset_index(drop=True, inplace=True)

# save results
dataframe_filename = os.path.join(saved_models_dir, experiment_id + '_dataframe_classifier_automatic.pkl')
df_all.to_pickle(dataframe_filename)

## Results analysis

# load results
dataframe_filename_0072 = os.path.join(saved_models_dir, experiment_id + '_dataframe_classifier_automatic.pkl')
df_0072 = pd.read_pickle(dataframe_filename_0072)

## show imbalance between classes
n_tot = len(df_0072['seg_type'])
n_wat = np.count_nonzero(df_0072['seg_type'] == 'wat')
n_non_wat = np.count_nonzero(df_0072['seg_type'] == 'non_wat')

print('Number of WAT cells: ' + str(n_wat) + ' (%0.1f' % (n_wat / n_tot * 100) + '%)')
print('Number of non-WAT objects: ' + str(n_non_wat) + ' (%0.1f' % (n_non_wat / n_tot * 100) + '%)')

## ROC

# classifier ROC (we make cell=1, other/brown=0 for clarity of the results)
fpr_0072, tpr_0072, thr_0072 = roc_curve(y_true=df_0072['seg_type'] == 'wat',
                                         y_score=1 - df_0072['seg_type_prop'])
roc_auc_0072 = auc(fpr_0072, tpr_0072)

# find point in the curve for False Positive Rate close to 10%
idx_0072 = np.argmin(np.abs(fpr_0072 - 0.1))

if DEBUG:
    # ROC curve before and after data augmentation
    plt.clf()
    plt.plot(fpr_0072, tpr_0072, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_0072)
    plt.scatter(fpr_0072[idx_0072], tpr_0072[idx_0072],
                label='Thr =  %0.3f, FPR = %0.3f, TPR = %0.3f'
                      % (thr_0072[idx_0072], fpr_0072[idx_0072], tpr_0072[idx_0072]),
                color='r')
    plt.tick_params(labelsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="lower right")
    plt.tight_layout()


# classifier confusion matrix
cytometer.utils.plot_confusion_matrix(y_true=df_0072['seg_type'] == 'wat',
                                      y_pred=1 - df_0072['seg_type_prop'] >= thr_0072[idx_0072],
                                      normalize=True,
                                      title='With data augmentation',
                                      xlabel='"WAT" predicted',
                                      ylabel='"WAT" is ground truth',
                                      cmap=plt.cm.Blues,
                                      colorbar=False)

## Boxplots

if DEBUG:
    plt.clf()
    idx_wat = df_0072['seg_type'] == 'wat'
    plt.boxplot([1 - df_0072['seg_type_prop'][np.logical_not(idx_wat)],
                1 - df_0072['seg_type_prop'][idx_wat]], labels=['Not WAT', 'WAT'])
    plt.plot([0.75, 2.25], [thr_0072[idx_0072], thr_0072[idx_0072]], 'r', linewidth=2)
    plt.tick_params(axis='both', labelsize=14)
    plt.ylabel('WAT pixels / Segmentation pixels', fontsize=14)
    plt.tight_layout()


'''
************************************************************************************************************************
AUTOMATIC SEGMENTATION VALIDATION BASED ON MANUAL CONTOURS:

  Apply segmentation trained with each 10 folds to the images in the other fold. This produces a lot of automatic
  segmentations.
  
  For each cell contour, find best automatic segmentation overlap.
  
  Compute Dice coefficient, area_contour, area_seg.
  
  Then, apply segmentation correction and recompute areas.

************************************************************************************************************************
'''

# Create dataframe with validation measures

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
            plt.imshow(im)
            plt.axis('off')
            plt.title('Histology', fontsize=14)

        # read the ground truth cell contours in the SVG file. This produces a list [contour_0, ..., contour_N-1]
        # where each contour_i = [(X_0, Y_0), ..., (X_P-1, X_P-1)]
        cell_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False,
                                                                minimum_npoints=3)
        contours = cell_contours

        print('Cells: ' + str(len(cell_contours)))
        print('')

        '''Automatic segmentation of image'''

        # filenames of contour and dmap models
        contour_model_filename = os.path.join(saved_models_dir, contour_model_basename + '_fold_' + str(i_fold) + '.h5')
        dmap_model_filename = os.path.join(saved_models_dir, dmap_model_basename + '_fold_' + str(i_fold) + '.h5')

        # segment histology
        labels, _ = cytometer.utils.segment_dmap_contour_v3(im, contour_model=contour_model_filename,
                                                            dmap_model=dmap_model_filename,
                                                            local_threshold_block_size=41, border_dilation=0)

        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(im)
            plt.axis('off')
            plt.title('Histology', fontsize=14)

            plt.subplot(222)
            plt.cla()
            plt.imshow(labels)
            plt.axis('off')
            plt.title('Segmentation', fontsize=14)

            plt.subplot(223)
            plt.cla()
            plt.imshow(im)
            plt.contour(labels, levels=np.unique(labels), colors='k')
            plt.axis('off')
            plt.title('Segmentation on histology', fontsize=14)

        # remove labels that touch the edges or that are too small
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
            raise ValueError('No labels produced!')

        # initialise dataframe to keep results: one cell per row, tagged with mouse metainformation
        df_im = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_tif),
                                                          values=[i_fold], values_tag='fold',
                                                          tags_to_keep=['id', 'ko', 'sex'])

        '''Find the automatic segmentation that best overlaps with each cell contour'''

        # loop contours
        for j, contour in enumerate(contours):

            # start dataframe row for this contour
            df = df_im.copy()
            df['im'] = i
            df['contour'] = j

            if DEBUG:
                # centre of current cell
                xy_c = (np.mean([p[0] for p in contour]), np.mean([p[1] for p in contour]))
                plt.text(xy_c[0], xy_c[1], 'j=' + str(j))
                plt.scatter(xy_c[0], xy_c[1])

                # close the contour for the plot
                contour_aux = contour.copy()
                contour_aux.append(contour[0])

                plt.clf()
                plt.imshow(im)
                plt.plot([p[0] for p in contour_aux], [p[1] for p in contour_aux], color='green')
                plt.axis('off')

            # rasterise object described by contour
            cell_seg_contour = Image.new("1", im.size, "black")  # I = 32-bit signed integer pixels
            draw = ImageDraw.Draw(cell_seg_contour)
            draw.polygon(contour, outline="white", fill="white")
            cell_seg_contour = np.array(cell_seg_contour, dtype=np.uint8)

            # find automatic segmentation that best overlaps contour
            lab = mode(labels[cell_seg_contour > 0])[0][0]

            if lab == 0:
                warnings.warn('Skipping. Contour j = ' + str(j) + ' overlaps with background segmentation lab = 0')
                continue

            # isolate that best automatic segmentation
            cell_seg = labels == lab

            # compute Dice coefficient
            contour_and_lab = np.logical_and(cell_seg, cell_seg_contour)
            dice = 2 * np.count_nonzero(contour_and_lab) \
                   / (np.count_nonzero(cell_seg) + np.count_nonzero(cell_seg_contour))

            # add to dataframe
            df['label_seg'] = lab
            df['dice'] = dice
            df['area_contour'] = np.count_nonzero(cell_seg_contour) * xres * yres  # um^2
            df['area_seg'] = np.count_nonzero(cell_seg) * xres * yres  # um^2

            '''Bounding boxes'''

            # compute bounding box for the automatic segmentation
            bbox_seg_x0, bbox_seg_y0, bbox_seg_xend, bbox_seg_yend = \
                cytometer.utils.bounding_box_with_margin(cell_seg, coordinates='xy', inc=1.00)
            bbox_seg_r0, bbox_seg_c0, bbox_seg_rend, bbox_seg_cend = \
                cytometer.utils.bounding_box_with_margin(cell_seg, coordinates='rc', inc=1.00)

            # this renaming here is redundant, but it'll be useful in later validation
            bbox_total_x0 = bbox_seg_x0
            bbox_total_y0 = bbox_seg_y0
            bbox_total_xend = bbox_seg_xend
            bbox_total_yend = bbox_seg_yend

            bbox_total_r0 = bbox_seg_r0
            bbox_total_c0 = bbox_seg_c0
            bbox_total_rend = bbox_seg_rend
            bbox_total_cend = bbox_seg_cend
            bbox_total = (bbox_total_r0, bbox_total_c0, bbox_total_rend, bbox_total_cend)

            if DEBUG:
                plt.clf()
                plt.imshow(im)
                plt.contour(cell_seg_contour, linewidths=1, colors='green')
                plt.contour(cell_seg, linewidths=1, colors='red')

                plt.plot((bbox_total_x0, bbox_total_xend, bbox_total_xend, bbox_total_x0, bbox_total_x0),
                         (bbox_total_y0, bbox_total_y0, bbox_total_yend, bbox_total_yend, bbox_total_y0),
                         color='green')
                plt.axis('off')

                plt.xlim(bbox_total_x0 - (bbox_total_xend - bbox_total_x0) * 0.1,
                         bbox_total_xend + (bbox_total_xend - bbox_total_x0) * 0.1)
                plt.ylim(bbox_total_yend + (bbox_total_yend - bbox_total_y0) * 0.1,
                         bbox_total_y0 - (bbox_total_yend - bbox_total_y0) * 0.1)

            '''Cropping and resizing of individual contour'''

            # crop image and masks according to bounding box of automatic segmentation
            window_im = cytometer.utils.extract_bbox(np.array(im), bbox_total)
            window_seg = cytometer.utils.extract_bbox(cell_seg.astype(np.uint8), bbox_total)
            window_seg_contour = cytometer.utils.extract_bbox(cell_seg_contour.astype(np.uint8), bbox_total)

            if DEBUG:
                plt.clf()
                plt.imshow(window_im)
                plt.contour(window_seg_contour, linewidths=1, colors='green')
                plt.contour(window_seg, linewidths=1, colors='red')
                plt.axis('off')

            # input to segmentation quality CNN: multiply histology by +1/-1 segmentation mask
            # NOTE: quality network 0053 expects values in [-255, 255], float32
            window_masked_im = \
                cytometer.utils.quality_model_mask(window_seg.astype(np.float32), im=window_im.astype(np.float32),
                                                   quality_model_type='-1_1')[0, :, :, :]

            if DEBUG:
                plt.clf()
                plt.imshow(window_im)
                plt.contour(np.any(window_masked_im > 0, axis=2), linewidths=1, colors='k')
                plt.imshow(np.any(window_masked_im > 0, axis=2), alpha=0.5)
                plt.axis('off')

            # scaling factors for the training image
            training_size = (training_window_len, training_window_len)
            scaling_factor = np.array(training_size) / np.array(window_seg.shape[0:2])
            window_pixel_size = np.array([xres, yres]) / scaling_factor  # (um, um)

            # resize the images to training window size
            window_im = cytometer.utils.resize(window_im, size=training_size, resample=Image.LINEAR)
            window_masked_im = cytometer.utils.resize(window_masked_im, size=training_size, resample=Image.LINEAR)
            window_seg = cytometer.utils.resize(window_seg, size=training_size, resample=Image.NEAREST)
            window_seg_contour = cytometer.utils.resize(window_seg_contour, size=training_size, resample=Image.NEAREST)

            # add dummy dimensions for keras
            window_im = np.expand_dims(window_im, axis=0)
            window_masked_im = np.expand_dims(window_masked_im, axis=0)
            window_seg = np.expand_dims(window_seg, axis=0)

            # correct types
            window_im = window_im.astype(np.float32) / 255.0

            # check sizes and types
            assert (window_im.ndim == 4 and window_im.dtype == np.float32
                    and np.min(window_im) >= 0 and np.max(window_im) <= 1.0)
            assert (window_masked_im.ndim == 4 and window_masked_im.dtype == np.float32
                    and np.min(window_masked_im) >= -255 and np.max(window_masked_im) <= 255)

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

            # compute Dice coefficient
            contour_and_lab_corrected = np.logical_and(window_seg_corrected, window_seg_contour)
            dice_corrected = 2 * np.count_nonzero(contour_and_lab_corrected) \
                             / (np.count_nonzero(window_seg_corrected) + np.count_nonzero(window_seg_contour))

            # add to dataframe
            df['dice_corrected'] = dice_corrected
            df['area_seg_corrected'] = area_seg_corrected

            if DEBUG:
                # plot segmentation correction
                plt.clf()
                plt.imshow(window_im[0, :, :, :])
                plt.contour(window_seg_contour, linewidths=1, colors='green')
                plt.contour(window_seg[0, :, :], linewidths=1, colors='red')
                plt.contour(window_seg_corrected, linewidths=1, colors='black')
                plt.axis('off')

            # append current results to global dataframe
            df_all = pd.concat([df_all, df])

        print('Time so far: ' + str(time.time() - time0) + ' s')

# reset indices
df_all.reset_index(drop=True, inplace=True)

# save results
dataframe_filename = os.path.join(saved_models_dir, experiment_id + '_dataframe_segmentation_automatic.pkl')
df_all.to_pickle(dataframe_filename)

## Results analysis

# load results
dataframe_filename_0072 = os.path.join(saved_models_dir, experiment_id + '_dataframe_segmentation_automatic.pkl')
df_0072 = pd.read_pickle(dataframe_filename_0072)

# remove contours that got matched to the background
df_0072 = df_0072.loc[df_0072['label_seg'] != 0, :]

# remove segmentations that are larger than 20,000 um^2, because the largest cell in the manual dataset is ~19,000 um^2
df_0072 = df_0072.loc[df_0072['area_seg'] <= 20e3, :]

# reset indices
df_0072.reset_index(drop=True, inplace=True)

# number of cells per fold
for i_fold in range(n_folds):
    print('fold = ' + str(i_fold) + ', ' + str(np.count_nonzero(df_0072['fold'] == i_fold)))

## Dice histograms

if DEBUG:
    plt.clf()
    plt.hist(df_0072['dice'], bins=100)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Dice coeff.', fontsize=14)
    plt.ylabel('Cell counts', fontsize=14)

print('Segmentations with Dice < 0.5: ' + str(np.count_nonzero(df_0072['dice'] < 0.5) / df_0072.shape[0]))

## Dice vs. area plots

if DEBUG:
    plt.clf()
    plt.scatter(df_0072['area_contour'], df_0072['dice'])

## Population cell area profiles

if DEBUG:
    plt.clf()
    plt.boxplot([df_0072['area_contour'], df_0072['area_seg'], df_0072['area_seg_corrected']], notch=True,
                labels=['Manual', 'Automatic', 'Corrected'])

## Area scatter plots

# linear regression
slope_0072_seg, intercept_0072_seg, \
r_value_0072_seg, p_value_0072_seg, std_err_0072_seg = \
    linregress(df_0072['area_contour'], df_0072['area_seg'])

slope_0072_seg_corrected, intercept_0072_seg_corrected, \
r_value_0072_seg_corrected, p_value_0072_seg_corrected, std_err_0072_seg_corrected = \
    linregress(df_0072['area_contour'], df_0072['area_seg_corrected'])

if DEBUG:
    plt.clf()
    # plt.subplot(211)
    plt.scatter(df_0072['area_contour'], df_0072['area_seg'], c=df_0072['dice'], s=1, cmap='RdBu')
    plt.plot([0, 20e3], [0, 20e3], color='darkorange', label='Identity')
    plt.plot([0, 20e3],
             [intercept_0072_seg, intercept_0072_seg + 20e3 * slope_0072_seg],
             color='C0', label='Linear regression')
    plt.xlabel('Manual segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.ylabel('Auto segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()
    cbar = plt.colorbar()
    cbar.ax.set_title('Dice coeff.')
    plt.tight_layout()

    # plt.subplot(212)
    plt.clf()
    plt.scatter(df_0072['area_contour'], df_0072['area_seg_corrected'], c=df_0072['dice_corrected'], s=1, cmap='RdBu')
    plt.plot([0, 20e3], [0, 20e3], color='darkorange', label='Identity')
    plt.plot([0, 20e3],
             [intercept_0072_seg_corrected, intercept_0072_seg_corrected + 20e3 * slope_0072_seg_corrected],
             color='C0', label='Linear regression')
    plt.xlabel('Manual segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.ylabel('Auto segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()
    cbar = plt.colorbar()
    cbar.ax.set_title('Dice coeff.')
    plt.tight_layout()

## Remove segmentations with Dice < 0.5

# remove segmentations that are larger than 20,000 um^2, because the largest cell in the manual dataset is ~19,000 um^2
df_0072 = df_0072.loc[df_0072['dice'] >= 0.5, :]

# reset indices
df_0072.reset_index(drop=True, inplace=True)

# number of cells per fold
for i_fold in range(n_folds):
    print('fold = ' + str(i_fold) + ', ' + str(np.count_nonzero(df_0072['fold'] == i_fold)))

## Area error boxplots

if DEBUG:
    plt.clf()
    plt.boxplot([df_0072['area_seg'] / df_0072['area_contour'],
                 df_0072['area_seg_corrected'] / df_0072['area_contour']], labels=['No correction', 'Corrected'],
                notch=True)
    plt.plot([0.75, 2.25], [1.0, 1.0], color='red')
    plt.tick_params(axis='both', labelsize=14)
    plt.ylabel('Auto segmentation area / Manual segmentation area', fontsize=14)
    plt.ylim(0, 4)
    plt.tight_layout()

## Area scatter plots

# linear regression
slope_0072_seg, intercept_0072_seg, \
r_value_0072_seg, p_value_0072_seg, std_err_0072_seg = \
    linregress(df_0072['area_contour'], df_0072['area_seg'])

slope_0072_seg_corrected, intercept_0072_seg_corrected, \
r_value_0072_seg_corrected, p_value_0072_seg_corrected, std_err_0072_seg_corrected = \
    linregress(df_0072['area_contour'], df_0072['area_seg_corrected'])

if DEBUG:
    plt.clf()
    # plt.subplot(211)
    plt.scatter(df_0072['area_contour'], df_0072['area_seg'], c=df_0072['dice'], s=2, cmap='RdBu',
                vmin=0.5, vmax=1.0)
    plt.plot([0, 20e3], [0, 20e3], color='darkorange', label='Identity')
    plt.plot([0, 20e3],
             [intercept_0072_seg, intercept_0072_seg + 20e3 * slope_0072_seg],
             color='C0', label='Linear regression')
    plt.xlabel('Manual segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.ylabel('Auto segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()
    cbar = plt.colorbar()
    cbar.ax.set_title('Dice coeff.')
    plt.tight_layout()

    # plt.subplot(212)
    plt.clf()
    plt.scatter(df_0072['area_contour'], df_0072['area_seg_corrected'], c=df_0072['dice_corrected'], s=2, cmap='RdBu',
                vmin=0.5, vmax=1.0)
    plt.plot([0, 20e3], [0, 20e3], color='darkorange', label='Identity')
    plt.plot([0, 20e3],
             [intercept_0072_seg_corrected, intercept_0072_seg_corrected + 20e3 * slope_0072_seg_corrected],
             color='C0', label='Linear regression')
    plt.xlabel('Manual segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.ylabel('Auto segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()
    cbar = plt.colorbar()
    cbar.ax.set_title('Dice coeff.')
    plt.tight_layout()

## Population cell area profiles

if DEBUG:
    plt.clf()
    plt.boxplot([df_0072['area_contour'], df_0072['area_seg'], df_0072['area_seg_corrected']], notch=True,
                labels=['Manual', 'Automatic', 'Corrected'])


'''
************************************************************************************************************************
AUTOMATIC SEGMENTATION VALIDATION (WITH CLASSIFIER) USING POPULATION AREA PROFILE:

  Apply segmentation trained with each 10 folds to the images in the other fold. This produces a lot of automatic
  segmentations.

  Apply segmentation correction and recompute areas.

  Apply classifier to each segmentation and corrected segmentation to decide whether it's accepted or rejected.

************************************************************************************************************************
'''

# rough segmentation parameters
downsample_factor = 8.0
dilation_size = 25
component_size_threshold = 1e6
hole_size_treshold = 8000

# Create dataframe with validation measures

time0 = time.time()
df_all = pd.DataFrame()

for i_fold in range(n_folds):

    print('## Fold ' + str(i_fold) + '/' + str(n_folds - 1))
    if (i_fold <= 6):
        print('Skipping, done')
        continue

    '''Load data
    '''

    # list of test files in this fold
    file_list_test = np.array(file_list)[idx_test_all[i_fold]]
    # file_list_train = np.array(file_list)[idx_train_all[i_fold]]

    # correct home directory in file paths
    file_list_test = cytometer.data.change_home_directory(list(file_list_test),
                                                          '/users/rittscher/rcasero', home, check_isfile=True)

    # load classifier model
    classifier_model_filename = os.path.join(saved_models_dir,
                                             classifier_model_basename + '_fold_' + str(i_fold) + '.h5')
    classifier_model = keras.models.load_model(classifier_model_filename)

    # load quality model
    quality_model_filename = os.path.join(saved_models_dir, quality_model_basename + '_fold_' + str(i_fold) + '.h5')
    quality_model = keras.models.load_model(quality_model_filename)

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
            plt.imshow(im)
            plt.axis('off')
            plt.title('Histology', fontsize=14)

        '''Rough segmentation'''

        histology_filename = os.path.basename(file_svg)
        aux = re.split('_row', histology_filename)
        histology_filename = aux[0] + '.ndpi'
        histology_filename = os.path.join(histology_dir, histology_filename)

        aux = aux[1].replace('.svg', '')
        aux = re.split('_', aux)
        row = np.int32(aux[1])
        col = np.int32(aux[3])

        # rough segmentation of the tissue in the full histology image
        rough_mask, im_downsampled = \
            cytometer.utils.rough_foreground_mask(histology_filename, downsample_factor=downsample_factor,
                                                  dilation_size=dilation_size,
                                                  component_size_threshold=component_size_threshold,
                                                  hole_size_treshold=hole_size_treshold,
                                                  return_im=True)

        # crop full histology to only the training image
        row_0 = np.int32(np.round((row - 500) / downsample_factor))
        row_end = row_0 + np.int32(np.round(im.size[0] / downsample_factor))
        col_0 = np.int32(np.round((col - 500) / downsample_factor))
        col_end = col_0 + np.int32(np.round(im.size[1] / downsample_factor))

        # crop rough mask and downsampled image
        im_crop = im_downsampled[row_0:row_end, col_0:col_end]
        rough_mask_crop = rough_mask[row_0:row_end, col_0:col_end]

        # upsample image and mask
        im_crop = Image.fromarray(im_crop)
        im_crop = im_crop.resize(size=(1001, 1001), resample=Image.LINEAR)
        im_crop = np.array(im_crop)
        rough_mask_crop = Image.fromarray(rough_mask_crop)
        rough_mask_crop = rough_mask_crop.resize(size=(1001, 1001), resample=Image.NEAREST)
        rough_mask_crop = np.array(rough_mask_crop)

        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(im)
            plt.axis('off')
            plt.subplot(222)
            plt.imshow(im_crop)
            plt.axis('off')
            plt.subplot(224)
            plt.imshow(rough_mask_crop)
            plt.axis('off')

            plt.clf()
            plt.imshow(im)
            plt.contour(rough_mask_crop, colors='k')

        '''Automatic segmentation of image'''

        # filenames of contour and dmap models
        contour_model_filename = os.path.join(saved_models_dir, contour_model_basename + '_fold_' + str(i_fold) + '.h5')
        dmap_model_filename = os.path.join(saved_models_dir, dmap_model_basename + '_fold_' + str(i_fold) + '.h5')

        # segment histology
        labels, _ = cytometer.utils.segment_dmap_contour_v3(im, contour_model=contour_model_filename,
                                                            dmap_model=dmap_model_filename)

        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(im)
            plt.axis('off')
            plt.title('Histology', fontsize=14)

            plt.subplot(222)
            plt.cla()
            plt.imshow(labels)
            plt.axis('off')
            plt.title('Segmentation', fontsize=14)

            plt.subplot(223)
            plt.cla()
            plt.imshow(im)
            plt.contour(labels, levels=np.unique(labels), colors='k')
            plt.axis('off')
            plt.title('Segmentation on histology', fontsize=14)

        # remove labels that touch the edges or that are too small
        labels \
            = cytometer.utils.clean_segmentation(labels, remove_edge_labels=True, min_cell_area=min_cell_area,
                                                 mask=rough_mask_crop)

        if DEBUG:
            plt.subplot(224)
            plt.cla()
            plt.imshow(im)
            plt.contour(labels, levels=np.unique(labels), colors='k')
            plt.contour(rough_mask_crop, levels=np.unique(labels), colors='r')
            plt.axis('off')
            plt.title('Cleaned segmentation', fontsize=14)

        # list of unique segmentation labels (remove background)
        labels_seg = list(np.unique(labels))
        if 0 in labels_seg:
            labels_seg.remove(0)

        if len(labels_seg) == 0:
            warnings.warn('No labels produced!')
            continue

        # initialise dataframe to keep results: one cell per row, tagged with mouse metainformation
        df_im = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_tif),
                                                          values=[i_fold], values_tag='fold',
                                                          tags_to_keep=['id', 'ko', 'sex'])

        '''Find the automatic segmentation that best overlaps with each cell contour'''

        # loop segmentation labels
        for j, lab in enumerate(labels_seg):

            # start dataframe row for this contour
            df = df_im.copy()
            df['im'] = i

            # isolate current segmentation
            cell_seg = (labels == lab).astype(np.uint8)

            if DEBUG:
                plt.clf()
                plt.imshow(im)
                plt.contour(cell_seg, colors='red')
                plt.axis('off')

            # add to dataframe
            df['label_seg'] = lab
            df['area_seg'] = np.count_nonzero(cell_seg) * xres * yres  # um^2

            '''Bounding boxes'''

            # compute bounding box for the automatic segmentation
            bbox_seg_x0, bbox_seg_y0, bbox_seg_xend, bbox_seg_yend = \
                cytometer.utils.bounding_box_with_margin(cell_seg, coordinates='xy', inc=1.00)
            bbox_seg_r0, bbox_seg_c0, bbox_seg_rend, bbox_seg_cend = \
                cytometer.utils.bounding_box_with_margin(cell_seg, coordinates='rc', inc=1.00)

            # this renaming here is redundant, but it'll be useful in later validation
            bbox_total_x0 = bbox_seg_x0
            bbox_total_y0 = bbox_seg_y0
            bbox_total_xend = bbox_seg_xend
            bbox_total_yend = bbox_seg_yend

            bbox_total_r0 = bbox_seg_r0
            bbox_total_c0 = bbox_seg_c0
            bbox_total_rend = bbox_seg_rend
            bbox_total_cend = bbox_seg_cend
            bbox_total = (bbox_total_r0, bbox_total_c0, bbox_total_rend, bbox_total_cend)

            if DEBUG:
                plt.clf()
                plt.imshow(im)
                plt.contour(cell_seg, linewidths=1, colors='red')

                plt.plot((bbox_total_x0, bbox_total_xend, bbox_total_xend, bbox_total_x0, bbox_total_x0),
                         (bbox_total_y0, bbox_total_y0, bbox_total_yend, bbox_total_yend, bbox_total_y0),
                         color='green')
                plt.axis('off')

                plt.xlim(bbox_total_x0 - (bbox_total_xend - bbox_total_x0) * 0.1,
                         bbox_total_xend + (bbox_total_xend - bbox_total_x0) * 0.1)
                plt.ylim(bbox_total_yend + (bbox_total_yend - bbox_total_y0) * 0.1,
                         bbox_total_y0 - (bbox_total_yend - bbox_total_y0) * 0.1)

            '''Cropping and resizing of individual contour'''

            # crop image and masks according to bounding box of automatic segmentation
            window_im = cytometer.utils.extract_bbox(np.array(im), bbox_total)
            window_seg = cytometer.utils.extract_bbox(cell_seg.astype(np.uint8), bbox_total)

            if DEBUG:
                plt.clf()
                plt.imshow(window_im)
                plt.contour(window_seg, linewidths=1, colors='red')
                plt.axis('off')

            '''Segmentation correction'''

            # input to segmentation quality CNN: multiply histology by +1/-1 segmentation mask
            # NOTE: quality network 0053 expects values in [-255, 255], float32
            window_masked_im = \
                cytometer.utils.quality_model_mask(window_seg.astype(np.float32), im=window_im.astype(np.float32),
                                                   quality_model_type='-1_1')[0, :, :, :]

            if DEBUG:
                plt.clf()
                plt.imshow(window_im)
                plt.contour(np.any(window_masked_im > 0, axis=2), linewidths=1, colors='k')
                plt.imshow(np.any(window_masked_im > 0, axis=2), alpha=0.5)
                plt.axis('off')

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

            # check sizes and types
            assert (window_im.ndim == 4 and window_im.dtype == np.float32
                    and np.min(window_im) >= 0 and np.max(window_im) <= 1.0)
            assert (window_masked_im.ndim == 4 and window_masked_im.dtype == np.float32
                    and np.min(window_masked_im) >= -255 and np.max(window_masked_im) <= 255)

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

            # add to dataframe
            df['area_seg_corrected'] = area_seg_corrected

            if DEBUG:
                # plot segmentation correction
                plt.clf()
                plt.imshow(window_im[0, :, :, :])
                plt.contour(window_seg[0, :, :], linewidths=1, colors='red')
                plt.contour(window_seg_corrected, linewidths=1, colors='black')
                plt.axis('off')

            '''Object classification as "WAT" or "other"'''

            # process histology for classification
            window_classifier_out = classifier_model.predict(window_im, batch_size=batch_size)

            # get classification label for each pixel
            window_classifier_class = np.argmax(window_classifier_out, axis=3)

            # proportion of "Other" pixels in the mask
            window_other_prop = np.count_nonzero(window_seg * window_classifier_class) \
                                / np.count_nonzero(window_seg)

            # add to dataframe row
            df['seg_type_prop'] = window_other_prop

            if DEBUG:
                # plot classification
                plt.clf()
                plt.imshow(window_classifier_class[0, :, :])
                plt.contour(window_seg[0, :, :], linewidths=1, colors='red')
                plt.title('"Other" prop = ' + str("{:.0f}".format(window_other_prop * 100)) + '%', fontsize=14)
                plt.axis('off')

            '''Wrap things up in this loop'''

            # append current results to global dataframe
            df_all = pd.concat([df_all, df])

        print('Time so far: ' + str(time.time() - time0) + ' s')

# reset indices
df_all.reset_index(drop=True, inplace=True)

# save results
dataframe_filename = os.path.join(saved_models_dir, experiment_id + '_dataframe_segmentation_profiles.pkl')
df_all.to_pickle(dataframe_filename)

## Results analysis

# load previous dataframe, where we already have the contour areas. Note that all rows here correspond to "Cell"
# contours, because only those contours were used in that experiment
dataframe_filename_0072 = os.path.join(saved_models_dir, experiment_id + '_dataframe_segmentation_automatic.pkl')
df_0072_manual = pd.read_pickle(dataframe_filename_0072)

# remove contours that got matched to the background
df_0072_manual = df_0072_manual.loc[df_0072_manual['label_seg'] != 0, :]

# remove segmentations that are larger than 20,000 um^2, because the largest cell in the manual dataset is ~19,000 um^2
df_0072_manual = df_0072_manual.loc[df_0072_manual['area_seg'] <= 20e3, :]

# reset indices
df_0072_manual.reset_index(drop=True, inplace=True)

# load this dataframe, where we have all the automatic segmentation objects, without the constraint of being matched
# to a manual contour
dataframe_filename_0072 = os.path.join(saved_models_dir, experiment_id + '_dataframe_segmentation_profiles.pkl')
df_0072_auto = pd.read_pickle(dataframe_filename_0072)

# remove segmentations that are larger than 20,000 um^2, because the largest cell in the manual dataset is ~19,000 um^2
df_0072_auto = df_0072_auto.loc[df_0072_auto['area_seg'] <= 20e3, :]

# reset indices
df_0072_auto.reset_index(drop=True, inplace=True)


# number of cells per fold
for i_fold in range(n_folds):
    print('fold = ' + str(i_fold) + ', ' + str(np.count_nonzero(df_0072_manual['fold'] == i_fold)) + ' (manual), '
          + str(np.count_nonzero(df_0072_auto['fold'] == i_fold)) + ' (auto), ')

## Boxplot of "cell" proportion in the automatic segmentation (in the manual segmentation, all "wat" contours are valid)

if DEBUG:
    plt.clf()
    plt.hist(1 - df_0072_auto['seg_type_prop'], bins=100)

print('Prop. classifier OK objects: ' + str(np.count_nonzero((1 - df_0072_auto['seg_type_prop']) >= 0.9) / len(df_0072_auto['seg_type_prop'])))

## Population cell area profiles

# objects that are cells are those with 90% of "wat" pixels
idx_cell_auto = (1 - df_0072_auto['seg_type_prop']) >= 0.9

if DEBUG:
    plt.clf()
    boxp = plt.boxplot([df_0072_manual['area_contour'],
                        df_0072_auto['area_seg'][idx_cell_auto],
                        df_0072_auto['area_seg_corrected'][idx_cell_auto]],
                       notch=True, labels=['Manual', 'Automatic,\nno correction', 'Automatic,\nwith correction'])
    plt.ylabel('Segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()

    # points of interest in the manual contours boxplot
    contour_perc_w0_manual = boxp['whiskers'][0].get_data()[1][1]
    contour_perc_25_manual = boxp['boxes'][0].get_data()[1][1]
    contour_perc_50_manual = boxp['medians'][0].get_data()[1][0]
    contour_perc_75_manual = boxp['boxes'][0].get_data()[1][5]
    contour_perc_wend_manual = boxp['whiskers'][1].get_data()[1][1]

    # points of interest in the auto, no correction boxplot
    contour_perc_w0_auto = boxp['whiskers'][2].get_data()[1][1]
    contour_perc_25_auto = boxp['boxes'][1].get_data()[1][1]
    contour_perc_50_auto = boxp['medians'][1].get_data()[1][0]
    contour_perc_75_auto = boxp['boxes'][1].get_data()[1][5]
    contour_perc_wend_auto = boxp['whiskers'][3].get_data()[1][1]

    # points of interest in the auto, correction boxplot
    contour_perc_w0_corrected = boxp['whiskers'][4].get_data()[1][1]
    contour_perc_25_corrected = boxp['boxes'][2].get_data()[1][1]
    contour_perc_50_corrected = boxp['medians'][2].get_data()[1][0]
    contour_perc_75_corrected = boxp['boxes'][2].get_data()[1][5]
    contour_perc_wend_corrected = boxp['whiskers'][5].get_data()[1][1]

    plt.ylim(-250, 6800)

    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_w0_manual, 'k--')
    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_25_manual, 'k--')
    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_50_manual, 'C1--')
    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_75_manual, 'k--')
    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_wend_manual, 'k--')

    # plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_w0_auto, 'k--')
    # plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_25_auto, 'k--')
    # plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_50_auto, 'C1--')
    # plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_75_auto, 'k--')
    # plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_wend_auto, 'k--')
    #
    # plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_w0_corrected, 'k--')
    # plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_25_corrected, 'k--')
    # plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_50_corrected, 'C1--')
    # plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_75_corrected, 'k--')
    # plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_wend_corrected, 'k--')

print('From manual to auto, no correction')
print('Bottom whisker: ' + str(100 * (contour_perc_w0_auto - contour_perc_w0_manual) / contour_perc_w0_manual) + '%')
print('25%: ' + str(100 * (contour_perc_25_auto - contour_perc_25_manual) / contour_perc_25_manual) + '%')
print('Median: ' + str(100 * (contour_perc_50_auto - contour_perc_50_manual) / contour_perc_50_manual) + '%')
print('75%: ' + str(100 * (contour_perc_75_auto - contour_perc_75_manual) / contour_perc_75_manual) + '%')
print('Top whisker: ' + str(100 * (contour_perc_wend_auto - contour_perc_wend_manual) / contour_perc_wend_manual) + '%')

print('From manual to auto, correction')
print('Bottom whisker: ' + str(100 * (contour_perc_w0_corrected - contour_perc_w0_manual) / contour_perc_w0_manual) + '%')
print('25%: ' + str(100 * (contour_perc_25_corrected - contour_perc_25_manual) / contour_perc_25_manual) + '%')
print('Median: ' + str(100 * (contour_perc_50_corrected - contour_perc_50_manual) / contour_perc_50_manual) + '%')
print('75%: ' + str(100 * (contour_perc_75_corrected - contour_perc_75_manual) / contour_perc_75_manual) + '%')
print('Top whisker: ' + str(100 * (contour_perc_wend_corrected - contour_perc_wend_manual) / contour_perc_wend_manual) + '%')

## Compare female/male cells sizes in manual and corrected datasets

# objects that are cells are those with 90% of "wat" pixels
idx_cell_manual_f = df_0072_manual['sex'] == 'f'
idx_cell_manual_m = np.logical_not(idx_cell_manual_f)
idx_cell_auto_f = np.logical_and((1 - df_0072_auto['seg_type_prop']) >= 0.9,
                                 df_0072_auto['sex'] == 'f')
idx_cell_auto_m = np.logical_and((1 - df_0072_auto['seg_type_prop']) >= 0.9,
                                 df_0072_auto['sex'] == 'm')

if DEBUG:
    plt.clf()
    boxp = plt.boxplot([df_0072_manual['area_contour'][idx_cell_manual_f],
                        df_0072_manual['area_contour'][idx_cell_manual_m],
                        df_0072_auto['area_seg_corrected'][idx_cell_auto_f],
                        df_0072_auto['area_seg_corrected'][idx_cell_auto_m]],
                       notch=True, labels=['Manual F', 'Manual M', 'Auto\ncorrected F', 'Auto\ncorrected M'],
                       positions=[1, 1.5, 2.5, 3])
    plt.ylabel('Segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()


'''
************************************************************************************************************************
FROM HERE ONWARDS, CODE NEEDS TO BE REWRITTEN
************************************************************************************************************************
'''

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
        contour_type_all = ['wat', ] * len(cell_contours) \
                           + ['other', ] * len(other_contours) \
                           + ['bat', ] * len(brown_contours)

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
            plt.subplot(222)
            plt.cla()
            plt.imshow(labels)
            plt.axis('off')
            plt.title('Segmentation', fontsize=14)

            plt.subplot(223)
            plt.cla()
            plt.imshow(im)
            plt.contour(labels, levels=np.unique(labels), colors='k')
            plt.axis('off')
            plt.title('Segmentation on histology', fontsize=14)

        # remove labels that touch the edges or that are too small
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

        '''
        ****************************************************************************************************
        Ground truth cell by cell processing
        ****************************************************************************************************
        '''

        # loop cell contours
        for j, contour in enumerate(contours):

            # start dataframe row for this contour
            df = df_im.copy()
            df['im'] = i
            df['contour'] = j

            if DEBUG:
                # centre of current cell
                # xy_c = (np.mean([p[0] for p in contour]), np.mean([p[1] for p in contour]))
                # plt.text(xy_c[0], xy_c[1], 'j=' + str(j))
                # plt.scatter(xy_c[0], xy_c[1])

                # close the contour for the plot
                contour_aux = contour.copy()
                contour_aux.append(contour[0])

                plt.clf()
                plt.imshow(im)
                plt.contour(labels, levels=np.unique(labels), colors='k')
                plt.plot([p[0] for p in contour_aux], [p[1] for p in contour_aux], color='green')
                plt.axis('off')

            # rasterise current ground truth segmentation
            cell_seg_contour = Image.new("1", im.size, "black")  # I = 32-bit signed integer pixels
            draw = ImageDraw.Draw(cell_seg_contour)
            draw.polygon(contour, outline="white", fill="white")
            cell_seg_contour = np.array(cell_seg_contour, dtype=np.uint8)

            if contour_type_all[j] == 'wat':  # contour type: white adipocyte tissue

                '''Match hand-traced segmentation to automatic segmentation'''

                # find the segmented label that best overlaps with the ground truth label
                match_out = \
                    cytometer.utils.match_overlapping_labels(labels_ref=labels, labels_test=cell_seg_contour,
                                                             allow_repeat_ref=False)

                # save correspondence to dataframe
                if len(match_out) == 1:
                    df['label'] = match_out[0]['lab_ref']

                    if DEBUG:
                        plt.contour(labels == df['label'][0], colors='r')

                    # remove matched segmentation from list of remaining segmentations
                    if df['label'][0] in labels_seg:
                        labels_seg.remove(df['label'][0])
                else:
                    warnings.warn('No correspondence found for contour: ' + str(j))
                    df['label'] = -1

            elif contour_type_all[j] in ['other', 'bat']:  # contour type: other cells or brown adipose tissue

                # it makes no sense to match an other or bat contour to a segmenation label, because they were
                # hand-traced without trying to follow any contours
                df['label'] = -1

            else:  # contour type: unknown

                raise ValueError('Unrecognised contour type: ' + contour_type_all[j])

            '''Bounding boxes'''

            # compute bounding box for the contour
            bbox_seg_contour_x0, bbox_seg_gtruth_y0, bbox_seg_contour_xend, bbox_seg_contour_yend = \
                cytometer.utils.bounding_box_with_margin(cell_seg_contour, coordinates='xy', inc=1.00)
            bbox_seg_contour_r0, bbox_seg_contour_c0, bbox_seg_contour_rend, bbox_seg_contour_cend = \
                cytometer.utils.bounding_box_with_margin(cell_seg_contour, coordinates='rc', inc=1.00)

            # isolate matched segmented label
            if df['label'][0] != -1:  # WAT cells have a corresponding segmentation
                cell_seg = (labels == df['label'][0]).astype(np.uint8)

                bbox_seg_x0, bbox_seg_y0, bbox_seg_xend, bbox_seg_yend = \
                    cytometer.utils.bounding_box_with_margin(cell_seg, coordinates='xy', inc=1.00)
                bbox_seg_r0, bbox_seg_c0, bbox_seg_rend, bbox_seg_cend = \
                    cytometer.utils.bounding_box_with_margin(cell_seg, coordinates='rc', inc=1.00)

                # add to dataframe: segmentation areas and Dice coefficient
                df['area_contour'] = match_out[0]['area_test'] * xres * yres  # um^2
                df['area_seg'] = match_out[0]['area_ref'] * xres * yres  # um^2
                df['dice'] = match_out[0]['dice']

            else:  # "other" and BAT have no corresponding segmentation

                # add to dataframe: contour area
                df['area_contour'] = np.count_nonzero(cell_seg_contour) * xres * yres  # um^2
                df['area_seg'] = np.nan
                df['dice'] = np.nan

            if DEBUG:
                plt.clf()
                plt.imshow(im)
                plt.contour(cell_seg_contour, linewidths=1, colors='green')

                plt.plot((bbox_seg_contour_x0, bbox_seg_contour_xend, bbox_seg_contour_xend, bbox_seg_contour_x0, bbox_seg_contour_x0),
                         (bbox_seg_gtruth_y0, bbox_seg_gtruth_y0, bbox_seg_contour_yend, bbox_seg_contour_yend, bbox_seg_gtruth_y0),
                         color='green')

                if df['label'][0] != -1:

                    plt.plot((bbox_seg_x0, bbox_seg_xend, bbox_seg_xend, bbox_seg_x0, bbox_seg_x0),
                             (bbox_seg_y0, bbox_seg_y0, bbox_seg_yend, bbox_seg_yend, bbox_seg_y0),
                             color='red')
                    plt.contour(cell_seg, linewidths=1, colors='red')

                    bbox_total_x0 = np.min((bbox_seg_contour_x0, bbox_seg_x0))
                    bbox_total_y0 = np.min((bbox_seg_gtruth_y0, bbox_seg_y0))
                    bbox_total_xend = np.max((bbox_seg_contour_xend, bbox_seg_xend))
                    bbox_total_yend = np.max((bbox_seg_contour_yend, bbox_seg_yend))

                else:

                    bbox_total_x0 = bbox_seg_contour_x0
                    bbox_total_y0 = bbox_seg_gtruth_y0
                    bbox_total_xend = bbox_seg_contour_xend
                    bbox_total_yend = bbox_seg_contour_yend

                plt.xlim(bbox_total_x0 - (bbox_total_xend - bbox_total_x0) * 0.1,
                         bbox_total_xend + (bbox_total_xend - bbox_total_x0) * 0.1)
                plt.ylim(bbox_total_yend + (bbox_total_yend - bbox_total_y0) * 0.1,
                         bbox_total_y0 - (bbox_total_yend - bbox_total_y0) * 0.1)
                plt.axis('off')

            # crop image and masks according to bounding box of automatic segmentation
            window_im = cytometer.utils.extract_bbox(np.array(im), (bbox_seg_r0, bbox_seg_c0, bbox_seg_rend, bbox_seg_cend))
            window_seg_contour = cytometer.utils.extract_bbox(cell_seg_contour, (bbox_seg_r0, bbox_seg_c0, bbox_seg_rend, bbox_seg_cend))
            if df['label'][0] != -1:
                window_seg = cytometer.utils.extract_bbox(cell_seg, (bbox_seg_r0, bbox_seg_c0, bbox_seg_rend, bbox_seg_cend))

            if DEBUG:
                plt.clf()
                plt.imshow(window_im)
                plt.contour(window_seg_contour, linewidths=1, colors='green')
                if df['label'][0] != -1:
                    plt.contour(window_seg, linewidths=1, colors='red')
                plt.axis('off')

            '''Cropping and resizing of individual cell'''

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
            window_seg_contour = cytometer.utils.resize(window_seg_contour, size=training_size, resample=Image.NEAREST)
            window_seg = cytometer.utils.resize(window_seg, size=training_size, resample=Image.NEAREST)

            # add dummy dimensions for keras
            window_im = np.expand_dims(window_im, axis=0)
            window_masked_im = np.expand_dims(window_masked_im, axis=0)
            window_seg = np.expand_dims(window_seg, axis=0)

            # correct types
            window_im = window_im.astype(np.float32) / 255.0
            window_seg = window_seg.astype(np.float32)
            window_seg_contour = window_seg_contour.astype(np.float32)

            # check sizes and types
            assert(window_im.ndim == 4 and window_im.dtype == np.float32
                   and np.min(window_im) >= 0 and np.max(window_im) <= 1.0)
            assert(window_masked_im.ndim == 4 and window_masked_im.dtype == np.float32
                   and np.min(window_masked_im) >= -255.0 and np.max(window_masked_im) <= 255.0)
            assert(window_seg.ndim == 3 and window_seg.dtype == np.float32)
            assert(window_seg_contour.ndim == 2 and window_seg_contour.dtype == np.float32)

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
                plt.contour(window_seg_contour, linewidths=1, colors='green')
                plt.contour(window_seg[0, :, :], linewidths=1, colors='red')
                plt.contour(window_seg_corrected, linewidths=1, colors='black')
                plt.axis('off')

            '''Object classification as "white adipocyte" or "other"'''

            # process histology for classification
            window_classifier_out = classifier_model.predict(window_im, batch_size=batch_size)

            # get classification label for each pixel
            window_classifier_class = np.argmax(window_classifier_out, axis=3)

            # proportion of "Other" pixels in the mask
            window_other_prop = np.count_nonzero(window_seg_corrected * window_classifier_class) \
                                / np.count_nonzero(window_seg_corrected)

            # add to dataframe row
            df['contour_type'] = contour_type_all[j]
            df['contour_type_prop'] = window_other_prop

            if DEBUG:
                # plot classification
                plt.clf()
                plt.imshow(window_classifier_class[0, :, :])
                # plt.contour(window_seg_gtruth, linewidths=1, colors='green')
                plt.contour(window_seg[0, :, :], linewidths=1, colors='red')
                plt.title('"Other" prop = ' + str("{:.0f}".format(window_other_prop * 100)) + '%', fontsize=14)
                plt.axis('off')

            # append current results to global dataframe
            df_all = pd.concat([df_all, df])

        print('Time so far: ' + str(time.time() - time0) + ' s')

        '''
        ****************************************************************************************************
        Automatic segmentations without ground truth
        ****************************************************************************************************
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
            df['area_contour'] = np.nan
            df['area_seg'] = np.count_nonzero(cell_seg) * xres * yres  # um^2
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
            window_seg_contour = window_seg_contour.astype(np.float32)

            # check sizes and types
            assert(window_im.ndim == 4 and window_im.dtype == np.float32
                   and np.min(window_im) >= 0 and np.max(window_im) <= 1.0)
            assert(window_masked_im.ndim == 4 and window_masked_im.dtype == np.float32
                   and np.min(window_masked_im) >= -255.0 and np.max(window_masked_im) <= 255.0)
            assert(window_seg.ndim == 3 and window_seg.dtype == np.float32)

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

            '''Object classification as "white adipocyte" or "other"'''

            # process histology for classification
            window_classifier_out = classifier_model.predict(window_im, batch_size=batch_size)

            # get classification label for each pixel
            window_classifier_class = np.argmax(window_classifier_out, axis=3)

            # proportion of "Other" pixels in the mask
            window_other_prop = np.count_nonzero(window_seg_corrected * window_classifier_class) \
                                / np.count_nonzero(window_seg_corrected)

            # add to dataframe row
            df['contour_type'] = -1
            df['contour_type_prop'] = window_other_prop

            if DEBUG:
                # plot classification
                plt.clf()
                plt.imshow(window_classifier_class[0, :, :])
                plt.contour(window_seg[0, :, :], linewidths=1, colors='red')
                plt.title('"Other" prop = ' + str("{:.0f}".format(window_other_prop * 100)) + '%', fontsize=14)
                plt.axis('off')

            # append current results to global dataframe
            df_all = pd.concat([df_all, df])

        print('Time so far: ' + str(time.time() - time0) + ' s')

# reset indices
df_all.reset_index(drop=True, inplace=True)

# save results
dataframe_filename = os.path.join(saved_models_dir, experiment_id + '_dataframe.pkl')
df_all.to_pickle(dataframe_filename)

'''Save segmentation results to image files
'''

# dataframe:
#  ['id', 'ko', 'sex', 'fold', 'im', 'contour', 'area_contour', 'area_seg',
#   'dice', 'area_seg_corrected', 'contour_type', 'contour_type_prop']

# load results
dataframe_filename_0072 = os.path.join(saved_models_dir, 'klf14_b6ntac_exp_0072_pipeline_validation_dataframe.pkl')
df_0072 = pd.read_pickle(dataframe_filename_0072)

for row in range(df_0072.shape[0]):

    pass


'''Dataframe analysis of segmentations with corresponding ground truth
'''

# dataframe:
#  ['id', 'ko', 'sex', 'fold', 'im', 'contour', 'area_contour', 'area_seg',
#   'dice', 'area_seg_corrected', 'contour_type', 'contour_type_prop']

## this experiment: classifier with data augmentation (only segmentations with corresponding ground truth)

# load results
dataframe_filename_0072 = os.path.join(saved_models_dir, 'klf14_b6ntac_exp_0072_pipeline_validation_dataframe.pkl')
df_0072 = pd.read_pickle(dataframe_filename_0072)

# remove rows of non-cell contours


# remove rows with segmentations that have no corresponding ground truth
idx = df_0072['contour_type'] != -1
df_0072 = df_0072.loc[idx, :]

# reject rows with very low Dice values, because that means that the ground truth contours doesn't really overlap with
# an automatic segmentation. Instead, it's probably just touching a nearby one
idx = df_0072['dice'] >= 0.5
df_0072 = df_0072.loc[idx, :]

# add columns for cell estimates. This could be computed on the fly in the plots, but this way makes the code below
# a bit easier to read, and less likely to have bugs if we forget the 1-x operation
df_0072['contour_type'] = 1 - df_0072['contour_type']
df_0072['cell_prop'] = 1 - df_0072['contour_type_prop']

# classifier ROC (we make cell=1, other=0 for clarity of the results)
fpr_0072, tpr_0072, thr_0072 = roc_curve(y_true=df_0072['contour_type'],
                                         y_score=df_0072['cell_prop'])
roc_auc_0072 = auc(fpr_0072, tpr_0072)

# find point in the curve for False Positive Rate close to 10%
idx_0072 = np.where(fpr_0072 > 0.1)[0][0]

## show imbalance between classes
n_wat = np.count_nonzero(df_0072['contour_type'] == 1)
n_non_wat = np.count_nonzero(df_0072['contour_type'] == 1)
print('Number of cell objects: ' + str(n_wat) + ' (%0.1f' % (n_wat / (n_wat + n_non_wat) * 100) + '%)')
print('Number of other objects: ' + str(n_non_wat) + ' (%0.1f' % (n_non_wat / (n_wat + n_non_wat) * 100) + '%)')

## plots for both classifiers

if DEBUG:
    # ROC curve before and after data augmentation
    plt.clf()
    plt.plot(fpr_0072, tpr_0072, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_0072)
    plt.scatter(fpr_0072[idx_0072], tpr_0072[idx_0072],
                label='Thr =  %0.3f, FPR = %0.3f, TPR = %0.3f'
                      % (thr_0072[idx_0072], fpr_0072[idx_0072], tpr_0072[idx_0072]),
                color='blue')
    plt.tick_params(labelsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="lower right")


# classifier confusion matrix
cytometer.utils.plot_confusion_matrix(y_true=df_0072['contour_type'],
                                      y_pred=df_0072['cell_prop'] >= thr_0072[idx_0072],
                                      normalize=True,
                                      title='With data augmentation',
                                      xlabel='"Cell" predicted',
                                      ylabel='"Cell" is ground truth',
                                      cmap=plt.cm.Blues,
                                      colorbar=False)

## segmentation correction

# indices of "Cell" objects
idx_cell = df_0072['contour_type'] == 1

# linear regression
slope_0072_seg, intercept_0072_seg, \
r_value_0072_seg, p_value_0072_seg, std_err_0072_seg = \
    linregress(df_0072.loc[idx_cell, 'area_contour'], df_0072.loc[idx_cell, 'area_seg'])

slope_0072_seg_corrected, intercept_0072_seg_corrected, \
r_value_0072_seg_corrected, p_value_0072_seg_corrected, std_err_0072_seg_corrected = \
    linregress(df_0072.loc[idx_cell, 'area_contour'], df_0072.loc[idx_cell, 'area_seg_corrected'])

if DEBUG:
    # linear regression: ground truth area vs. best matching segmentation area
    # No area correction
    plt.clf()
    plt.scatter(df_0072.loc[idx_cell, 'area_contour'], df_0072.loc[idx_cell, 'area_seg'], label='')
    plt.plot([0, 20e3], [0, 20e3], color='darkorange', label='Identity')
    plt.plot([0, 20e3],
             [intercept_0072_seg, intercept_0072_seg + 20e3 * slope_0072_seg],
             color='red', label='Linear regression')
    plt.xlabel('Ground truth area ($\mu$m$^2$)', fontsize=14)
    plt.ylabel('Segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()

    # linear regression: ground truth area vs. best matching segmentation area
    # Area correction
    plt.clf()
    plt.scatter(df_0072.loc[idx_cell, 'area_contour'], df_0072.loc[idx_cell, 'area_seg_corrected'], label='')
    plt.plot([0, 20e3], [0, 20e3], color='darkorange', label='Identity')
    plt.plot([0, 20e3],
             [intercept_0072_seg_corrected, intercept_0072_seg_corrected + 20e3 * slope_0072_seg_corrected],
             color='red', label='Linear regression')
    plt.xlabel('Ground truth area ($\mu$m$^2$)', fontsize=14)
    plt.ylabel('Segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()

## population area profiles

if DEBUG:

    # hand vs. automatic vs. corrected segmentation
    plt.clf()
    plt.hist(df_0072.loc[idx_cell, 'area_contour'], density=True, bins=50, histtype='step',
             label='Hand segmentation')
    plt.hist(df_0072.loc[idx_cell, 'area_seg'], density=True, bins=50, histtype='step',
             label='Automatic segmentation')
    plt.hist(df_0072.loc[idx_cell, 'area_seg_corrected'], density=True, bins=50, histtype='step',
             label='Corrected segmentation')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.legend()

## reload this experiment, including segmentations without corresponding ground truth

# load results
dataframe_filename_0072 = os.path.join(saved_models_dir, 'klf14_b6ntac_exp_0064_pipeline_validation_dataframe.pkl')
df_0064_all = pd.read_pickle(dataframe_filename_0072)

# new column with proportion of cell pixels in segmentation, for convenience
df_0064_all['cell_prop'] = 1 - df_0064_all['contour_type_prop']

# population profiles for automatic segmentations and automatic classification
idx_cell_auto = df_0064_all['cell_prop'] >= thr_0072[idx_0072]

# median values
print('Ground truth median area: %0.0f ' % np.median(df_0072.loc[idx_cell, 'area_contour']) + r'(um^2)')
print('Pipeline median area: %0.0f ' % np.median(df_0064_all.loc[idx_cell_auto, 'area_seg_corrected']) + r'(um^2)')

if DEBUG:

    # histograms: hand vs. full test windows
    plt.clf()
    plt.hist(df_0072.loc[idx_cell, 'area_contour'], density=True, bins=50, histtype='step',
             label='Hand segmentations')
    plt.hist(df_0064_all.loc[idx_cell_auto, 'area_seg_corrected'], density=True, bins=50, histtype='step',
             label='Classifier segmentations')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.legend()

    # boxplots: hand vs. full test windows
    plt.clf()
    plt.boxplot((df_0072.loc[idx_cell, 'area_contour'],
                 df_0064_all.loc[idx_cell_auto, 'area_seg_corrected']), notch=True,
                labels=('Hand segmentations', 'Classifier segmentations'))
    plt.ylabel('Segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(-840, 6800)
