"""
Validate pipeline v7 with 10-fold cross validation:
 * data generation
   * training images (*0076*)
   * non-overlap training images (*0077*)
   * augmented training images (*0078*)
   * k-folds + extra "other" for classifier (*0094*)
 * segmentation
   * dmap (*0086*)
   * contour from dmap (*0091*)
 * classifier (*0095*)
 * segmentation correction (*0089*) networks""
 * validation (0096)"
"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0096_pipeline_v7_validation'

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
# from enum import IntEnum
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import scipy.stats as stats
from mlxtend.evaluate import permutation_test
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from enum import IntEnum

# # limit number of GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

import cytometer.utils
import cytometer.data

# import tensorflow as tf
# # limit GPU memory used
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# set_session(tf.Session(config=config))

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
max_cell_area = 100e3
min_mask_overlap = 0.8
phagocytosis = True
min_class_prop = 0.5
correction_window_len = 401
correction_smoothing = 11
batch_size = 2

# rough_foreground_mask() parameters
downsample_factor = 8.0
dilation_size = 25
component_size_threshold = 0
hole_size_treshold = 8000

# types of pixels
class PixelType(IntEnum):
    UNDETERMINED = 0
    WAT = 1
    OTHER = 2

'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
ndpi_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')

training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')
# saved_figures_dir = os.path.join(root_data_dir, 'figures')
saved_figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
metainfo_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')

# k-folds file
saved_kfolds_filename = 'klf14_b6ntac_exp_0094_generate_extra_training_images.pickle'

# model names
dmap_model_basename = 'klf14_b6ntac_exp_0086_cnn_dmap'
contour_model_basename = 'klf14_b6ntac_exp_0091_cnn_contour_after_dmap'
classifier_model_basename = 'klf14_b6ntac_exp_0095_cnn_tissue_classifier_fcn'
correction_model_basename = 'klf14_b6ntac_exp_0089_cnn_segmentation_correction_overlapping_scaled_contours'

'''Load folds'''

# load list of images, and indices for training vs. testing indices
saved_kfolds_filename = os.path.join(saved_models_dir, saved_kfolds_filename)
with open(saved_kfolds_filename, 'rb') as f:
    aux = pickle.load(f)
file_svg_list = aux['file_list']
idx_test_all = aux['idx_test']
idx_train_all = aux['idx_train']

# correct home directory
file_svg_list = [x.replace('/users/rittscher/rcasero', home) for x in file_svg_list]
file_svg_list = [x.replace('/home/rcasero', home) for x in file_svg_list]

# number of images
n_im = len(file_svg_list)

# number of folds
n_folds = len(idx_test_all)

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

'''
************************************************************************************************************************
Prepare the testing data:

  This is computed once, and then saved to 
  'klf14_b6ntac_exp_0096_pipeline_v7_validation_data.npz'.
  In subsequent runs, the data is loaded from that file.

  Apply classifier trained with each 10 folds to the other fold. 
************************************************************************************************************************
'''

'''Load the test data
'''

# file name for pre-computed data
data_filename = os.path.join(saved_models_dir, experiment_id + '_data.npz')

if os.path.isfile(data_filename):

    # Note: If an image has no contours, it's corresponding im_array_all, ..., out_mask_all will be all zeros

    # load pre-computed data
    aux = np.load(data_filename)
    im_array_all = aux['im_array_all']
    rough_mask_all = aux['rough_mask_all']
    out_class_all = aux['out_class_all']
    out_mask_all = aux['out_mask_all']
    del aux

else:  # pre-compute the validation data and save to file

    # start timer
    t0 = time.time()

    # init output
    im_array_all = np.zeros(shape=(len(file_svg_list), 1001, 1001, 3), dtype=np.float32)
    rough_mask_all = np.zeros(shape=(len(file_svg_list), 1001, 1001), dtype=np.bool)
    out_class_all = np.zeros(shape=(len(file_svg_list), 1001, 1001, 1), dtype=np.float32)
    out_mask_all = np.zeros(shape=(len(file_svg_list), 1001, 1001), dtype=np.float32)

    # loop files with hand traced contours
    for i, file_svg in enumerate(file_svg_list):

        '''Read histology training window
        '''

        print('file ' + str(i) + '/' + str(len(file_svg_list) - 1) + ': ' + os.path.basename(file_svg))

        # change file extension from .svg to .tif
        file_tif = file_svg.replace('.svg', '.tif')

        # open histology training image
        im = Image.open(file_tif)

        # make array copy
        im_array = np.array(im)

        if DEBUG:
            enhancer = ImageEnhance.Contrast(im)
            enhanced_im = enhancer.enhance(4.0)

            plt.clf()
            plt.imshow(im)

            plt.clf()
            plt.imshow(enhanced_im)

        '''Rough segmentation'''

        histology_filename = os.path.basename(file_svg)
        aux = re.split('_row', histology_filename)
        histology_filename = aux[0] + '.ndpi'
        histology_filename = os.path.join(ndpi_dir, histology_filename)

        aux = aux[1].replace('.svg', '')
        aux = re.split('_', aux)
        row = np.int32(aux[1])
        col = np.int32(aux[3])

        # rough segmentation of the tissue in the full histology image (not just the training window)
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
            plt.contour(rough_mask_crop, colors='k')

        '''Read contours
        '''

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
        contour_type = [np.zeros(shape=(len(cell_contours),), dtype=np.uint8),  # 0: white-adipocyte
                        np.ones(shape=(len(other_contours),), dtype=np.uint8),  # 1: other types of tissue
                        np.ones(shape=(len(brown_contours),), dtype=np.uint8)]  # 1: brown cells (treated as "other" tissue)
        contour_type = np.concatenate(contour_type)

        print('Cells: ' + str(len(cell_contours)))
        print('Other: ' + str(len(other_contours)))
        print('Brown: ' + str(len(brown_contours)))

        if (len(contours) == 0):  # there are no contours

            # blank outputs for this image
            im_array = np.zeros(shape=im_crop.shape, dtype=np.uint8)
            out_class = np.zeros(shape=im_crop.shape[0:2], dtype=np.uint8)
            out_mask = np.zeros(shape=im_crop.shape[0:2], dtype=np.uint8)

        else:  # there are contours

            # initialise arrays for training
            out_class = np.zeros(shape=im_array.shape[0:2], dtype=np.uint8)
            out_mask = np.zeros(shape=im_array.shape[0:2], dtype=np.uint8)

            if DEBUG:
                plt.clf()
                plt.imshow(im_array)
                plt.scatter((im_array.shape[1] - 1) / 2.0, (im_array.shape[0] - 1) / 2.0)

            # loop ground truth cell contours
            for j, contour in enumerate(contours):

                if DEBUG:
                    plt.clf()
                    plt.imshow(im_array)
                    plt.plot([p[0] for p in contour], [p[1] for p in contour])
                    xy_c = (np.mean([p[0] for p in contour]), np.mean([p[1] for p in contour]))
                    plt.scatter(xy_c[0], xy_c[1])

                # rasterise current ground truth segmentation
                cell_seg_gtruth = Image.new("1", im_array.shape[0:2][::-1], "black")  # I = 32-bit signed integer pixels
                draw = ImageDraw.Draw(cell_seg_gtruth)
                draw.polygon(contour, outline="white", fill="white")
                cell_seg_gtruth = np.array(cell_seg_gtruth, dtype=np.bool)

                if DEBUG:
                    plt.clf()
                    plt.subplot(121)
                    plt.imshow(im_array)
                    plt.plot([p[0] for p in contour], [p[1] for p in contour])
                    xy_c = (np.mean([p[0] for p in contour]), np.mean([p[1] for p in contour]))
                    plt.scatter(xy_c[0], xy_c[1])
                    plt.subplot(122)
                    plt.imshow(im_array)
                    plt.contour(cell_seg_gtruth.astype(np.uint8))

                # add current object to training output and mask
                out_mask[cell_seg_gtruth] = 1
                out_class[cell_seg_gtruth] = contour_type[j]

            # end for j, contour in enumerate(contours):

        if DEBUG:
            plt.clf()
            plt.subplot(121)
            plt.imshow(im_array)
            plt.contour(out_mask.astype(np.uint8), colors='r')
            plt.title('Mask', fontsize=14)
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(im_array)
            plt.contour(out_class.astype(np.uint8), colors='k')
            plt.title('Class', fontsize=14)
            plt.axis('off')
            plt.tight_layout()

        # convert to expected types
        im_array = im_array.astype(np.float32)
        rough_mask_crop = rough_mask_crop.astype(np.bool)
        out_class = out_class.astype(np.float32)
        out_mask = out_mask.astype(np.float32)

        # scale image intensities from [0, 255] to [0.0, 1.0]
        im_array /= 255

        # append input/output/mask for later use in training
        im_array_all[i, ...] = im_array
        rough_mask_all[i, ...] = rough_mask_crop
        out_class_all[i, :, :, 0] = out_class
        out_mask_all[i, ...] = out_mask

        print('Time so far: ' + str("{:.1f}".format(time.time() - t0)) + ' s')

    # save results to avoid having to recompute them every time
    np.savez(data_filename, im_array_all=im_array_all, rough_mask_all=rough_mask_all, out_class_all=out_class_all,
             out_mask_all=out_mask_all)


'''
************************************************************************************************************************
Object-wise classification validation (new way)

Here we label the pixels of the image as WAT/no-WAT/unlabelled according to ground truth.

Then we apply automatic segmentation. We count the proportion of WAT pixels in each object. That value (e.g. 0.34) gets 
assigned to each pixel inside. That's then used as scores that are compared to the ground truth labelling to for the
ROC. 
************************************************************************************************************************
'''

# correct home directory in file paths
file_svg_list = cytometer.data.change_home_directory(list(file_svg_list), '/home/rcasero', home, check_isfile=True)
file_svg_list = cytometer.data.change_home_directory(list(file_svg_list), '/users/rittscher/rcasero', home, check_isfile=True)

# load data computed in the previous section
data_filename = os.path.join(saved_models_dir, experiment_id + '_data.npz')
with np.load(data_filename) as data:
    im_array_all = data['im_array_all']
    out_class_all = 1 - data['out_class_all']  # encode as 0: other, 1: WAT
    out_mask_all = data['out_mask_all']

filename_pixel_gtruth = os.path.join(saved_figures_dir, 'klf14_b6ntac_exp_0096_pixel_gtruth.npz')
if os.path.isfile(filename_pixel_gtruth):

    aux = np.load(filename_pixel_gtruth)
    pixel_gtruth_class = aux['pixel_gtruth_class']
    pixel_gtruth_prop = aux['pixel_gtruth_prop']

else:

    # start timer
    t0 = time.time()

    # init output vectors
    pixel_gtruth_class = []
    pixel_gtruth_prop = []

    for i_fold in range(len(idx_test_all)):

        print('# Fold ' + str(i_fold) + '/' + str(len(idx_test_all) - 1))

        # test image indices. These indices refer to file_list
        idx_test = idx_test_all[i_fold]

        # list of test files (used later for the dataframe)
        file_svg_list_test = np.array(file_svg_list)[idx_test]

        print('## len(idx_test) = ' + str(len(idx_test)))

        # extract testing data
        im_array_test = im_array_all[idx_test, :, :, :]
        out_class_test = out_class_all[idx_test, :, :, :]
        out_mask_test = out_mask_all[idx_test, :, :]

        # loop test images
        for i, file_svg in enumerate(file_svg_list_test):

            print('# Fold ' + str(i_fold) + '/' + str(len(idx_test_all) - 1) + ', i = '
                  + str(i) + '/' + str(len(idx_test) - 1))

            ''' Ground truth contours '''

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
            cell_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell',
                                                                    add_offset_from_filename=False,
                                                                    minimum_npoints=3)
            other_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Other',
                                                                     add_offset_from_filename=False,
                                                                     minimum_npoints=3)
            brown_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Brown',
                                                                     add_offset_from_filename=False,
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

            # create dataframe for this image
            im_idx = [idx_test_all[i_fold][i], ] * len(contours)  # absolute index of current test image
            df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_tif),
                                                                  values=im_idx, values_tag='im',
                                                                  tags_to_keep=['id', 'ko_parent', 'sex'])
            df_common['contour'] = range(len(contours))
            df_common['type'] = contour_type_all

            '''Label pixels of image as either WAT/non-WAT'''

            if DEBUG:

                plt.clf()
                plt.subplot(121)
                plt.imshow(im)
                # plt.contour(out_mask_test[i, :, :], colors='r')
                plt.axis('off')
                plt.title('Histology', fontsize=14)
                plt.subplot(122)
                plt.imshow(im)
                first_wat = True
                first_other = True
                for j, contour in enumerate(contours):
                    # close the contour for the plot
                    contour_aux = contour.copy()
                    contour_aux.append(contour[0])
                    if first_wat and contour_type_all[j] == 'wat':
                        plt.plot([p[0] for p in contour_aux], [p[1] for p in contour_aux], color='C0', linewidth=2,
                                 label='WAT contour')
                        first_wat = False
                    elif contour_type_all[j] == 'wat':
                        plt.plot([p[0] for p in contour_aux], [p[1] for p in contour_aux], color='C0', linewidth=2)
                    elif first_other and contour_type_all[j] != 'wat':
                        plt.plot([p[0] for p in contour_aux], [p[1] for p in contour_aux], color='C1', linewidth=2,
                                 label='Other contour')
                        first_other = False
                    else:
                        plt.plot([p[0] for p in contour_aux], [p[1] for p in contour_aux], color='C1', linewidth=2)
                plt.legend()
                plt.axis('off')
                plt.title('Manual contours', fontsize=14)
                plt.tight_layout()

                # plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_manual_contours.svg'))

        # names of contour, dmap and tissue classifier models
        contour_model_filename = \
            os.path.join(saved_models_dir, contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
        dmap_model_filename = \
            os.path.join(saved_models_dir, dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
        classifier_model_filename = \
            os.path.join(saved_models_dir, classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')

        # segment histology and classify pixels
        pred_seg_test, pred_class_test, _ \
            = cytometer.utils.segment_dmap_contour_v6(im_array_test,
                                                      dmap_model=dmap_model_filename,
                                                      contour_model=contour_model_filename,
                                                      classifier_model=classifier_model_filename,
                                                      border_dilation=0, batch_size=batch_size)

        # loop input images
        for i in range(len(file_svg_list_test)):

            # extract current segmentation and classification, for ease of use
            aux_seg = pred_seg_test[i, ...]
            aux_class = pred_class_test[i, :, :, 0]
            aux_prop = np.zeros(shape=aux_seg.shape, dtype=np.float32)

            # for each label, get the proportion of WAT pixels. Then, assign that value to all pixels in the label
            for lab in np.unique(pred_seg_test[i, ...]):

                # pixels in the current label
                idx = aux_seg == lab

                # proportion of WAT pixels in the label
                prop = np.count_nonzero(aux_class[idx]) / np.count_nonzero(idx)

                # transfer proportion value to all pixels of the label
                aux_prop[idx] = prop

            # transfer this image's results to output vectors
            aux_mask = out_mask_test[i, ...] > 0
            pixel_gtruth_prop.append(aux_prop[aux_mask])
            aux_gtruth = out_class_test[i, :, :, 0]
            pixel_gtruth_class.append(aux_gtruth[aux_mask])

            if DEBUG:
                plt.subplot(121)
                plt.cla()
                plt.imshow(aux_prop)
                plt.colorbar()
                plt.contour(pred_seg_test[i, ...], levels=np.unique(pred_seg_test[i, ...]), colors='w')
                plt.axis('off')

    # convert output lists to vectors
    pixel_gtruth_class = np.hstack(pixel_gtruth_class)
    pixel_gtruth_prop = np.hstack(pixel_gtruth_prop)

    if DEBUG:
        plt.clf()
        plt.boxplot((pixel_gtruth_prop[pixel_gtruth_class == 0],
                     pixel_gtruth_prop[pixel_gtruth_class == 1]))

    # save data for later
    np.savez_compressed(filename_pixel_gtruth,
                        pixel_gtruth_class=pixel_gtruth_class, pixel_gtruth_prop=pixel_gtruth_prop)

# pixel score thresholds
# ROC curve
fpr, tpr, thr = roc_curve(y_true=pixel_gtruth_class, y_score=pixel_gtruth_prop)
roc_auc = auc(fpr, tpr)

# we fix the min_class_prop threshold to what is used in the pipeline
thr_target = 0.5
tpr_target = np.interp(thr_target, thr[::-1], tpr[::-1])
fpr_target = np.interp(thr_target, thr[::-1], fpr[::-1])

# # we fix the FPR (False Positive Rate) and interpolate the TPR (True Positive Rate) on the ROC curve
# fpr_target = 0.05
# tpr_target = np.interp(fpr_target, fpr, tpr)
# thr_target = np.interp(fpr_target, fpr, thr)

if DEBUG:
    plt.clf()
    plt.plot(fpr, tpr)
    plt.scatter(fpr_target, tpr_target, color='C0', s=100,
                label='Object score thr. =  %0.2f, FPR = %0.0f%%, TPR = %0.0f%%'
                      % (thr_target, fpr_target * 100, tpr_target * 100))
    plt.tick_params(labelsize=14)
    plt.xlabel('Pixel WAT False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('Pixel WAT True Positive Rate (TPR)', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})
    plt.tight_layout()

'''
************************************************************************************************************************
Segmentation validation with pipeline v7.

Loop manual contours and find overlaps with automatically segmented contours. Compute cell areas and prop. of WAT
pixels.
************************************************************************************************************************
'''

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import scipy.stats.mstats
import more_itertools, itertools
import statsmodels.formula.api as smf

# correct home directory in file paths
file_svg_list = cytometer.data.change_home_directory(list(file_svg_list), '/home/rcasero', home, check_isfile=True)
file_svg_list = cytometer.data.change_home_directory(list(file_svg_list), '/users/rittscher/rcasero', home, check_isfile=True)

## compute and save results (you can skip this section if this has been done before, and go straight where you load the
## results)

# load data computed in the previous section
data_filename = os.path.join(saved_models_dir, experiment_id + '_data.npz')
with np.load(data_filename) as data:
    im_array_all = data['im_array_all']
    rough_mask_all = data['rough_mask_all']
    out_class_all = 1 - data['out_class_all']  # encode as 0: other, 1: WAT
    out_mask_all = data['out_mask_all']

# start timer
t0 = time.time()

# init dataframes
df_manual_all = pd.DataFrame()
df_auto_all = pd.DataFrame()

# output filenames for the loop
dataframe_manual_filename = os.path.join(saved_models_dir, experiment_id + '_test_pipeline_manual.pkl')
dataframe_auto_filename = os.path.join(saved_models_dir, experiment_id + '_test_pipeline_auto.pkl')

# this loop doesn't need to be run if it was computed already, and were saved to output files
if not os.path.isfile(dataframe_manual_filename) or not os.path.isfile(dataframe_auto_filename):

    for i_fold in range(len(idx_test_all)):

        ''' Get the images/masks/classification that were not used for training of this particular fold '''

        print('# Fold ' + str(i_fold) + '/' + str(len(idx_test_all) - 1))

        # test and training image indices. These indices refer to file_list
        idx_test = idx_test_all[i_fold]

        # list of test files (used later for the dataframe)
        file_list_test = np.array(file_svg_list)[idx_test]

        print('## len(idx_test) = ' + str(len(idx_test)))

        # split data into training and testing
        im_array_test = im_array_all[idx_test, :, :, :]
        rough_mask_test = rough_mask_all[idx_test, :, :]
        out_class_test = out_class_all[idx_test, :, :, :]
        out_mask_test = out_mask_all[idx_test, :, :]

        ''' Segmentation into non-overlapping objects '''

        # names of contour, dmap and tissue classifier models
        contour_model_filename = \
            os.path.join(saved_models_dir, contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
        dmap_model_filename = \
            os.path.join(saved_models_dir, dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
        classifier_model_filename = \
            os.path.join(saved_models_dir, classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')

        # segment histology
        pred_seg_test, pred_class_test, _ \
            = cytometer.utils.segment_dmap_contour_v6(im_array_test,
                                                      dmap_model=dmap_model_filename,
                                                      contour_model=contour_model_filename,
                                                      classifier_model=classifier_model_filename,
                                                      border_dilation=0, batch_size=batch_size)

        if DEBUG:
            i = 0
            plt.clf()
            plt.subplot(221)
            plt.cla()
            plt.imshow(im_array_test[i, :, :, :])
            plt.axis('off')
            plt.subplot(222)
            plt.cla()
            plt.imshow(im_array_test[i, :, :, :])
            plt.contourf(pred_class_test[i, :, :, 0].astype(np.float32), alpha=0.5)
            plt.axis('off')
            plt.subplot(223)
            plt.cla()
            plt.imshow(im_array_test[i, :, :, :])
            plt.contour(pred_seg_test[i, :, :], levels=np.unique(pred_seg_test[i, :, :]), colors='k')
            plt.axis('off')
            plt.subplot(224)
            plt.cla()
            plt.imshow(im_array_test[i, :, :, :])
            plt.contourf(pred_class_test[i, :, :, 0].astype(np.float32), alpha=0.5)
            plt.contour(pred_seg_test[i, :, :], levels=np.unique(pred_seg_test[i, :, :]), colors='k')
            plt.axis('off')
            plt.tight_layout()

        # clean segmentation: remove labels that are too small or that don't overlap enough with
        # the rough foreground mask
        pred_seg_test, _ \
            = cytometer.utils.clean_segmentation(pred_seg_test, min_cell_area=min_cell_area, max_cell_area=max_cell_area,
                                                 remove_edge_labels=False,
                                                 mask=rough_mask_test, min_mask_overlap=min_mask_overlap,
                                                 phagocytosis=True, labels_class=None)

        if DEBUG:
            plt.clf()
            aux = np.stack((rough_mask_test[i, :, :],) * 3, axis=2)
            plt.imshow(im_array_test[i, :, :, :] * aux)
            plt.contour(pred_seg_test[i, ...], levels=np.unique(pred_seg_test[i, ...]), colors='k')
            plt.axis('off')

        ''' Split image into individual labels and correct segmentation to take overlaps into account '''

        (window_seg_test, window_im_test, window_class_test, window_rough_mask_test), index_list, scaling_factor_list \
            = cytometer.utils.one_image_per_label_v2((pred_seg_test, im_array_test,
                                                      pred_class_test[:, :, :, 0].astype(np.uint8),
                                                      rough_mask_test.astype(np.uint8)),
                                                     resize_to=(training_window_len, training_window_len),
                                                     resample=(Image.NEAREST, Image.LINEAR, Image.NEAREST, Image.NEAREST),
                                                     only_central_label=True)

        # correct segmentations
        correction_model_filename = os.path.join(saved_models_dir,
                                                 correction_model_basename + '_model_fold_' + str(i_fold) + '.h5')
        window_seg_corrected_test = cytometer.utils.correct_segmentation(im=window_im_test, seg=window_seg_test,
                                                                         correction_model=correction_model_filename,
                                                                         model_type='-1_1', batch_size=batch_size,
                                                                         smoothing=11)

        if DEBUG:
            j = 113
            plt.clf()
            plt.imshow(window_im_test[j, ...])
            plt.contour(window_seg_test[j, ...], colors='r')
            plt.contour(window_seg_corrected_test[j, ...], colors='g')

        # loop test images
        for i in range(len(idx_test)):

            print('# Fold ' + str(i_fold) + '/' + str(len(idx_test_all) - 1) + ', i = '
                  + str(i) + '/' + str(len(idx_test) - 1))

            ''' Contours '''

            # file with the contours
            file_svg = file_list_test[i]

            # open histology testing image
            file_tif = file_svg.replace('.svg', '.tif')
            im = Image.open(file_tif)

            if DEBUG:
                # plots for poster

                # histology slice
                plt.clf()
                plt.imshow(im)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_pipeline_im_fold_%d_i_%d.svg' % (i_fold, i)),
                            bbox_inches='tight', pad_inches=0)

                # WAT classifier
                plt.clf()
                plt.imshow(pred_class_test[i, :, :, 0])
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_pipeline_pred_class_test_fold_%d_i_%d.svg' % (i_fold, i)),
                            bbox_inches='tight', pad_inches=0)

            # read pixel size information
            xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
            yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

            # read the ground truth cell contours in the SVG file. This produces a list [contour_0, ..., contour_N-1]
            # where each contour_i = [(X_0, Y_0), ..., (X_P-1, X_P-1)]
            contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False,
                                                               minimum_npoints=3)

            print('Cells: ' + str(len(contours)))
            print('')

            ''' Labels '''

            # proportion of WAT pixels in each label
            pred_lab_test, pred_prop_test = cytometer.utils.prop_of_pixels_in_label(lab=pred_seg_test[i, :, :],
                                                                                    mask=pred_class_test[i, :, :, 0])

            # auxiliary variable to make accessing the labels in this image more easily
            pred_wat_seg_test_i = pred_seg_test[i, :, :]

            # initialise dataframe to keep results: one cell per row, tagged with mouse metainformation
            df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_tif),
                                                                  values=[i_fold], values_tag='fold',
                                                                  tags_to_keep=['id', 'ko_parent', 'sex', 'genotype'])

            # cells on the edge
            labels_edge = cytometer.utils.edge_labels(pred_seg_test[i, :, :])

            ''' All automatic labels loop '''
            lab_no_edge_unique = np.unique(pred_seg_test[i, :, :])
            lab_no_edge_unique = set(lab_no_edge_unique) - {0}  # remove background label
            lab_no_edge_unique = list(np.sort(list(lab_no_edge_unique - set(labels_edge))))
            for lab in lab_no_edge_unique:

                # get the array position of the automatic segmentation
                idx = np.where([x == (i, lab) for x in index_list])[0][0]

                # area of the corrected segmentation
                (sy, sx) = scaling_factor_list[idx]
                area_corrected = np.count_nonzero(window_seg_corrected_test[idx, :, :]) * xres * yres / (sx * sy)

                # proportion of WAT pixels in the corrected segmentations
                wat_prop_corrected \
                    = np.count_nonzero(window_seg_corrected_test[idx, :, :] * window_class_test[idx, :, :]) \
                      / np.count_nonzero(window_seg_corrected_test[idx, :, :])

                # start dataframe row for this contour
                df_auto = df_common.copy()
                df_auto['im'] = i
                df_auto['lab_auto'] = lab
                df_auto['area_auto'] = np.count_nonzero(pred_seg_test[i, :, :] == lab) * xres * yres  # um^2
                df_auto['area_corrected'] = area_corrected  # um^2
                df_auto['wat_prop_auto'] = pred_prop_test[pred_lab_test == lab]
                df_auto['wat_prop_corrected'] = wat_prop_corrected
                df_auto['auto_is_edge_cell'] = lab in labels_edge

                # concatenate current row to general dataframe
                df_auto_all = df_auto_all.append(df_auto, ignore_index=True)

            ''' Only manual contours regardless of whether they have a corresponding auto label loop '''
            for j, contour in enumerate(contours):

                # start dataframe row for this contour
                df_manual = df_common.copy()
                df_manual['im'] = i
                df_manual['contour'] = j

                # manual segmentation: rasterise object described by contour
                cell_seg_contour = Image.new("1", im_array_test.shape[1:3], "black")  # I = 32-bit signed integer pixels
                draw = ImageDraw.Draw(cell_seg_contour)
                draw.polygon(contour, outline="white", fill="white")
                cell_seg_contour = np.array(cell_seg_contour, dtype=np.bool)

                # plt.contour(cell_seg_contour, colors='r')  ###########################
                # plt.pause(3)  #############################

                # find automatic segmentation that best overlaps contour
                import scipy.stats
                lab_best_overlap = stats.mode(pred_wat_seg_test_i[cell_seg_contour]).mode[0]

                if lab_best_overlap == 0:  # the manual contour overlaps the background more than any automatic segmentation
                    warnings.warn('Skipping. Contour j = ' + str(j) + ' overlaps with background segmentation lab = 0')

                    # add to dataframe
                    df_manual['lab_auto'] = 0
                    df_manual['dice_auto'] = np.NaN
                    df_manual['area_manual'] = np.count_nonzero(cell_seg_contour) * xres * yres  # um^2
                    df_manual['area_auto'] = np.NaN
                    df_manual['area_corrected'] = np.NaN
                    df_manual['wat_prop_auto'] = np.NaN
                    df_manual['wat_prop_corrected'] = np.NaN
                    df_manual['auto_is_edge_cell'] = np.NaN

                else:

                    # get the array position of the automatic segmentation
                    idx = np.where([x == (i, lab_best_overlap) for x in index_list])[0][0]

                    # compute Dice coefficient between manual and automatic segmentation with best overlap
                    cell_best_overlap = pred_wat_seg_test_i == lab_best_overlap  # auto segmentation (best match)
                    # cell_seg_contour  # manual segmentation
                    intersect_auto_manual = cell_best_overlap * cell_seg_contour
                    dice_auto = 2 * np.count_nonzero(intersect_auto_manual) \
                                / (np.count_nonzero(cell_best_overlap) + np.count_nonzero(cell_seg_contour))

                    # crop the manual contour, using the same cropping window that was used for the automatic contour
                    (window_auto, window_manual), _, scaling_factor \
                        = cytometer.utils.one_image_per_label_v2((np.expand_dims(cell_best_overlap, axis=0).astype(np.uint8),
                                                                  np.expand_dims(cell_seg_contour, axis=0).astype(np.uint8)),
                                                                 resize_to=(training_window_len, training_window_len),
                                                                 resample=(Image.NEAREST, Image.NEAREST),
                                                                 only_central_label=True)

                    if DEBUG:
                        # plot for poster

                        # segmentation labels
                        plt.clf()
                        plt.imshow(pred_seg_test[i, :, :])
                        plt.contour(pred_seg_test[i, :, :] == lab_best_overlap, colors='r', linewidths=4)
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(os.path.join(saved_figures_dir,
                                                 'exp_0096_pipeline_labels_fold_%d_i_%d_j_%d.svg' % (i_fold, i, j)),
                                    bbox_inches='tight', pad_inches=0)

                        # one cell segmentations: manual, auto, corrected
                        plt.clf()
                        plt.imshow(window_im_test[idx, ...])
                        cntr1 = plt.contour(window_manual[0, :, :], linewidths=5, colors='g', linestyles='solid')
                        cntr2 = plt.contour(window_auto[0, :, :], linewidths=3, colors='r', linestyles='dotted')
                        cntr3 = plt.contour(window_seg_corrected_test[idx, :, :], linewidths=3, colors='k', linestyles='dotted')
                        h1, _ = cntr1.legend_elements()
                        h2, _ = cntr2.legend_elements()
                        h3, _ = cntr3.legend_elements()
                        plt.legend([h1[0], h2[0], h3[0]],
                                   ['Manual (%0.0f $\mu$m$^2$)' % (np.count_nonzero(window_manual[0, :, :]) * xres * yres),
                                    'Auto (%0.0f $\mu$m$^2$)' % (np.count_nonzero(window_auto[0, :, :]) * xres * yres),
                                    'Corrected (%0.0f $\mu$m$^2$)' % (np.count_nonzero(window_seg_corrected_test[idx, :, :]) * xres * yres)],
                                   fontsize=12)
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(os.path.join(saved_figures_dir,
                                                 'exp_0096_pipeline_window_seg_fold_%d_i_%d_j_%d.svg' % (i_fold, i, j)),
                                    bbox_inches='tight', pad_inches=0)

                    # double-check that the cropping of the automatic segmentation we get here is the same we got before the
                    # j loop
                    assert(np.all(window_auto[0, :, :] == window_seg_test[idx, :, :]))

                    # proportion of WAT pixels in the corrected segmentations
                    wat_prop_corrected \
                        = np.count_nonzero(window_seg_corrected_test[idx, :, :] * window_class_test[idx, :, :]) \
                    / np.count_nonzero(window_seg_corrected_test[idx, :, :])

                    # compute Dice coefficient between manual and corrected segmentation with best overlap
                    a = np.count_nonzero(window_manual[0, :, :])
                    b = np.count_nonzero(window_seg_corrected_test[idx, :, :])
                    a_n_b = np.count_nonzero(window_manual[0, :, :] * window_seg_corrected_test[idx, :, :])
                    dice_corrected = 2 * a_n_b / (a + b)

                    # area of the corrected segmentation
                    (sy, sx) = scaling_factor_list[idx]
                    area_corrected = np.count_nonzero(window_seg_corrected_test[idx, :, :]) * xres * yres / (sx * sy)

                    # add to dataframe
                    df_manual['label_seg'] = lab_best_overlap
                    df_manual['dice_auto'] = dice_auto
                    df_manual['dice_corrected'] = dice_corrected
                    df_manual['area_manual'] = np.count_nonzero(cell_seg_contour) * xres * yres  # um^2
                    df_manual['area_auto'] = np.count_nonzero(cell_best_overlap) * xres * yres  # um^2
                    df_manual['area_corrected'] = area_corrected  # um^2
                    df_manual['wat_prop_auto'] = pred_prop_test[pred_lab_test == lab_best_overlap]
                    df_manual['wat_prop_corrected'] = wat_prop_corrected
                    df_manual['auto_is_edge_cell'] = lab_best_overlap in labels_edge

                    if DEBUG:
                        plt.clf()
                        plt.subplot(121)
                        plt.imshow(window_im_test[idx, ...])
                        cntr1 = plt.contour(window_manual[0, :, :], colors='g')
                        cntr2 = plt.contour(window_seg_test[idx, ...], colors='k')
                        cntr3 = plt.contour(window_seg_corrected_test[idx, ...], colors='r')
                        h1, _ = cntr1.legend_elements()
                        h2, _ = cntr2.legend_elements()
                        h3, _ = cntr3.legend_elements()
                        plt.legend([h1[0], h2[0], h3[0]], ['Manual', 'Auto', 'Corrected'])
                        plt.title('Histology and segmentation', fontsize=14)
                        plt.axis('off')
                        plt.subplot(122)
                        plt.imshow(window_class_test[idx, ...].astype(np.uint8), vmin=0, vmax=1)
                        plt.contour(window_manual[0, :, :], colors='g')
                        plt.contour(window_seg_test[idx, ...], colors='k')
                        plt.contour(window_seg_corrected_test[idx, ...], colors='r')
                        plt.title('Prop$_{\mathrm{WAT, corrected}}$ = %0.1f%%'
                                  % (100 * df_manual['wat_prop_auto']), fontsize=14)
                        plt.axis('off')
                        plt.tight_layout()

                # concatenate current row to general dataframe
                df_manual_all = df_manual_all.append(df_manual, ignore_index=True)

        # clear keras session to prevent each segmentation iteration from getting slower. Note that this forces us to
        # reload the models every time
        K.clear_session()

        # end of fold loop
        print('Time so far: ' + str("{:.1f}".format(time.time() - t0)) + ' s')

    # save results to avoid having to recompute them every time (1.4 h on 2 Titan RTX GPUs)
    df_manual_all.to_pickle(dataframe_manual_filename)
    df_auto_all.to_pickle(dataframe_auto_filename)

## Analyse results: Manual data

# load dataframe with manual segmentations matched or not to automatic segmentations
df_manual_all = pd.read_pickle(dataframe_manual_filename)

# study of hand traced areas, all of them
manual_quart = scipy.stats.mstats.hdquantiles(df_manual_all['area_manual'], prob=[0.25, 0.50, 0.75]).data
print('Hand traced cells:')
print('min = ' + str(np.min(df_manual_all['area_manual'])) + ' um^2')
print('max = ' + str(np.max(df_manual_all['area_manual'])) + ' um^2')
print('quartiles = ' + str(manual_quart) + ' um^2')
# print('min = ' + str(np.min(df_manual_all['area_manual']) / (xres * yres)) + ' pixel')
# print('max = ' + str(np.max(df_manual_all['area_manual']) / (xres * yres)) + ' pixel')
# print('quartiles = ' + str(manual_quart / (xres * yres)) + ' pixel')

# cells where there's a manual and automatic segmentation reasonable overlap, even if it's poor
idx_manual_auto_overlap = df_manual_all['dice_auto'] > 0.5  # this ignores NaNs

# boxplots of manual/auto/corrected areas. This is just a sanity check. Note that this only includes automatic
# segmentations that already overlap with manual segmentations, so the results should be quite good
plt.clf()
bp = plt.boxplot((df_manual_all['area_manual'][idx_manual_auto_overlap] / 1e3,
                  df_manual_all['area_auto'][idx_manual_auto_overlap] / 1e3,
                  df_manual_all['area_corrected'][idx_manual_auto_overlap] / 1e3),
                 positions=[1, 2, 3], notch=True, labels=['Hand traced', 'Auto', 'Corrected'])

# points of interest from the boxplots
bp_poi = cytometer.utils.boxplot_poi(bp)

plt.plot([0.75, 3.25], [bp_poi[0, 2], ] * 2, 'C1', linestyle='dotted')  # manual median
plt.plot([0.75, 3.25], [bp_poi[0, 1], ] * 2, 'k', linestyle='dotted')  # manual Q1
plt.plot([0.75, 3.25], [bp_poi[0, 3], ] * 2, 'k', linestyle='dotted')  # manual Q3
plt.tick_params(axis="both", labelsize=14)
plt.ylabel('Area ($\cdot 10^{3} \mu$m$^2$)', fontsize=14)
plt.ylim(-700 / 1e3, 10000 / 1e3)
plt.tight_layout()

# manual quartile values
plt.text(1.20, bp_poi[0, 3] + .1, '%0.1f' % (bp_poi[0, 3]), fontsize=12, color='k')
plt.text(1.20, bp_poi[0, 2] + .1, '%0.1f' % (bp_poi[0, 2]), fontsize=12, color='C1')
plt.text(1.20, bp_poi[0, 1] + .1, '%0.1f' % (bp_poi[0, 1]), fontsize=12, color='k')

# auto quartile values
plt.text(2.20, bp_poi[1, 3] + .1 - .3, '%0.1f' % (bp_poi[1, 3]), fontsize=12, color='k')
plt.text(2.20, bp_poi[1, 2] + .1 - .3, '%0.1f' % (bp_poi[1, 2]), fontsize=12, color='C1')
plt.text(2.20, bp_poi[1, 1] + .1 - .4, '%0.1f' % (bp_poi[1, 1]), fontsize=12, color='k')

# corrected quartile values
plt.text(3.20, bp_poi[2, 3] + .1 - .1, '%0.1f' % (bp_poi[2, 3]), fontsize=12, color='k')
plt.text(3.20, bp_poi[2, 2] + .1 + .0, '%0.1f' % (bp_poi[2, 2]), fontsize=12, color='C1')
plt.text(3.20, bp_poi[2, 1] + .1 + .0, '%0.1f' % (bp_poi[2, 1]), fontsize=12, color='k')

plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_boxplots_manual_dataset.svg'))
plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_boxplots_manual_dataset.png'))

# Wilcoxon sign-ranked tests of whether manual areas are significantly different to auto/corrected areas
idx = idx_manual_auto_overlap
print('Manual mean ± std = ' + str(np.mean(df_manual_all['area_manual'][idx])) + ' ± '
      + str(np.std(df_manual_all['area_manual'][idx])))
print('Auto mean ± std = ' + str(np.mean(df_manual_all['area_auto'][idx])) + ' ± '
      + str(np.std(df_manual_all['area_auto'][idx])))
print('Corrected mean ± std = ' + str(np.mean(df_manual_all['area_corrected'][idx])) + ' ± '
      + str(np.std(df_manual_all['area_corrected'][idx])))

# Wilcoxon signed-rank test to check whether the medians are significantly different
w, p = scipy.stats.wilcoxon(df_manual_all['area_manual'][idx_manual_auto_overlap],
                            df_manual_all['area_auto'][idx_manual_auto_overlap])
print('Manual vs. auto, W = ' + str(w) + ', p = ' + str(p))

w, p = scipy.stats.wilcoxon(df_manual_all['area_manual'][idx_manual_auto_overlap],
                            df_manual_all['area_corrected'][idx_manual_auto_overlap])
print('Manual vs. corrected, W = ' + str(w) + ', p = ' + str(p))


# boxplots of area error
plt.clf()
bp = plt.boxplot((df_manual_all['area_auto'][idx_manual_auto_overlap] / 1e3
                  - df_manual_all['area_manual'][idx_manual_auto_overlap] / 1e3,
                  df_manual_all['area_corrected'][idx_manual_auto_overlap] / 1e3
                  - df_manual_all['area_manual'][idx_manual_auto_overlap] / 1e3),
                 positions=[1, 2], notch=True, labels=['Auto -\nHand traced', 'Corrected -\nHand traced'])

plt.plot([0.75, 2.25], [0, 0], 'k', 'linewidth', 2)
plt.xlim(0.5, 2.5)
plt.ylim(-1.4, 1.1)

# points of interest from the boxplots
bp_poi = cytometer.utils.boxplot_poi(bp)

# manual quartile values
plt.text(1.10, bp_poi[0, 2], '%0.2f' % (bp_poi[0, 2]), fontsize=12, color='C1')
plt.text(2.10, bp_poi[1, 2], '%0.2f' % (bp_poi[1, 2]), fontsize=12, color='C1')

plt.tick_params(axis="both", labelsize=14)
plt.ylabel('Area error ($\cdot 10^{3} \mu$m$^2$)', fontsize=14)
plt.tight_layout()

plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_error_boxplots_manual_dataset.svg'))
plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_error_boxplots_manual_dataset.png'))

# boxplots of area error
plt.clf()
bp = plt.boxplot((df_manual_all['area_auto'][idx_manual_auto_overlap] / 1e3
                  - df_manual_all['area_manual'][idx_manual_auto_overlap] / 1e3,
                  df_manual_all['area_corrected'][idx_manual_auto_overlap] / 1e3
                  - df_manual_all['area_manual'][idx_manual_auto_overlap] / 1e3),
                 positions=[1, 2], notch=True, labels=['Auto -\nHand traced', 'Corrected -\nHand traced'])

plt.plot([0.75, 2.25], [0, 0], 'k', 'linewidth', 2)
plt.xlim(0.5, 2.5)
plt.ylim(-1.4, 1.1)

# points of interest from the boxplots
bp_poi = cytometer.utils.boxplot_poi(bp)

# manual quartile values
plt.text(1.10, bp_poi[0, 2], '%0.2f' % (bp_poi[0, 2]), fontsize=12, color='C1')
plt.text(2.10, bp_poi[1, 2], '%0.2f' % (bp_poi[1, 2]), fontsize=12, color='C1')

plt.tick_params(axis="both", labelsize=14)
plt.ylabel('Area error ($\cdot 10^{3} \mu$m$^2$)', fontsize=14)
plt.tight_layout()

plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_error_boxplots_manual_dataset.svg'))
plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_error_boxplots_manual_dataset.png'))

# compute Dice coefficient mean, median, std
print('Auto')
print('Median: ' + str(np.median(df_manual_all['dice_auto'][idx_manual_auto_overlap])))
print('Mean: ' + str(np.mean(df_manual_all['dice_auto'][idx_manual_auto_overlap])))
print('Std: ' + str(np.std(df_manual_all['dice_auto'][idx_manual_auto_overlap])))
print('Corrected')
print('Median: ' + str(np.median(df_manual_all['dice_corrected'][idx_manual_auto_overlap])))
print('Mean: ' + str(np.mean(df_manual_all['dice_corrected'][idx_manual_auto_overlap])))
print('Std: ' + str(np.std(df_manual_all['dice_corrected'][idx_manual_auto_overlap])))

# compute median and std
auto_err_med = scipy.stats.mstats.hdquantiles(df_manual_all['area_auto'][idx_manual_auto_overlap] / 1e3
                  - df_manual_all['area_manual'][idx_manual_auto_overlap] / 1e3, prob=0.5).data[0]
auto_err_med_std = scipy.stats.mstats.hdquantiles_sd(df_manual_all['area_auto'][idx_manual_auto_overlap] / 1e3
                  - df_manual_all['area_manual'][idx_manual_auto_overlap] / 1e3, prob=0.5).data[0]
corrected_err_med = scipy.stats.mstats.hdquantiles(df_manual_all['area_corrected'][idx_manual_auto_overlap] / 1e3
                  - df_manual_all['area_manual'][idx_manual_auto_overlap] / 1e3, prob=0.5).data[0]
corrected_err_med_std = scipy.stats.mstats.hdquantiles_sd(df_manual_all['area_corrected'][idx_manual_auto_overlap] / 1e3
                  - df_manual_all['area_manual'][idx_manual_auto_overlap] / 1e3, prob=0.5).data[0]
print('Segmentation error')
print('Auto - hand traced = ' + str(auto_err_med) + ' 土 ' + str(auto_err_med_std) + ' um^2')
print('Corrected - hand traced = ' + str(corrected_err_med) + ' 土 ' + str(corrected_err_med_std) + ' um^2')


# 2D density plot
if DEBUG:
    plt.clf()
    sns.kdeplot(df_manual_all['area_manual'][idx_manual_auto_overlap],
                df_manual_all['area_auto'][idx_manual_auto_overlap] / df_manual_all['area_manual'][idx_manual_auto_overlap],
                cmap="Reds", shade=True)
    sns.kdeplot(df_manual_all['area_manual'][idx_manual_auto_overlap],
                df_manual_all['area_corrected'][idx_manual_auto_overlap] / df_manual_all['area_manual'][idx_manual_auto_overlap],
                cmap="Blues", shade=True)


# area vs. WAT proportion
if DEBUG:
    plt.clf()
    plt.scatter(df_manual_all['wat_prop_auto'], df_manual_all['area_auto'], s=4)
    plt.xlabel('Prop. WAT pixels', fontsize=14)
    plt.ylabel('Area ($\mu$m$^2$)', fontsize=14)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()

# histogram of area correction factor

if DEBUG:
    # objects selected for the plot
    idx = idx_manual_auto_overlap

    # median and std of ratios
    q1_auto = np.quantile(df_manual_all['area_auto'][idx] / df_manual_all['area_manual'][idx], 0.25)
    q1_corrected = np.quantile(df_manual_all['area_corrected'][idx] / df_manual_all['area_manual'][idx], 0.25)
    med_auto = np.median(df_manual_all['area_auto'][idx] / df_manual_all['area_manual'][idx])
    med_corrected = np.median(df_manual_all['area_corrected'][idx] / df_manual_all['area_manual'][idx])
    q3_auto = np.quantile(df_manual_all['area_auto'][idx] / df_manual_all['area_manual'][idx], 0.75)
    q3_corrected = np.quantile(df_manual_all['area_corrected'][idx] / df_manual_all['area_manual'][idx], 0.75)

    plt.clf()
    plt.hist(df_manual_all['area_auto'][idx] / df_manual_all['area_manual'][idx], bins=51, histtype='step', linewidth=3,
             density=True,
             label='Auto / Hand traced area\nQ1, Q2, Q3 = %0.2f, %0.2f, %0.2f'
                   % (q1_auto, med_auto, q3_auto))
    plt.hist(df_manual_all['area_corrected'][idx] / df_manual_all['area_manual'][idx], bins=51, histtype='step', linewidth=3,
             density=True,
             label='Corrected / Hand traced area\nQ1, Q2, Q3 = %0.2f, %0.2f, %0.2f'
                   % (q1_corrected, med_corrected, q3_corrected))
    plt.plot([1, 1], [0, 3.5], 'k', linewidth=2)
    plt.legend(fontsize=12)
    plt.xlabel('Pipeline / Hand traced area', fontsize=14)
    plt.ylabel('Histogram density', fontsize=14)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_areas_ratio_hist.svg'))
    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_areas_ratio_hist.png'))

if DEBUG:
    plt.clf()
    plt.boxplot(df_manual_all['area_manual'][idx] - df_manual_all['area_auto'][idx])

## median and CI of segmentation auto area error vs. hand traced area
## Note: If we perform a sign test to see whether the median = 0, we would assume a binomial distribution of number of
## values < median, and with a Gaussian approximation to the binomial distribution, we'd be performing a normal null
## hypothesis test. which corresponds to a CI-95% of -1.96*std, +1.96*std around the median value.
## https://youtu.be/dLTvZUrs-CI?t=463

df_manual_all['area_auto_manual_diff'] = df_manual_all['area_auto'] - df_manual_all['area_manual']

# sort manual areas from smallest to largest
idx = np.argsort(df_manual_all['area_manual'])
x = np.array(df_manual_all['area_manual'][idx])
y = np.array(df_manual_all['area_auto_manual_diff'][idx])

# remove NaNs
idx = ~np.isnan(x) & ~np.isnan(y)
x = x[idx]
y = y[idx]

# bin the points so that each bin has the same number of points (roughly)
median_window_size = 150
padding = [None] * (median_window_size - 6)
idx_split = more_itertools.windowed(itertools.chain(padding, range(len(x)), padding), n=median_window_size)
x_bins = [scipy.stats.mstats.hdquantiles(x[np.array(idx)[~np.equal(idx, None)].astype(np.int)],
                                         prob=[0.50], axis=0).data[0] for idx in idx_split]

idx_split = more_itertools.windowed(itertools.chain(padding, range(len(y)), padding), n=median_window_size)
y_bins_q2 = [scipy.stats.mstats.hdquantiles(y[np.array(idx)[~np.equal(idx, None)].astype(np.int)],
                                            prob=[0.50], axis=0).data[0] for idx in idx_split]

idx_split = more_itertools.windowed(itertools.chain(padding, range(len(y)), padding), n=median_window_size)
y_bins_q2_std = [scipy.stats.mstats.hdquantiles_sd(y[np.array(idx)[~np.equal(idx, None)].astype(np.int)],
                                                   prob=[0.50], axis=0).data[0] for idx in idx_split]

x_bins = np.array(x_bins)
y_bins_q2 = np.array(y_bins_q2)
y_bins_q2_std = np.array(y_bins_q2_std)

# 95% confidence interval for the median estimate
y_bins_ci_lo = y_bins_q2 - 1.96 * y_bins_q2_std
y_bins_ci_hi = y_bins_q2 + 1.96 * y_bins_q2_std

if DEBUG:
    plt.clf()
    plt.plot([x_bins[0] * 1e-3, x_bins[-1] * 1e-3], [0, 0], 'k')
    plt.plot(x_bins * 1e-3, 100 * (y_bins_q2 / x_bins - 1) , 'r')

if DEBUG:
    plt.clf()
    plt.plot([x_bins[0] * 1e-3, x_bins[-1] * 1e-3], [0, 0], 'k')
    plt.scatter(df_manual_all['area_manual'] * 1e-3, df_manual_all['area_auto_manual_diff'] * 1e-3, s=1)
    plt.fill_between(x_bins * 1e-3, y_bins_ci_lo * 1e-3, y_bins_ci_hi * 1e-3, facecolor='r', alpha=0.5)
    plt.plot(x_bins * 1e-3, y_bins_q2 * 1e-3, 'r')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Area$_{ht}$ ($10^3\ \mu m^2$)', fontsize=14)
    plt.ylabel('Area$_{auto}$ - Area$_{ht}$ ($10^3\ \mu m^2$)', fontsize=14)
    plt.xlim(x_bins[0] * 1e-3, x_bins[-1] * 1e-3)
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_auto_manual_error.svg'))
    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_auto_manual_error.png'))

    # zoom in
    plt.xlim(-0.25, 14)
    plt.ylim(-2.6, 2.5)
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_auto_manual_error_zoom.svg'))
    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_auto_manual_error_zoom.png'))

# auto area: plot area error as ratio
df_ols = pd.DataFrame()
df_ols['area_manual_bins'] = x_bins
df_ols['area_auto_manual_diff_bins_q2'] = y_bins_q2
df_ols['area_auto_diff_ratio_bins_q2'] = df_ols['area_auto_manual_diff_bins_q2'] / df_ols['area_manual_bins']

# median of median error ratios for cells > 950 um^2
area_error_ratio_q2 = np.median(df_ols['area_auto_diff_ratio_bins_q2'][df_ols['area_manual_bins'] > 950])

# range of error ratios where we consider the error acceptable
area_error_ratio_q2_lo = area_error_ratio_q2 - 10 / 100
area_error_ratio_q2_hi = area_error_ratio_q2 + 10 / 100

# plot
plt.clf()
plt.scatter(df_manual_all['area_manual'] * 1e-3,
            df_manual_all['area_auto_manual_diff'] / df_manual_all['area_manual'] * 100, s=1)
plt.plot([df_ols['area_manual_bins'].iloc[0] * 1e-3, df_ols['area_manual_bins'].iloc[-1] * 1e-3],
         [area_error_ratio_q2 * 100, area_error_ratio_q2 * 100], 'k', linewidth=2)
plt.plot([df_ols['area_manual_bins'].iloc[0] * 1e-3, df_ols['area_manual_bins'].iloc[-1] * 1e-3],
         [area_error_ratio_q2_lo * 100, area_error_ratio_q2_lo * 100], 'k--', linewidth=2)
plt.plot([df_ols['area_manual_bins'].iloc[0] * 1e-3, df_ols['area_manual_bins'].iloc[-1] * 1e-3],
         [area_error_ratio_q2_hi * 100, area_error_ratio_q2_hi * 100], 'k--', linewidth=2)
plt.plot(df_ols['area_manual_bins'] * 1e-3, df_ols['area_auto_diff_ratio_bins_q2'] * 100, 'r')
plt.fill_between(df_ols['area_manual_bins'] * 1e-3,
                 y_bins_ci_lo / df_ols['area_manual_bins'] * 100, y_bins_ci_hi / df_ols['area_manual_bins'] * 100,
                 facecolor='r', alpha=0.5)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel('Area$_{ht}$ ($10^3\ \mu m^2$)', fontsize=14)
plt.ylabel('Area$_{auto}$ / Area$_{ht}$ - 1 (%)', fontsize=14)
plt.xlim(-0.25, 11)
plt.ylim(-75, 100)
plt.tight_layout()

plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_auto_diff_ratio_error.svg'))
plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_auto_diff_ratio_error.png'))

# zoom in
plt.ylim(-22, 18)
plt.tight_layout()

plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_auto_diff_ratio_error_zoom.svg'))
plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_auto_diff_ratio_error_zoom.png'))

# compute what proportion of cells are poorly segmented
ecdf = sm.distributions.empirical_distribution.ECDF(df_manual_all['area_manual'])
cell_area_threshold = 780
print('Unusuable segmentations = ' + str(ecdf(cell_area_threshold)))


## median and CI of segmentation corrected area error vs. hand traced area
## Note: If we perform a sign test to see whether the median = 0, we would assume a binomial distribution of number of
## values < median, and with a Gaussian approximation to the binomial distribution, we'd be performing a normal null
## hypothesis test. which corresponds to a CI-95% of -1.96*std, +1.96*std around the median value.
## https://youtu.be/dLTvZUrs-CI?t=463

df_manual_all['area_corrected_manual_diff'] = df_manual_all['area_corrected'] - df_manual_all['area_manual']

# sort manual areas from smallest to largest
idx = np.argsort(df_manual_all['area_manual'])
x = np.array(df_manual_all['area_manual'][idx])
y = np.array(df_manual_all['area_corrected_manual_diff'][idx])

# remove NaNs
idx = ~np.isnan(x) & ~np.isnan(y)
x = x[idx]
y = y[idx]

# bin the points so that each bin has the same number of points (roughly)
median_window_size = 150
padding = [None] * (median_window_size - 6)
idx_split = more_itertools.windowed(itertools.chain(padding, range(len(x)), padding), n=median_window_size)
x_bins = [scipy.stats.mstats.hdquantiles(x[np.array(idx)[~np.equal(idx, None)].astype(np.int)],
                                         prob=[0.50], axis=0).data[0] for idx in idx_split]

idx_split = more_itertools.windowed(itertools.chain(padding, range(len(y)), padding), n=median_window_size)
y_bins_q2 = [scipy.stats.mstats.hdquantiles(y[np.array(idx)[~np.equal(idx, None)].astype(np.int)],
                                            prob=[0.50], axis=0).data[0] for idx in idx_split]

idx_split = more_itertools.windowed(itertools.chain(padding, range(len(y)), padding), n=median_window_size)
y_bins_q2_std = [scipy.stats.mstats.hdquantiles_sd(y[np.array(idx)[~np.equal(idx, None)].astype(np.int)],
                                                   prob=[0.50], axis=0).data[0] for idx in idx_split]

x_bins = np.array(x_bins)
y_bins_q2 = np.array(y_bins_q2)
y_bins_q2_std = np.array(y_bins_q2_std)

# 95% confidence interval for the median estimate
y_bins_ci_lo = y_bins_q2 - 1.96 * y_bins_q2_std
y_bins_ci_hi = y_bins_q2 + 1.96 * y_bins_q2_std

# plot for paper
if DEBUG:
    plt.clf()
    plt.plot([x_bins[0] * 1e-3, x_bins[-1] * 1e-3], [0, 0], 'k')
    plt.plot(x_bins * 1e-3, 100 * (y_bins_q2 / x_bins - 1) , 'r')

# plot for paper
if DEBUG:
    plt.clf()
    plt.plot([x_bins[0] * 1e-3, x_bins[-1] * 1e-3], [0, 0], 'k')
    plt.scatter(df_manual_all['area_manual'] * 1e-3, df_manual_all['area_corrected_manual_diff'] * 1e-3, s=1)
    plt.fill_between(x_bins * 1e-3, y_bins_ci_lo * 1e-3, y_bins_ci_hi * 1e-3, facecolor='r', alpha=0.5)
    plt.plot(x_bins * 1e-3, y_bins_q2 * 1e-3, 'r')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Area$_{ht}$ ($10^3\ \mu m^2$)', fontsize=14)
    plt.ylabel('Area$_{auto}$ - Area$_{ht}$ ($10^3\ \mu m^2$)', fontsize=14)
    plt.xlim(x_bins[0] * 1e-3, x_bins[-1] * 1e-3)
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_corrected_manual_error.svg'))
    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_corrected_manual_error.png'))

    plt.ylim(-0.5, 2.5)
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_corrected_manual_error_zoom.svg'))
    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_corrected_manual_error_zoom.png'))

# corrected area: plot area error as ratio
df_ols = pd.DataFrame()
df_ols['area_manual_bins'] = x_bins
df_ols['area_corrected_manual_diff_bins_q2'] = y_bins_q2
df_ols['area_corrected_diff_ratio_bins_q2'] = df_ols['area_corrected_manual_diff_bins_q2'] / df_ols['area_manual_bins']

# median of median error ratios for cells > 950 um^2
area_error_ratio_q2 = np.median(df_ols['area_corrected_diff_ratio_bins_q2'][df_ols['area_manual_bins'] > 950])

# range of error ratios where we consider the error acceptable
area_error_ratio_q2_lo = area_error_ratio_q2 - 10 / 100
area_error_ratio_q2_hi = area_error_ratio_q2 + 10 / 100

# plot
plt.clf()
plt.scatter(df_manual_all['area_manual'] * 1e-3,
            df_manual_all['area_corrected_manual_diff'] / df_manual_all['area_manual'] * 100, s=1)
plt.plot([df_ols['area_manual_bins'].iloc[0] * 1e-3, df_ols['area_manual_bins'].iloc[-1] * 1e-3],
         [area_error_ratio_q2 * 100, area_error_ratio_q2 * 100], 'k', linewidth=2)
plt.plot([df_ols['area_manual_bins'].iloc[0] * 1e-3, df_ols['area_manual_bins'].iloc[-1] * 1e-3],
         [area_error_ratio_q2_lo * 100, area_error_ratio_q2_lo * 100], 'k--', linewidth=2)
plt.plot([df_ols['area_manual_bins'].iloc[0] * 1e-3, df_ols['area_manual_bins'].iloc[-1] * 1e-3],
         [area_error_ratio_q2_hi * 100, area_error_ratio_q2_hi * 100], 'k--', linewidth=2)
plt.plot(df_ols['area_manual_bins'] * 1e-3, df_ols['area_corrected_diff_ratio_bins_q2'] * 100, 'r')
plt.fill_between(df_ols['area_manual_bins'] * 1e-3,
                 y_bins_ci_lo / df_ols['area_manual_bins'] * 100, y_bins_ci_hi / df_ols['area_manual_bins'] * 100,
                 facecolor='r', alpha=0.5)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel('Area$_{ht}$ ($10^3\ \mu m^2$)', fontsize=14)
plt.ylabel('Area$_{corrected}$ / Area$_{ht}$ - 1 (%)', fontsize=14)
plt.xlim(-0.25, 11)
plt.ylim(-75, 100)
plt.tight_layout()

plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_corrected_diff_ratio_error.svg'))
plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_corrected_diff_ratio_error.png'))

# zoom in
plt.ylim(-10, 20)
plt.tight_layout()

plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_corrected_diff_ratio_error_zoom.svg'))
plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_corrected_diff_ratio_error_zoom.png'))

# compute what proportion of cells are poorly segmented
ecdf = sm.distributions.empirical_distribution.ECDF(df_manual_all['area_manual'])
cell_area_threshold = 780
print('Unusuable segmentations = ' + str(ecdf(cell_area_threshold)))

## old code

# # test whether area errors are similar in different stratifications of the data ...
# model = sm.formula.ols('area_corrected_manual ~ C(genotype) + C(ko_parent) + C(sex)', data=df_manual_all, subset=idx).fit()
# print(model.summary())
#
# # ... PAT vs. MAT
# model = sm.formula.ols('area_corrected_manual ~ area_manual', data=df_manual_all, subset=idx).fit()
# print(model.summary())

# # cells larger than 1250 um^2
# cell_area_threshold = 1250
# idx = np.where(df_ols['area_manual_bins'] >= cell_area_threshold)[0]
# model = sm.formula.ols('area_auto_ratio_bins ~ area_manual_bins', data=df_ols, subset=idx).fit()
# print(model.summary())
#
# print('R^2 = ' + str(model.rsquared) + ', F(' + str(model.df_model)
#       + ', ' + str(model.df_resid) + ') = ' + str(model.fvalue) + ', p = ' + str(model.f_pvalue))
# print('beta = ' + str(model.params['area_manual_bins']) + ' 土 ' + str(model.bse['area_manual_bins']))
#
# y_right = model.predict(df_ols['area_manual_bins'])
#
# # cells between 740 and 1800 um^2
# idx = np.where((df_ols['area_manual_bins'] >= 740)
#             & (df_ols['area_manual_bins'] <= 1220))[0]
# model = sm.formula.ols('area_auto_ratio_bins ~ area_manual_bins', data=df_ols, subset=idx).fit()
# print(model.summary())
#
# print('R^2 = ' + str(model.rsquared) + ', F(' + str(model.df_model)
#       + ', ' + str(model.df_resid) + ') = ' + str(model.fvalue) + ', p = ' + str(model.f_pvalue))
# print('beta = ' + str(model.params['area_manual_bins']) + ' 土 ' + str(model.bse['area_manual_bins']))
#
# y_left = model.predict(df_ols['area_manual_bins'])




# linear regression

# objects selected for the plot
idx = idx_manual_auto_overlap

# linear regressions
slope_manual_auto, intercept_manual_auto, \
r_value_manual_auto, p_value_manual_auto, std_err_manual_auto = \
    stats.linregress(df_manual_all['area_manual'][idx],
                     df_manual_all['area_auto'][idx])
slope_manual_corrected, intercept_manual_corrected, \
r_value_manual_corrected, p_value_manual_corrected, std_err_manual_corrected = \
    stats.linregress(df_manual_all['area_manual'][idx],
                     df_manual_all['area_corrected'][idx])

plt.clf()
plt.plot([0, 20], [0, 20], 'g',
         path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()],
         label=r'Identity ($\alpha=$ 1.00, $\beta=$0.00)')
plt.scatter(df_manual_all['area_manual'][idx] / 1e3,
            df_manual_all['area_auto'][idx] / 1e3, s=4, color='C0')
plt.plot([0, 20], (np.array([0, 20e3]) * slope_manual_auto + intercept_manual_auto) / 1e3, color='C0',
         path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()],
         label=r'Auto ($\alpha=$ %0.2f, $\beta=$ %0.2f)' % (slope_manual_auto, intercept_manual_auto))
plt.scatter(df_manual_all['area_manual'][idx] / 1e3,
            df_manual_all['area_corrected'][idx] / 1e3, s=4, color='C1')
plt.plot([0, 20], (np.array([0, 20e3]) * slope_manual_corrected + intercept_manual_corrected) / 1e3, color='C1',
         path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()],
         label=r'Corrected ($\alpha=$ %0.2f, $\beta=$ %0.2f)' % (slope_manual_corrected, intercept_manual_corrected))
plt.legend(fontsize=12)
plt.xlabel('Manual Area ($\cdot 10^{3} \mu$m$^2$)', fontsize=14)
plt.ylabel('Auto/Corrected area ($\cdot 10^{3} \mu$m$^2$)', fontsize=14)
plt.tick_params(axis="both", labelsize=14)

## Analyse results: Automatic data

# other_broken_svg_file_list = [
#     '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04_row_010716_col_008924.svg',
#     '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.1a  50-16 C1 - 2016-02-02 09.12.41_row_028156_col_018596.svg',
#     '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52_row_014628_col_069148.svg',
#     '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52_row_019340_col_017348.svg',
#     '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32_row_034628_col_040116.svg',
#     '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32_row_035948_col_041492.svg'
# ]
# plt.imshow(Image.open(other_broken_svg_file_list[4].replace('.svg', '.tif')))

# # remove fold/image pairs that contain only "other" tissue or broken cells
# other_broken_fold_im = [(2, 6), (6, 0), (8, 3), (8, 4), (7, 3), (7, 4)]
# idx_manual_slice_with_wat = np.logical_not([x in other_broken_fold_im
#                                             for x in zip(df_manual_all['fold'], df_manual_all['im'])])
#
# # remove fold/image pairs that contain substantial areas with broken cells or poor segmentations in general
# poor_seg_fold_im = [(4, 5), (5, 0), (3, 1), (6, 3), (4, 2), (3, 5)]
# idx_manual_slice_with_acceptable_seg = np.logical_not([x in poor_seg_fold_im
#                                                        for x in zip(df_manual_all['fold'], df_manual_all['im'])])

# load dataframe with all auto segmentations
df_auto_all = pd.read_pickle(dataframe_auto_filename)

# load dataframe with manual segmentations matched to automatic segmentations
data_manual_filename = os.path.join(saved_models_dir, experiment_id + '_test_pipeline_manual.pkl')
df_manual_all = pd.read_pickle(data_manual_filename)

# boolean vectors to select subsets of rows from the dataframe
# idx_auto_wat = np.array(df_auto_all['wat_prop_auto'] >= 0.59) # this threshold computed above in the Object-wise validation
# idx_corrected_wat = np.array(df_auto_all['wat_prop_corrected'] >= 0.59) # ditto
idx_auto_wat = np.array(df_auto_all['wat_prop_auto'] >= 0.5) # threshold that was used in the experiments
idx_corrected_wat = np.array(df_auto_all['wat_prop_corrected'] >= 0.5) # ditto
idx_auto_not_large = np.array(df_auto_all['area_auto'] < 20e3)
idx_corrected_not_large = np.array(df_auto_all['area_corrected'] < 20e3)
idx_auto_not_edge = np.logical_not(df_auto_all['auto_is_edge_cell'])

# indices of automatic segmentations accepted for analysis
idx_auto = idx_auto_wat * idx_auto_not_large * idx_auto_not_edge
idx_corrected = idx_corrected_wat * idx_corrected_not_large * idx_auto_not_edge

# plot not used in paper
if DEBUG:
    # boxplots of manual/auto/corrected areas.
    plt.clf()
    bp = plt.boxplot((df_manual_all['area_manual'] / 1e3,
                      df_auto_all['area_auto'][idx_auto] / 1e3,
                      df_auto_all['area_corrected'][idx_corrected] / 1e3),
                     positions=[1, 2, 3], notch=True, labels=['Hand traced', 'Auto', 'Corrected'])

    # points of interest from the boxplots
    bp_poi = cytometer.utils.boxplot_poi(bp)

    plt.plot([0.75, 3.25], [bp_poi[0, 2], ] * 2, 'C1', linestyle='dotted')  # manual median
    plt.plot([0.75, 3.25], [bp_poi[0, 1], ] * 2, 'k', linestyle='dotted')  # manual Q1
    plt.plot([0.75, 3.25], [bp_poi[0, 3], ] * 2, 'k', linestyle='dotted')  # manual Q3
    plt.tick_params(axis="both", labelsize=14)
    plt.ylabel('Area ($\cdot 10^{3} \mu$m$^2$)', fontsize=14)
    plt.ylim(-700 / 1e3, 10000 / 1e3)
    plt.tight_layout()

    # manual quartile values
    plt.text(1.20, bp_poi[0, 3] + .1, '%0.1f' % (bp_poi[0, 3]), fontsize=12, color='k')
    plt.text(1.20, bp_poi[0, 2] + .1, '%0.1f' % (bp_poi[0, 2]), fontsize=12, color='C1')
    plt.text(1.20, bp_poi[0, 1] + .1, '%0.1f' % (bp_poi[0, 1]), fontsize=12, color='k')

    # auto quartile values
    plt.text(2.20, bp_poi[1, 3] + .1 - .3, '%0.1f' % (bp_poi[1, 3]), fontsize=12, color='k')
    plt.text(2.20, bp_poi[1, 2] + .1 + .1, '%0.1f' % (bp_poi[1, 2]), fontsize=12, color='C1')
    plt.text(2.20, bp_poi[1, 1] + .1 + .02, '%0.1f' % (bp_poi[1, 1]), fontsize=12, color='k')

    # corrected quartile values
    plt.text(3.20, bp_poi[2, 3] + .1 - .1, '%0.1f' % (bp_poi[2, 3]), fontsize=12, color='k')
    plt.text(3.20, bp_poi[2, 2] + .1 + .0, '%0.1f' % (bp_poi[2, 2]), fontsize=12, color='C1')
    plt.text(3.20, bp_poi[2, 1] + .1 - .1, '%0.1f' % (bp_poi[2, 1]), fontsize=12, color='k')

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_area_boxplots.svg'))

''' 
************************************************************************************************************************
Linear Mixed-Effects Model analysis (sqrt_area ~ sex + ko + other variables + random(mouse|window))
************************************************************************************************************************
'''

# load dataframe with manual segmentations matched to automatic segmentations
data_manual_filename = os.path.join(saved_models_dir, experiment_id + '_test_pipeline_manual.pkl')
df_manual_all = pd.read_pickle(data_manual_filename)

if DEBUG:
    # show that data is now normal
    plt.clf()
    ax = plt.subplot(311)
    prob = stats.probplot(df_manual_all.area_manual, dist=stats.norm, plot=ax)
    plt.title(r'Areas ($\mu m^2$)')
    ax = plt.subplot(312)
    prob = stats.probplot(np.sqrt(df_manual_all.area_manual), dist=stats.norm, plot=ax)
    plt.title(r'Square root transformation of areas $\left(\sqrt{\mu m^2}\right)$')
    ax = plt.subplot(313)
    plt.hist(np.sqrt(df_manual_all.area_manual), bins=50)
    plt.tight_layout()

# for the mixed-effects linear model, we want the KO variable to be ordered, so that it's PAT=0, MAT=1 in terms of
# genetic risk, and the sex variable to be ordered in the sense that males have larger cells than females
df_manual_all['ko_parent'] = df_manual_all['ko_parent'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
df_manual_all['sex'] = df_manual_all['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))

# create column for sqrt(data)
df_manual_all['sqrt_area_manual'] = np.sqrt(df_manual_all['area_manual'])

# create column to have IDs for each image
df_manual_all['image_id'] = df_manual_all['id'] + df_manual_all['im'].astype(str)

# Mixed-effects linear model
vc = {'image_id': '0 + C(image_id)'}  # image_id is a random effected nested inside mouse_id
md = smf.mixedlm('sqrt_area_manual ~ sex + ko', vc_formula=vc, re_formula='1', groups='id', data=df_manual_all)
mdf = md.fit()
print(mdf.summary())

# pretty print of the model

# \sqrt{\mathrm{area}}$ &\sim \mathrm{sex} \cdot \alpha_\mathrm{sex} + \mathrm{ko} \cdot \beta_\mathrm{ko} + \mathrm{random}(\mathrm{mouse}|\mathrm{image})

# \alpha_\mathrm{sex} = \begin{cases}0,&\mathrm{female}\\1,&\mathrm{male}\end{cases}

# \beta_\mathrm{ko} = \begin{cases}0,&\mathrm{PAT}\\1,&\mathrm{MAT}\end{cases}

'''
************************************************************************************************************************
Statistical analysis of manual data (using Harrell-Davis quantile estimates)
************************************************************************************************************************
'''

## Analyse results: Manual data (regardless of whether they match an automatic segmentation)

# load data computed in the previous section
data_filename = os.path.join(saved_models_dir, experiment_id + '_classifier_by_object.pkl')
df_manual_all = pd.read_pickle(data_filename)

# keep only WAT cells
df_manual_all = df_manual_all.loc[df_manual_all['type'] == 'wat', :]
df_manual_all = df_manual_all.drop('type', axis=1)

# drop the 'wat_prop_X' columns
col_to_drop = df_manual_all.columns
col_to_drop = col_to_drop[['wat_prop_' in x for x in col_to_drop]]
df_manual_all = df_manual_all.drop(col_to_drop, axis=1)

# WARNING! Here the PAT/MAT column is called "ko", but in other experiments we renamed it to "ko_parent"
df_manual_all = df_manual_all.rename(columns={'ko': 'ko_parent'})

## extra files that were not used in the CNN training and segmentation, but that they were added so that we could have
## representative ECDFs for animals that were undersampled

file_svg_list_extra = [
    os.path.join(training_data_dir, 'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_020824_col_018688.svg'),
    os.path.join(training_data_dir, 'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_013256_col_007952.svg'),
    os.path.join(training_data_dir, 'KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46_row_006040_col_005272.svg'),
    os.path.join(training_data_dir, 'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_012680_col_023936.svg'),
    os.path.join(training_data_dir, 'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_017360_col_024712.svg')
]
for i, file_svg in enumerate(file_svg_list_extra):

    # open histology testing image
    file_tif = file_svg.replace('.svg', '.tif')
    im = Image.open(file_tif)

    # read pixel size information
    xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
    yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

    # read the ground truth cell contours in the SVG file. This produces a list [contour_0, ..., contour_N-1]
    # where each contour_i = [(X_0, Y_0), ..., (X_P-1, X_P-1)]
    contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False,
                                                       minimum_npoints=3)

    print('Cells: ' + str(len(contours)))
    print('')

    # create dataframe for this image
    im_idx = 'extra_' + str(i)
    df_manual = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_tif),
                                                          values=[im_idx] * len(contours), values_tag='im',
                                                          tags_to_keep=['id', 'ko_parent', 'sex'])
    df_manual['contour'] = range(0, len(contours))
    df_manual['area'] = np.nan

    for j, contour in enumerate(contours):

        # manual segmentation: rasterise object described by contour
        cell_seg_contour = Image.new("1", im.size, "black")  # I = 32-bit signed integer pixels
        draw = ImageDraw.Draw(cell_seg_contour)
        draw.polygon(contour, outline="white", fill="white")
        cell_seg_contour = np.array(cell_seg_contour, dtype=np.bool)

        df_manual.loc[j, 'area'] = np.count_nonzero(cell_seg_contour) * xres * yres

    # append contours from this image to the total dataframe
    df_manual_all = df_manual_all.append(df_manual, ignore_index=True, sort=False)

# number of cells in each stratum

idx_f_mat = np.logical_and(df_manual_all['sex'] == 'f', df_manual_all['ko_parent'] == 'MAT')
idx_f_pat = np.logical_and(df_manual_all['sex'] == 'f', df_manual_all['ko_parent'] == 'PAT')
idx_m_mat = np.logical_and(df_manual_all['sex'] == 'm', df_manual_all['ko_parent'] == 'MAT')
idx_m_pat = np.logical_and(df_manual_all['sex'] == 'm', df_manual_all['ko_parent'] == 'PAT')

print('f PAT = ' + str(np.count_nonzero(idx_f_pat)))
print('f MAT = ' + str(np.count_nonzero(idx_f_mat)))
print('m PAT = ' + str(np.count_nonzero(idx_m_pat)))
print('m MAT = ' + str(np.count_nonzero(idx_m_mat)))

## boxplots of PAT vs MAT in male

if DEBUG:
    plt.clf()
    plt.boxplot([df_manual_all['area'][idx_f_pat] * 1e-3,
                 df_manual_all['area'][idx_f_mat] * 1e-3,
                 df_manual_all['area'][idx_m_pat] * 1e-3,
                 df_manual_all['area'][idx_m_mat] * 1e-3
                 ],
                labels=['f/PAT', 'f/MAT', 'm/PAT', 'm/MAT'],
                positions=[1, 2, 3.5, 4.5],
                notch=True)
    plt.ylim(-875/1e3, 11)
    plt.ylabel('Area ($\cdot 10^{-3} \mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()

# compute all percentiles
quantiles = np.linspace(0.1, 0.9, 9)
area_perc_f_pat = stats.mstats.hdquantiles(df_manual_all['area'][idx_f_pat], prob=quantiles)
area_perc_f_mat = stats.mstats.hdquantiles(df_manual_all['area'][idx_f_mat], prob=quantiles)
area_perc_m_pat = stats.mstats.hdquantiles(df_manual_all['area'][idx_m_pat], prob=quantiles)
area_perc_m_mat = stats.mstats.hdquantiles(df_manual_all['area'][idx_m_mat], prob=quantiles)

if DEBUG:
    plt.clf()
    plt.plot(quantiles, area_perc_f_pat, label='Female PAT')
    plt.plot(quantiles, area_perc_m_pat, label='Male PAT')
    plt.ylabel('Area ($\cdot 10^3 \mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='best', prop={'size': 12})
    plt.tight_layout()

# list of animals in each group
unique_ids_f_pat = np.unique(df_manual_all[idx_f_pat]['id'])
unique_ids_m_pat = np.unique(df_manual_all[idx_m_pat]['id'])
unique_ids_f_mat = np.unique(df_manual_all[idx_f_mat]['id'])
unique_ids_m_mat = np.unique(df_manual_all[idx_m_mat]['id'])

area_perc_f_pat = np.zeros(shape=(len(unique_ids_f_pat), len(quantiles)))
area_perc_m_pat = np.zeros(shape=(len(unique_ids_m_pat), len(quantiles)))
area_perc_f_mat = np.zeros(shape=(len(unique_ids_f_mat), len(quantiles)))
area_perc_m_mat = np.zeros(shape=(len(unique_ids_m_mat), len(quantiles)))

# loop animals to compute quantiles for each animal. Each row corresponds to an animal. Each column, to a quantile.

# F/PAT
for i, id in enumerate(unique_ids_f_pat):
    # indices of cells that correspond to the current animal
    idx = np.logical_and(idx_f_pat, df_manual_all['id'] == id)

    # compute percentiles
    area_perc_f_pat[i, :] = stats.mstats.hdquantiles(df_manual_all['area'][idx], prob=quantiles)

# M/PAT
for i, id in enumerate(unique_ids_m_pat):
    # indices of cells that correspond to the current animal
    idx = np.logical_and(idx_m_pat, df_manual_all['id'] == id)

    # compute percentiles
    area_perc_m_pat[i, :] = stats.mstats.hdquantiles(df_manual_all['area'][idx], prob=quantiles)

# F/MAT
for i, id in enumerate(unique_ids_f_mat):
    # indices of cells that correspond to the current animal
    idx = np.logical_and(idx_f_mat, df_manual_all['id'] == id)

    # compute percentiles
    area_perc_f_mat[i, :] = stats.mstats.hdquantiles(df_manual_all['area'][idx], prob=quantiles)

# M/MAT
for i, id in enumerate(unique_ids_m_mat):
    # indices of cells that correspond to the current animal
    idx = np.logical_and(idx_m_mat, df_manual_all['id'] == id)

    # compute percentiles
    area_perc_m_mat[i, :] = stats.mstats.hdquantiles(df_manual_all['area'][idx], prob=quantiles)

# PAT
if DEBUG:
    plt.clf()
    [plt_f_pat, _, _, _, _] = plt.plot(quantiles, np.transpose(area_perc_f_pat) * 1e-3, color='C0', linewidth=3)
    [plt_m_pat, _, _, _] = plt.plot(quantiles, np.transpose(area_perc_m_pat) * 1e-3, color='C1', linewidth=3)
    plt.xlabel('Cell population quantile', fontsize=14)
    plt.ylabel('Area ($\cdot 10^3 \mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend([plt_f_pat, plt_m_pat], ['Female PAT', 'Male PAT'], loc='upper left', prop={'size': 12})
    plt.ylim(-0.25, 11)
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_pat_female_vs_male.svg'))
    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_pat_female_vs_male.png'))

# MAT
if DEBUG:
    plt.clf()
    [plt_f_mat, _, _, _, _] = plt.plot(quantiles, np.transpose(area_perc_f_mat) * 1e-3, color='C2', linewidth=3)
    [plt_m_mat, _, _, _, _] = plt.plot(quantiles, np.transpose(area_perc_m_mat) * 1e-3, color='C3', linewidth=3)
    plt.xlabel('Cell population quantile', fontsize=14)
    plt.ylabel('Area ($\cdot 10^3 \mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend([plt_f_mat, plt_m_mat], ['Female MAT', 'Male MAT'], loc='upper left', prop={'size': 12})
    plt.ylim(-0.25, 11)
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_mat_female_vs_male.svg'))
    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_mat_female_vs_male.png'))

# Female
if DEBUG:
    plt.clf()
    [plt_f_pat, _, _, _, _] = plt.plot(quantiles, np.transpose(area_perc_f_pat) * 1e-3, color='C0', linewidth=3)
    [plt_f_mat, _, _, _, _] = plt.plot(quantiles, np.transpose(area_perc_f_mat) * 1e-3, color='C2', linewidth=3)
    plt.xlabel('Cell population quantile', fontsize=14)
    plt.ylabel('Area ($\cdot 10^3 \mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend([plt_f_pat, plt_f_mat], ['Female PAT', 'Female MAT'], loc='upper left', prop={'size': 12})
    plt.ylim(-0.25, 11)
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_female_pat_vs_mat.svg'))
    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_female_pat_vs_mat.png'))

# Male
if DEBUG:
    plt.clf()
    [plt_m_pat, _, _, _] = plt.plot(quantiles, np.transpose(area_perc_m_pat) * 1e-3, color='C1', linewidth=3)
    [plt_m_mat, _, _, _, _] = plt.plot(quantiles, np.transpose(area_perc_m_mat) * 1e-3, color='C3', linewidth=3)
    plt.xlabel('Cell population quantile', fontsize=14)
    plt.ylabel('Area ($\cdot 10^3 \mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend([plt_m_pat, plt_m_mat], ['Male PAT', 'Male MAT'], loc='upper left', prop={'size': 12})
    plt.ylim(-0.25, 11)
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_male_pat_vs_mat.svg'))
    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_male_pat_vs_mat.png'))

# compute variability of area values for each quantile
area_interval_f_pat = stats.mstats.hdquantiles(area_perc_f_pat, prob=[0.25, 0.5, 0.75], axis=0)
area_interval_m_pat = stats.mstats.hdquantiles(area_perc_m_pat, prob=[0.25, 0.5, 0.75], axis=0)
area_interval_f_mat = stats.mstats.hdquantiles(area_perc_f_mat, prob=[0.25, 0.5, 0.75], axis=0)
area_interval_m_mat = stats.mstats.hdquantiles(area_perc_m_mat, prob=[0.25, 0.5, 0.75], axis=0)

# PAT (females and males)
if DEBUG:
    plt.clf()
    plt.plot(quantiles, area_interval_f_pat[1, :] * 1e-3, 'C0', linewidth=3, label='Female PAT median & Q1-Q3 interval')
    plt.fill_between(quantiles, area_interval_f_pat[0, :] * 1e-3, area_interval_f_pat[2, :] * 1e-3,
                     facecolor='C0', alpha=0.3)
    plt.plot(quantiles, area_interval_f_pat[0, :] * 1e-3, 'C0', linewidth=1)
    plt.plot(quantiles, area_interval_f_pat[2, :] * 1e-3, 'C0', linewidth=1)

    plt.plot(quantiles, area_interval_m_pat[1, :] * 1e-3, 'C1', linewidth=3, label='Male PAT median & Q1-Q3 interval')
    plt.fill_between(quantiles, area_interval_m_pat[0, :] * 1e-3, area_interval_m_pat[2, :] * 1e-3,
                     facecolor='C1', alpha=0.3)
    plt.plot(quantiles, area_interval_m_pat[0, :] * 1e-3, 'C1', linewidth=1)
    plt.plot(quantiles, area_interval_m_pat[2, :] * 1e-3, 'C1', linewidth=1)

    plt.xlabel('Quantile', fontsize=14)
    plt.ylabel('White adipocyte area ($\cdot 10^3 \mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='best', prop={'size': 12})
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_pat_female_vs_male_bands.svg'))
    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_pat_female_vs_male_bands.png'))

# MAT (females and males)
if DEBUG:
    plt.clf()
    plt.plot(quantiles, area_interval_f_mat[1, :] * 1e-3, 'C2', linewidth=3, label='Female MAT median & Q1-Q3 interval')
    plt.fill_between(quantiles, area_interval_f_mat[0, :] * 1e-3, area_interval_f_mat[2, :] * 1e-3,
                     facecolor='C0', alpha=0.3)
    plt.plot(quantiles, area_interval_f_mat[0, :] * 1e-3, 'C2', linewidth=1)
    plt.plot(quantiles, area_interval_f_mat[2, :] * 1e-3, 'C2', linewidth=1)

    plt.plot(quantiles, area_interval_m_mat[1, :] * 1e-3, 'C3', linewidth=3, label='Male MAT median & Q1-Q3 interval')
    plt.fill_between(quantiles, area_interval_m_mat[0, :] * 1e-3, area_interval_m_mat[2, :] * 1e-3,
                     facecolor='C1', alpha=0.3)
    plt.plot(quantiles, area_interval_m_mat[0, :] * 1e-3, 'C3', linewidth=1)
    plt.plot(quantiles, area_interval_m_mat[2, :] * 1e-3, 'C3', linewidth=1)

    plt.xlabel('Quantile', fontsize=14)
    plt.ylabel('White adipocyte area ($\cdot 10^3 \mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='best', prop={'size': 12})
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_mat_female_vs_male_bands.svg'))
    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_mat_female_vs_male_bands.png'))

# Female (PAT and MAT)
if DEBUG:
    plt.clf()
    plt.plot(quantiles, area_interval_f_pat[1, :] * 1e-3, 'C0', linewidth=3, label='Female PAT median & Q1-Q3 interval')
    plt.fill_between(quantiles, area_interval_f_pat[0, :] * 1e-3, area_interval_f_pat[2, :] * 1e-3,
                     facecolor='C0', alpha=0.3)
    plt.plot(quantiles, area_interval_f_pat[0, :] * 1e-3, 'C0', linewidth=1)
    plt.plot(quantiles, area_interval_f_pat[2, :] * 1e-3, 'C0', linewidth=1)

    plt.plot(quantiles, area_interval_f_mat[1, :] * 1e-3, 'C2', linewidth=3, label='Female MAT median & Q1-Q3 interval')
    plt.fill_between(quantiles, area_interval_f_mat[0, :] * 1e-3, area_interval_f_mat[2, :] * 1e-3,
                     facecolor='C1', alpha=0.3)
    plt.plot(quantiles, area_interval_f_mat[0, :] * 1e-3, 'C2', linewidth=1)
    plt.plot(quantiles, area_interval_f_mat[2, :] * 1e-3, 'C2', linewidth=1)

    plt.xlabel('Quantile', fontsize=14)
    plt.ylabel('White adipocyte area ($\cdot 10^3 \mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='best', prop={'size': 12})
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_female_pat_vs_mat_bands.svg'))
    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_female_pat_vs_mat_bands.png'))

# Male (PAT and MAT)
if DEBUG:
    plt.clf()
    plt.plot(quantiles, area_interval_m_pat[1, :] * 1e-3, 'C1', linewidth=3, label='Male PAT median & Q1-Q3 interval')
    plt.fill_between(quantiles, area_interval_m_pat[0, :] * 1e-3, area_interval_m_pat[2, :] * 1e-3,
                     facecolor='C0', alpha=0.3)
    plt.plot(quantiles, area_interval_m_pat[0, :] * 1e-3, 'C1', linewidth=1)
    plt.plot(quantiles, area_interval_m_pat[2, :] * 1e-3, 'C1', linewidth=1)

    plt.plot(quantiles, area_interval_m_mat[1, :] * 1e-3, 'C3', linewidth=3, label='Male MAT median & Q1-Q3 interval')
    plt.fill_between(quantiles, area_interval_m_mat[0, :] * 1e-3, area_interval_m_mat[2, :] * 1e-3,
                     facecolor='C1', alpha=0.3)
    plt.plot(quantiles, area_interval_m_mat[0, :] * 1e-3, 'C3', linewidth=1)
    plt.plot(quantiles, area_interval_m_mat[2, :] * 1e-3, 'C3', linewidth=1)

    plt.xlabel('Quantile', fontsize=14)
    plt.ylabel('White adipocyte area ($\cdot 10^3 \mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='best', prop={'size': 12})
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_male_pat_vs_mat_bands.svg'))
    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_male_pat_vs_mat_bands.png'))

# test whether the median values are different enough between female vs. male
func = lambda x, y: np.abs(stats.mstats.hdquantiles(x, prob=0.5, axis=0).data[0]
                           - stats.mstats.hdquantiles(y, prob=0.5, axis=0).data[0])
# func = lambda x, y: np.abs(np.mean(x) - np.mean(y))

pval_perc_f2m_pat = np.zeros(shape=(len(quantiles),))
for i, q in enumerate(quantiles):
    pval_perc_f2m_pat[i] = permutation_test(x=area_perc_f_pat[:, i], y=area_perc_m_pat[:, i],
                                            func=func, method='exact', seed=None)

pval_perc_f2m_mat = np.zeros(shape=(len(quantiles),))
for i, q in enumerate(quantiles):
    pval_perc_f2m_mat[i] = permutation_test(x=area_perc_f_mat[:, i], y=area_perc_m_mat[:, i],
                                            func=func, method='exact', seed=None)

# multitest correction using Hochberg a.k.a. Simes-Hochberg method
_, pval_perc_f2m_pat, _, _ = multipletests(pval_perc_f2m_pat, method='simes-hochberg', alpha=0.05, returnsorted=False)
_, pval_perc_f2m_mat, _, _ = multipletests(pval_perc_f2m_mat, method='simes-hochberg', alpha=0.05, returnsorted=False)

# plot the median difference and the population quantiles at which the difference is significant
if DEBUG:
    plt.clf()
    idx = pval_perc_f2m_pat < 0.05
    delta_a_f2m_pat = (area_interval_m_pat[1, :] - area_interval_f_pat[1, :]) / area_interval_f_pat[1, :] # difference of medians
    plt.stem(quantiles[idx], 100 * delta_a_f2m_pat[idx],
             markerfmt='ko', linefmt='k-', basefmt='k',
             label='p-val$_{\mathrm{PAT}}$ < 0.05')
    plt.plot(quantiles, 100 * delta_a_f2m_pat, 'k', linewidth=3, label='PAT female to male')
    plt.xlabel('Cell population quantile', fontsize=14)
    plt.ylabel('Area change (%)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='best', prop={'size': 12})
    plt.ylim(-5, 145)
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_change_female_2_male_pat.svg'))
    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_change_female_2_male_pat.png'))

    plt.clf()
    idx = pval_perc_f2m_mat < 0.05
    delta_a_f2m_mat = (area_interval_m_mat[1, :] - area_interval_f_mat[1, :]) / area_interval_f_mat[1, :] # difference of medians
    plt.stem(quantiles[idx], 100 * delta_a_f2m_mat[idx],
             markerfmt='ko', linefmt='k-', basefmt='k',
             label='p-val$_{\mathrm{MAT}}$ < 0.05')
    plt.plot(quantiles, 100 * delta_a_f2m_mat, 'k', linewidth=3, label='MAT female to male')
    plt.xlabel('Cell population quantile', fontsize=14)
    plt.ylabel('Area change (%)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='best', prop={'size': 12})
    plt.ylim(-5, 145)
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_change_female_2_male_mat.svg'))
    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_change_female_2_male_mat.png'))

# test whether the median values are different enough between PAT vs. MAT
pval_perc_f_pat2mat = np.zeros(shape=(len(quantiles),))
for i, q in enumerate(quantiles):
    pval_perc_f_pat2mat[i] = permutation_test(x=area_perc_f_pat[:, i], y=area_perc_f_mat[:, i],
                                              func=func, method='exact', seed=None)

pval_perc_m_pat2mat = np.zeros(shape=(len(quantiles),))
for i, q in enumerate(quantiles):
    pval_perc_m_pat2mat[i] = permutation_test(x=area_perc_m_pat[:, i], y=area_perc_m_mat[:, i],
                                              func=func, method='exact', seed=None)

# multitest correction using Hochberg a.k.a. Simes-Hochberg method
_, pval_perc_f_pat2mat, _, _ = multipletests(pval_perc_f_pat2mat, method='simes-hochberg', alpha=0.05, returnsorted=False)
_, pval_perc_m_pat2mat, _, _ = multipletests(pval_perc_m_pat2mat, method='simes-hochberg', alpha=0.05, returnsorted=False)

# plot the median difference and the population quantiles at which the difference is significant
if DEBUG:
    plt.clf()
    idx = pval_perc_f_pat2mat < 0.05
    delta_a_f_pat2mat = (area_interval_f_mat[1, :] - area_interval_f_pat[1, :]) / area_interval_f_pat[1, :] # difference of medians
    if np.any(idx):
        plt.stem(quantiles[idx], 100 * delta_a_f_pat2mat[idx],
                 markerfmt='ko', linefmt='k-', basefmt='k',
                 label='p-val$_{\mathrm{PAT}}$ < 0.05')
    plt.plot(quantiles, 100 * delta_a_f_pat2mat, 'k', linewidth=3, label='Female PAT to MAT')
    plt.xlabel('Cell population quantile', fontsize=14)
    plt.ylabel('Area change (%)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='lower right', prop={'size': 12})
    plt.ylim(-24.5, -5.5)
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_change_pat_2_mat_female.svg'))
    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_change_pat_2_mat_female.png'))

    plt.clf()
    idx = pval_perc_m_pat2mat < 0.05
    delta_a_m_pat2mat = (area_interval_m_mat[1, :] - area_interval_m_pat[1, :]) / area_interval_m_pat[1, :] # difference of medians
    if np.any(idx):
        plt.stem(quantiles[idx], 100 * delta_a_m_pat2mat[idx],
                 markerfmt='ko', linefmt='k-', basefmt='k',
                 label='p-val$_{\mathrm{MAT}}$ < 0.05')
    plt.plot(quantiles, 100 * delta_a_m_pat2mat, 'k', linewidth=3, label='Male PAT to MAT')
    plt.xlabel('Cell population quantile', fontsize=14)
    plt.ylabel('Area change (%)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='lower right', prop={'size': 12})
    plt.ylim(-24.5, -5.5)
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_change_pat_2_mat_male.svg'))
    plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_cell_area_change_pat_2_mat_male.png'))
