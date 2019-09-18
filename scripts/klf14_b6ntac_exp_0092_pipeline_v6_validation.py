"""
Validate pipeline v6:
 * data generation
   * training images (*0076*)
   * non-overlap training images (*0077*)
   * augmented training images (*0078*)
   * k-folds (*0079*)
 * segmentation
   * dmap (*0086*)
   * contour from dmap (0091)
 * classifier (*0088*)
 * segmentation correction (0089) networks"
 * validation (0092)
"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0092_pipeline_v6_validation'

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
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.metrics import roc_curve, auc
from scipy.stats import linregress, mode
from enum import IntEnum

# # limit number of GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

import cytometer.utils
import cytometer.data
import tensorflow as tf

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
min_cell_area = 75
median_size = 0
closing_size = 11
contour_seed_threshold = 0.005
batch_size = 2
local_threshold_block_size = 41

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

# k-folds file
saved_kfolds_filename = 'klf14_b6ntac_exp_0079_generate_kfolds.pickle'

# model names
dmap_model_basename = 'klf14_b6ntac_exp_0086_cnn_dmap'
contour_model_basename = 'klf14_b6ntac_exp_0091_cnn_contour_after_dmap'
classifier_model_basename = 'klf14_b6ntac_exp_0088_cnn_tissue_classifier_fcn'
correction_model_basename = 'klf14_b6ntac_exp_0089_cnn_segmentation_correction_overlapping_scaled_contours'

# HACK: use older models while the new ones are computing, so that we can develop the code in the mean time
# classifier_model_basename = 'klf14_b6ntac_exp_0074_cnn_tissue_classifier_fcn'

'''Load folds'''

# load list of images, and indices for training vs. testing indices
saved_kfolds_filename = os.path.join(saved_models_dir, saved_kfolds_filename)
with open(saved_kfolds_filename, 'rb') as f:
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

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(root_data_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

'''
************************************************************************************************************************
Prepare the testing data:

  This is computed once, and then saved to 
  'klf14_b6ntac_exp_0092_pipeline_v6_validation_data.npz'.
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

        print('file ' + str(i) + '/' + str(len(file_svg_list) - 1))

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
Areas of manual contours. This computes a subset of section "Object-wise classification validation", just the manual 
contour results (if you just need this subset, it's a lot faster than computing all the results in the other section). 
************************************************************************************************************************
'''

# start timer
t0 = time.time()

# init
df_all = pd.DataFrame()

for i_fold in range(len(idx_test_all)):

    print('# Fold ' + str(i_fold) + '/' + str(len(idx_test_all) - 1))

    # test and training image indices. These indices refer to file_list
    idx_test = idx_test_all[i_fold]

    # list of test files (used later for the dataframe)
    file_list_test = np.array(file_svg_list)[idx_test]

    print('## len(idx_test) = ' + str(len(idx_test)))

    # split data into training and testing
    im_array_test = im_array_all[idx_test, :, :, :]

    # out_class_train = out_class_all[idx_train, :, :, :]
    out_class_test = out_class_all[idx_test, :, :, :]

    # out_mask_train = out_mask_all[idx_train, :, :]
    out_mask_test = out_mask_all[idx_test, :, :]

    # loop test images
    for i, file_svg in enumerate(file_list_test):

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

        # create dataframe for this image
        im_idx = [idx_test_all[i_fold][i], ] * len(contours)  # absolute index of current test image
        df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_tif),
                                                              values=im_idx, values_tag='im',
                                                              tags_to_keep=['id', 'ko', 'sex'])
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

        # loop contours
        for j, contour in enumerate(contours):

            if DEBUG:
                # close the contour for the plot
                contour_aux = contour.copy()
                contour_aux.append(contour[0])

                plt.cla()
                plt.imshow(im)
                plt.plot([p[0] for p in contour_aux], [p[1] for p in contour_aux], color='C0')
                plt.axis('off')
                plt.title('Histology', fontsize=14)
                plt.axis('off')

            # rasterise object described by contour
            cell_seg_contour = Image.new("1", im.size, "black")  # I = 32-bit signed integer pixels
            draw = ImageDraw.Draw(cell_seg_contour)
            draw.polygon(contour, outline="white", fill="white")
            cell_seg_contour = np.array(cell_seg_contour, dtype=np.uint8)

            # compute area of object
            df_common.loc[j, 'area'] = np.count_nonzero(cell_seg_contour) * xres * yres

        # concatenate current dataframe to general dataframe
        df_all = df_all.append(df_common)

# reindex the dataframe
df_all.reset_index(drop=True, inplace=True)

# save results
data_filename = os.path.join(saved_models_dir, experiment_id + '_manual_contour_areas.pkl')
df_all.to_pickle(data_filename)

# load results
data_filename = os.path.join(saved_models_dir, experiment_id + '_manual_contour_areas.pkl')
df_all = pd.read_pickle(data_filename)

# WAT objects, female and male
idx_wat_female = np.array(df_all['type'] == 'wat') * np.array(df_all['sex'] == 'f')
idx_wat_male = np.array(df_all['type'] == 'wat') * np.array(df_all['sex'] == 'm')

if DEBUG:
    plt.clf()
    plt.hist(df_all['area'][idx_wat_female], bins=50, histtype='step')
    plt.hist(df_all['area'][idx_wat_male], bins=50, histtype='step')


'''
************************************************************************************************************************
Object-wise classification validation
************************************************************************************************************************
'''

# correct home directory in file paths
file_svg_list = cytometer.data.change_home_directory(list(file_svg_list), '/users/rittscher/rcasero', home, check_isfile=True)

# load data computed in the previous section
data_filename = os.path.join(saved_models_dir, experiment_id + '_data.npz')
with np.load(data_filename) as data:
    im_array_all = data['im_array_all']
    out_class_all = 1 - data['out_class_all']  # encode as 0: other, 1: WAT
    out_mask_all = data['out_mask_all']

# start timer
t0 = time.time()

# init
df_all = pd.DataFrame()

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

        ''' Tissue classification (applied pixel by pixel to the whole image) '''

        # load classification model
        classifier_model_filename = os.path.join(saved_models_dir,
                                                 classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')
        classifier_model = keras.models.load_model(classifier_model_filename)

        # reshape model input
        classifier_model = cytometer.utils.change_input_size(classifier_model, batch_shape=im_array_test.shape)

        # apply classification to test data
        pred_class_test = classifier_model.predict(np.expand_dims(im_array_test[i, ...], axis=0), batch_size=batch_size)

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
            plt.subplot(121)
            plt.imshow(im)
            plt.axis('off')
            plt.title('Histology', fontsize=14)
            plt.subplot(122)
            plt.cla()
            plt.imshow(pred_class_test[0, :, :, 0], cmap='plasma')
            plt.title('Pixel WAT score', fontsize=14)
            plt.axis('off')
            plt.tight_layout()

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

        # create dataframe for this image
        im_idx = [idx_test_all[i_fold][i], ] * len(contours)  # absolute index of current test image
        df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_tif),
                                                              values=im_idx, values_tag='im',
                                                              tags_to_keep=['id', 'ko', 'sex'])
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

        # loop contours
        for j, contour in enumerate(contours):

            if DEBUG:
                # close the contour for the plot
                contour_aux = contour.copy()
                contour_aux.append(contour[0])

                plt.subplot(121)
                plt.cla()
                plt.imshow(im)
                plt.plot([p[0] for p in contour_aux], [p[1] for p in contour_aux], color='C0')
                plt.axis('off')
                plt.title('Histology', fontsize=14)
                plt.axis('off')
                plt.subplot(122)
                plt.cla()
                plt.imshow(pred_class_test[0, :, :, 0], cmap='plasma')
                plt.plot([p[0] for p in contour_aux], [p[1] for p in contour_aux], color='w')
                plt.title('Pixel WAT score', fontsize=14)
                plt.axis('off')
                plt.tight_layout()

            # rasterise object described by contour
            cell_seg_contour = Image.new("1", im.size, "black")  # I = 32-bit signed integer pixels
            draw = ImageDraw.Draw(cell_seg_contour)
            draw.polygon(contour, outline="white", fill="white")
            cell_seg_contour = np.array(cell_seg_contour, dtype=np.uint8)

            # compute area of object
            df_common.loc[j, 'area'] = np.count_nonzero(cell_seg_contour) * xres * yres

            # get scores from within the object
            aux = pred_class_test[0, :, :, 0]  # other = 0, wat = 1
            wat_scores = aux[cell_seg_contour == 1]

            # compute proportions for different thresholds of Otherness
            prop = np.linspace(0, 100, 101)
            for p in prop:
                # e.g. df_im.loc[j, 'wat_prop_55'] = np.count_nonzero(wat_scores > 0.55) / len(wat_scores)
                df_common.loc[j, 'wat_prop_' + str(int(p))] = np.count_nonzero(wat_scores > (p / 100)) / len(wat_scores)

            if DEBUG:
                # close the contour for the plot
                contour_aux = contour.copy()
                contour_aux.append(contour[0])

                plt.clf()
                plt.subplot(221)
                plt.imshow(im)
                plt.axis('off')
                plt.title('Histology and segmented object', fontsize=14)
                plt.plot([p[0] for p in contour_aux], [p[1] for p in contour_aux], color='k', linewidth=2)
                plt.axis('off')
                plt.subplot(222)
                plt.imshow(pred_class_test[0, :, :, 0], cmap='plasma', vmin=0, vmax=1)
                plt.colorbar(shrink=1)
                plt.title('Pixel WAT scores', fontsize=14)
                plt.axis('off')
                plt.subplot(223)
                plt.imshow(pred_class_test[0, :, :, 0] > 0.50)
                plt.title('Pixel threshold, score > 0.50', fontsize=14)
                plt.axis('off')
                plt.subplot(224)
                plt.imshow(pred_class_test[0, :, :, 0] > 0.50)
                plt.plot([p[0] for p in contour_aux], [p[1] for p in contour_aux], color='g', linewidth=2)
                plt.title('Prop$_{\mathrm{WAT}}$ = %0.1f%%\nWAT if Prop$_{\mathrm{WAT}}$ > 50%%'
                          % (100 * df_common.loc[j, 'wat_prop_50']), fontsize=14)
                plt.axis('off')
                plt.tight_layout()

            if DEBUG:
                # close the contour for the plot
                contour_aux = contour.copy()
                contour_aux.append(contour[0])

                plt.clf()
                plt.subplot(231)
                plt.cla()
                plt.imshow(im)
                plt.contour(cell_seg_contour, colors='C0')
                plt.axis('off')
                plt.title('Histology', fontsize=14)
                plt.axis('off')
                plt.subplot(232)
                plt.cla()
                plt.imshow(pred_class_test[0, :, :, 0], cmap='plasma', vmin=0.0, vmax=1.0)
                cb = plt.colorbar(shrink=0.65)
                cb.ax.tick_params(labelsize=12)
                plt.contour(cell_seg_contour, colors='w')
                plt.title('Pixel WAT score', fontsize=14)
                plt.axis('off')
                plt.subplot(234)
                plt.cla()
                plt.imshow(pred_class_test[0, :, :, 0] > 0.50, cmap='plasma')
                plt.contour(cell_seg_contour, colors='r')
                aux = df_common.loc[j, 'wat_prop_50'] * 100
                plt.title('WAT score > 0.50\nProp$_{\mathrm{WAT}}$ = %0.1f%%' % aux, fontsize=14)
                plt.axis('off')
                plt.subplot(235)
                plt.cla()
                plt.imshow(pred_class_test[0, :, :, 0] > 0.44, cmap='plasma')
                plt.contour(cell_seg_contour, colors='r')
                aux = df_common.loc[j, 'wat_prop_44'] * 100
                plt.title('WAT score > 0.44\nProp$_{\mathrm{WAT}}$ = %0.1f%%' % aux, fontsize=14)
                plt.axis('off')
                plt.subplot(236)
                plt.cla()
                plt.imshow(pred_class_test[0, :, :, 0] > 0.56, cmap='plasma')
                plt.contour(cell_seg_contour, colors='r')
                aux = df_common.loc[j, 'wat_prop_56'] * 100
                plt.title('WAT score > 0.56\nProp$_{\mathrm{WAT}}$ = %0.1f%%' % aux, fontsize=14)
                plt.axis('off')
                plt.tight_layout()

        # concatenate current dataframe to general dataframe
        df_all = df_all.append(df_common, ignore_index=True)

# save results
data_filename = os.path.join(saved_models_dir, experiment_id + '_classifier_by_object.pkl')
df_all.to_pickle(data_filename)

''' Analyse results '''

# load data computed in the previous section
data_filename = os.path.join(saved_models_dir, experiment_id + '_classifier_by_object.pkl')
df_all = pd.read_pickle(data_filename)

# df_all['wat_prop_10']: proportions of pixels within the image with score > 0.10

y_wat_true = df_all['type'] == 'wat'

if DEBUG:

    # init outputs
    tpr_target = []
    obj_thr_target = []
    roc_auc = []

    # pixel score thresholds
    pix_thr = np.array(range(50, 101))
    for p in pix_thr:
        # ROC curve
        fpr, tpr, thr = roc_curve(y_true=y_wat_true, y_score=df_all['wat_prop_' + str(p)])
        roc_auc.append(auc(fpr, tpr))

        # we fix the FPR (False Positive Rate) and interpolate the TPR (True Positive Rate) on the ROC curve
        fpr_target = 0.10
        tpr_target.append(np.interp(fpr_target, fpr, tpr))

        # we also interpolate the corresponding object proportion threshold
        aux = np.interp(fpr_target, fpr, thr)
        aux = np.min((1.0, aux))
        obj_thr_target.append(aux)
    tpr_target = np.array(tpr_target)
    obj_thr_target = np.array(obj_thr_target)

    # maximum TPR for the FPR = 0.05
    idx_max_tpr = np.argmax(tpr_target)

    plt.clf()
    plt.subplot(121)
    plt.plot(pix_thr / 100, tpr_target * 100)
    plt.plot([pix_thr[idx_max_tpr] / 100,] * 2, [5, tpr_target[idx_max_tpr] * 100], '--C0')
    plt.scatter([pix_thr[idx_max_tpr] / 100], [tpr_target[idx_max_tpr] * 100],
                label='Pixel WAT score thr. = %0.2f\nObject WAT FPR = %0.0f%%\nObject WAT TPR = %0.0f%%'
                      % (pix_thr[idx_max_tpr] / 100, fpr_target * 100, tpr_target[idx_max_tpr] * 100), s=100)
    plt.tick_params(labelsize=14)
    plt.xlabel('Pixel WAT score threshold', fontsize=14)
    plt.ylabel('Object WAT TPR (%%) for FPR = %0.0f%%' % (fpr_target * 100), fontsize=14)
    plt.legend(loc='best', prop={'size': 12})

    plt.subplot(122)
    plt.plot(pix_thr / 100, obj_thr_target * 100, 'C0')
    plt.plot([pix_thr[idx_max_tpr] / 100,] * 2, [0, obj_thr_target[idx_max_tpr] * 100], '--C0')
    plt.scatter([pix_thr[idx_max_tpr] / 100], [obj_thr_target[idx_max_tpr] * 100],
                label='Pixel WAT score thr. = %0.2f\nObject WAT score thr. = %0.1f%%'
                      % (pix_thr[idx_max_tpr] / 100, obj_thr_target[idx_max_tpr] * 100), s=100)
    plt.tick_params(labelsize=14)
    plt.xlabel('Pixel WAT score threshold', fontsize=14)
    plt.ylabel('Object WAT score threshold (%)', fontsize=14)
    plt.legend(loc='best', prop={'size': 12})
    plt.tight_layout()

if DEBUG:
    # show problem of the ROC not having data points for low FPR values
    plt.clf()
    p = 50
    fpr, tpr, thr = roc_curve(y_true=y_wat_true, y_score=df_all['wat_prop_' + str(p)])
    plt.plot(fpr, tpr, label='Pixel thr. = 0.50')
    p = 62
    fpr, tpr, thr = roc_curve(y_true=y_wat_true, y_score=df_all['wat_prop_' + str(p)])
    plt.plot(fpr, tpr, label='Pixel thr. = 0.62')
    plt.plot([.1, .1], [0, 1], '--', color='black')
    plt.tick_params(labelsize=14)
    plt.xlabel('FPR', fontsize=14)
    plt.ylabel('TPR', fontsize=14)
    plt.legend(loc='best', prop={'size': 12})
    plt.tight_layout()

if DEBUG:
    plt.clf()
    plt.plot(pix_thr / 100, roc_auc)

if DEBUG:
    # classifier confusion matrix
    idx = np.where(pix_thr == 50)[0][0]
    cytometer.utils.plot_confusion_matrix(y_true=y_wat_true,
                                          y_pred=df_all['wat_prop_50'] >= obj_thr_target[idx],
                                          normalize=True,
                                          title='Object classifier',
                                          xlabel='Predicted',
                                          ylabel='Ground truth',
                                          cmap=plt.cm.Blues,
                                          colorbar=False)
    plt.xticks([0, 1], ('Other', 'WAT'))
    plt.yticks([0, 1], ('Other', 'WAT'))
    plt.tight_layout()


'''
************************************************************************************************************************
Pixel-wise classification validation
************************************************************************************************************************
'''

# load data computed in the previous section
data_filename = os.path.join(saved_models_dir, experiment_id + '_data.npz')
with np.load(data_filename) as data:
    im_array_all = data['im_array_all']
    out_class_all = 1 - data['out_class_all']  # encode as 0: other, 1: WAT
    out_mask_all = data['out_mask_all']

# init output
predict_class_test_all = np.zeros(shape=out_class_all.shape, dtype=np.float32)

for i_fold in range(len(idx_test_all)):

    print('# Fold ' + str(i_fold) + '/' + str(len(idx_test_all) - 1))

    # test and training image indices. These indices refer to file_list
    idx_test = idx_test_all[i_fold]
    idx_train = idx_train_all[i_fold]

    print('## len(idx_train) = ' + str(len(idx_train)))
    print('## len(idx_test) = ' + str(len(idx_test)))

    # split data into training and testing
    im_array_train = im_array_all[idx_train, :, :, :]
    im_array_test = im_array_all[idx_test, :, :, :]

    out_class_train = out_class_all[idx_train, :, :, :]
    out_class_test = out_class_all[idx_test, :, :, :]

    out_mask_train = out_mask_all[idx_train, :, :]
    out_mask_test = out_mask_all[idx_test, :, :]

    # load classification model
    classifier_model_filename = os.path.join(saved_models_dir, classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    classifier_model = keras.models.load_model(classifier_model_filename)

    # reshape model input
    classifier_model = cytometer.utils.change_input_size(classifier_model, batch_shape=im_array_test.shape)

    # apply classification to test data
    predict_class_test = classifier_model.predict(im_array_test, batch_size=batch_size)

    # append data for total output
    predict_class_test_all[idx_test, ...] = predict_class_test

    if DEBUG:
        for i in range(len(idx_test)):

            plt.clf()
            plt.subplot(221)
            plt.imshow(im_array_test[i, :, :, :])
            plt.contour(out_mask_test[i, :, :].astype(np.uint8), colors='r')
            plt.title('i = ' + str(i) + ', Mask', fontsize=14)
            plt.axis('off')
            plt.subplot(222)
            plt.imshow(im_array_test[i, :, :, :])
            plt.imshow(out_class_test[i, :, :, 0].astype(np.uint8), alpha=0.5)
            plt.title('Class', fontsize=14)
            plt.axis('off')
            plt.subplot(212)
            plt.imshow(im_array_test[i, :, :, :])
            plt.imshow(predict_class_test[i, :, :, 0], alpha=0.5)
            plt.title('Predicted class', fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.pause(5)


# save results
data_filename = os.path.join(saved_models_dir, experiment_id + '_pixel_classifier.npz')
np.savez(data_filename, predict_class_test_all=predict_class_test_all)

''' Analyse results '''

# load data computed in previous sections
data_filename = os.path.join(saved_models_dir, experiment_id + '_data.npz')
with np.load(data_filename) as data:
    im_array_all = data['im_array_all']
    out_class_all = 1 - data['out_class_all']  # encode as 0: other, 1: WAT
    out_mask_all = data['out_mask_all']

data_filename = os.path.join(saved_models_dir, experiment_id + '_pixel_classifier.npz')
with np.load(data_filename) as data:
    predict_class_test_all = data['predict_class_test_all']

# vectors of labelled pixels (WAT / Other)
out_mask_all = out_mask_all.astype(np.bool)
out_class_all = out_class_all[:, :, :, 0]
predict_class_test_all = predict_class_test_all[:, :, :, 0]  # wat = larger score
y_wat_true = out_class_all[out_mask_all]
y_wat_predict = predict_class_test_all[out_mask_all]

# classifier ROC (we make WAT=1, other=0 for clarity of the results)
fpr, tpr, thr = roc_curve(y_true=y_wat_true, y_score=y_wat_predict)
roc_auc = auc(fpr, tpr)

# interpolate values for pixel score thr = 0.50
thr_target = 0.50
tpr_target = np.interp(thr_target, thr[::-1], tpr[::-1])
fpr_target = np.interp(thr_target, thr[::-1], fpr[::-1])

if DEBUG:
    # ROC curve
    plt.clf()
    plt.plot(fpr * 100, tpr * 100, color='C0', lw=2, label='Pixel ROC. Area = %0.2f' % roc_auc)
    plt.scatter(fpr_target * 100, tpr_target * 100,
                label='Pixel score thr. =  %0.2f, FPR = %0.0f%%, TPR = %0.0f%%'
                      % (thr_target, fpr_target * 100, tpr_target * 100),
                color='C0', s=100)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Pixel WAT False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('Pixel WAT True Positive Rate (TPR)', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 12})
    plt.tight_layout()

    # classifier confusion matrix
    cytometer.utils.plot_confusion_matrix(y_true=y_wat_true,
                                          y_pred=y_wat_predict >= thr_target,
                                          normalize=True,
                                          title='Pixel classifier',
                                          xlabel='Predicted',
                                          ylabel='Ground truth',
                                          cmap=plt.cm.Blues,
                                          colorbar=False)
    plt.xticks([0, 1], ('Other', 'WAT'))
    plt.yticks([0, 1], ('Other', 'WAT'))
    plt.tight_layout()


'''
************************************************************************************************************************
Apply the pipeline v6 to training histology images (segmentation, classification)
************************************************************************************************************************
'''

# correct home directory in file paths
file_svg_list = cytometer.data.change_home_directory(list(file_svg_list), '/users/rittscher/rcasero', home, check_isfile=True)

# load data computed in the previous section
data_filename = os.path.join(saved_models_dir, experiment_id + '_data.npz')
with np.load(data_filename) as data:
    im_array_all = data['im_array_all']
    rough_mask_all = data['rough_mask_all']
    out_class_all = 1 - data['out_class_all']  # encode as 0: other, 1: WAT
    out_mask_all = data['out_mask_all']

# start timer
t0 = time.time()

# init
df_all = pd.DataFrame()

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

    # contour, dmap and tissue classifier models
    contour_model_filename = \
        os.path.join(saved_models_dir, contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    dmap_model_filename = \
        os.path.join(saved_models_dir, dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    classifier_model_filename = \
        os.path.join(saved_models_dir, classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')

    # segment histology
    pred_seg_test, pred_class_test, _ \
        = cytometer.utils.segment_dmap_contour_v4(im_array_test,
                                                  contour_model=contour_model_filename,
                                                  dmap_model=dmap_model_filename,
                                                  classifier_model=classifier_model_filename,
                                                  border_dilation=0)

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

    # clean segmentation: remove labels that touch the edges, that are too small or that don't overlap enough with
    # the rough foreground mask
    pred_seg_test \
        = cytometer.utils.clean_segmentation(pred_seg_test, remove_edge_labels=False, min_cell_area=min_cell_area,
                                             mask=rough_mask_test, phagocytosis=False)

    if DEBUG:
        plt.clf()
        aux = np.stack((rough_mask_test[i, :, :],) * 3, axis=2)
        plt.imshow(im_array_test[i, :, :, :] * aux)
        plt.contour(pred_seg_test[0, ...], levels=np.unique(pred_seg_test[0, ...]), colors='k')
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

    ''' Compare automatic segmentations to manual segmentations '''

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
                                                              tags_to_keep=['id', 'ko', 'sex'])

        for j, contour in enumerate(contours):

            # start dataframe row for this contour
            df = df_common.copy()
            df['im'] = i
            df['contour'] = j

            # manual segmentation: rasterise object described by contour
            cell_seg_contour = Image.new("1", im_array_test.shape[1:3], "black")  # I = 32-bit signed integer pixels
            draw = ImageDraw.Draw(cell_seg_contour)
            draw.polygon(contour, outline="white", fill="white")
            cell_seg_contour = np.array(cell_seg_contour, dtype=np.bool)

            # find automatic segmentation that best overlaps contour
            lab_best_overlap = mode(pred_wat_seg_test_i[cell_seg_contour]).mode[0]

            if lab_best_overlap == 0:  # the manual contour overlaps the background more than any automatic segmentation
                warnings.warn('Skipping. Contour j = ' + str(j) + ' overlaps with background segmentation lab = 0')

                # add to dataframe
                df['lab_auto'] = 0
                df['dice_auto'] = np.NaN
                df['area_manual'] = np.count_nonzero(cell_seg_contour) * xres * yres  # um^2
                df['area_auto'] = np.NaN
                df['area_corrected'] = np.NaN
                df['wat_prop_auto'] = np.NaN
                df['wat_prop_corrected'] = np.NaN

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
                df['label_seg'] = lab_best_overlap
                df['dice_auto'] = dice_auto
                df['dice_corrected'] = dice_corrected
                df['area_manual'] = np.count_nonzero(cell_seg_contour) * xres * yres  # um^2
                df['area_auto'] = np.count_nonzero(cell_best_overlap) * xres * yres  # um^2
                df['area_corrected'] = area_corrected  # um^2
                df['wat_prop_auto'] = pred_prop_test[pred_lab_test == lab_best_overlap]
                df['wat_prop_corrected'] = wat_prop_corrected

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
                              % (100 * df['wat_prop_auto']), fontsize=14)
                    plt.axis('off')
                    plt.tight_layout()

            # concatenate current dataframe to general dataframe
            df_all = df_all.append(df, ignore_index=True)

    # end of image loop
    print('Time so far: ' + str("{:.1f}".format(time.time() - t0)) + ' s')

# save results to avoid having to recompute them every time (58 min on 2 Titan RTX GPUs)
dataframe_filename = os.path.join(saved_models_dir, experiment_id + '_test_pipeline.pkl')
df_all.to_pickle(dataframe_filename)

## Analyse results

# load dataframe with automatic segmentations, classifications, areas, etc
data_filename = os.path.join(saved_models_dir, experiment_id + '_test_pipeline.pkl')
df_all = pd.read_pickle(data_filename)

idx_wat = df_all['wat_prop_auto'] >= 0.95
idx_not_large = df_all['area_auto'] < 20e3
idx_good_segmentation = df_all['dice_auto'] >= 0.5

# area vs. WAT proportion
plt.clf()
plt.scatter(df_all['wat_prop_auto'], df_all['area_auto'], s=4)
plt.xlabel('Prop. WAT pixels', fontsize=14)
plt.ylabel('Area ($\mu$m$^2$)', fontsize=14)
plt.tick_params(axis="both", labelsize=14)

# compare automatic and corrected areas to manual areas

# objects selected for the plot
idx = idx_wat * idx_not_large

# linear regressions
slope_manual_auto, intercept_manual_auto, \
r_value_manual_auto, p_value_manual_auto, std_err_manual_auto = \
    linregress(df_all['area_manual'][idx],
               df_all['area_auto'][idx])
slope_manual_corrected, intercept_manual_corrected, \
r_value_manual_corrected, p_value_manual_corrected, std_err_manual_corrected = \
    linregress(df_all['area_manual'][idx],
               df_all['area_corrected'][idx])

plt.clf()
plt.plot([0, 20], [0, 20], 'g',
         path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()],
         label=r'Identity ($\alpha=$ 1.00, $\beta=$0.00)')
plt.scatter(df_all['area_manual'][idx] / 1e3,
            df_all['area_auto'][idx] / 1e3, s=4, color='C0')
plt.plot([0, 20], (np.array([0, 20e3]) * slope_manual_auto + intercept_manual_auto) / 1e3, color='C0',
         path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()],
         label=r'Auto ($\alpha=$ %0.2f, $\beta=$ %0.2f)' % (slope_manual_auto, intercept_manual_auto))
plt.scatter(df_all['area_manual'][idx] / 1e3,
            df_all['area_corrected'][idx] / 1e3, s=4, color='C1')
plt.plot([0, 20], (np.array([0, 20e3]) * slope_manual_corrected + intercept_manual_corrected) / 1e3, color='C1',
         path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()],
         label=r'Corrected ($\alpha=$ %0.2f, $\beta=$ %0.2f)' % (slope_manual_corrected, intercept_manual_corrected))
plt.legend(fontsize=14)
plt.xlabel('Manual Area ($\cdot 10^{3} \mu$m$^2$)', fontsize=14)
plt.ylabel('Auto/Corrected area ($\cdot 10^{3} \mu$m$^2$)', fontsize=14)
plt.tick_params(axis="both", labelsize=14)

# compare automatic and corrected areas to manual areas (keeping only segmentations with Dice >= 0.5)

# objects selected for the plot
idx = idx_wat * idx_not_large * idx_good_segmentation

# linear regressions
slope_manual_auto, intercept_manual_auto, \
r_value_manual_auto, p_value_manual_auto, std_err_manual_auto = \
    linregress(df_all['area_manual'][idx],
               df_all['area_auto'][idx])
slope_manual_corrected, intercept_manual_corrected, \
r_value_manual_corrected, p_value_manual_corrected, std_err_manual_corrected = \
    linregress(df_all['area_manual'][idx],
               df_all['area_corrected'][idx])

plt.clf()
plt.plot([0, 20e3], [0, 20e3], 'g',
         path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()],
         label=r'Identity ($\alpha=$ 1.00, $\beta=$0.00)')
plt.scatter(df_all['area_manual'][idx],
            df_all['area_auto'][idx], s=4, color='C0')
plt.plot([0, 20e3], np.array([0, 20e3]) * slope_manual_auto + intercept_manual_auto, color='C0',
         path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()],
         label=r'Auto ($\alpha=$ %0.2f, $\beta=$ %0.2f)' % (slope_manual_auto, intercept_manual_auto))
plt.scatter(df_all['area_manual'][idx],
            df_all['area_corrected'][idx], s=4, color='C1')
plt.plot([0, 20e3], np.array([0, 20e3]) * slope_manual_corrected + intercept_manual_corrected, color='C1',
         path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()],
         label=r'Corrected ($\alpha=$ %0.2f, $\beta=$ %0.2f)' % (slope_manual_corrected, intercept_manual_corrected))
plt.legend(fontsize=14)
plt.xlabel('Manual Area ($\mu$m$^2$)', fontsize=14)
plt.ylabel('Auto/Corrected area ($\mu$m$^2$)', fontsize=14)
plt.tick_params(axis="both", labelsize=14)

####################################################################################################################
## CODE BELOW STILL NEEDS TO BE FIXED
####################################################################################################################

# WAT contour areas
idx_wat_manual = df_all_manual['type'] == 'wat'

# Automatic segmentation objects that we classify as WAT
idx_wat_auto = np.array(df_all['prop_wat'] > 0.715) * np.array(df_all['prop_rough_mask'] > 0.9)

if DEBUG:
    plt.clf()
    plt.scatter(np.array(df_all['prop_wat']), np.array(df_all['area_corrected']), s=10)

    plt.clf()
    plt.hist2d(np.array(df_all['prop_wat']), np.array(df_all['area_corrected']), bins=[10, 100])
    plt.xlabel('Prop_WAT', fontsize=14)
    plt.ylabel('Segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.tick_params(axis="both", labelsize=14)

if DEBUG:
    plt.clf()
    boxp = plt.boxplot((df_all_manual['area'][idx_wat_manual],
                        df_all['area'][idx_wat_auto],
                        df_all['area_corrected'][idx_wat_auto]),
                       labels=('Manual', 'Automatic\nsegmentation', 'Corrected\nsegmentation'),
                       notch=True)
    plt.tick_params(axis="both", labelsize=14)
    plt.ylabel('Segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.tight_layout()

    plt.ylim(-500, 7800)

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

    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_w0_manual, 'k--')
    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_25_manual, 'k--')
    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_50_manual, 'C1--')
    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_75_manual, 'k--')
    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_wend_manual, 'k--')

    plt.text(1.89, 1250, '%0.1f%%' % (100 * (contour_perc_50_auto - contour_perc_50_manual) / contour_perc_50_manual),
             fontsize=12, color='C1')
    plt.text(2.84, 2220, '+%0.1f%%' % (100 * (contour_perc_50_corrected - contour_perc_50_manual) / contour_perc_50_manual),
             fontsize=12, color='C1')

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
