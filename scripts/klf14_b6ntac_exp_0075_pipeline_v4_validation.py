"""
Validate pipeline v4:
 * segmentation
   * dmap (0056)
   * contour (0070)
 * classifier (0074)
 * segmentation correction (0053) networks

"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0075_pipeline_v4_validation'

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
from PIL import Image, ImageDraw, ImageEnhance
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
local_threshold_block_size = 41

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
classifier_model_basename = 'klf14_b6ntac_exp_0074_cnn_tissue_classifier_fcn'
correction_model_basename = 'klf14_b6ntac_exp_0053_cnn_quality_network_fcn_overlapping_scaled_contours'

# load list of images, and indices for training vs. testing indices
kfold_filename = os.path.join(saved_models_dir, 'klf14_b6ntac_exp_0055_cnn_contour_kfold_info.pickle')
with open(kfold_filename, 'rb') as f:
    aux = pickle.load(f)
file_list = aux['file_list']
idx_test_all = aux['idx_test']
idx_train_all = aux['idx_train']

# number of images
n_im = len(file_list)

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(root_data_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

'''
************************************************************************************************************************
Prepare the testing data:

  You can skip this if this section has already been run and saved to 
  'klf14_b6ntac_exp_0075_pipeline_v4_validation_data.npz'.

  Apply classifier trained with each 10 folds to the other fold. 
************************************************************************************************************************
'''

'''Load the test data
'''

# start timer
t0 = time.time()

# init output
im_array_all = []
rough_mask_all = []
out_class_all = []
out_mask_all = []
contour_type_all = []
i_all = []

# correct home directory in file paths
file_list = cytometer.data.change_home_directory(list(file_list), '/users/rittscher/rcasero', home, check_isfile=True)

# loop files with hand traced contours
for i, file_svg in enumerate(file_list):

    '''Read histology training window
    '''

    print('file ' + str(i) + '/' + str(len(file_list) - 1))

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
    histology_filename = os.path.join(histology_dir, histology_filename)

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
    contour_type_all.append(contour_type)

    print('Cells: ' + str(len(cell_contours)))
    print('Other: ' + str(len(other_contours)))
    print('Brown: ' + str(len(brown_contours)))

    if (len(contours) == 0):
        print('No contours... skipping')
        continue

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

    # add dummy dimensions for keras
    im_array = np.expand_dims(im_array, axis=0)
    rough_mask_crop = np.expand_dims(rough_mask_crop, axis=0)
    out_class = np.expand_dims(out_class, axis=0)
    out_class = np.expand_dims(out_class, axis=3)
    out_mask = np.expand_dims(out_mask, axis=0)

    # convert to expected types
    im_array = im_array.astype(np.float32)
    rough_mask_crop = rough_mask_crop.astype(np.bool)
    out_class = out_class.astype(np.float32)
    out_mask = out_mask.astype(np.float32)

    # scale image intensities from [0, 255] to [0.0, 1.0]
    im_array /= 255

    # append input/output/mask for later use in training
    im_array_all.append(im_array)
    rough_mask_all.append(rough_mask_crop)
    out_class_all.append(out_class)
    out_mask_all.append(out_mask)
    i_all.append(i)

    print('Time so far: ' + str("{:.1f}".format(time.time() - t0)) + ' s')

# collapse lists into arrays
im_array_all = np.concatenate(im_array_all)
rough_mask_all = np.concatenate(rough_mask_all)
out_class_all = np.concatenate(out_class_all)
out_mask_all = np.concatenate(out_mask_all)

# save results to avoid having to recompute them every time
data_filename = os.path.join(saved_models_dir, experiment_id + '_data.npz')
np.savez(data_filename, im_array_all=im_array_all, rough_mask_all=rough_mask_all, out_class_all=out_class_all,
         out_mask_all=out_mask_all, i_all=i_all)

'''
************************************************************************************************************************
Object-wise classification
************************************************************************************************************************
'''

# correct home directory in file paths
file_list = cytometer.data.change_home_directory(list(file_list), '/users/rittscher/rcasero', home, check_isfile=True)

# load data computed in the previous section
data_filename = os.path.join(saved_models_dir, experiment_id + '_data.npz')
with np.load(data_filename) as data:
    im_array_all = data['im_array_all']
    out_class_all = data['out_class_all']
    out_mask_all = data['out_mask_all']
    i_all = data['i_all']

# start timer
t0 = time.time()

# init
df_all = pd.DataFrame()

for i_fold in range(len(idx_test_all)):

    print('# Fold ' + str(i_fold) + '/' + str(len(idx_test_all) - 1))

    # test and training image indices. These indices refer to file_list
    idx_test = idx_test_all[i_fold]
    # idx_train = idx_train_all[i_fold]

    # list of test files (used later for the dataframe)
    file_list_test = np.array(file_list)[idx_test]

    # map the indices from file_list to im_array_all (there's an image that had no WAT or Other contours and was
    # skipped)
    idx_lut = np.full(shape=(len(file_list), ), fill_value=-1, dtype=idx_test.dtype)
    idx_lut[i_all] = range(len(i_all))
    # idx_train = idx_lut[idx_train]
    idx_test = idx_lut[idx_test]

    # print('## len(idx_train) = ' + str(len(idx_train)))
    print('## len(idx_test) = ' + str(len(idx_test)))

    # split data into training and testing
    # im_array_train = im_array_all[idx_train, :, :, :]
    im_array_test = im_array_all[idx_test, :, :, :]

    # out_class_train = out_class_all[idx_train, :, :, :]
    out_class_test = out_class_all[idx_test, :, :, :]

    # out_mask_train = out_mask_all[idx_train, :, :]
    out_mask_test = out_mask_all[idx_test, :, :]

    # loop test images
    for i, file_svg in enumerate(file_list_test):

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
            plt.imshow(pred_class_test[0, :, :, 1], cmap='plasma')
            plt.title('Classifier score', fontsize=14)
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
        df_im = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_tif),
                                                          values=im_idx, values_tag='im',
                                                          tags_to_keep=['id', 'ko', 'sex'])
        df_im['contour'] = range(len(contours))
        df_im['type'] = contour_type_all

        '''Label pixels of image as either WAT/non-WAT'''

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
                plt.imshow(pred_class_test[0, :, :, 1], cmap='plasma')
                plt.plot([p[0] for p in contour_aux], [p[1] for p in contour_aux], color='w')
                plt.title('Classifier score', fontsize=14)
                plt.axis('off')
                plt.tight_layout()

            # rasterise object described by contour
            cell_seg_contour = Image.new("1", im.size, "black")  # I = 32-bit signed integer pixels
            draw = ImageDraw.Draw(cell_seg_contour)
            draw.polygon(contour, outline="white", fill="white")
            cell_seg_contour = np.array(cell_seg_contour, dtype=np.uint8)

            # get scores from within the object
            aux = pred_class_test[0, :, :, 1]
            scores = aux[cell_seg_contour == 1]

            # compute proportions for different thresholds of Otherness
            df_im.loc[j, 'prop_20'] = np.count_nonzero(scores > 0.2) / len(scores)
            df_im.loc[j, 'prop_25'] = np.count_nonzero(scores > 0.25) / len(scores)
            df_im.loc[j, 'prop_30'] = np.count_nonzero(scores > 0.3) / len(scores)
            df_im.loc[j, 'prop_35'] = np.count_nonzero(scores > 0.35) / len(scores)
            df_im.loc[j, 'prop_40'] = np.count_nonzero(scores > 0.4) / len(scores)

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
                plt.imshow(pred_class_test[0, :, :, 1], cmap='plasma')
                cb = plt.colorbar(shrink=0.8)
                cb.ax.tick_params(labelsize=12)
                plt.contour(cell_seg_contour, colors='w')
                plt.title('Pixel-wise classifier score', fontsize=14)
                plt.axis('off')
                plt.subplot(234)
                plt.cla()
                plt.imshow(pred_class_test[0, :, :, 1] > 0.2, cmap='plasma')
                plt.contour(cell_seg_contour, colors='r')
                aux = df_im.loc[j, 'prop_20']*100
                plt.title('Prop(Score > 0.2) = %0.1f%%' % aux, fontsize=14)
                plt.axis('off')
                plt.subplot(235)
                plt.cla()
                plt.imshow(pred_class_test[0, :, :, 1] > 0.3, cmap='plasma')
                plt.contour(cell_seg_contour, colors='r')
                aux = df_im.loc[j, 'prop_30']*100
                plt.title('Prop(Score > 0.3) = %0.1f%%' % aux, fontsize=14)
                plt.axis('off')
                plt.subplot(236)
                plt.cla()
                plt.imshow(pred_class_test[0, :, :, 1] > 0.4, cmap='plasma')
                plt.contour(cell_seg_contour, colors='r')
                aux = df_im.loc[j, 'prop_40']*100
                plt.title('Prop(Score > 0.4) = %0.1f%%' % aux, fontsize=14)
                plt.axis('off')
                plt.tight_layout()

        # concatenate current dataframe to general dataframe
        df_all = df_all.append(df_im)

# reindex the dataframe
df_all.reset_index(drop=True, inplace=True)

# save results
data_filename = os.path.join(saved_models_dir, experiment_id + '_classifier_by_object.npz')
np.savez(data_filename, df_all=df_all)

''' Analyse results '''

y_true = df_all['type'] == 'other'
y_score_20 = df_all['prop_20']
y_score_25 = df_all['prop_25']
y_score_30 = df_all['prop_30']
y_score_35 = df_all['prop_35']
y_score_40 = df_all['prop_40']

# classifier ROC (True = "other", and score is a measure of "otherness")
fpr_20, tpr_20, thr_20 = roc_curve(y_true=y_true, y_score=y_score_20)
roc_auc_20 = auc(fpr_20, tpr_20)

fpr_25, tpr_25, thr_25 = roc_curve(y_true=y_true, y_score=y_score_25)
roc_auc_25 = auc(fpr_25, tpr_25)

fpr_30, tpr_30, thr_30 = roc_curve(y_true=y_true, y_score=y_score_30)
roc_auc_30 = auc(fpr_30, tpr_30)

fpr_35, tpr_35, thr_35 = roc_curve(y_true=y_true, y_score=y_score_35)
roc_auc_35 = auc(fpr_35, tpr_35)

fpr_40, tpr_40, thr_40 = roc_curve(y_true=y_true, y_score=y_score_40)
roc_auc_40 = auc(fpr_40, tpr_40)

# find point in the curve for softmax score thr > 0.5
idx_thr_20 = np.where(thr_20 > 0.5)[0][-1]

if DEBUG:
    # find point in the curve for softmax score thr > 0.5
    idx_thr_20 = np.where(thr_20 > 0.5)[0][-1]

    # ROC curve before and after data augmentation
    plt.clf()
    plt.plot(fpr_20, tpr_20, color='C0', lw=2, label='Area = %0.2f' % roc_auc_20)
    plt.plot(fpr_25, tpr_25, color='C1', lw=2, label='Area = %0.2f' % roc_auc_25)
    plt.plot(fpr_30, tpr_30, color='C2', lw=2, label='Area = %0.2f' % roc_auc_30)
    plt.plot(fpr_35, tpr_35, color='C3', lw=2, label='Area = %0.2f' % roc_auc_35)
    plt.plot(fpr_40, tpr_40, color='C4', lw=2, label='Area = %0.2f' % roc_auc_40)
    plt.scatter(fpr_20[idx_thr_20], tpr_20[idx_thr_20],
                label='Thr =  %0.3f, FPR = %0.3f, TPR = %0.3f'
                      % (thr_20[idx_thr_20], fpr_20[idx_thr_20], tpr_20[idx_thr_20]),
                color='blue', s=200)
    plt.tick_params(labelsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="lower right")
    plt.tight_layout()

plt.clf()
plt.boxplot((df_all['prop_20'][df_all['type'] == 'wat'], df_all['prop_20'][df_all['type'] != 'wat'],
             df_all['prop_30'][df_all['type'] == 'wat'], df_all['prop_30'][df_all['type'] != 'wat'],
             df_all['prop_40'][df_all['type'] == 'wat'], df_all['prop_40'][df_all['type'] != 'wat']),
            labels=('WAT\n>20%', 'Other\n>20%', 'WAT\n>30%', 'Other\n>30%', 'WAT\n>40%', 'Other\n>40%'),
            positions=(1, 2, 4, 5, 7, 8), notch=True)
plt.xlabel('Contour type according to pixel-wise threshold', fontsize=14)
plt.ylabel('Prop(Other)', fontsize=14)
plt.tick_params(axis="both", labelsize=14)
plt.tight_layout()

'''
************************************************************************************************************************
Apply the pipeline v4 to training histology images (segmentation, classification)
************************************************************************************************************************
'''

# correct home directory in file paths
file_list = cytometer.data.change_home_directory(list(file_list), '/users/rittscher/rcasero', home, check_isfile=True)

# load data computed in the previous section
data_filename = os.path.join(saved_models_dir, experiment_id + '_data.npz')
with np.load(data_filename) as data:
    im_array_all = data['im_array_all']
    rough_mask_all = data['rough_mask_all']
    out_class_all = data['out_class_all']
    out_mask_all = data['out_mask_all']
    i_all = data['i_all']

# start timer
t0 = time.time()

# init
df_all = pd.DataFrame()

for i_fold in range(len(idx_test_all)):

    print('# Fold ' + str(i_fold) + '/' + str(len(idx_test_all) - 1))

    # test and training image indices. These indices refer to file_list
    idx_test = idx_test_all[i_fold]
    # idx_train = idx_train_all[i_fold]

    # list of test files (used later for the dataframe)
    file_list_test = np.array(file_list)[idx_test]

    # map the indices from file_list to im_array_all (there's an image that had no WAT or Other contours and was
    # skipped)
    idx_lut = np.full(shape=(len(file_list), ), fill_value=-1, dtype=idx_test.dtype)
    idx_lut[i_all] = range(len(i_all))
    # idx_train = idx_lut[idx_train]
    idx_test = idx_lut[idx_test]

    # print('## len(idx_train) = ' + str(len(idx_train)))
    print('## len(idx_test) = ' + str(len(idx_test)))

    # split data into training and testing
    # im_array_train = im_array_all[idx_train, :, :, :]
    im_array_test = im_array_all[idx_test, :, :, :]

    # rough_mask_train = rough_mask_all[idx_train, :, :]
    rough_mask_test = rough_mask_all[idx_test, :, :]

    # out_class_train = out_class_all[idx_train, :, :, :]
    out_class_test = out_class_all[idx_test, :, :, :]

    # out_mask_train = out_mask_all[idx_train, :, :]
    out_mask_test = out_mask_all[idx_test, :, :]

    # loop test images
    for i in range(len(idx_test)):

        print('# Fold ' + str(i_fold) + '/' + str(len(idx_test_all) - 1) + ', i = '
              + str(i) + '/' + str(pred_seg_test.shape[0] - 1))

        # open full resolution histology slide
        file_tif = file_list_test[i].replace('.svg', '.tif')
        im = Image.open(file_tif)

        # read pixel size information
        xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
        yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

        ''' Tissue classification (applied pixel by pixel to the whole image) '''

        # load classification model
        classifier_model_filename = os.path.join(saved_models_dir,
                                                 classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')
        classifier_model = keras.models.load_model(classifier_model_filename)

        # reshape model input
        classifier_model = cytometer.utils.change_input_size(classifier_model, batch_shape=im_array_test.shape)

        # apply classification to test data
        pred_class_test = classifier_model.predict(np.expand_dims(im_array_test[i, ...], axis=0), batch_size=batch_size)

        if DEBUG:
            plt.clf()
            plt.subplot(231)
            aux = np.stack((rough_mask_test[i, :, :], ) * 3, axis=2)
            plt.imshow(im_array_test[i, :, :, :] * aux)
            plt.contour(out_mask_test[i, :, :].astype(np.uint8), colors='r')
            plt.title('Training mask', fontsize=14)
            plt.axis('off')
            plt.subplot(232)
            plt.imshow(im_array_test[i, :, :, :])
            plt.imshow(out_class_test[i, :, :, 0].astype(np.uint8), alpha=0.5)
            plt.title('Ground truth class', fontsize=14)
            plt.axis('off')
            plt.subplot(233)
            plt.imshow(im_array_test[i, :, :, :])
            plt.imshow(pred_class_test[0, :, :, 1], alpha=0.5)
            plt.title('Softmax score', fontsize=14)
            plt.axis('off')
            plt.subplot(234)
            plt.imshow(im_array_test[i, :, :, :])
            plt.imshow(pred_class_test[0, :, :, 1] > 0.20, alpha=0.5)
            plt.title('Score > 0.20', fontsize=14)
            plt.axis('off')
            plt.subplot(235)
            plt.imshow(im_array_test[i, :, :, :])
            plt.imshow(pred_class_test[0, :, :, 1] > 0.30, alpha=0.5)
            plt.title('Score > 0.30', fontsize=14)
            plt.axis('off')
            plt.subplot(236)
            plt.imshow(im_array_test[i, :, :, :])
            plt.imshow(pred_class_test[0, :, :, 1] > 0.40, alpha=0.5)
            plt.title('Score > 0.40', fontsize=14)
            plt.axis('off')
            plt.tight_layout()

        ''' Segmentation into non-overlapping objects '''

        # contour and dmap models
        contour_model_filename = os.path.join(saved_models_dir, contour_model_basename + '_fold_' + str(i_fold) + '.h5')
        dmap_model_filename = os.path.join(saved_models_dir, dmap_model_basename + '_fold_' + str(i_fold) + '.h5')

        # segment histology
        pred_seg_test, _ = cytometer.utils.segment_dmap_contour_v3(np.expand_dims(im_array_test[i, ...], axis=0),
                                                                   contour_model=contour_model_filename,
                                                                   dmap_model=dmap_model_filename,
                                                                   local_threshold_block_size=local_threshold_block_size,
                                                                   border_dilation=0)

        # clean segmentation: remove labels that touch the edges, that are too small or that don't overlap enough with
        # the rough foreground mask
        pred_seg_test \
            = cytometer.utils.clean_segmentation(pred_seg_test, remove_edge_labels=True, min_cell_area=min_cell_area,
                                                 mask=rough_mask_test, phagocytosis=True)

        ''' Split image into individual labels and correct segmentation to take overlaps into account '''

        # load correction model
        correction_model_filename = os.path.join(saved_models_dir,
                                                 correction_model_basename + '_model_fold_' + str(i_fold) + '.h5')
        correction_model = keras.models.load_model(correction_model_filename)

        # split segmentation into separate labels, and scale to same size
        (window_seg_test, window_im_test, window_class_test), index_list, scaling_factor_list \
            = cytometer.utils.one_image_per_label_v2((pred_seg_test, im_array_test, pred_class_test[:, :, :, 1]),
                                                     resize_to=(training_window_len, training_window_len),
                                                     resample=(Image.NEAREST, Image.LINEAR, Image.NEAREST),
                                                     only_central_label=True)

        # correct segmentations
        window_seg_corrected_test = cytometer.utils.correct_segmentation(im=window_im_test * 255, seg=window_seg_test,
                                                                         correction_model=correction_model,
                                                                         model_type='-1_1', batch_size=batch_size,
                                                                         smoothing=11)

        if DEBUG:
            for j in range(window_seg_test.shape[0]):
                plt.clf()
                plt.subplot(221)
                plt.imshow(window_im_test[j, ...])
                plt.contour(window_seg_test[j, ...], colors='k')
                plt.contour(window_seg_corrected_test[j, ...], colors='r')
                plt.title('Histology', fontsize=14)
                plt.axis('off')
                plt.subplot(222)
                plt.imshow(window_class_test[j, ...] > 0.2)
                plt.contour(window_seg_test[j, ...], colors='k')
                plt.contour(window_seg_corrected_test[j, ...], colors='r')
                plt.title('Classifier > 0.2', fontsize=14)
                plt.axis('off')
                plt.subplot(223)
                plt.imshow(window_class_test[j, ...] > 0.3)
                plt.contour(window_seg_test[j, ...], colors='k')
                plt.contour(window_seg_corrected_test[j, ...], colors='r')
                plt.title('Classifier > 0.3', fontsize=14)
                plt.axis('off')
                plt.subplot(224)
                plt.imshow(window_class_test[j, ...] > 0.4)
                plt.contour(window_seg_test[j, ...], colors='k')
                plt.contour(window_seg_corrected_test[j, ...], colors='r')
                plt.title('Classifier > 0.4', fontsize=14)
                plt.axis('off')
                # plt.tight_layout()
                plt.pause(5)

        ''' Quantitative measures '''

        # list of labels (for convenience)
        labels_unique_ref = [x[1] for x in index_list]

        # count number of pixels for each non-overlap label
        labels_count_ref = np.count_nonzero(window_seg_test, axis=(1, 2))

        # largest label
        lab_max = np.max(labels_unique_ref)

        # count number of pixels for each corrected label
        corrected_count_ref = np.count_nonzero(window_seg_corrected_test, axis=(1, 2))

        # pixel size in the resized images (note: they are provided as (sr, sc) in index_list
        sx = np.array([x[1] for x in scaling_factor_list])
        sy = np.array([x[0] for x in scaling_factor_list])

        # create dataframe for this image
        im_idx = [idx_test_all[i_fold][i], ] * len(labels_unique_ref)  # absolute index of current test image
        df_im = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_tif),
                                                          values=im_idx, values_tag='im',
                                                          tags_to_keep=['id', 'ko', 'sex'])
        df_im['lab'] = labels_unique_ref
        df_im['area'] = labels_count_ref * xres * yres / sx / sy
        df_im['area_corrected'] = corrected_count_ref * xres * yres / sx / sy

        if DEBUG:
            plt.clf()
            plt.scatter(df_im['area'], df_im['area_corrected'])
            aux = np.max(df_im['area_corrected'])
            plt.plot([0, aux], [0, aux], 'C1')

        ## compute proportion of "Mask" pixels in each automatically segmented label

        def pixel_proportion(lab_max, labels, mask, labels_unique_ref, labels_count_ref):
            # init vector to get proportion of "Mask" pixels in each label
            labels_prop = np.zeros(shape=(lab_max + 1,))
            # compute proportion of "Mask" pixels to total number of pixels in each label
            labels_unique, labels_count = np.unique(labels * mask, return_counts=True)
            labels_prop[labels_unique] = labels_count
            labels_prop[labels_unique_ref] /= labels_count_ref
            return labels_prop[labels_unique_ref]

        df_im['prop_mask'] = pixel_proportion(lab_max,
                                              labels=pred_seg_test[0, :, :],
                                              mask=out_mask_test[i, :, :].astype(np.int32),
                                              labels_unique_ref=labels_unique_ref,
                                              labels_count_ref=labels_count_ref)

        ## compute proportion of ground truth "Other" pixels in each automatically segmented label

        df_im['prop_other_gtruth'] = pixel_proportion(lab_max,
                                                      labels=pred_seg_test[0, :, :],
                                                      mask=out_class_test[i, :, :, 0].astype(np.int32),
                                                      labels_unique_ref=labels_unique_ref,
                                                      labels_count_ref=labels_count_ref)

        ## compute proportion of classifier "Other" pixels in each automatically segmented label for several softmax
        ## thresholds

        df_im['prop_other_seg_20'] = pixel_proportion(lab_max,
                                                      labels=pred_seg_test[0, :, :],
                                                      mask=pred_class_test[0, :, :, 1] > 0.20,
                                                      labels_unique_ref=labels_unique_ref,
                                                      labels_count_ref=labels_count_ref)

        df_im['prop_other_seg_30'] = pixel_proportion(lab_max,
                                                      labels=pred_seg_test[0, :, :],
                                                      mask=pred_class_test[0, :, :, 1] > 0.30,
                                                      labels_unique_ref=labels_unique_ref,
                                                      labels_count_ref=labels_count_ref)

        df_im['prop_other_seg_40'] = pixel_proportion(lab_max,
                                                      labels=pred_seg_test[0, :, :],
                                                      mask=pred_class_test[0, :, :, 1] > 0.40,
                                                      labels_unique_ref=labels_unique_ref,
                                                      labels_count_ref=labels_count_ref)


        # contatenate current dataframe to general dataframe
        df_all = df_all.append(df_im)

        if DEBUG:
            plt.clf()
            plt.subplot(231)
            plt.imshow(im_array_test[i, :, :, :])
            plt.contour(out_mask_test[i, :, :].astype(np.uint8), colors='r')
            plt.title('Training mask', fontsize=14)
            plt.axis('off')
            plt.subplot(232)
            plt.imshow(im_array_test[i, :, :, :])
            plt.contour(pred_seg_test[0, :, :], levels=np.unique(pred_seg_test[0, :, :]), colors='k')
            plt.contour(out_mask_test[i, :, :].astype(np.uint8), colors='r')
            plt.title('Cleaned automatic segs', fontsize=14)
            plt.axis('off')
            plt.subplot(233)
            plt.imshow(im_array_test[i, :, :, :])
            plt.contour(pred_seg_test[0, :, :], levels=np.unique(pred_seg_test[0, :, :]), colors='k')
            plt.imshow(out_class_test[i, :, :, 0], alpha=0.5)
            plt.title('Class ground truth', fontsize=14)
            plt.axis('off')
            plt.subplot(234)
            plt.imshow(im_array_test[i, :, :, :])
            plt.contour(pred_seg_test[0, :, :], levels=np.unique(pred_seg_test[0, :, :]), colors='k')
            plt.imshow(pred_class_test[0, :, :, 1] > 0.20, alpha=0.5)
            plt.title('Classifier thr > 0.20', fontsize=14)
            plt.axis('off')
            plt.subplot(235)
            plt.imshow(im_array_test[i, :, :, :])
            plt.contour(pred_seg_test[0, :, :], levels=np.unique(pred_seg_test[0, :, :]), colors='k')
            plt.imshow(pred_class_test[0, :, :, 1] > 0.30, alpha=0.5)
            plt.title('Classifier thr > 0.30', fontsize=14)
            plt.axis('off')
            plt.subplot(236)
            plt.imshow(im_array_test[i, :, :, :])
            plt.contour(pred_seg_test[0, :, :], levels=np.unique(pred_seg_test[0, :, :]), colors='k')
            plt.imshow(pred_class_test[0, :, :, 1] > 0.40, alpha=0.5)
            plt.title('Classifier thr > 0.40', fontsize=14)
            plt.axis('off')
            plt.tight_layout()

    # end of image loop
    print('Time so far: ' + str("{:.1f}".format(time.time() - t0)) + ' s')

# reindex the dataframe
df_all.reset_index(drop=True, inplace=True)

# save results to avoid having to recompute them every time (730 s on 2 Titan RTX GPUs)
data_filename = os.path.join(saved_models_dir, experiment_id + '_test_pipeline.npz')
np.savez(data_filename, df_all=df_all)

'''
************************************************************************************************************************
OLD CODE BELOW THIS (it needs to be rewritten or deleted)
************************************************************************************************************************
'''


'''
************************************************************************************************************************
Analyse automatic segmentation and classification results 
************************************************************************************************************************
'''

# load data computed in the previous section
data_filename = os.path.join(saved_models_dir, experiment_id + '_seg_test.npz')
with np.load(data_filename, allow_pickle=True) as data:
    im_array_test_all = data['im_array_test_all']
    out_class_test_all = data['out_class_test_all']
    out_mask_test_all = data['out_mask_test_all']
    pred_class_test_all = data['pred_class_test_all']
    df_all = data['df_all']

# vectors of pixels where we know whether they are WAT or Other
out_mask_test_all = out_mask_test_all.astype(np.bool)
out_class_test_all = out_class_test_all[:, :, :, 0]
y_true = out_class_test_all[out_mask_test_all]
pred_class_test_all = pred_class_test_all[:, :, :, 1]
y_predict = pred_class_test_all[out_mask_test_all]

# classifier ROC (we make cell=1, other=0 for clarity of the results)
fpr, tpr, thr = roc_curve(y_true=y_true, y_score=y_predict)
roc_auc = auc(fpr, tpr)

# find point in the curve for softmax score thr > 0.5
idx_thr = np.where(thr > 0.5)[0][-1]

if DEBUG:
    # find point in the curve for softmax score thr > 0.5
    idx_thr = np.where(thr > 0.5)[0][-1]

    # ROC curve before and after data augmentation
    plt.clf()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.scatter(fpr[idx_thr], tpr[idx_thr],
                label='Thr =  %0.3f, FPR = %0.3f, TPR = %0.3f'
                      % (thr[idx_thr], fpr[idx_thr], tpr[idx_thr]),
                color='blue', s=200)
    plt.tick_params(labelsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="lower right")
    plt.tight_layout()

    # boxplot
    plt.clf()
    plt.boxplot([y_predict[y_true == 0], y_predict[y_true == 1]], labels=('WAT/Background', 'Other'),
                notch=True)
    plt.plot([0.75, 2.25], [thr[idx_thr], ] * 2, 'r', linewidth=2)
    plt.xlabel('Ground truth class', fontsize=14)
    plt.ylabel('Softmax prediction', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()

    # classifier confusion matrix
    cytometer.utils.plot_confusion_matrix(y_true=y_true,
                                          y_pred=y_predict >= thr[idx_thr],
                                          normalize=True,
                                          title='Tissue classifier',
                                          xlabel='Predicted',
                                          ylabel='Ground truth',
                                          cmap=plt.cm.Blues,
                                          colorbar=False)
    plt.xticks([0, 1], ('Cell/\nBg', 'Other'))
    plt.yticks([0, 1], ('Cell/\nBg', 'Other'))
    plt.tight_layout()

    # find point in the curve for softmax score thr > 0.4
    idx_thr = np.where(thr > 0.4)[0][-1]

    # ROC curve before and after data augmentation
    plt.clf()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.scatter(fpr[idx_thr], tpr[idx_thr],
                label='Thr =  %0.3f, FPR = %0.3f, TPR = %0.3f'
                      % (thr[idx_thr], fpr[idx_thr], tpr[idx_thr]),
                color='blue', s=200)
    plt.tick_params(labelsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="lower right")
    plt.tight_layout()

    # boxplot
    plt.clf()
    plt.boxplot([y_predict[y_true == 0], y_predict[y_true == 1]], labels=('WAT/Background', 'Other'),
                notch=True)
    plt.plot([0.75, 2.25], [thr[idx_thr], ] * 2, 'r', linewidth=2)
    plt.xlabel('Ground truth class', fontsize=14)
    plt.ylabel('Softmax prediction', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()

    # classifier confusion matrix
    cytometer.utils.plot_confusion_matrix(y_true=y_true,
                                          y_pred=y_predict >= thr[idx_thr],
                                          normalize=True,
                                          title='Tissue classifier',
                                          xlabel='Predicted',
                                          ylabel='Ground truth',
                                          cmap=plt.cm.Blues,
                                          colorbar=False)
    plt.xticks([0, 1], ('Cell/\nBg', 'Other'))
    plt.yticks([0, 1], ('Cell/\nBg', 'Other'))
    plt.tight_layout()

# find point in the curve for softmax score thr > 0.5
idx_thr = np.where(thr > 0.4)[0][-1]

if DEBUG:
    # ROC curve before and after data augmentation
    plt.clf()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.scatter(fpr[idx_thr], tpr[idx_thr],
                label='Thr =  %0.3f, FPR = %0.3f, TPR = %0.3f'
                      % (thr[idx_thr], fpr[idx_thr], tpr[idx_thr]),
                color='blue', s=200)
    plt.tick_params(labelsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="lower right")
    plt.tight_layout()

    # boxplot
    plt.clf()
    plt.boxplot([y_predict[y_true == 0], y_predict[y_true == 1]], labels=('WAT/Background', 'Other'),
                notch=True)
    plt.plot([0.75, 2.25], [thr[idx_thr],] * 2, 'r', linewidth=2)
    plt.xlabel('Ground truth class', fontsize=14)
    plt.ylabel('Softmax prediction', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()

    # classifier confusion matrix
    cytometer.utils.plot_confusion_matrix(y_true=y_true,
                                          y_pred=y_predict >= thr[idx_thr],
                                          normalize=True,
                                          title='Tissue classifier',
                                          xlabel='Predicted',
                                          ylabel='Ground truth',
                                          cmap=plt.cm.Blues,
                                          colorbar=False)
    plt.xticks([0, 1], ('Cell/\nBg', 'Other'))
    plt.yticks([0, 1], ('Cell/\nBg', 'Other'))
    plt.tight_layout()











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

        # remove labels that touch the edges, that are too small or that are completely surrounded by another cell
        labels \
            = cytometer.utils.clean_segmentation(labels, remove_edge_labels=True, min_cell_area=min_cell_area,
                                                 phagocytosis=True)

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
    quality_model_filename = os.path.join(saved_models_dir, correction_model_basename + '_fold_' + str(i_fold) + '.h5')
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
    plt.tight_layout()

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
    boxp = plt.boxplot([df_0072['area_seg'] / df_0072['area_contour'],
                        df_0072['area_seg_corrected'] / df_0072['area_contour']], labels=['No correction', 'Corrected'],
                       notch=True)
    plt.plot([0.75, 2.25], [1.0, 1.0], color='red')
    plt.tick_params(axis='both', labelsize=14)
    plt.ylabel('Auto segmentation area / Manual segmentation area', fontsize=14)
    plt.ylim(0, 4)
    plt.tight_layout()

# points of interest in the auto, no correction boxplot
contour_perc_50_auto = boxp['medians'][0].get_data()[1][0]

# points of interest in the auto, correction boxplot
contour_perc_50_corrected = boxp['medians'][1].get_data()[1][0]

print('Median from auto segmentation: ' + str(contour_perc_50_auto * 100) + '%')
print('Median from corrected segmentation: ' + str(contour_perc_50_corrected * 100) + '%')

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
    boxp = plt.boxplot([df_0072['area_contour'], df_0072['area_seg'], df_0072['area_seg_corrected']], notch=True,
                       labels=['Manual', 'Automatic', 'Corrected'])
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylabel('Auto segmentation area ($\mu$m$^2$)', fontsize=14)
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

    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_w0_manual, 'k--')
    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_25_manual, 'k--')
    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_50_manual, 'C1--')
    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_75_manual, 'k--')
    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_wend_manual, 'k--')

    plt.ylim(-250, 7500)

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

    if i_fold <= 6:
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
    quality_model_filename = os.path.join(saved_models_dir, correction_model_basename + '_fold_' + str(i_fold) + '.h5')
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
        labels = cytometer.utils.clean_segmentation(labels, remove_edge_labels=True, min_cell_area=min_cell_area,
                                                    mask=rough_mask_crop, min_mask_overlap=0.6, phagocytosis=True)

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

        '''Iterate segmentation labels'''

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

            # process (histology * mask) for segmentation correction
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
                plt.contour(window_seg[0, :, :], linewidths=1, colors='greenyellow')
                plt.contour(window_seg_corrected, linewidths=1, colors='white')
                plt.axis('off')

            '''Object classification as "WAT" or "other"'''

            # process histology for classification
            window_classifier_out = classifier_model.predict(window_im, batch_size=batch_size)

            # get classification label for each pixel
            window_classifier_class = np.argmax(window_classifier_out, axis=3)

            # proportion of "Other" pixels in the mask
            window_other_prop = np.count_nonzero(window_seg_corrected * window_classifier_class[0, :, :]) \
                                / np.count_nonzero(window_seg)

            # add to dataframe row
            df['seg_type_prop'] = window_other_prop

            if DEBUG:
                # plot classification
                plt.clf()
                plt.imshow(window_classifier_class[0, :, :])
                plt.contour(window_seg_corrected, linewidths=1, colors='red')
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
          + str(np.count_nonzero(df_0072_auto['fold'] == i_fold)) + ' (auto)')

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

    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_w0_manual, 'k--')
    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_25_manual, 'k--')
    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_50_manual, 'C1--')
    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_75_manual, 'k--')
    plt.plot([0.75, 3.25], np.array([1, 1]) * contour_perc_wend_manual, 'k--')

    plt.ylim(-250, 6800)

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
ANALYSIS OF MANUAL CONTOUR DATA:

************************************************************************************************************************
'''

time0 = time.time()
df_all = pd.DataFrame()

# correct home directory in file paths
file_list = cytometer.data.change_home_directory(list(file_list), '/users/rittscher/rcasero', home, check_isfile=True)

for i, file_svg in enumerate(file_list):

    print('file ' + str(i) + '/' + str(len(file_list) - 1))

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

    # initialise dataframe to keep results: one cell per row, tagged with mouse metainformation
    df_im = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_tif),
                                                      values=[i], values_tag='file',
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

        # add to dataframe
        df['area_contour'] = np.count_nonzero(cell_seg_contour) * xres * yres

        # append current results to global dataframe
        df_all = pd.concat([df_all, df])

    print('Time so far: ' + str(time.time() - time0) + ' s')

# reset indices
df_all.reset_index(drop=True, inplace=True)

## boxplots of PAT vs MAT in male

idx_f_mat = np.logical_and(df_all['sex'] == 'f', df_all['ko'] == 'MAT')
idx_f_pat = np.logical_and(df_all['sex'] == 'f', df_all['ko'] == 'PAT')
idx_m_mat = np.logical_and(df_all['sex'] == 'm', df_all['ko'] == 'MAT')
idx_m_pat = np.logical_and(df_all['sex'] == 'm', df_all['ko'] == 'PAT')

if DEBUG:
    plt.clf()
    plt.boxplot([df_all['area_contour'][idx_m_pat],
                 df_all['area_contour'][idx_m_mat]],
                labels=['PAT', 'MAT'],
                notch=True)
    plt.ylim(-875, 10e3)
    plt.title('Male', fontsize=14)
    plt.ylabel('Segmentation area ($\mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()

# difference in medians
area_median_m_pat = np.median(df_all['area_contour'][idx_m_pat])
area_median_m_mat = np.median(df_all['area_contour'][idx_m_mat])

print((area_median_m_pat - area_median_m_mat) / area_median_m_pat)

# compute all percentiles
area_perc_m_pat = np.percentile(df_all['area_contour'][idx_m_pat], range(0, 101))
area_perc_m_mat = np.percentile(df_all['area_contour'][idx_m_mat], range(0, 101))

area_change_m_pat2mat = (area_perc_m_mat - area_perc_m_pat) / area_perc_m_pat

if DEBUG:
    plt.clf()
    plt.plot(range(0, 101), area_change_m_pat2mat * 100)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Population percentile', fontsize=14)
    plt.ylabel('Area change (%) from PAT to MAT', fontsize=14)
    plt.ylim(-25, 0)
    plt.tight_layout()

# permutation testing
quantiles, pval, reject = cytometer.utils.compare_ecdfs(area_perc_m_pat, area_perc_m_mat, alpha=0.05,
                                                        num_perms=10000, multitest_method=None)
quantiles_corr, pval_corr, reject_corr = cytometer.utils.compare_ecdfs(area_perc_m_pat, area_perc_m_mat, alpha=0.05,
                                                                       num_perms=10000, multitest_method='fdr_by')

if DEBUG:
    plt.clf()
    plt.subplot(211)
    plt.plot(range(0, 101), area_change_m_pat2mat * 100)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Population percentile', fontsize=14)
    plt.ylabel('Area change (%) from PAT to MAT', fontsize=14)
    plt.ylim(-25, 0)

    plt.subplot(212)
    plt.plot(range(0, 101), pval, label='Uncorrected')
    plt.plot(range(0, 101), pval_corr, label='Benjamini-Yekutieli')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Population percentile', fontsize=14)
    plt.ylabel('p-value', fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()
