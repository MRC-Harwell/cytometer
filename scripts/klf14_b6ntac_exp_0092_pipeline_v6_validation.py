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
import scipy.stats as stats
from enum import IntEnum
import statsmodels.formula.api as smf

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
saved_figures_dir = os.path.join(root_data_dir, 'figures')

# k-folds file
saved_kfolds_filename = 'klf14_b6ntac_exp_0079_generate_kfolds.pickle'

# model names
dmap_model_basename = 'klf14_b6ntac_exp_0086_cnn_dmap'
contour_model_basename = 'klf14_b6ntac_exp_0091_cnn_contour_after_dmap'
classifier_model_basename = 'klf14_b6ntac_exp_0088_cnn_tissue_classifier_fcn'
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
df_manual_all = pd.DataFrame()

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
            plt.tight_layout()

            plt.savefig(os.path.join(saved_figures_dir, 'exp_0092_manual_contours.svg'))

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
        df_manual_all = df_manual_all.append(df_common, ignore_index=True)

# save results
data_filename = os.path.join(saved_models_dir, experiment_id + '_classifier_by_object.pkl')
df_manual_all.to_pickle(data_filename)

''' Analyse results '''

# load data computed in the previous section
data_filename = os.path.join(saved_models_dir, experiment_id + '_classifier_by_object.pkl')
df_manual_all = pd.read_pickle(data_filename)

# df_all['wat_prop_10']: proportions of pixels within the image with score > 0.10

y_wat_true = df_manual_all['type'] == 'wat'

if DEBUG:

    # init outputs
    tpr_target = []
    obj_thr_target = []
    roc_auc = []

    # pixel score thresholds
    pix_thr = np.array(range(50, 101))
    for p in pix_thr:
        # ROC curve
        fpr, tpr, thr = roc_curve(y_true=y_wat_true, y_score=df_manual_all['wat_prop_' + str(p)])
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
    fpr, tpr, thr = roc_curve(y_true=y_wat_true, y_score=df_manual_all['wat_prop_' + str(p)])
    plt.plot(fpr, tpr, label='Pixel thr. = 0.50')
    p = 62
    fpr, tpr, thr = roc_curve(y_true=y_wat_true, y_score=df_manual_all['wat_prop_' + str(p)])
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
    # ROC for pixel threshold = 50%
    plt.clf()
    fpr, tpr, thr = roc_curve(y_true=y_wat_true, y_score=df_manual_all['wat_prop_50'])
    plt.plot(fpr[1:], tpr[1:], label='Pixel thr. = 0.50', linewidth=2, color='C0')
    plt.scatter(fpr[1], tpr[1], color='C0', s=50)
    plt.text(0.21, 0.82, r'Prop. WAT pixels$\geq$%0.2f'
                         '\n'
                         r'FPR=%0.2f, TPR=%0.2f' % (thr[1], fpr[1], tpr[1]), fontsize=14)
    # plt.xlim(0.0, 1.0)
    # plt.ylim(0.0, 1.0)
    plt.tick_params(labelsize=14)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0092_roc_object_classification.svg'))

    # classifier confusion matrix
    idx = np.where(pix_thr == 50)[0][0]
    cytometer.utils.plot_confusion_matrix(y_true=y_wat_true,
                                          y_pred=df_manual_all['wat_prop_50'] >= obj_thr_target[idx],
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
Segmentation validation with pipeline v6.

Loop manual contours and find overlaps with automatically segmented contours. Compute cell areas and prop. of WAT
pixels.
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

# init dataframes
df_manual_all = pd.DataFrame()
df_auto_all = pd.DataFrame()

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

    # clean segmentation: remove labels that are too small or that don't overlap enough with
    # the rough foreground mask
    pred_seg_test \
        = cytometer.utils.clean_segmentation(pred_seg_test, remove_edge_labels=False, min_cell_area=min_cell_area,
                                             mask=rough_mask_test, phagocytosis=False)

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
            plt.savefig(os.path.join(saved_figures_dir, 'exp_0092_pipeline_im_fold_%d_i_%d.svg' % (i_fold, i)),
                        bbox_inches='tight', pad_inches=0)

            # WAT classifier
            plt.clf()
            plt.imshow(pred_class_test[i, :, :, 0])
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(saved_figures_dir, 'exp_0092_pipeline_pred_class_test_fold_%d_i_%d.svg' % (i_fold, i)),
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
                                                              tags_to_keep=['id', 'ko', 'sex'])

        # cells on the edge
        labels_edge = cytometer.utils.edge_labels(pred_seg_test[i, :, :])

        ''' All automatic labels loop '''
        lab_no_edge_unique = np.unique(pred_seg_test[i, :, :])
        lab_no_edge_unique = set(lab_no_edge_unique) - set([0])  # remove background label
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

        ''' Only manual contours and their corresponding auto labels loop '''
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
                                             'exp_0092_pipeline_labels_fold_%d_i_%d_j_%d.svg' % (i_fold, i, j)),
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
                                             'exp_0092_pipeline_window_seg_fold_%d_i_%d_j_%d.svg' % (i_fold, i, j)),
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

    # end of image loop
    print('Time so far: ' + str("{:.1f}".format(time.time() - t0)) + ' s')

# save results to avoid having to recompute them every time (15 min on 2 Titan RTX GPUs)
dataframe_manual_filename = os.path.join(saved_models_dir, experiment_id + '_test_pipeline_manual.pkl')
df_manual_all.to_pickle(dataframe_manual_filename)

dataframe_auto_filename = os.path.join(saved_models_dir, experiment_id + '_test_pipeline_auto.pkl')
df_auto_all.to_pickle(dataframe_auto_filename)

## Analyse results: Manual data

# load dataframe with manual segmentations matched to automatic segmentations
data_manual_filename = os.path.join(saved_models_dir, experiment_id + '_test_pipeline_manual.pkl')
df_manual_all = pd.read_pickle(data_manual_filename)

# boolean vectors to select subsets of rows from the dataframe
idx_manual_wat = np.array(df_manual_all['wat_prop_auto'] >= 0.95)
idx_manual_not_large = np.array(df_manual_all['area_auto'] < 20e3)
idx_manual_good_segmentation = np.array(df_manual_all['dice_auto'] >= 0.5)
idx_manual_not_edge = np.logical_not(df_manual_all['auto_is_edge_cell'])

# remove fold/image pairs that contain only "other" tissue or broken cells
other_broken_fold_im = [(2, 6), (6, 0), (8, 3), (8, 4), (7, 3), (7, 4)]
idx_manual_slice_with_wat = np.logical_not([x in other_broken_fold_im
                                            for x in zip(df_manual_all['fold'], df_manual_all['im'])])

# remove fold/image pairs that contain substantial areas with broken cells or poor segmentations in general
poor_seg_fold_im = [(4, 5), (5, 0), (3, 1), (6, 3), (4, 2), (3, 5)]
idx_manual_slice_with_acceptable_seg = np.logical_not([x in poor_seg_fold_im
                                                       for x in zip(df_manual_all['fold'], df_manual_all['im'])])

# area vs. WAT proportion
plt.clf()
plt.scatter(df_manual_all['wat_prop_auto'], df_manual_all['area_auto'], s=4)
plt.xlabel('Prop. WAT pixels', fontsize=14)
plt.ylabel('Area ($\mu$m$^2$)', fontsize=14)
plt.tick_params(axis="both", labelsize=14)
plt.tight_layout()

# histogram of area correction factor

# objects selected for the plot
idx = idx_manual_wat * idx_manual_not_large * idx_manual_good_segmentation

# median and std of ratios
med_auto = np.median(df_manual_all['area_auto'][idx] / df_manual_all['area_manual'][idx])
med_corrected = np.median(df_manual_all['area_corrected'][idx] / df_manual_all['area_manual'][idx])
q1_auto = np.quantile(df_manual_all['area_auto'][idx] / df_manual_all['area_manual'][idx], 0.25)
q1_corrected = np.quantile(df_manual_all['area_corrected'][idx] / df_manual_all['area_manual'][idx], 0.25)
q3_auto = np.quantile(df_manual_all['area_auto'][idx] / df_manual_all['area_manual'][idx], 0.75)
q3_corrected = np.quantile(df_manual_all['area_corrected'][idx] / df_manual_all['area_manual'][idx], 0.75)

plt.clf()
plt.hist(df_manual_all['area_auto'][idx] / df_manual_all['area_manual'][idx], bins=51, histtype='step', linewidth=3,
         density=True,
         label='Auto / Manual area\nQ1, Q2, Q3 = %0.2f, %0.2f, %0.2f'
               % (q1_auto, med_auto, q3_auto))
plt.hist(df_manual_all['area_corrected'][idx] / df_manual_all['area_manual'][idx], bins=51, histtype='step', linewidth=3,
         density=True,
         label='Corrected / Manual area\nQ1, Q2, Q3 = %0.2f, %0.2f, %0.2f'
               % (q1_corrected, med_corrected, q3_corrected))
plt.plot([1, 1], [0, 3.3], 'k', linewidth=2)
plt.legend(fontsize=14)
plt.xlabel('Pipeline / Manual Area', fontsize=14)
plt.ylabel('Histogram density', fontsize=14)
plt.tick_params(axis="both", labelsize=14)
plt.tight_layout()

plt.savefig(os.path.join(saved_figures_dir, 'exp_0092_areas_ratio_hist.svg'))

# compare automatic and corrected areas to manual areas

# objects selected for the plot
idx = idx_manual_wat * idx_manual_not_large

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
plt.legend(fontsize=14)
plt.xlabel('Manual Area ($\cdot 10^{3} \mu$m$^2$)', fontsize=14)
plt.ylabel('Auto/Corrected area ($\cdot 10^{3} \mu$m$^2$)', fontsize=14)
plt.tick_params(axis="both", labelsize=14)

# compare automatic and corrected areas to manual areas (keeping only segmentations with Dice >= 0.5)

# objects selected for the plot
idx = idx_manual_wat * idx_manual_not_large * idx_manual_good_segmentation * idx_manual_slice_with_wat

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
plt.plot([0, 20e3], [0, 20e3], 'g',
         path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()],
         label=r'Identity ($\alpha=$ 1.00, $\beta=$0.00)')
plt.scatter(df_manual_all['area_manual'][idx],
            df_manual_all['area_auto'][idx], s=4, color='C0')
plt.plot([0, 20e3], np.array([0, 20e3]) * slope_manual_auto + intercept_manual_auto, color='C0',
         path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()],
         label=r'Auto ($\alpha=$ %0.2f, $\beta=$ %0.2f)' % (slope_manual_auto, intercept_manual_auto))
plt.scatter(df_manual_all['area_manual'][idx],
            df_manual_all['area_corrected'][idx], s=4, color='C1')
plt.plot([0, 20e3], np.array([0, 20e3]) * slope_manual_corrected + intercept_manual_corrected, color='C1',
         path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()],
         label=r'Corrected ($\alpha=$ %0.2f, $\beta=$ %0.2f)' % (slope_manual_corrected, intercept_manual_corrected))
plt.legend(fontsize=14)
plt.xlabel('Manual Area ($\mu$m$^2$)', fontsize=14)
plt.ylabel('Auto/Corrected area ($\mu$m$^2$)', fontsize=14)
plt.tick_params(axis="both", labelsize=14)

# plot for poster

# objects selected for the plot
idx = idx_manual_wat * idx_manual_not_large * idx_manual_slice_with_wat * idx_manual_slice_with_acceptable_seg * idx_manual_not_edge * idx_manual_good_segmentation

# boxplots of cell populations
plt.clf()
bp = plt.boxplot([df_manual_all['area_manual'][idx], df_manual_all['area_auto'][idx], df_manual_all['area_corrected'][idx]],
                 positions=[1, 2, 3], notch=True, labels=['Manual', 'Auto', 'Corrected'])

# points of interest in the manual contours boxplot
bp_perc_w0_manual = bp['whiskers'][0].get_data()[1][1]
bp_perc_25_manual = bp['boxes'][0].get_data()[1][1]
bp_perc_50_manual = bp['medians'][0].get_data()[1][0]
bp_perc_75_manual = bp['boxes'][0].get_data()[1][5]
bp_perc_wend_manual = bp['whiskers'][1].get_data()[1][1]

plt.plot([0.75, 3.25], [bp_perc_50_manual, ] * 2, 'C1', linestyle='dotted')
plt.plot([0.75, 3.25], [bp_perc_25_manual, ] * 2, 'k', linestyle='dotted')
plt.plot([0.75, 3.25], [bp_perc_75_manual, ] * 2, 'k', linestyle='dotted')
plt.tick_params(axis="both", labelsize=14)
plt.ylabel('Area ($\mu$m$^2$)', fontsize=14)
plt.ylim(-800, 8500)


## Analyse results: Automatic data

# load dataframe with all auto segmentations
data_auto_filename = os.path.join(saved_models_dir, experiment_id + '_test_pipeline_auto.pkl')
df_auto_all = pd.read_pickle(data_auto_filename)

# load dataframe with manual segmentations matched to automatic segmentations
data_manual_filename = os.path.join(saved_models_dir, experiment_id + '_test_pipeline_manual.pkl')
df_manual_all = pd.read_pickle(data_manual_filename)

# boolean vectors to select subsets of rows from the dataframe
idx_manual_wat = np.array(df_manual_all['wat_prop_auto'] >= 0.95)
idx_manual_not_large = np.array(df_manual_all['area_auto'] < 20e3)
idx_manual_good_segmentation = np.array(df_manual_all['dice_auto'] >= 0.5)
idx_manual_not_edge = np.logical_not(df_manual_all['auto_is_edge_cell'])

idx_auto_wat = np.array(df_auto_all['wat_prop_auto'] >= 0.95)
idx_auto_not_large = np.array(df_auto_all['area_auto'] < 20e3)
idx_auto_not_edge = np.logical_not(df_auto_all['auto_is_edge_cell'])

# objects selected for the plot
idx_auto = idx_auto_wat * idx_auto_not_large * idx_auto_not_edge

# boxplots of cell populations
plt.clf()
bp = plt.boxplot([df_manual_all['area_manual'] * 1e-3, df_auto_all['area_auto'][idx_auto] * 1e-3,
                 df_auto_all['area_corrected'][idx_auto] * 1e-3],
                 positions=[1, 2, 3], notch=True, labels=['Manual', 'Auto', 'Corrected'])

# points of interest in the manual contours boxplot
# bp_perc_w0_manual = bp['whiskers'][0].get_data()[1][1]
bp_perc_25_manual = bp['boxes'][0].get_data()[1][1]
bp_perc_50_manual = bp['medians'][0].get_data()[1][0]
bp_perc_75_manual = bp['boxes'][0].get_data()[1][5]
# bp_perc_wend_manual = bp['whiskers'][1].get_data()[1][1]

plt.plot([0.75, 3.25], [bp_perc_50_manual, ] * 2, 'C1', linestyle='dotted')
plt.plot([0.75, 3.25], [bp_perc_25_manual, ] * 2, 'k', linestyle='dotted')
plt.plot([0.75, 3.25], [bp_perc_75_manual, ] * 2, 'k', linestyle='dotted')
plt.tick_params(axis="both", labelsize=14)
plt.ylabel('Area ($\cdot 10^{3} \mu$m$^2$)', fontsize=14)
plt.ylim(-800 * 1e-3, 8500 * 1e-3)
plt.tight_layout()

plt.text(1.20, bp_perc_25_manual + .1, 'Q1=%0.1f' % (bp_perc_25_manual), fontsize=12, color='k')
plt.text(1.20, bp_perc_50_manual + .1, 'Q2=%0.1f' % (bp_perc_50_manual), fontsize=12, color='k')
plt.text(1.20, bp_perc_75_manual + .1, 'Q3=%0.1f' % (bp_perc_75_manual), fontsize=12, color='k')

bp_perc_25_manual = bp['boxes'][1].get_data()[1][1]
bp_perc_50_manual = bp['medians'][1].get_data()[1][0]
bp_perc_75_manual = bp['boxes'][1].get_data()[1][5]

plt.text(2.20, bp_perc_25_manual + .1 - 0.25, '%0.1f' % (bp_perc_25_manual), fontsize=12, color='k')
plt.text(2.20, bp_perc_50_manual + .1 - 0.1, '%0.1f' % (bp_perc_50_manual), fontsize=12, color='k')
plt.text(2.20, bp_perc_75_manual + .1 - 0.05, '%0.1f' % (bp_perc_75_manual), fontsize=12, color='k')

bp_perc_25_manual = bp['boxes'][2].get_data()[1][1]
bp_perc_50_manual = bp['medians'][2].get_data()[1][0]
bp_perc_75_manual = bp['boxes'][2].get_data()[1][5]

plt.text(3.20, bp_perc_25_manual + .1, '%0.1f' % (bp_perc_25_manual), fontsize=12, color='k')
plt.text(3.20, bp_perc_50_manual + .2, '%0.1f' % (bp_perc_50_manual), fontsize=12, color='k')
plt.text(3.20, bp_perc_75_manual + .2, '%0.1f' % (bp_perc_75_manual), fontsize=12, color='k')

plt.savefig(os.path.join(saved_figures_dir, 'exp_0092_area_boxplots.svg'))

''' 
************************************************************************************************************************
Linear Mixed Effects Model analysis (sqrt_area ~ sex + ko + other variables + random(mouse|window))
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
df_manual_all['ko'] = df_manual_all['ko'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
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
Statistical analysis of manual data
************************************************************************************************************************
'''

## Analyse results: Manual data

# load dataframe with manual segmentations matched to automatic segmentations
data_manual_filename = os.path.join(saved_models_dir, experiment_id + '_test_pipeline_manual.pkl')
df_manual_all = pd.read_pickle(data_manual_filename)

## boxplots of PAT vs MAT in male

idx_f_mat = np.logical_and(df_manual_all['sex'] == 'f', df_manual_all['ko'] == 'MAT')
idx_f_pat = np.logical_and(df_manual_all['sex'] == 'f', df_manual_all['ko'] == 'PAT')
idx_m_mat = np.logical_and(df_manual_all['sex'] == 'm', df_manual_all['ko'] == 'MAT')
idx_m_pat = np.logical_and(df_manual_all['sex'] == 'm', df_manual_all['ko'] == 'PAT')

if DEBUG:
    plt.clf()
    plt.boxplot([df_manual_all['area_manual'][idx_f_pat] * 1e-3,
                 df_manual_all['area_manual'][idx_f_mat] * 1e-3,
                 df_manual_all['area_manual'][idx_m_pat] * 1e-3,
                 df_manual_all['area_manual'][idx_m_mat] * 1e-3
                 ],
                labels=['f/PAT', 'f/MAT', 'm/PAT', 'm/MAT'],
                positions=[1, 2, 3.5, 4.5],
                notch=True)
    plt.ylim(-875/1e3, 11)
    plt.ylabel('Area ($\cdot 10^{-3} \mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0092_area_boxplots_by_sex_ko.svg'))

# difference in medians
area_median_m_pat = np.median(df_manual_all['area_manual'][idx_m_pat])
area_median_m_mat = np.median(df_manual_all['area_manual'][idx_m_mat])

print((area_median_m_pat - area_median_m_mat) / area_median_m_pat)

# compute all percentiles
area_perc_f_pat = np.percentile(df_manual_all['area_manual'][idx_f_pat], range(1, 100))
area_perc_f_mat = np.percentile(df_manual_all['area_manual'][idx_f_mat], range(1, 100))
area_perc_m_pat = np.percentile(df_manual_all['area_manual'][idx_m_pat], range(1, 100))
area_perc_m_mat = np.percentile(df_manual_all['area_manual'][idx_m_mat], range(1, 100))

# area change from PAT to MAT
area_change_f_pat2mat = (area_perc_f_mat - area_perc_f_pat) / area_perc_f_pat
area_change_m_pat2mat = (area_perc_m_mat - area_perc_m_pat) / area_perc_m_pat

# permutation testing
quantiles_f_pat2mat, pval_f_pat2mat, reject_f_pat2mat \
    = cytometer.utils.compare_ecdfs(df_manual_all['area_manual'][idx_f_pat], df_manual_all['area_manual'][idx_f_mat],
                                    alpha=0.05, num_perms=10000, multitest_method=None,
                                    rng_seed=0, resampling_method='bootstrap')
quantiles_corr_f_pat2mat, pval_corr_f_pat2mat, reject_corr_f_pat2mat \
    = cytometer.utils.compare_ecdfs(x=df_manual_all['area_manual'][idx_f_pat], y=df_manual_all['area_manual'][idx_f_mat],
                                    alpha=0.05, num_perms=10000, multitest_method='fdr_by',
                                    rng_seed=0, resampling_method='bootstrap')
quantiles_m_pat2mat, pval_m_pat2mat, reject_m_pat2mat \
    = cytometer.utils.compare_ecdfs(df_manual_all['area_manual'][idx_m_pat], df_manual_all['area_manual'][idx_m_mat],
                                    alpha=0.05, num_perms=10000, multitest_method=None,
                                    rng_seed=0, resampling_method='bootstrap')
quantiles_corr_m_pat2mat, pval_corr_m_pat2mat, reject_corr_m_pat2mat \
    = cytometer.utils.compare_ecdfs(df_manual_all['area_manual'][idx_m_pat], df_manual_all['area_manual'][idx_m_mat],
                                    alpha=0.05, num_perms=10000, multitest_method='fdr_by',
                                    rng_seed=0, resampling_method='bootstrap')

# remove points for 0% and 100%
quantiles_f_pat2mat = quantiles_f_pat2mat[1:100]
pval_f_pat2mat = pval_f_pat2mat[1:100]
reject_f_pat2mat = reject_f_pat2mat[1:100]

quantiles_corr_f_pat2mat = quantiles_corr_f_pat2mat[1:100]
pval_corr_f_pat2mat = pval_corr_f_pat2mat[1:100]
reject_corr_f_pat2mat = reject_corr_f_pat2mat[1:100]

quantiles_m_pat2mat = quantiles_m_pat2mat[1:100]
pval_m_pat2mat = pval_m_pat2mat[1:100]
reject_m_pat2mat = reject_m_pat2mat[1:100]

quantiles_corr_m_pat2mat = quantiles_corr_m_pat2mat[1:100]
pval_corr_m_pat2mat = pval_corr_m_pat2mat[1:100]
reject_corr_m_pat2mat = reject_corr_m_pat2mat[1:100]

if DEBUG:
    x = np.array(range(1, 100))

    plt.clf()
    plt.plot(range(1, 100), area_change_f_pat2mat * 100, color='C0', linewidth=3, label='Female, p-val $< 0.05$')
    plt.scatter(x[~reject_f_pat2mat], area_change_f_pat2mat[~reject_f_pat2mat] * 100,
                marker='o', color='C0', s=10, linewidths=10, label='Female, p-val $\geq 0.05$')
    plt.plot(range(1, 100), area_change_m_pat2mat * 100, color='C1', linewidth=3, label='Male, p-val $< 0.05$')
    plt.scatter(x[~reject_m_pat2mat], area_change_m_pat2mat[~reject_m_pat2mat] * 100,
                marker='o', color='C1', s=10, linewidths=10, label='Male, p-val $\geq 0.05$')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Population percentile (%)', fontsize=14)
    plt.ylabel('Area change (%) from PAT to MAT', fontsize=14)
    plt.ylim(-30, -5)
    plt.legend(loc='best', prop={'size': 12})
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0092_area_change_pat_to_mat.svg'))

# area change from female to male
area_change_pat_f2m = (area_perc_m_pat - area_perc_f_pat) / area_perc_m_pat
area_change_mat_f2m = (area_perc_m_mat - area_perc_f_mat) / area_perc_m_mat

# permutation testing
quantiles_pat_f2m, pval_pat_f2m, reject_pat_f2m \
    = cytometer.utils.compare_ecdfs(df_manual_all['area_manual'][idx_f_pat], df_manual_all['area_manual'][idx_m_pat],
                                    alpha=0.05, num_perms=10000, multitest_method=None,
                                    rng_seed=0, resampling_method='bootstrap')
quantiles_corr_pat_f2m, pval_corr_pat_f2m, reject_corr_pat_f2m \
    = cytometer.utils.compare_ecdfs(df_manual_all['area_manual'][idx_f_pat], df_manual_all['area_manual'][idx_m_pat],
                                    alpha=0.05, num_perms=10000, multitest_method='fdr_by',
                                    rng_seed=0, resampling_method='bootstrap')
quantiles_mat_f2m, pval_mat_f2m, reject_mat_f2m \
    = cytometer.utils.compare_ecdfs(df_manual_all['area_manual'][idx_f_mat], df_manual_all['area_manual'][idx_m_mat],
                                    alpha=0.05, num_perms=10000, multitest_method=None,
                                    rng_seed=0, resampling_method='bootstrap')
quantiles_corr_mat_f2m, pval_corr_mat_f2m, reject_corr_mat_f2m \
    = cytometer.utils.compare_ecdfs(df_manual_all['area_manual'][idx_f_mat], df_manual_all['area_manual'][idx_m_mat],
                                    alpha=0.05, num_perms=10000, multitest_method='fdr_by',
                                    rng_seed=0, resampling_method='bootstrap')

# remove points for 0% and 100%
quantiles_pat_f2m = quantiles_pat_f2m[1:100]
pval_pat_f2m = pval_pat_f2m[1:100]
reject_pat_f2m = reject_pat_f2m[1:100]

quantiles_corr_pat_f2m = quantiles_corr_pat_f2m[1:100]
pval_corr_pat_f2m = pval_corr_pat_f2m[1:100]
reject_corr_pat_f2m = reject_corr_pat_f2m[1:100]

quantiles_mat_f2m = quantiles_mat_f2m[1:100]
pval_mat_f2m = pval_mat_f2m[1:100]
reject_mat_f2m = reject_mat_f2m[1:100]

quantiles_corr_mat_f2m = quantiles_corr_mat_f2m[1:100]
pval_corr_mat_f2m = pval_corr_mat_f2m[1:100]
reject_corr_mat_f2m = reject_corr_mat_f2m[1:100]

if DEBUG:
    x = np.array(range(1, 100))

    plt.clf()
    plt.plot(range(1, 100), area_change_pat_f2m * 100, color='C0', linewidth=3, label='PAT, p-val $< 0.05$')
    plt.scatter(x[~reject_pat_f2m], area_change_pat_f2m[~reject_pat_f2m] * 100,
                marker='o', color='C0', s=10, linewidths=10, label='PAT, p-val $\geq 0.05$')
    plt.plot(range(1, 100), area_change_mat_f2m * 100, color='C1', linewidth=3, label='MAT, p-val $< 0.05$')
    plt.scatter(x[~reject_mat_f2m], area_change_mat_f2m[~reject_mat_f2m] * 100,
                marker='o', color='C1', s=10, linewidths=10, label='MAT, p-val $\geq 0.05$')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Population percentile (%)', fontsize=14)
    plt.ylabel('Area change (%) from female to male', fontsize=14)
    plt.ylim(50, 77)
    plt.legend(loc='best', prop={'size': 12})
    plt.tight_layout()

    plt.savefig(os.path.join(saved_figures_dir, 'exp_0092_area_change_f_to_m.svg'))

# permutation testing
quantiles, pval, reject = cytometer.utils.compare_ecdfs(area_perc_m_pat, area_perc_m_mat, alpha=0.05,
                                                        num_perms=10000, multitest_method=None,
                                                        rng_seed=0, resampling_method='bootstrap')
quantiles_corr, pval_corr, reject_corr = cytometer.utils.compare_ecdfs(area_perc_m_pat, area_perc_m_mat, alpha=0.05,
                                                                       num_perms=10000, multitest_method='fdr_by',
                                                                       rng_seed=0, resampling_method='bootstrap')

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
