"""
Segmentation validation of pipeline v8.

Loop manual contours and find overlaps with automatically segmented contours. Compute cell areas and prop. of WAT
pixels.

Changes over klf14_b6ntac_exp_0096_pipeline_v7_validation:
* Validation is done with new contours method match_overlapping_contours() instead of old labels method
  match_overlapping_labels().
* Instead of going step by step with the pipeline, we use the whole segmentation_pipeline6() function.
* Segmentation clean up has changed a bit to match the cleaning in v8.

Changes over klf14_b6ntac_exp_0108_pipeline_v7_validation:
* Add colour correction so that we have pipeline v8.
"""


# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0109_pipeline_v8_validation'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import sys
if os.path.join(home, 'Software/cytometer') not in sys.path:
    sys.path.extend([os.path.join(home, 'Software/cytometer')])
import numpy as np
import openslide
import pickle
import pandas as pd
import PIL
import matplotlib.pyplot as plt
import scipy

# limit number of GPUs
if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
    print('Limiting visible CUDA devices to: ' + os.environ['CUDA_VISIBLE_DEVICES'])

# force tensorflow environment
os.environ['KERAS_BACKEND'] = 'tensorflow'

import cytometer.data
import cytometer.utils

import keras.backend as K

DEBUG = False
SAVE_FIGS = False

# image enhancer
enhance_contrast = 4.0

# segmentation parameters
min_cell_area = 200  # pixel
max_cell_area = 200e3  # pixel
min_mask_overlap = 0.8
phagocytosis = True
min_class_prop = 0.6
correction_window_len = 401
correction_smoothing = 11

# downsampled slide parameters
downsample_factor_goal = 16  # approximate value, that may vary a bit in each histology file

# data paths
histology_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
histology_ext = '.ndpi'
area2quantile_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v8')
saved_models_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v8')
annotations_dir = os.path.join(home, 'bit/cytometer_data/aida_data_Klf14_v8/annotations')
metainfo_dir = os.path.join(home, 'Data/cytometer_data/klf14')
paper_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')

# file with RGB modes from all training data
klf14_training_colour_histogram_file = os.path.join(saved_models_dir, 'klf14_training_colour_histogram.npz')

# k-folds file
saved_kfolds_filename = 'klf14_b6ntac_exp_0094_generate_extra_training_images.pickle'

# model names
dmap_model_basename = 'klf14_b6ntac_exp_0086_cnn_dmap'
contour_model_basename = 'klf14_b6ntac_exp_0091_cnn_contour_after_dmap'
classifier_model_basename = 'klf14_b6ntac_exp_0095_cnn_tissue_classifier_fcn'
correction_model_basename = 'klf14_b6ntac_exp_0089_cnn_segmentation_correction_overlapping_scaled_contours'

# files to save dataframes with segmentation validation to
dataframe_auto_filename = os.path.join(paper_dir, experiment_id + '_segmentation_validation_auto.csv')
dataframe_corrected_filename = os.path.join(paper_dir, experiment_id + '_segmentation_validation_corrected.csv')

# statistical mode of the background colour (typical colour value) from the training dataset
with np.load(klf14_training_colour_histogram_file) as data:
    mode_r_target = data['mode_r']
    mode_g_target = data['mode_g']
    mode_b_target = data['mode_b']

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

# associate a fold to each training file
fold = -np.ones(shape=(n_im,))  # initialise with -1 values in case a training file has no fold associated to it
for i_fold in range(n_folds):
    fold[idx_test_all[i_fold]] = i_fold

# init dataframes to contain the comparison between hand traced and automatically segmented cells
dataframe_columns = ['file_svg_idx', 'test_idx', 'test_area', 'ref_idx', 'ref_area', 'dice', 'hausdorff']
df_auto_all = pd.DataFrame(columns=dataframe_columns)
df_corrected_all = pd.DataFrame(columns=dataframe_columns)

for i, file_svg in enumerate(file_svg_list):

    print('File ' + str(i) + '/' + str(len(file_svg_list) - 1) + ': ' + os.path.basename(file_svg))

    # load hand traced contours
    cells = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False,
                                                    minimum_npoints=3)
    other_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Other', add_offset_from_filename=False,
                                                             minimum_npoints=3) +\
                     cytometer.data.read_paths_from_svg_file(file_svg, tag='Brown', add_offset_from_filename=False,
                                                             minimum_npoints=3)

    # load training image
    file_im = file_svg.replace('.svg', '.tif')
    im = np.array(PIL.Image.open(file_im))

    if DEBUG:
        plt.clf()
        plt.imshow(im)

    # open histology file the training file was extracted from
    file_whole_slide = os.path.basename(file_svg).split('_row_')[0]
    file_whole_slide = os.path.join(histology_dir, file_whole_slide + '.ndpi')
    im_whole_slide = openslide.OpenSlide(file_whole_slide)

    # get downsampled image
    downsample_level = im_whole_slide.get_best_level_for_downsample(downsample_factor_goal)
    im_downsampled = im_whole_slide.read_region(location=(0, 0), level=downsample_level,
                                                size=im_whole_slide.level_dimensions[downsample_level])
    im_downsampled = np.array(im_downsampled)
    im_downsampled = im_downsampled[:, :, 0:3]

    # estimate the colour mode of the downsampled image, so that we can correct the image tint to match the KLF14
    # training dataset. We apply the same correction to each tile, to avoid that a tile with e.g. only muscle gets
    # overcorrected
    mode_r_tile = scipy.stats.mode(im_downsampled[:, :, 0], axis=None).mode[0]
    mode_g_tile = scipy.stats.mode(im_downsampled[:, :, 1], axis=None).mode[0]
    mode_b_tile = scipy.stats.mode(im_downsampled[:, :, 2], axis=None).mode[0]

    # correct tint of the tile to match KLF14 training data
    im[:, :, 0] = im[:, :, 0] + (mode_r_target - mode_r_tile)
    im[:, :, 1] = im[:, :, 1] + (mode_g_target - mode_g_tile)
    im[:, :, 2] = im[:, :, 2] + (mode_b_target - mode_b_tile)

    # names of contour, dmap and tissue classifier models
    dmap_model_filename = \
        os.path.join(saved_models_dir, dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    contour_model_filename = \
        os.path.join(saved_models_dir, contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    correction_model_filename = \
        os.path.join(saved_models_dir, correction_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    classifier_model_filename = \
        os.path.join(saved_models_dir, classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')

    # segment histology, split into individual objects, and apply segmentation correction
    labels, labels_class, todo_edge, \
    window_im, window_labels, window_labels_corrected, window_labels_class, index_list, scaling_factor_list \
        = cytometer.utils.segmentation_pipeline6(im=im,
                                                 dmap_model=dmap_model_filename,
                                                 contour_model=contour_model_filename,
                                                 correction_model=correction_model_filename,
                                                 classifier_model=classifier_model_filename,
                                                 min_cell_area=min_cell_area,
                                                 max_cell_area=max_cell_area,
                                                 remove_edge_labels=False,
                                                 #mask=istissue_tile,
                                                 #min_mask_overlap=min_mask_overlap,
                                                 phagocytosis=phagocytosis,
                                                 min_class_prop=min_class_prop,
                                                 correction_window_len=correction_window_len,
                                                 correction_smoothing=correction_smoothing,
                                                 return_bbox=True, return_bbox_coordinates='xy')

    # convert labels in single-cell images to contours (points), and add offset so that the contour coordinates are
    # referred to the whole image
    if len(index_list) == 0:
        offset_xy = np.array([])
    else:
        offset_xy = index_list[:, [2, 3]]  # index_list: [i, lab, x0, y0, xend, yend]
    contours_auto = cytometer.utils.labels2contours(window_labels, offset_xy=offset_xy,
                                                    scaling_factor_xy=scaling_factor_list)
    contours_corrected = cytometer.utils.labels2contours(window_labels_corrected, offset_xy=offset_xy,
                                                         scaling_factor_xy=scaling_factor_list)

    # plot hand traced contours vs. segmented contours
    if DEBUG:
        enhancer = PIL.ImageEnhance.Contrast(PIL.Image.fromarray(im))
        tile_enhanced = np.array(enhancer.enhance(enhance_contrast))

        # without overlap
        plt.clf()
        plt.imshow(tile_enhanced)
        for j in range(len(cells)):
            cell = np.array(cells[j])
            plt.fill(cell[:, 0], cell[:, 1], edgecolor='C0', fill=False)
            plt.text(np.mean(cell[:, 0]), np.mean(cell[:, 1]), str(j))
        for j in range(len(contours_auto)):
            plt.fill(contours_auto[j][:, 0], contours_auto[j][:, 1], edgecolor='C1', fill=False)
            plt.text(np.mean(contours_auto[j][:, 0]), np.mean(contours_auto[j][:, 1]), str(j))

        # with overlap
        plt.clf()
        plt.imshow(tile_enhanced)
        for j in range(len(cells)):
            cell = np.array(cells[j])
            plt.fill(cell[:, 0], cell[:, 1], edgecolor='C0', fill=False)
        for j in range(len(contours_corrected)):
            plt.fill(contours_corrected[j][:, 0], contours_corrected[j][:, 1], edgecolor='C1', fill=False)
            plt.text(np.mean(contours_corrected[j][:, 0]), np.mean(contours_corrected[j][:, 1]), str(j))

    # match segmented contours to hand traced contours
    df_auto = cytometer.utils.match_overlapping_contours(contours_ref=cells, contours_test=contours_auto,
                                                         allow_repeat_ref=False, return_unmatched_refs=True)
    df_corrected = cytometer.utils.match_overlapping_contours(contours_ref=cells, contours_test=contours_corrected,
                                                              allow_repeat_ref=False, return_unmatched_refs=True)

    # aggregate results from this image into total dataframes
    df_auto['file_svg_idx'] = i
    df_corrected['file_svg_idx'] = i
    df_auto_all = pd.concat([df_auto_all, df_auto], ignore_index=True)
    df_corrected_all = pd.concat([df_corrected_all, df_corrected], ignore_index=True)

    # save dataframes to file
    df_auto_all.to_csv(dataframe_auto_filename, index=False)
    df_auto_all.to_csv(dataframe_corrected_filename, index=False)

    # clear keras session to prevent each segmentation iteration from getting slower. Note that this forces us to
    # reload the models every time
    K.clear_session()

if DEBUG:
    plt.clf()
    plt.scatter(df_auto_all['ref_area'], df_auto_all['test_area'] / df_auto_all['ref_area'] - 1)

    plt.clf()
    plt.scatter(df_corrected_all['ref_area'], df_corrected_all['test_area'] / df_corrected_all['ref_area'] - 1)

