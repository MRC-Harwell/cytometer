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
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
min_class_prop = 0.65
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
figures_dir = os.path.join(paper_dir, 'figures')

# file with RGB modes from all training data
klf14_training_colour_histogram_file = os.path.join(saved_models_dir, 'klf14_training_colour_histogram.npz')

# k-folds file
saved_kfolds_filename = 'klf14_b6ntac_exp_0094_generate_extra_training_images.pickle'

# model names
dmap_model_basename = 'klf14_b6ntac_exp_0086_cnn_dmap'
contour_model_basename = 'klf14_b6ntac_exp_0091_cnn_contour_after_dmap'
classifier_model_basename = 'klf14_b6ntac_exp_0095_cnn_tissue_classifier_fcn'
correction_model_basename = 'klf14_b6ntac_exp_0089_cnn_segmentation_correction_overlapping_scaled_contours'

# files to save dataframes with segmentation validation to.
# "v2" means that we are going to use "klf14_b6ntac_training_v2" as the hand traced contours
dataframe_auto_filename = os.path.join(paper_dir, experiment_id + '_segmentation_validation_auto_v2.csv')
dataframe_corrected_filename = os.path.join(paper_dir, experiment_id + '_segmentation_validation_corrected_v2.csv')

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

# get v2 of the hand traced contours
file_svg_list = [x.replace('/klf14_b6ntac_training/', '/klf14_b6ntac_training_v2/') for x in file_svg_list]

# number of images
n_im = len(file_svg_list)

# number of folds
n_folds = len(idx_test_all)

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# associate a fold to each training file
fold = -np.ones(shape=(n_im,), dtype=np.int32)  # initialise with -1 values in case a training file has no fold associated to it
for i_fold in range(n_folds):
    fold[idx_test_all[i_fold]] = i_fold
del i_fold

########################################################################################################################
## Find matches between hand traced contours and pipeline segmentations
########################################################################################################################

# init dataframes to contain the comparison between hand traced and automatically segmented cells
dataframe_columns = ['file_svg_idx', 'test_idx', 'test_area', 'ref_idx', 'ref_area', 'dice', 'hausdorff']
df_auto_all = pd.DataFrame(columns=dataframe_columns)
df_corrected_all = pd.DataFrame(columns=dataframe_columns)

for i, file_svg in enumerate(file_svg_list):

    print('File ' + str(i) + '/' + str(len(file_svg_list) - 1) + ': ' + os.path.basename(file_svg))

    # load hand traced contours
    cells = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False,
                                                    minimum_npoints=3)

    # no hand traced cells in this image
    if len(cells) == 0:
        continue

    # load training image
    file_im = file_svg.replace('.svg', '.tif')
    im = PIL.Image.open(file_im)

    # read pixel size information
    xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
    yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

    im = np.array(im)

    if DEBUG:
        plt.clf()
        plt.imshow(im)

    # colour correction using the whole slide the training image was extracted from
    file_whole_slide = os.path.basename(file_svg).split('_row_')[0]
    file_whole_slide = os.path.join(histology_dir, file_whole_slide + '.ndpi')
    im_whole_slide = openslide.OpenSlide(file_whole_slide)

    downsample_level = im_whole_slide.get_best_level_for_downsample(downsample_factor_goal)
    im_downsampled = im_whole_slide.read_region(location=(0, 0), level=downsample_level,
                                                size=im_whole_slide.level_dimensions[downsample_level])
    im_downsampled = np.array(im_downsampled)
    im_downsampled = im_downsampled[:, :, 0:3]

    mode_r_tile = scipy.stats.mode(im_downsampled[:, :, 0], axis=None).mode[0]
    mode_g_tile = scipy.stats.mode(im_downsampled[:, :, 1], axis=None).mode[0]
    mode_b_tile = scipy.stats.mode(im_downsampled[:, :, 2], axis=None).mode[0]

    im[:, :, 0] = im[:, :, 0] + (mode_r_target - mode_r_tile)
    im[:, :, 1] = im[:, :, 1] + (mode_g_target - mode_g_tile)
    im[:, :, 2] = im[:, :, 2] + (mode_b_target - mode_b_tile)

    # names of contour, dmap and tissue classifier models
    i_fold = fold[i]
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
                                                         allow_repeat_ref=False, return_unmatched_refs=True,
                                                         xres=xres, yres=yres)
    df_corrected = cytometer.utils.match_overlapping_contours(contours_ref=cells, contours_test=contours_corrected,
                                                              allow_repeat_ref=False, return_unmatched_refs=True,
                                                              xres=xres, yres=yres)

    # aggregate results from this image into total dataframes
    df_auto['file_svg_idx'] = i
    df_corrected['file_svg_idx'] = i
    df_auto_all = pd.concat([df_auto_all, df_auto], ignore_index=True)
    df_corrected_all = pd.concat([df_corrected_all, df_corrected], ignore_index=True)

    # save dataframes to file
    df_auto_all.to_csv(dataframe_auto_filename, index=False)
    df_corrected_all.to_csv(dataframe_corrected_filename, index=False)

    # clear keras session to prevent each segmentation iteration from getting slower. Note that this forces us to
    # reload the models every time
    K.clear_session()

if DEBUG:
    plt.clf()
    plt.scatter(df_auto_all['ref_area'], df_auto_all['test_area'] / df_auto_all['ref_area'] - 1)

    plt.clf()
    plt.scatter(df_corrected_all['ref_area'], df_corrected_all['test_area'] / df_corrected_all['ref_area'] - 1)

########################################################################################################################
## Comparison of cell sizes: hand traced vs. auto vs. corrected
## Note: If we perform a sign test to see whether the median = 0, we would assume a binomial distribution of number of
## values < median, and with a Gaussian approximation to the binomial distribution, we'd be performing a normal null
## hypothesis test. which corresponds to a CI-95% of -1.96*std, +1.96*std around the median value.
## https://youtu.be/dLTvZUrs-CI?t=463
########################################################################################################################

import scipy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

## Auxiliary function to load a dataframe with matched cell areas

def load_dataframe(dataframe_filename):

    # read dataframe
    df_all = pd.read_csv(dataframe_filename)

    # remove hand traced cells with no auto match, as we don't need those here
    df_all.dropna(subset=['test_idx'], inplace=True)

    # remove very low Dice indices, as those indicate overlap with a neighbour, rather than a proper segmentation
    df_all = df_all[df_all['dice'] >= 0.5]

    # sort manual areas from smallest to largest
    df_all.sort_values(by=['ref_area'], ascending=True, ignore_index=True, inplace=True)

    # compute area error for convenience
    df_all['test_ref_area_diff'] = df_all['test_area'] - df_all['ref_area']
    df_all['test_ref_area_err'] = np.array(df_all['test_ref_area_diff'] / df_all['ref_area'])

    return df_all

## Boxplots comparing cell populations in hand traced vs. pipeline segmentations

df_auto_all = load_dataframe(dataframe_auto_filename)
df_corrected_all = load_dataframe(dataframe_corrected_filename)

plt.clf()
bp = plt.boxplot((df_auto_all['ref_area'] / 1e3,
                  df_auto_all['test_area'] / 1e3,
                  df_corrected_all['test_area'] / 1e3),
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

plt.savefig(os.path.join(figures_dir, 'exp_0109_area_boxplots_manual_dataset.svg'))
plt.savefig(os.path.join(figures_dir, 'exp_0109_area_boxplots_manual_dataset.png'))

# Wilcoxon sign-ranked tests of whether manual areas are significantly different to auto/corrected areas
print('Manual mean ± std = ' + str(np.mean(df_auto_all['ref_area'])) + ' ± '
      + str(np.std(df_auto_all['ref_area'])))
print('Auto mean ± std = ' + str(np.mean(df_auto_all['test_area'])) + ' ± '
      + str(np.std(df_auto_all['test_area'])))
print('Corrected mean ± std = ' + str(np.mean(df_corrected_all['test_area'])) + ' ± '
      + str(np.std(df_corrected_all['test_area'])))

# Wilcoxon signed-rank test to check whether the medians are significantly different
w, p = scipy.stats.wilcoxon(df_auto_all['ref_area'],
                            df_auto_all['test_area'])
print('Manual vs. auto, W = ' + str(w) + ', p = ' + str(p))

w, p = scipy.stats.wilcoxon(df_corrected_all['ref_area'],
                            df_corrected_all['test_area'])
print('Manual vs. corrected, W = ' + str(w) + ', p = ' + str(p))


# boxplots of area error
plt.clf()
bp = plt.boxplot(((df_auto_all['test_area'] / df_auto_all['ref_area'] - 1) * 100,
                  (df_corrected_all['test_area'] / df_corrected_all['ref_area'] - 1) * 100),
                 positions=[1, 2], notch=True, labels=['Auto vs.\nHand traced', 'Corrected vs.\nHand traced'])
# bp = plt.boxplot((df_auto_all['test_area'] / 1e3 - df_auto_all['ref_area'] / 1e3,
#                   df_corrected_all['test_area'] / 1e3 - df_corrected_all['ref_area'] / 1e3),
#                  positions=[1, 2], notch=True, labels=['Auto -\nHand traced', 'Corrected -\nHand traced'])

plt.plot([0.75, 2.25], [0, 0], 'k', 'linewidth', 2)
plt.xlim(0.5, 2.5)
# plt.ylim(-1.4, 1.1)

plt.ylim(-40, 40)

# points of interest from the boxplots
bp_poi = cytometer.utils.boxplot_poi(bp)

# manual quartile values
plt.text(1.10, bp_poi[0, 2], '%0.2f' % (bp_poi[0, 2]), fontsize=12, color='C1')
plt.text(2.10, bp_poi[1, 2], '%0.2f' % (bp_poi[1, 2]), fontsize=12, color='C1')

plt.tick_params(axis="both", labelsize=14)
plt.ylabel('Area$_{pipeline}$ / Area$_{ht} - 1$ ($\%$)', fontsize=14)
plt.tight_layout()

plt.savefig(os.path.join(figures_dir, 'exp_0109_area_error_boxplots_manual_dataset.svg'))
plt.savefig(os.path.join(figures_dir, 'exp_0109_area_error_boxplots_manual_dataset.png'))

## Segmentation error vs. cell size plots, with Gaussian process regression

# load dataframes to file
for output in ['auto', 'corrected']:

    if output == 'auto':
        df_all = load_dataframe(dataframe_auto_filename)
    elif output == 'corrected':
        df_all = load_dataframe(dataframe_corrected_filename)
    else:
        raise ValueError('Output must be "auto" or "corrected"')

    # convert ref_area to quantiles
    n_quantiles = 1001
    quantiles = np.linspace(0, 1, n_quantiles)
    ref_area_q = scipy.stats.mstats.hdquantiles(df_all['ref_area'], prob=quantiles)
    f = scipy.interpolate.interp1d(ref_area_q, quantiles)
    df_all['ref_area_quantile'] = f(df_all['ref_area'])

    # estimate the std of area errors, which will be used as a measure of noise for the Gaussian process. Then assign
    # the value alpha=std**2 to each point within the bin, to later use in GaussianProcessRegressor
    bin_std, bin_edges, binnumber = \
        scipy.stats.binned_statistic(df_all['ref_area_quantile'], df_all['test_ref_area_err'], statistic='std', bins=100)
    df_all['alpha'] = bin_std[binnumber - 1] ** 2

    # Gaussian process regression of the segmentation errors
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(0.01, (0.01/1000, 1)) + C(1.0, (1e-3, 1e3))
    kernel = C(1.0, (1e-2, 1e3)) * RBF(0.1, (0.1/1000, 1)) + C(1.0, (1e-2, 1e3))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=df_all['alpha'], n_restarts_optimizer=10)
    gp.fit(np.atleast_2d(df_all['ref_area_quantile']).T,
           np.array(df_all['test_ref_area_err']))
    x = quantiles
    y_pred, sigma = gp.predict(x.reshape(-1, 1), return_std=True)

    if DEBUG:
        print('kernel: ' + str(gp.kernel))
        for h in range(len(gp.kernel.hyperparameters)):
            print('Gaussian process hyperparameter ' + str(h) + ': ' + str(10**gp.kernel.theta[h]) + ', '
                  + str(gp.kernel.hyperparameters[h]))

    # plot segmentation errors
    plt.clf()
    plt.scatter(df_all['ref_area'] * 1e-3, np.array(df_all['test_ref_area_err']) * 100, s=2)
    plt.plot(ref_area_q * 1e-3, y_pred * 100, 'r', linewidth=2)
    plt.fill(np.concatenate([ref_area_q * 1e-3, ref_area_q[::-1] * 1e-3]),
             np.concatenate([100 * (y_pred - 1.9600 * sigma),
                             100 * (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='r', ec='None', label='95% confidence interval')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Area$_{ht}$ ($10^3\ \mu m^2$)', fontsize=14)
    plt.ylabel('Area$_{' + output + '}$ / Area$_{ht} - 1$ (%)', fontsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'exp_0109_area_' + output + '_manual_error.svg'))
    plt.savefig(os.path.join(figures_dir, 'exp_0109_area_' + output + '_manual_error.png'))

    if output == 'auto':
        plt.ylim(-14, 0)
    elif output == 'corrected':
        plt.ylim(0, 8)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'exp_0109_area_' + output + '_manual_error_zoom.svg'))
    plt.savefig(os.path.join(figures_dir, 'exp_0109_area_' + output + '_manual_error_zoom.png'))

'''
************************************************************************************************************************
Object-wise classification validation (new way)

Here we label the pixels of the image as WAT/no-WAT/unlabelled according to ground truth.

Then we apply automatic segmentation. We count the proportion of WAT pixels in each object. That value (e.g. 0.34) gets 
assigned to each pixel inside. That's then used as scores that are compared to the ground truth labelling to for the
ROC. 
************************************************************************************************************************
'''

import keras
import shapely, shapely.ops
import rasterio, rasterio.features

for i, file_svg in enumerate(file_svg_list):

    print('File ' + str(i) + '/' + str(len(file_svg_list) - 1) + ': ' + os.path.basename(file_svg))

    # load hand traced contours
    cells = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False,
                                                    minimum_npoints=3)
    other = cytometer.data.read_paths_from_svg_file(file_svg, tag='Other', add_offset_from_filename=False,
                                                    minimum_npoints=3)
    other += cytometer.data.read_paths_from_svg_file(file_svg, tag='Brown', add_offset_from_filename=False,
                                                     minimum_npoints=3)
    contours = cells + other

    if (len(contours) == 0):
        raise RuntimeError('No contours')

    # load training image
    file_im = file_svg.replace('.svg', '.tif')
    im = PIL.Image.open(file_im)

    # read pixel size information
    xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
    yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

    im = np.array(im)

    if DEBUG:
        plt.clf()
        plt.imshow(im)
        for j in range(len(cells)):
            cell = np.array(cells[j])
            plt.fill(cell[:, 0], cell[:, 1], edgecolor='C0', fill=False)
            plt.text(np.mean(cell[:, 0]), np.mean(cell[:, 1]), str(j))

    # # make a list with the type of cell each contour is classified as
    # contour_type = [np.zeros(shape=(len(cells),), dtype=np.uint8),  # 0: white-adipocyte
    #                 np.ones(shape=(len(other),), dtype=np.uint8)]  # 1: other/brown types of tissue
    # contour_type = np.concatenate(contour_type)

    print('Cells: ' + str(len(cells)))
    print('Other/Brown: ' + str(len(other)))

    # rasterise mask of all cell pixels
    cells = [shapely.geometry.Polygon(x) for x in cells]
    other = [shapely.geometry.Polygon(x) for x in other]
    cell_mask_ground_truth = rasterio.features.rasterize(cells, out_shape=im.shape[0:2], fill=0, default_value=1)
    other_mask_ground_truth = rasterio.features.rasterize(other, out_shape=im.shape[0:2], fill=0, default_value=1)
    all_mask_ground_truth = rasterio.features.rasterize(cells + other, out_shape=im.shape[0:2], fill=0, default_value=1)

    if DEBUG:
        plt.figure()
        plt.clf()
        plt.subplot(221)
        plt.imshow(cell_mask_ground_truth)
        plt.title('Cells')
        plt.axis('off')
        plt.subplot(222)
        plt.imshow(other_mask_ground_truth)
        plt.title('Other')
        plt.axis('off')
        plt.subplot(223)
        plt.imshow(all_mask_ground_truth)
        plt.title('Combined')
        plt.axis('off')

    # colour correction using the whole slide the training image was extracted from
    file_whole_slide = os.path.basename(file_svg).split('_row_')[0]
    file_whole_slide = os.path.join(histology_dir, file_whole_slide + '.ndpi')
    im_whole_slide = openslide.OpenSlide(file_whole_slide)

    downsample_level = im_whole_slide.get_best_level_for_downsample(downsample_factor_goal)
    im_downsampled = im_whole_slide.read_region(location=(0, 0), level=downsample_level,
                                                size=im_whole_slide.level_dimensions[downsample_level])
    im_downsampled = np.array(im_downsampled)
    im_downsampled = im_downsampled[:, :, 0:3]

    mode_r_tile = scipy.stats.mode(im_downsampled[:, :, 0], axis=None).mode[0]
    mode_g_tile = scipy.stats.mode(im_downsampled[:, :, 1], axis=None).mode[0]
    mode_b_tile = scipy.stats.mode(im_downsampled[:, :, 2], axis=None).mode[0]

    im[:, :, 0] = im[:, :, 0] + (mode_r_target - mode_r_tile)
    im[:, :, 1] = im[:, :, 1] + (mode_g_target - mode_g_tile)
    im[:, :, 2] = im[:, :, 2] + (mode_b_target - mode_b_tile)

    # pixel classification of histology image
    i_fold = fold[i]
    classifier_model_filename = \
        os.path.join(saved_models_dir, classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    classifier_model = keras.models.load_model(classifier_model_filename)
    classifier_model = cytometer.utils.change_input_size(classifier_model, batch_shape=(1,) + im.shape)

    labels_class = classifier_model.predict(np.expand_dims(im.astype(np.float32) / 255.0, axis=0))
    labels_class = labels_class[0, :, :, 0] > 0.5

    if DEBUG:
        plt.subplot(224)
        plt.imshow(labels_class)
        plt.title('Class prediction')
        plt.axis('off')




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

# calculate FPR and TPR for different thresholds
fpr_interp = np.interp([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8], thr[::-1], fpr[::-1])
tpr_interp = np.interp([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8], thr[::-1], tpr[::-1])
print('thr: ' + str([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]))
print('fpr: ' + str(fpr_interp))
print('tpr: ' + str(tpr_interp))

# we fix the min_class_prop threshold to what is used in the pipeline
thr_target = 0.5
tpr_target = np.interp(thr_target, thr[::-1], tpr[::-1])
fpr_target = np.interp(thr_target, thr[::-1], fpr[::-1])

# # we fix the FPR (False Positive Rate) and interpolate the TPR (True Positive Rate) on the ROC curve
# fpr_target = 0.05
# tpr_target = np.interp(fpr_target, fpr, tpr)
# thr_target = np.interp(fpr_target, fpr, thr)

# plot ROC curve for the Tissue classifier (computer pixel-wise for the object-classification error)
plt.clf()
plt.plot(fpr, tpr)
plt.scatter(fpr_target, tpr_target, color='C0', s=100,
            label='$Z_{obj} \geq$ %0.2f, FPR = %0.0f%%, TPR = %0.0f%%'
                  % (thr_target, fpr_target * 100, tpr_target * 100))
plt.tick_params(labelsize=14)
plt.xlabel('Pixel WAT False Positive Rate (FPR)', fontsize=14)
plt.ylabel('Pixel WAT True Positive Rate (TPR)', fontsize=14)
plt.legend(loc="lower right", prop={'size': 12})
plt.tight_layout()

plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_pipeline_roc_tissue_cnn_pixelwise.svg'),
            bbox_inches='tight', pad_inches=0)
plt.savefig(os.path.join(saved_figures_dir, 'exp_0096_pipeline_roc_tissue_cnn_pixelwise.png'),
            bbox_inches='tight', pad_inches=0)

