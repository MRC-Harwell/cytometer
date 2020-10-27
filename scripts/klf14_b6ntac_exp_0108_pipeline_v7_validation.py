"""
Segmentation validation of pipeline v7 with 10-fold cross validation:
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

Loop manual contours and find overlaps with automatically segmented contours. Compute cell areas and prop. of WAT
pixels.

Changes over klf14_b6ntac_exp_0096_pipeline_v7_validation:
* Validation is done with new contours method match_overlapping_contours() instead of old labels method
  match_overlapping_labels().
* Instead of going step by step with the pipeline, we use the whole segmentation_pipeline6() function.
* Segmentation clean up has changed a bit to match the cleaning in v8.
"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0108_pipeline_v7_validation'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import sys
if os.path.join(home, 'Software/cytometer') not in sys.path:
    sys.path.extend([os.path.join(home, 'Software/cytometer')])
import numpy as np
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
min_class_prop = 0.65
correction_window_len = 401
correction_smoothing = 11

# data paths
histology_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
histology_ext = '.ndpi'
area2quantile_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v7')
saved_models_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v7')
annotations_dir = os.path.join(home, 'bit/cytometer_data/aida_data_Klf14_v7/annotations')
metainfo_dir = os.path.join(home, 'Data/cytometer_data/klf14')
paper_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')
figures_dir = os.path.join(paper_dir, 'figures')

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
fold = -np.ones(shape=(n_im,))  # initialise with -1 values in case a training file has no fold associated to it
for i_fold in range(n_folds):
    fold[idx_test_all[i_fold]] = i_fold

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
            plt.text(np.mean(cell[:, 0]) - 8, np.mean(cell[:, 1]) + 8, str(j))
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

    # match segmented contours to hand traced contours.
    # Note:
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

plt.savefig(os.path.join(figures_dir, 'exp_0108_area_boxplots_manual_dataset.svg'))
plt.savefig(os.path.join(figures_dir, 'exp_0108_area_boxplots_manual_dataset.png'))

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

plt.savefig(os.path.join(figures_dir, 'exp_0108_area_error_boxplots_manual_dataset.svg'))
plt.savefig(os.path.join(figures_dir, 'exp_0108_area_error_boxplots_manual_dataset.png'))

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

    plt.savefig(os.path.join(figures_dir, 'exp_0108_area_' + output + '_manual_error.svg'))
    plt.savefig(os.path.join(figures_dir, 'exp_0108_area_' + output + '_manual_error.png'))

    if output == 'auto':
        plt.ylim(-14, 0)
    elif output == 'corrected':
        plt.ylim(0, 10)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'exp_0108_area_' + output + '_manual_error_zoom.svg'))
    plt.savefig(os.path.join(figures_dir, 'exp_0108_area_' + output + '_manual_error_zoom.png'))


# # compute what proportion of cells are poorly segmented
# ecdf = sm.distributions.empirical_distribution.ECDF(df_manual_all['area_manual'])
# cell_area_threshold = 780
# print('Unusuable segmentations = ' + str(ecdf(cell_area_threshold)))
