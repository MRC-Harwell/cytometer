"""
Generate figures for the DeepCytometer paper.

Code cannibalised from:
* klf14_b6ntac_exp_0097_full_slide_pipeline_v7.py

"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0099_paper_figures'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import sys
import pickle
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import cytometer.utils

# Filter out INFO & WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # limit number of GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

os.environ['KERAS_BACKEND'] = 'tensorflow'

import time
import openslide
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.signal import fftconvolve
from cytometer.utils import rough_foreground_mask, bspline_resample
from cytometer.data import append_paths_to_aida_json_file, write_paths_to_aida_json_file
import PIL
import tensorflow as tf
import keras
from keras import backend as K
from skimage.measure import regionprops
import shutil
import itertools
from shapely.geometry import Polygon

# # limit GPU memory used
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# set_session(tf.Session(config=config))

DEBUG = False
SAVE_FIGS = False

root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
data_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
training_dir = os.path.join(home, root_data_dir, 'klf14_b6ntac_training')
seg_dir = os.path.join(home, root_data_dir, 'klf14_b6ntac_seg')
# figures_dir = os.path.join(root_data_dir, 'figures')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')
results_dir = os.path.join(root_data_dir, 'klf14_b6ntac_results')
annotations_dir = os.path.join(home, 'Software/AIDA/dist/data/annotations')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')

# k-folds file
saved_extra_kfolds_filename = 'klf14_b6ntac_exp_0094_generate_extra_training_images.pickle'

# model names
dmap_model_basename = 'klf14_b6ntac_exp_0086_cnn_dmap'
contour_model_basename = 'klf14_b6ntac_exp_0091_cnn_contour_after_dmap'
classifier_model_basename = 'klf14_b6ntac_exp_0095_cnn_tissue_classifier_fcn'
correction_model_basename = 'klf14_b6ntac_exp_0089_cnn_segmentation_correction_overlapping_scaled_contours'

# full resolution image window and network expected receptive field parameters
fullres_box_size = np.array([2751, 2751])
receptive_field = np.array([131, 131])

# rough_foreground_mask() parameters
downsample_factor = 8.0
dilation_size = 25
component_size_threshold = 1e6
hole_size_treshold = 8000

# contour parameters
contour_downsample_factor = 0.1
bspline_k = 1

# block_split() parameters in downsampled image
block_len = np.ceil((fullres_box_size - receptive_field) / downsample_factor)
block_overlap = np.ceil((receptive_field - 1) / 2 / downsample_factor).astype(np.int)

# segmentation parameters
min_cell_area = 1500
max_cell_area = 100e3
min_mask_overlap = 0.8
phagocytosis = True
min_class_prop = 0.5
correction_window_len = 401
correction_smoothing = 11
batch_size = 16

# segmentation correction parameters

# load list of images, and indices for training vs. testing indices
saved_kfolds_filename = os.path.join(saved_models_dir, saved_extra_kfolds_filename)
with open(saved_kfolds_filename, 'rb') as f:
    aux = pickle.load(f)
file_svg_list = aux['file_list']
idx_test_all = aux['idx_test']
idx_train_all = aux['idx_train']

# correct home directory
file_svg_list = [x.replace('/users/rittscher/rcasero', home) for x in file_svg_list]
file_svg_list = [x.replace('/home/rcasero', home) for x in file_svg_list]

# loop the folds to get the ndpi files that correspond to testing of each fold
ndpi_files_test_list = {}
for i_fold in range(len(idx_test_all)):
    # list of .svg files for testing
    file_svg_test = np.array(file_svg_list)[idx_test_all[i_fold]]

    # list of .ndpi files that the .svg windows came from
    file_ndpi_test = [os.path.basename(x).replace('.svg', '') for x in file_svg_test]
    file_ndpi_test = np.unique([x.split('_row')[0] for x in file_ndpi_test])

    # add to the dictionary {file: fold}
    for file in file_ndpi_test:
        ndpi_files_test_list[file] = i_fold

########################################################################################################################
## Plots of get_next_roi_to_process()
########################################################################################################################

# File 4/19: KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04. Fold = 2
i_file = 4
# File 10/19: KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38. Fold = 5
i_file = 10

ndpi_file_kernel = list(ndpi_files_test_list.keys())[i_file]

# for i_file, ndpi_file_kernel in enumerate(ndpi_files_test_list):

# fold  where the current .ndpi image was not used for training
i_fold = ndpi_files_test_list[ndpi_file_kernel]

print('File ' + str(i_file) + '/' + str(len(ndpi_files_test_list) - 1) + ': ' + ndpi_file_kernel
      + '. Fold = ' + str(i_fold))

# make full path to ndpi file
ndpi_file = os.path.join(data_dir, ndpi_file_kernel + '.ndpi')

contour_model_file = os.path.join(saved_models_dir, contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
dmap_model_file = os.path.join(saved_models_dir, dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
classifier_model_file = os.path.join(saved_models_dir,
                                     classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')
correction_model_file = os.path.join(saved_models_dir,
                                     correction_model_basename + '_model_fold_' + str(i_fold) + '.h5')

# name of file to save annotations
annotations_file = os.path.basename(ndpi_file)
annotations_file = os.path.splitext(annotations_file)[0]
annotations_file = os.path.join(annotations_dir, annotations_file + '_exp_0097.json')

# name of file to save areas and contours
results_file = os.path.basename(ndpi_file)
results_file = os.path.splitext(results_file)[0]
results_file = os.path.join(results_dir, results_file + '_exp_0097.npz')

# rough segmentation of the tissue in the image
lores_istissue0, im_downsampled = rough_foreground_mask(ndpi_file, downsample_factor=downsample_factor,
                                                        dilation_size=dilation_size,
                                                        component_size_threshold=component_size_threshold,
                                                        hole_size_treshold=hole_size_treshold,
                                                        return_im=True)

if DEBUG:
    plt.clf()
    plt.imshow(im_downsampled)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_histology_i_file_' + str(i_file) + '.png'),
                bbox_inches='tight')

    plt.clf()
    plt.imshow(im_downsampled)
    plt.contour(lores_istissue0, colors='k')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_rough_mask_i_file_' + str(i_file) + '.png'),
                bbox_inches='tight')

# segmentation copy, to keep track of what's left to do
lores_istissue = lores_istissue0.copy()

# open full resolution histology slide
im = openslide.OpenSlide(ndpi_file)

# pixel size
assert(im.properties['tiff.ResolutionUnit'] == 'centimeter')
xres = 1e-2 / float(im.properties['tiff.XResolution'])
yres = 1e-2 / float(im.properties['tiff.YResolution'])

# # init empty list to store area values and contour coordinates
# areas_all = []
# contours_all = []

# keep extracting histology windows until we have finished
step = -1
time_0 = time_curr = time.time()
while np.count_nonzero(lores_istissue) > 0:

    # next step (it starts from 0)
    step += 1

    time_prev = time_curr
    time_curr = time.time()

    print('File ' + str(i_file) + '/' + str(len(ndpi_files_test_list) - 1) + ': step ' +
          str(step) + ': ' +
          str(np.count_nonzero(lores_istissue)) + '/' + str(np.count_nonzero(lores_istissue0)) + ': ' +
          "{0:.1f}".format(100.0 - np.count_nonzero(lores_istissue) / np.count_nonzero(lores_istissue0) * 100) +
          '% completed: ' +
          'step time ' + "{0:.2f}".format(time_curr - time_prev) + ' s' +
          ', total time ' + "{0:.2f}".format(time_curr - time_0) + ' s')

    ## Code extracted from:
    ## get_next_roi_to_process()

    # variables for get_next_roi_to_process()
    seg = lores_istissue.copy()
    downsample_factor = downsample_factor
    max_window_size = fullres_box_size
    border = np.round((receptive_field - 1) / 2)

    # convert to np.array so that we can use algebraic operators
    max_window_size = np.array(max_window_size)
    border = np.array(border)

    # convert segmentation mask to [0, 1]
    seg = (seg != 0).astype('int')

    # approximate measures in the downsampled image (we don't round them)
    lores_max_window_size = max_window_size / downsample_factor
    lores_border = border / downsample_factor

    # kernels that flipped correspond to top line and left line. They need to be pre-flipped
    # because the convolution operation internally flips them (two flips cancel each other)
    kernel_top = np.zeros(shape=np.round(lores_max_window_size - 2 * lores_border).astype('int'))
    kernel_top[int((kernel_top.shape[0] - 1) / 2), :] = 1
    kernel_left = np.zeros(shape=np.round(lores_max_window_size - 2 * lores_border).astype('int'))
    kernel_left[:, int((kernel_top.shape[1] - 1) / 2)] = 1

    if DEBUG:
        plt.clf()
        plt.imshow(kernel_top)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_kernel_top_i_file_' + str(i_file) + '.png'),
                    bbox_inches='tight')

        plt.imshow(kernel_left)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_kernel_left_i_file_' + str(i_file) + '.png'),
                    bbox_inches='tight')

    seg_top = np.round(fftconvolve(seg, kernel_top, mode='same'))
    seg_left = np.round(fftconvolve(seg, kernel_left, mode='same'))

    # window detections
    detection_idx = np.nonzero(seg_left * seg_top)

    # set top-left corner of the box = top-left corner of first box detected
    lores_first_row = detection_idx[0][0]
    lores_first_col = detection_idx[1][0]

    # first, we look within a window with the maximum size
    lores_last_row = detection_idx[0][0] + lores_max_window_size[0] - 2 * lores_border[0]
    lores_last_col = detection_idx[1][0] + lores_max_window_size[1] - 2 * lores_border[1]

    # second, if the segmentation is smaller than the window, we reduce the window size
    window = seg[lores_first_row:int(np.round(lores_last_row)), lores_first_col:int(np.round(lores_last_col))]

    idx = np.any(window, axis=1)  # reduce rows size
    last_segmented_pixel_len = np.max(np.where(idx))
    lores_last_row = detection_idx[0][0] + np.min((lores_max_window_size[0] - 2 * lores_border[0],
                                                   last_segmented_pixel_len))

    idx = np.any(window, axis=0)  # reduce cols size
    last_segmented_pixel_len = np.max(np.where(idx))
    lores_last_col = detection_idx[1][0] + np.min((lores_max_window_size[1] - 2 * lores_border[1],
                                                   last_segmented_pixel_len))

    # save coordinates for plot (this is only for a figure in the paper and doesn't need to be done in the real
    # implementation)
    lores_first_col_bak = lores_first_col
    lores_first_row_bak = lores_first_row
    lores_last_col_bak = lores_last_col
    lores_last_row_bak = lores_last_row

    # add a border around the window
    lores_first_row = np.max([0, lores_first_row - lores_border[0]])
    lores_first_col = np.max([0, lores_first_col - lores_border[1]])

    lores_last_row = np.min([seg.shape[0], lores_last_row + lores_border[0]])
    lores_last_col = np.min([seg.shape[1], lores_last_col + lores_border[1]])

    # convert low resolution indices to high resolution
    first_row = np.int(np.round(lores_first_row * downsample_factor))
    last_row = np.int(np.round(lores_last_row * downsample_factor))
    first_col = np.int(np.round(lores_first_col * downsample_factor))
    last_col = np.int(np.round(lores_last_col * downsample_factor))

    # round down indices in downsampled segmentation
    lores_first_row = int(lores_first_row)
    lores_last_row = int(lores_last_row)
    lores_first_col = int(lores_first_col)
    lores_last_col = int(lores_last_col)

    # load window from full resolution slide
    tile = im.read_region(location=(first_col, first_row), level=0,
                          size=(last_col - first_col, last_row - first_row))
    tile = np.array(tile)
    tile = tile[:, :, 0:3]

    # interpolate coarse tissue segmentation to full resolution
    istissue_tile = lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col]
    istissue_tile = cytometer.utils.resize(istissue_tile, size=(last_col - first_col, last_row - first_row),
                                           resample=PIL.Image.NEAREST)

    if DEBUG:
        plt.clf()
        plt.imshow(tile)
        plt.imshow(istissue_tile, alpha=0.5)
        plt.contour(istissue_tile, colors='k')
        plt.title('Yellow: Tissue. Purple: Background')
        plt.axis('off')

    # clear keras session to prevent each segmentation iteration from getting slower. Note that this forces us to
    # reload the models every time
    K.clear_session()

    # segment histology, split into individual objects, and apply segmentation correction
    labels, labels_class, todo_edge, \
    window_im, window_labels, window_labels_corrected, window_labels_class, index_list, scaling_factor_list \
        = cytometer.utils.segmentation_pipeline6(tile,
                                                 dmap_model=dmap_model_file,
                                                 contour_model=contour_model_file,
                                                 correction_model=correction_model_file,
                                                 classifier_model=classifier_model_file,
                                                 min_cell_area=min_cell_area,
                                                 mask=istissue_tile,
                                                 min_mask_overlap=min_mask_overlap,
                                                 phagocytosis=phagocytosis,
                                                 min_class_prop=min_class_prop,
                                                 correction_window_len=correction_window_len,
                                                 correction_smoothing=correction_smoothing,
                                                 return_bbox=True, return_bbox_coordinates='xy',
                                                 batch_size=batch_size)

    # downsample "to do" mask so that the rough tissue segmentation can be updated
    lores_todo_edge = PIL.Image.fromarray(todo_edge.astype(np.uint8))
    lores_todo_edge = lores_todo_edge.resize((lores_last_col - lores_first_col,
                                              lores_last_row - lores_first_row),
                                             resample=PIL.Image.NEAREST)
    lores_todo_edge = np.array(lores_todo_edge)

    # update coarse tissue mask (this is only necessary here to plot figures for the paper. In the actual code,
    # the coarse mask gets directly updated, without this intermediate step)
    seg_updated = seg.copy()
    seg_updated[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = lores_todo_edge

    if DEBUG:
        plt.clf()
        fig = plt.imshow(seg, cmap='Greys')
        plt.contour(seg_left * seg_top > 0, colors='r')
        rect = Rectangle((lores_first_col, lores_first_row),
                         lores_last_col - lores_first_col, lores_last_row - lores_first_row,
                         alpha=0.5, facecolor='g', edgecolor='g', zorder=2)
        fig.axes.add_patch(rect)
        rect2 = Rectangle((lores_first_col_bak, lores_first_row_bak),
                          lores_last_col_bak - lores_first_col_bak, lores_last_row_bak - lores_first_row_bak,
                          alpha=1.0, facecolor=None, fill=False, edgecolor='g', lw=1, zorder=3)
        fig.axes.add_patch(rect2)
        plt.scatter(detection_idx[1][0], detection_idx[0][0], color='k', s=5, zorder=3)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_fftconvolve_i_file_' + str(i_file) +
                                 '_step_' + str(step) + '.png'),
                    bbox_inches='tight')

    if DEBUG:
        plt.clf()
        fig = plt.imshow(seg, cmap='Greys')
        plt.contour(seg_left * seg_top > 0, colors='r')
        plt.contour(seg_updated, colors='w', zorder=4)
        rect = Rectangle((lores_first_col, lores_first_row),
                         lores_last_col - lores_first_col, lores_last_row - lores_first_row,
                         alpha=0.5, facecolor='g', edgecolor='g', zorder=2)
        fig.axes.add_patch(rect)
        rect2 = Rectangle((lores_first_col_bak, lores_first_row_bak),
                          lores_last_col_bak - lores_first_col_bak, lores_last_row_bak - lores_first_row_bak,
                          alpha=1.0, facecolor=None, fill=False, edgecolor='g', lw=3, zorder=3)
        fig.axes.add_patch(rect2)
        plt.scatter(detection_idx[1][0], detection_idx[0][0], color='k', s=5, zorder=3)
        plt.axis('off')
        plt.tight_layout()
        plt.xlim(int(lores_first_col - 50), int(lores_last_col + 50))
        plt.ylim(int(lores_last_row + 50), int(lores_first_row - 50))
        plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_fftconvolve_detail_i_file_' + str(i_file) +
                                 '_step_' + str(step) + '.png'),
                    bbox_inches='tight')

        # update coarse tissue mask for next iteration
        lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = lores_todo_edge

########################################################################################################################
## Whole code
########################################################################################################################

# File 4/19: KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04. Fold = 2
i_file = 4
ndpi_file_kernel = list(ndpi_files_test_list.keys())[i_file]

# for i_file, ndpi_file_kernel in enumerate(ndpi_files_test_list):

    # fold  where the current .ndpi image was not used for training
    i_fold = ndpi_files_test_list[ndpi_file_kernel]

    print('File ' + str(i_file) + '/' + str(len(ndpi_files_test_list) - 1) + ': ' + ndpi_file_kernel
          + '. Fold = ' + str(i_fold))

    # make full path to ndpi file
    ndpi_file = os.path.join(data_dir, ndpi_file_kernel + '.ndpi')

    contour_model_file = os.path.join(saved_models_dir, contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    dmap_model_file = os.path.join(saved_models_dir, dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    classifier_model_file = os.path.join(saved_models_dir,
                                         classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    correction_model_file = os.path.join(saved_models_dir,
                                         correction_model_basename + '_model_fold_' + str(i_fold) + '.h5')

    # name of file to save annotations
    annotations_file = os.path.basename(ndpi_file)
    annotations_file = os.path.splitext(annotations_file)[0]
    annotations_file = os.path.join(annotations_dir, annotations_file + '_exp_0097.json')

    # name of file to save areas and contours
    results_file = os.path.basename(ndpi_file)
    results_file = os.path.splitext(results_file)[0]
    results_file = os.path.join(results_dir, results_file + '_exp_0097.npz')

    # rough segmentation of the tissue in the image
    lores_istissue0, im_downsampled = rough_foreground_mask(ndpi_file, downsample_factor=downsample_factor,
                                                            dilation_size=dilation_size,
                                                            component_size_threshold=component_size_threshold,
                                                            hole_size_treshold=hole_size_treshold,
                                                            return_im=True)

    if DEBUG:
        plt.clf()
        plt.imshow(im_downsampled)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_histology_i_file_' + str(i_file) + '.png'),
                    bbox_inches='tight')

        plt.clf()
        plt.imshow(im_downsampled)
        plt.contour(lores_istissue0, colors='k')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_rough_mask_i_file_' + str(i_file) + '.png'),
                    bbox_inches='tight')

    # segmentation copy, to keep track of what's left to do
    lores_istissue = lores_istissue0.copy()

    # open full resolution histology slide
    im = openslide.OpenSlide(ndpi_file)

    # pixel size
    assert(im.properties['tiff.ResolutionUnit'] == 'centimeter')
    xres = 1e-2 / float(im.properties['tiff.XResolution'])
    yres = 1e-2 / float(im.properties['tiff.YResolution'])

    # # init empty list to store area values and contour coordinates
    # areas_all = []
    # contours_all = []

    # keep extracting histology windows until we have finished
    step = -1
    time_0 = time_curr = time.time()
    while np.count_nonzero(lores_istissue) > 0:

        # next step (it starts from 0)
        step += 1

        time_prev = time_curr
        time_curr = time.time()

        print('File ' + str(i_file) + '/' + str(len(ndpi_files_test_list) - 1) + ': step ' +
              str(step) + ': ' +
              str(np.count_nonzero(lores_istissue)) + '/' + str(np.count_nonzero(lores_istissue0)) + ': ' +
              "{0:.1f}".format(100.0 - np.count_nonzero(lores_istissue) / np.count_nonzero(lores_istissue0) * 100) +
              '% completed: ' +
              'step time ' + "{0:.2f}".format(time_curr - time_prev) + ' s' +
              ', total time ' + "{0:.2f}".format(time_curr - time_0) + ' s')









        # get indices for the next histology window to process
        (first_row, last_row, first_col, last_col), \
        (lores_first_row, lores_last_row, lores_first_col, lores_last_col) = \
            cytometer.utils.get_next_roi_to_process(lores_istissue, downsample_factor=downsample_factor,
                                                    max_window_size=fullres_box_size,
                                                    border=np.round((receptive_field-1)/2))

        # load window from full resolution slide
        tile = im.read_region(location=(first_col, first_row), level=0,
                              size=(last_col - first_col, last_row - first_row))
        tile = np.array(tile)
        tile = tile[:, :, 0:3]

        # interpolate coarse tissue segmentation to full resolution
        istissue_tile = lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col]
        istissue_tile = cytometer.utils.resize(istissue_tile, size=(last_col - first_col, last_row - first_row),
                                               resample=PIL.Image.NEAREST)

        if DEBUG:
            plt.clf()
            plt.imshow(tile)
            plt.imshow(istissue_tile, alpha=0.5)
            plt.contour(istissue_tile, colors='k')
            plt.title('Yellow: Tissue. Purple: Background')
            plt.axis('off')

        # clear keras session to prevent each segmentation iteration from getting slower. Note that this forces us to
        # reload the models every time
        K.clear_session()

        # segment histology, split into individual objects, and apply segmentation correction
        labels, labels_class, todo_edge, \
        window_im, window_labels, window_labels_corrected, window_labels_class, index_list, scaling_factor_list \
            = cytometer.utils.segmentation_pipeline6(tile,
                                                     dmap_model=dmap_model_file,
                                                     contour_model=contour_model_file,
                                                     correction_model=correction_model_file,
                                                     classifier_model=classifier_model_file,
                                                     min_cell_area=min_cell_area,
                                                     mask=istissue_tile,
                                                     min_mask_overlap=min_mask_overlap,
                                                     phagocytosis=phagocytosis,
                                                     min_class_prop=min_class_prop,
                                                     correction_window_len=correction_window_len,
                                                     correction_smoothing=correction_smoothing,
                                                     return_bbox=True, return_bbox_coordinates='xy',
                                                     batch_size=batch_size)

        # if no cells found, wipe out current window from tissue segmentation, and go to next iteration. Otherwise we'd
        # enter an infinite loop
        if len(index_list) == 0:
            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0
            # contours_all.append([])
            # areas_all.append([])
            # np.savez(results_file, contours=contours_all, areas=areas_all, lores_istissue=lores_istissue)
            continue

        if DEBUG:
            j = 4
            plt.clf()
            plt.subplot(221)
            plt.imshow(tile[:, :, :])
            plt.title('Histology', fontsize=16)
            plt.axis('off')
            plt.subplot(222)
            plt.imshow(tile[:, :, :])
            plt.contour(labels, levels=np.unique(labels), colors='C0')
            plt.contourf(todo_edge, colors='C2', levels=[0.5, 1])
            plt.title('Full segmentation', fontsize=16)
            plt.axis('off')
            plt.subplot(212)
            plt.imshow(window_im[j, :, :, :])
            plt.contour(window_labels[j, :, :], colors='C0')
            plt.contour(window_labels_corrected[j, :, :], colors='C1')
            plt.title('Crop around object and corrected segmentation', fontsize=16)
            plt.axis('off')
            plt.tight_layout()

        # downsample "to do" mask so that the rough tissue segmentation can be updated
        lores_todo_edge = PIL.Image.fromarray(todo_edge.astype(np.uint8))
        lores_todo_edge = lores_todo_edge.resize((lores_last_col - lores_first_col,
                                                  lores_last_row - lores_first_row),
                                                 resample=PIL.Image.NEAREST)
        lores_todo_edge = np.array(lores_todo_edge)

        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col])
            plt.title('Low res tissue mask', fontsize=16)
            plt.axis('off')
            plt.subplot(222)
            plt.imshow(istissue_tile)
            plt.title('Full res tissue mask', fontsize=16)
            plt.axis('off')
            plt.subplot(223)
            plt.imshow(todo_edge)
            plt.title('Full res left over tissue', fontsize=16)
            plt.axis('off')
            plt.subplot(224)
            plt.imshow(lores_todo_edge)
            plt.title('Low res left over tissue', fontsize=16)
            plt.axis('off')
            plt.tight_layout()

        # convert overlap labels in cropped images to contours (points), and add cropping window offset so that the
        # contours are in the tile-window coordinates
        offset_xy = index_list[:, [2, 3]]  # index_list: [i, lab, x0, y0, xend, yend]
        contours = cytometer.utils.labels2contours(window_labels_corrected, offset_xy=offset_xy,
                                                  scaling_factor_xy=scaling_factor_list)

        if DEBUG:
            plt.clf()
            plt.imshow(tile)
            for j in range(len(contours)):
                plt.fill(contours[j][:, 0], contours[j][:, 1], edgecolor='C0', fill=False)

        # compute non-overlap cell areas
        props = regionprops(labels)
        p_label = [p['label'] for p in props]
        p_area = np.array([p['area'] for p in props])
        areas = p_area * xres * yres  # (m^2)

        # downsample contours for AIDA annotations file
        lores_contours = []
        lores_areas = []
        for c in contours:
            lores_c = bspline_resample(c, factor=contour_downsample_factor, min_n=10, k=bspline_k, is_closed=True)
            lores_contours.append(lores_c)

            # compute overlap cell areas
            lores_areas.append(Polygon(lores_c).area * xres * yres)  # (m^2)

        if DEBUG:
            plt.clf()
            plt.imshow(tile)
            for j in range(len(contours)):
                plt.fill(lores_contours[j][:, 0], lores_contours[j][:, 1], edgecolor='C1', fill=False)

        # add tile offset, so that contours are in full slide coordinates
        for j in range(len(contours)):
            lores_contours[j][:, 0] += first_col
            lores_contours[j][:, 1] += first_row

        # give a colour that is proportional to the cell area
        lores_areas_sqrt = np.sqrt(np.array(lores_areas))
        hue = np.interp(lores_areas_sqrt,
                        [0, np.sqrt(20e3 * 1e-12)],
                        [0, 270])

        # add segmented contours to annotations file
        if os.path.isfile(annotations_file):
            append_paths_to_aida_json_file(annotations_file, lores_contours, hue=hue, pretty_print=True)
        elif len(contours) > 0:
            fp = open(annotations_file, 'w')
            write_paths_to_aida_json_file(fp, lores_contours, hue=hue, pretty_print=True)
            fp.close()

        # # add contours to list of all contours for the image
        # contours_all.append(lores_contours)
        # areas_all.append(areas)

        # update the tissue segmentation mask with the current window
        if np.all(lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] == lores_todo_edge):
            # if the mask remains identical, wipe out the whole window, as otherwise we'd have an
            # infinite loop
            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0
        else:
            # if the mask has been updated, use it to update the total tissue segmentation
            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = lores_todo_edge

        # # save results after every window computation
        # np.savez(results_file, contours=contours_all, areas=areas_all, lores_istissue=lores_istissue)

# end of "keep extracting histology windows until we have finished"

# if we run the script with qsub on the cluster, the standard output is in file
# klf14_b6ntac_exp_0001_cnn_dmap_contour.sge.sh.oPID where PID is the process ID
# Save it to saved_models directory
log_filename = os.path.join(saved_models_dir, experiment_id + '.log')
stdout_filename = os.path.join(home, 'Software', 'cytometer', 'scripts', experiment_id + '.sge.sh.o*')
stdout_filename = glob.glob(stdout_filename)[0]
if stdout_filename and os.path.isfile(stdout_filename):
    shutil.copy2(stdout_filename, log_filename)
else:
    # if we ran the script with nohup in linux, the standard output is in file nohup.out.
    # Save it to saved_models directory
    log_filename = os.path.join(saved_models_dir, experiment_id + '.log')
    nohup_filename = os.path.join(home, 'Software', 'cytometer', 'scripts', 'nohup.out')
    if os.path.isfile(nohup_filename):
        shutil.copy2(nohup_filename, log_filename)

