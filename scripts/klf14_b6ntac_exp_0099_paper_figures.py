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
sys.path.extend([os.path.join(home, 'Software/cytometer')])

########################################################################################################################
## Plots of get_next_roi_to_process()
########################################################################################################################

import pickle
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
from matplotlib.patches import Rectangle
from scipy.signal import fftconvolve
from cytometer.utils import rough_foreground_mask
import PIL
from keras import backend as K

LIMIT_GPU_MEMORY = False

# limit GPU memory used
if LIMIT_GPU_MEMORY:
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    set_session(tf.Session(config=config))

DEBUG = False
SAVE_FIGS = False

root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
data_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
training_dir = os.path.join(home, root_data_dir, 'klf14_b6ntac_training')
seg_dir = os.path.join(home, root_data_dir, 'klf14_b6ntac_seg')
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
## Show examples of what each deep CNN do (code cannibilised from the "inspect" scripts of the networks)
########################################################################################################################

import pickle
import warnings

# other imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

import cytometer.data
import cytometer.utils
import cytometer.model_checkpoint_parallel
import tensorflow as tf

from PIL import Image, ImageDraw

LIMIT_GPU_MEMORY = False

# limit GPU memory used
if LIMIT_GPU_MEMORY:
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    set_session(tf.Session(config=config))

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False

'''Directories and filenames'''

# data paths
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
klf14_training_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training')
klf14_training_non_overlap_data_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training_non_overlap')
klf14_training_augmented_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training_augmented')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
saved_models_dir = os.path.join(klf14_root_data_dir, 'saved_models')

saved_kfolds_filename = 'klf14_b6ntac_exp_0079_generate_kfolds.pickle'

# model names
dmap_model_basename = 'klf14_b6ntac_exp_0086_cnn_dmap'
contour_model_basename = 'klf14_b6ntac_exp_0091_cnn_contour_after_dmap'
classifier_model_basename = 'klf14_b6ntac_exp_0095_cnn_tissue_classifier_fcn'
correction_model_basename = 'klf14_b6ntac_exp_0089_cnn_segmentation_correction_overlapping_scaled_contours'


'''Load folds'''

# load list of images, and indices for training vs. testing indices
contour_model_kfold_filename = os.path.join(saved_models_dir, saved_kfolds_filename)
with open(contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
svg_file_list = aux['file_list']
idx_test_all = aux['idx_test']
idx_train_all = aux['idx_train']

if DEBUG:
    for i, file in enumerate(svg_file_list):
        print(str(i) + ': ' + file)

# correct home directory
svg_file_list = [x.replace('/home/rcasero', home) for x in svg_file_list]

# KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06_row_017204_col_019444.tif (fold 5 for testing. No .svg)
# KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06_row_009644_col_061660.tif (fold 5 for testing. No .svg)
# KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00_row_019228_col_015060.svg (fold 7 for testing. With .svg)

# find which fold the testing image belongs to
np.where(['36.1c' in x for x in svg_file_list])
idx_test_all[7]

# TIFF files that correspond to the SVG files (without augmentation)
im_orig_file_list = []
for i, file in enumerate(svg_file_list):
    im_orig_file_list.append(file.replace('.svg', '.tif'))
    im_orig_file_list[i] = os.path.join(os.path.dirname(im_orig_file_list[i]) + '_augmented',
                                        'im_seed_nan_' + os.path.basename(im_orig_file_list[i]))

    # check that files exist
    if not os.path.isfile(file):
        # warnings.warn('i = ' + str(i) + ': File does not exist: ' + os.path.basename(file))
        warnings.warn('i = ' + str(i) + ': File does not exist: ' + file)
    if not os.path.isfile(im_orig_file_list[i]):
        # warnings.warn('i = ' + str(i) + ': File does not exist: ' + os.path.basename(im_orig_file_list[i]))
        warnings.warn('i = ' + str(i) + ': File does not exist: ' + im_orig_file_list[i])

'''Inspect model results'''

# for i_fold, idx_test in enumerate(idx_test_all):
i_fold = 7; idx_test = idx_test_all[i_fold]

print('Fold ' + str(i_fold) + '/' + str(len(idx_test_all)-1))

'''Load data'''

# split the data list into training and testing lists
im_test_file_list, im_train_file_list = cytometer.data.split_list(im_orig_file_list, idx_test)

# load the test data (im, dmap, mask)
test_dataset, test_file_list, test_shuffle_idx = \
    cytometer.data.load_datasets(im_test_file_list, prefix_from='im', prefix_to=['im', 'dmap', 'mask', 'contour'],
                                 nblocks=1, shuffle_seed=None)

# fill in the little gaps in the mask
kernel = np.ones((3, 3), np.uint8)
for i in range(test_dataset['mask'].shape[0]):
    test_dataset['mask'][i, :, :, 0] = cv2.dilate(test_dataset['mask'][i, :, :, 0].astype(np.uint8),
                                                  kernel=kernel, iterations=1)

# load dmap model, and adjust input size
saved_model_filename = os.path.join(saved_models_dir, dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
dmap_model = keras.models.load_model(saved_model_filename)
if dmap_model.input_shape[1:3] != test_dataset['im'].shape[1:3]:
    dmap_model = cytometer.utils.change_input_size(dmap_model, batch_shape=test_dataset['im'].shape)

# estimate dmaps
pred_dmap = dmap_model.predict(test_dataset['im'], batch_size=4)

if DEBUG:
    for i in range(test_dataset['im'].shape[0]):
        plt.clf()
        plt.subplot(221)
        plt.imshow(test_dataset['im'][i, :, :, :])
        plt.axis('off')
        plt.subplot(222)
        plt.imshow(test_dataset['dmap'][i, :, :, 0])
        plt.axis('off')
        plt.subplot(223)
        plt.imshow(test_dataset['mask'][i, :, :, 0])
        plt.axis('off')
        plt.subplot(224)
        plt.imshow(pred_dmap[i, :, :, 0])
        plt.axis('off')

if DEBUG:

    # KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00_row_019228_col_015060.svg
    i = 2

    plt.clf()
    plt.imshow(test_dataset['im'][i, :, :, :])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

    plt.clf()
    plt.imshow(test_dataset['dmap'][i, :, :, 0])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'dmap_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

    plt.clf()
    plt.imshow(pred_dmap[i, :, :, 0])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pred_dmap_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

# load dmap to contour model, and adjust input size
saved_model_filename = os.path.join(saved_models_dir, contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
contour_model = keras.models.load_model(saved_model_filename)
if contour_model.input_shape[1:3] != pred_dmap.shape[1:3]:
    contour_model = cytometer.utils.change_input_size(contour_model, batch_shape=pred_dmap.shape)

# estimate contours
pred_contour = contour_model.predict(pred_dmap, batch_size=4)

if DEBUG:
    plt.clf()
    plt.imshow(test_dataset['contour'][i, :, :, 0])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'contour_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

    plt.clf()
    plt.imshow(pred_contour[i, :, :, 0])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pred_contour_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

# load classifier model, and adjust input size
saved_model_filename = os.path.join(saved_models_dir, classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')
classifier_model = keras.models.load_model(saved_model_filename)
if classifier_model.input_shape[1:3] != test_dataset['im'].shape[1:3]:
    classifier_model = cytometer.utils.change_input_size(classifier_model, batch_shape=test_dataset['im'].shape)

# estimate pixel-classification
pred_class = classifier_model.predict(test_dataset['im'], batch_size=4)

if DEBUG:
    plt.clf()
    plt.imshow(pred_class[i, :, :, 0])
    plt.contour(pred_class[i, :, :, 0] > 0.5, colors='r', linewidhts=3)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pred_class_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

    plt.clf()
    plt.imshow(pred_class[i, :, :, 0] > 0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pred_class_thresh_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

## create classifier ground truth

# print('file ' + str(i) + '/' + str(len(file_svg_list) - 1))

# init output
im_array_all = []
out_class_all = []
out_mask_all = []
contour_type_all = []

file_tif = os.path.join(klf14_training_dir, os.path.basename(im_test_file_list[i]))
file_tif = file_tif.replace('im_seed_nan_', '')

# change file extension from .svg to .tif
file_svg = file_tif.replace('.tif', '.svg')

# open histology training image
im = Image.open(file_tif)

# make array copy
im_array = np.array(im)

# read the ground truth cell contours in the SVG file. This produces a list [contour_0, ..., contour_N-1]
# where each contour_i = [(X_0, Y_0), ..., (X_P-1, X_P-1)]
cell_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False,
                                                        minimum_npoints=3)
other_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Other', add_offset_from_filename=False,
                                                         minimum_npoints=3)
brown_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Brown', add_offset_from_filename=False,
                                                         minimum_npoints=3)
background_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Background',
                                                              add_offset_from_filename=False,
                                                              minimum_npoints=3)
contours = cell_contours + other_contours + brown_contours + background_contours

# make a list with the type of cell each contour is classified as
contour_type = [np.zeros(shape=(len(cell_contours),), dtype=np.uint8),  # 0: white-adipocyte
                np.ones(shape=(len(other_contours),), dtype=np.uint8),  # 1: other types of tissue
                np.ones(shape=(len(brown_contours),), dtype=np.uint8),  # 1: brown cells (treated as "other" tissue)
                np.zeros(shape=(len(background_contours),), dtype=np.uint8)]  # 0: background
contour_type = np.concatenate(contour_type)
contour_type_all.append(contour_type)

print('Cells: ' + str(len(cell_contours)))
print('Other: ' + str(len(other_contours)))
print('Brown: ' + str(len(brown_contours)))
print('Background: ' + str(len(background_contours)))

# initialise arrays for training
out_class = np.zeros(shape=im_array.shape[0:2], dtype=np.uint8)
out_mask = np.zeros(shape=im_array.shape[0:2], dtype=np.uint8)

# loop ground truth cell contours
for j, contour in enumerate(contours):

    plt.plot([p[0] for p in contour], [p[1] for p in contour])
    plt.text(contour[0][0], contour[0][1], str(j))

    if DEBUG:
        plt.clf()

        plt.subplot(121)
        plt.imshow(im_array)
        plt.plot([p[0] for p in contour], [p[1] for p in contour])
        xy_c = (np.mean([p[0] for p in contour]), np.mean([p[1] for p in contour]))
        plt.scatter(xy_c[0], xy_c[1])

    # rasterise current ground truth segmentation
    cell_seg_gtruth = Image.new("1", im_array.shape[0:2][::-1], "black")  # I = 32-bit signed integer pixels
    draw = ImageDraw.Draw(cell_seg_gtruth)
    draw.polygon(contour, outline="white", fill="white")
    cell_seg_gtruth = np.array(cell_seg_gtruth, dtype=np.bool)

    # we are going to save the ground truth segmentation of the cell that we are going to later use in the figures
    if j == 106:
        cell_seg_gtruth_106 = cell_seg_gtruth.copy()

    if DEBUG:
        plt.subplot(122)
        plt.cla()
        plt.imshow(im_array)
        plt.contour(cell_seg_gtruth.astype(np.uint8))

    # add current object to training output and mask
    out_mask[cell_seg_gtruth] = 1
    out_class[cell_seg_gtruth] = contour_type[j]

if DEBUG:
    plt.clf()
    aux = (1- out_class).astype(np.float32)
    aux = np.ma.masked_where(out_mask < 0.5, aux)
    plt.imshow(aux)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'class_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

## Segmentation correction CNN

# segmentation parameters
min_cell_area = 1500
max_cell_area = 100e3
min_mask_overlap = 0.8
phagocytosis = True
min_class_prop = 0.5
correction_window_len = 401
correction_smoothing = 11
batch_size = 2

# segment histology
labels, labels_class, _ \
    = cytometer.utils.segment_dmap_contour_v6(im_array,
                                              contour_model=contour_model, dmap_model=dmap_model,
                                              classifier_model=classifier_model,
                                              border_dilation=0)
labels = labels[0, :, :]
labels_class = labels_class[0, :, :, 0]

if DEBUG:
    plt.clf()
    plt.imshow(labels)

# remove labels that touch the edges, that are too small or too large, don't overlap enough with the tissue mask,
# are fully surrounded by another label or are not white adipose tissue
labels, todo_edge = cytometer.utils.clean_segmentation(
    labels, min_cell_area=min_cell_area, max_cell_area=max_cell_area,
    remove_edge_labels=True, mask=None, min_mask_overlap=min_mask_overlap,
    phagocytosis=phagocytosis,
    labels_class=labels_class, min_class_prop=min_class_prop)


# split image into individual labels
im_array = np.expand_dims(im_array, axis=0)
labels = np.expand_dims(labels, axis=0)
labels_class = np.expand_dims(labels_class, axis=0)
cell_seg_gtruth_106 = np.expand_dims(cell_seg_gtruth_106, axis=0)
window_mask = None
(window_labels, window_im, window_labels_class, window_cell_seg_gtruth_106), index_list, scaling_factor_list \
    = cytometer.utils.one_image_per_label_v2((labels, im_array, labels_class, cell_seg_gtruth_106.astype(np.uint8)),
                                             resize_to=(correction_window_len, correction_window_len),
                                             resample=(Image.NEAREST, Image.LINEAR, Image.NEAREST, Image.NEAREST),
                                             only_central_label=True, return_bbox=False)

# load correction model
saved_model_filename = os.path.join(saved_models_dir, correction_model_basename + '_model_fold_' + str(i_fold) + '.h5')
correction_model = keras.models.load_model(saved_model_filename)
if correction_model.input_shape[1:3] != window_im.shape[1:3]:
    correction_model = cytometer.utils.change_input_size(correction_model, batch_shape=window_im.shape)

# multiply image by mask
window_im_masked = cytometer.utils.quality_model_mask(
    np.expand_dims(window_labels, axis=-1), im=window_im, quality_model_type='-1_1')

# process (histology * mask) to estimate which pixels are underestimated and which overestimated in the segmentation
window_im_masked = correction_model.predict(window_im_masked, batch_size=batch_size)

# compute the correction to be applied to the segmentation
correction = (window_im[:, :, :, 0].copy() * 0).astype(np.float32)
correction[window_im_masked[:, :, :, 0] >= 0.5] = 1  # the segmentation went too far
correction[window_im_masked[:, :, :, 0] <= -0.5] = -1  # the segmentation fell short

if DEBUG:
    j = 0

    plt.clf()
    plt.imshow(correction[j, :, :])
    # plt.contour(window_labels[j, ...], colors='r', linewidths=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pred_correction_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

    plt.clf()
    plt.imshow(correction[j, :, :])
    plt.contour(window_labels[j, ...], colors='r', linewidths=1)
    plt.contour(window_cell_seg_gtruth_106[j, ...], colors='w', linewidths=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(
        os.path.join(figures_dir, 'pred_correction_gtruth_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
        bbox_inches='tight')

# correct segmentation (full operation)
window_im = window_im.astype(np.float32)
window_im /= 255.0
window_labels_corrected = cytometer.utils.correct_segmentation(
    im=window_im, seg=window_labels,
    correction_model=correction_model, model_type='-1_1',
    smoothing=correction_smoothing,
    batch_size=batch_size)

if DEBUG:
    j = 0

    plt.clf()
    plt.imshow(window_im[j, ...])
    plt.contour(window_labels[j, ...], colors='r', linewidths=3)
    plt.text(185, 210, '+1', fontsize=30)
    plt.text(116, 320, '-1', fontsize=30)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'im_for_correction_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

    plt.clf()
    plt.imshow(window_im[j, ...])
    plt.contour(window_labels_corrected[j, ...], colors='g', linewidths=3)
    plt.text(185, 210, '+1', fontsize=30)
    plt.text(116, 320, '-1', fontsize=30)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'corrected_seg_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

    aux = np.array(contours[j])
    plt.plot(aux[:, 0], aux[:, 1])

########################################################################################################################
## Images for cytometer_v7.svg schematic (pipeline diagram)
########################################################################################################################

