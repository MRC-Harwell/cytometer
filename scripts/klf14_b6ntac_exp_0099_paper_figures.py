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
## Explore training/test data of different folds
########################################################################################################################

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K

import cytometer
import cytometer.data
import tensorflow as tf

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
ndpi_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
metainfo_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')

saved_kfolds_filename = 'klf14_b6ntac_exp_0079_generate_kfolds.pickle'
saved_extra_kfolds_filename = 'klf14_b6ntac_exp_0094_generate_extra_training_images.pickle'


# original dataset used in pipelines up to v6 + extra "other" tissue images
kfold_filename = os.path.join(saved_models_dir, saved_extra_kfolds_filename)
with open(kfold_filename, 'rb') as f:
    aux = pickle.load(f)
file_svg_list = aux['file_list']
idx_test_all = aux['idx_test']
idx_train_all = aux['idx_train']

# correct home directory
file_svg_list = [x.replace('/users/rittscher/rcasero', home) for x in file_svg_list]
file_svg_list = [x.replace('/home/rcasero', home) for x in file_svg_list]

# number of images
n_im = len(file_svg_list)

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

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


if DEBUG:
    # list of NDPI files
    for key in ndpi_files_test_list.keys():
        print(key)

# init dataframe to aggregate training numbers of each mouse
table = pd.DataFrame(columns=['Cells', 'Other', 'Background', 'Windows', 'Windows with cells'])

# loop files with hand traced contours
for i, file_svg in enumerate(file_svg_list):

    print('file ' + str(i) + '/' + str(len(file_svg_list) - 1) + ': ' + os.path.basename(file_svg))

    # read the ground truth cell contours in the SVG file. This produces a list [contour_0, ..., contour_N-1]
    # where each contour_i = [(X_0, Y_0), ..., (X_P-1, X_P-1)]
    cell_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False,
                                                            minimum_npoints=3)
    other_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Other', add_offset_from_filename=False,
                                                             minimum_npoints=3)
    brown_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Brown', add_offset_from_filename=False,
                                                             minimum_npoints=3)
    background_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Background', add_offset_from_filename=False,
                                                                  minimum_npoints=3)
    contours = cell_contours + other_contours + brown_contours + background_contours

    # make a list with the type of cell each contour is classified as
    contour_type = [np.zeros(shape=(len(cell_contours),), dtype=np.uint8),  # 0: white-adipocyte
                    np.ones(shape=(len(other_contours),), dtype=np.uint8),  # 1: other types of tissue
                    np.ones(shape=(len(brown_contours),), dtype=np.uint8),  # 1: brown cells (treated as "other" tissue)
                    np.zeros(shape=(len(background_contours),), dtype=np.uint8)] # 0: background
    contour_type = np.concatenate(contour_type)

    print('Cells: ' + str(len(cell_contours)) + '. Other: ' + str(len(other_contours))
          + '. Brown: ' + str(len(brown_contours)) + '. Background: ' + str(len(background_contours)))

    # create dataframe for this image
    df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_svg),
                                                          values=[i,], values_tag='i',
                                                          tags_to_keep=['id', 'ko_parent', 'sex'])

    # mouse ID as a string
    id = df_common['id'].values[0]
    sex = df_common['sex'].values[0]
    ko = df_common['ko_parent'].values[0]

    # row to add to the table
    df = pd.DataFrame(
        [(sex, ko,
          len(cell_contours), len(other_contours) + len(brown_contours), len(background_contours), 1, int(len(cell_contours)>0))],
        columns=['Sex', 'Genotype', 'Cells', 'Other', 'Background', 'Windows', 'Windows with cells'], index=[id])

    if id in table.index:

        num_cols = ['Cells', 'Other', 'Background', 'Windows', 'Windows with cells']
        table.loc[id, num_cols] = (table.loc[id, num_cols] + df.loc[id, num_cols])

    else:

        table = table.append(df, sort=False, ignore_index=False, verify_integrity=True)

# alphabetical order by mouse IDs
table = table.sort_index()

# total number of sampled windows
print('Total number of windows = ' + str(np.sum(table['Windows'])))
print('Total number of windows with cells = ' + str(np.sum(table['Windows with cells'])))

# total number of "Other" and background areas
print('Total number of Other areas = ' + str(np.sum(table['Other'])))
print('Total number of Background areas = ' + str(np.sum(table['Background'])))

# aggregate by sex and genotype
idx_f = table['Sex'] == 'f'
idx_m = table['Sex'] == 'm'
idx_pat = table['Genotype'] == 'PAT'
idx_mat = table['Genotype'] == 'MAT'

print('f PAT: ' + str(np.sum(table.loc[idx_f * idx_pat, 'Cells'])))
print('f MAT: ' + str(np.sum(table.loc[idx_f * idx_mat, 'Cells'])))
print('m PAT: ' + str(np.sum(table.loc[idx_m * idx_pat, 'Cells'])))
print('m MAT: ' + str(np.sum(table.loc[idx_m * idx_mat, 'Cells'])))

# find folds that test images belong to
for i_file, ndpi_file_kernel in enumerate(ndpi_files_test_list):

    # fold  where the current .ndpi image was not used for training
    i_fold = ndpi_files_test_list[ndpi_file_kernel]

    print('File ' + str(i_file) + '/' + str(len(ndpi_files_test_list) - 1) + ': ' + ndpi_file_kernel
          + '. Fold = ' + str(i_fold))

# mean and std of mouse weight
weight_f_mat = [22.07, 26.39, 30.65, 24.28, 27.72]
weight_f_pat = [31.42, 29.25, 27.18, 23.69, 21.20]
weight_m_mat = [46.19, 40.87, 40.02, 41.98, 34.52, 36.08]
weight_m_pat = [36.55, 40.77, 36.98, 36.11]

print('f MAT: mean = ' + str(np.mean(weight_f_mat)) + ', std = ' + str(np.std(weight_f_mat)))
print('f PAT: mean = ' + str(np.mean(weight_f_pat)) + ', std = ' + str(np.std(weight_f_pat)))
print('m MAT: mean = ' + str(np.mean(weight_m_mat)) + ', std = ' + str(np.std(weight_m_mat)))
print('m PAT: mean = ' + str(np.mean(weight_m_pat)) + ', std = ' + str(np.std(weight_m_pat)))

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
import math

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

# KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00_row_019228_col_015060.svg
i = 2

if DEBUG:

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

if DEBUG:
    plt.clf()
    plt.imshow(im_array)
    plt.contour(labels, levels=np.unique(labels), colors='k')
    plt.contourf(labels == 0)

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
## Plots of segmented full slides with quantile colourmaps
########################################################################################################################

# This is done in klf14_b6ntac_exp_0098_full_slide_size_analysis_v7

########################################################################################################################
## Segmentation validation
########################################################################################################################

# This is done in klf14_b6ntac_exp_0096_pipeline_v7_validation.py

########################################################################################################################
## Cell populations from automatically segmented images in two depots: SQWAT and GWAT.
## This section needs to be run for each of the depots. But the results are saved, so in later sections, it's possible
## to get all the data together
########################################################################################################################

import matplotlib.pyplot as plt
import cytometer.data
from shapely.geometry import Polygon
import openslide
import numpy as np
import scipy.stats
import pandas as pd
from mlxtend.evaluate import permutation_test
from statsmodels.stats.multitest import multipletests
import math

# directories
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
annotations_dir = os.path.join(home, 'Software/AIDA/dist/data/annotations')
ndpi_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
metainfo_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')

DEBUG = False

depot = 'sqwat'
# depot = 'gwat'

permutation_sample_size = 9  # the factorial of this number is the number of repetitions in the permutation tests

if depot == 'sqwat':
    # SQWAT: list of annotation files
    json_annotation_files = [
        'KLF14-B6NTAC 36.1d PAT 99-16 C1 - 2016-02-11 11.48.31.json',
        'KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46.json',
        'KLF14-B6NTAC-MAT-17.1a  44-16 C1 - 2016-02-01 11.14.17.json',
        'KLF14-B6NTAC-MAT-17.1e  48-16 C1 - 2016-02-01 16.27.05.json',
        'KLF14-B6NTAC-MAT-18.2a  57-16 C1 - 2016-02-03 09.10.17.json',
        'KLF14-B6NTAC-PAT-37.3c  414-16 C1 - 2016-03-15 17.15.41.json',
        'KLF14-B6NTAC-MAT-18.1d  53-16 C1 - 2016-02-02 14.32.03.json',
        'KLF14-B6NTAC-MAT-17.2b  65-16 C1 - 2016-02-04 10.24.22.json',
        'KLF14-B6NTAC-MAT-17.2g  69-16 C1 - 2016-02-04 16.15.05.json',
        'KLF14-B6NTAC 37.1a PAT 106-16 C1 - 2016-02-12 16.21.00.json',
        'KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06.json',
        # 'KLF14-B6NTAC-PAT-37.2d  411-16 C1 - 2016-03-15 12.42.26.json',
        'KLF14-B6NTAC-MAT-17.2a  64-16 C1 - 2016-02-04 09.17.52.json',
        'KLF14-B6NTAC-MAT-16.2f  216-16 C1 - 2016-02-18 10.28.27.json',
        'KLF14-B6NTAC-MAT-17.1d  47-16 C1 - 2016-02-01 15.25.53.json',
        'KLF14-B6NTAC-MAT-16.2e  215-16 C1 - 2016-02-18 09.19.26.json',
        'KLF14-B6NTAC 36.1g PAT 102-16 C1 - 2016-02-11 17.20.14.json',
        'KLF14-B6NTAC-37.1g PAT 112-16 C1 - 2016-02-16 13.33.09.json',
        'KLF14-B6NTAC-38.1e PAT 94-16 C1 - 2016-02-10 12.13.10.json',
        'KLF14-B6NTAC-MAT-18.2d  60-16 C1 - 2016-02-03 13.13.57.json',
        'KLF14-B6NTAC-MAT-18.2g  63-16 C1 - 2016-02-03 16.58.52.json',
        'KLF14-B6NTAC-MAT-18.2f  62-16 C1 - 2016-02-03 15.46.15.json',
        'KLF14-B6NTAC-MAT-18.1b  51-16 C1 - 2016-02-02 09.59.16.json',
        'KLF14-B6NTAC-MAT-19.2c  220-16 C1 - 2016-02-18 17.03.38.json',
        'KLF14-B6NTAC-MAT-18.1f  55-16 C1 - 2016-02-02 16.14.30.json',
        'KLF14-B6NTAC-PAT-36.3b  412-16 C1 - 2016-03-15 14.37.55.json',
        'KLF14-B6NTAC-MAT-16.2c  213-16 C1 - 2016-02-17 14.51.18.json',
        'KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32.json',
        'KLF14-B6NTAC 36.1e PAT 100-16 C1 - 2016-02-11 14.06.56.json',
        'KLF14-B6NTAC-MAT-18.1c  52-16 C1 - 2016-02-02 12.26.58.json',
        'KLF14-B6NTAC-MAT-18.2b  58-16 C1 - 2016-02-03 11.10.52.json',
        'KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38.json',
        'KLF14-B6NTAC-PAT-39.2d  454-16 C1 - 2016-03-17 14.33.38.json',
        'KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00.json',
        'KLF14-B6NTAC-MAT-18.2e  61-16 C1 - 2016-02-03 14.19.35.json',
        'KLF14-B6NTAC-MAT-19.2g  222-16 C1 - 2016-02-25 15.13.00.json',
        'KLF14-B6NTAC-PAT-37.2a  406-16 C1 - 2016-03-14 12.01.56.json',
        'KLF14-B6NTAC 36.1j PAT 105-16 C1 - 2016-02-12 14.33.33.json',
        'KLF14-B6NTAC-37.1b PAT 107-16 C1 - 2016-02-15 11.43.31.json',
        'KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04.json',
        'KLF14-B6NTAC-MAT-19.2f  217-16 C1 - 2016-02-18 11.48.16.json',
        'KLF14-B6NTAC-MAT-17.2d  67-16 C1 - 2016-02-04 12.34.32.json',
        'KLF14-B6NTAC-MAT-18.3c  218-16 C1 - 2016-02-18 13.12.09.json',
        'KLF14-B6NTAC-PAT-37.3a  413-16 C1 - 2016-03-15 15.54.12.json',
        'KLF14-B6NTAC-MAT-19.1a  56-16 C1 - 2016-02-02 17.23.31.json',
        'KLF14-B6NTAC-37.1h PAT 113-16 C1 - 2016-02-16 15.14.09.json',
        'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53.json',
        'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52.json',
        'KLF14-B6NTAC-37.1e PAT 110-16 C1 - 2016-02-15 17.33.11.json',
        'KLF14-B6NTAC-MAT-17.2f  68-16 C1 - 2016-02-04 15.05.54.json',
        'KLF14-B6NTAC 36.1h PAT 103-16 C1 - 2016-02-12 10.15.22.json',
        # 'KLF14-B6NTAC-PAT-39.1h  453-16 C1 - 2016-03-17 11.38.04.json',
        'KLF14-B6NTAC-MAT-16.2b  212-16 C1 - 2016-02-17 12.49.00.json',
        'KLF14-B6NTAC-MAT-17.1f  49-16 C1 - 2016-02-01 17.51.46.json',
        'KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11.json',
        'KLF14-B6NTAC-MAT-16.2a  211-16 C1 - 2016-02-17 11.46.42.json',
        'KLF14-B6NTAC-38.1f PAT 95-16 C1 - 2016-02-10 14.41.44.json',
        'KLF14-B6NTAC-PAT-36.3a  409-16 C1 - 2016-03-15 10.18.46.json',
        'KLF14-B6NTAC-MAT-19.2b  219-16 C1 - 2016-02-18 15.41.38.json',
        'KLF14-B6NTAC-MAT-17.1b  45-16 C1 - 2016-02-01 12.23.50.json',
        'KLF14-B6NTAC 36.1f PAT 101-16 C1 - 2016-02-11 15.23.06.json',
        'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33.json',
        'KLF14-B6NTAC-37.1d PAT 109-16 C1 - 2016-02-15 15.19.08.json',
        'KLF14-B6NTAC-MAT-18.2c  59-16 C1 - 2016-02-03 11.56.52.json',
        'KLF14-B6NTAC-PAT-37.2f  405-16 C1 - 2016-03-14 10.58.34.json',
        'KLF14-B6NTAC-PAT-37.2e  408-16 C1 - 2016-03-14 16.23.30.json',
        'KLF14-B6NTAC-MAT-19.2e  221-16 C1 - 2016-02-25 14.00.14.json',
        # 'KLF14-B6NTAC-PAT-37.2c  407-16 C1 - 2016-03-14 14.13.54.json',
        # 'KLF14-B6NTAC-PAT-37.2b  410-16 C1 - 2016-03-15 11.24.20.json',
        'KLF14-B6NTAC-PAT-37.4b  419-16 C1 - 2016-03-17 10.22.54.json',
        'KLF14-B6NTAC-37.1c PAT 108-16 C1 - 2016-02-15 14.49.45.json',
        'KLF14-B6NTAC-MAT-18.1a  50-16 C1 - 2016-02-02 09.12.41.json',
        'KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38.json',
        'KLF14-B6NTAC-PAT-37.2h  418-16 C1 - 2016-03-16 17.01.17.json',
        'KLF14-B6NTAC-MAT-17.2c  66-16 C1 - 2016-02-04 11.46.39.json',
        'KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52.json',
        'KLF14-B6NTAC-37.1f PAT 111-16 C2 - 2016-02-16 11.26 (1).json',
        'KLF14-B6NTAC-PAT 37.2b 410-16 C4 - 2020-02-14 10.27.23.json',
        'KLF14-B6NTAC-PAT 37.2c 407-16 C4 - 2020-02-14 10.15.57.json',
        # 'KLF14-B6NTAC-PAT 37.2d 411-16 C4 - 2020-02-14 10.34.10.json'
    ]
elif depot == 'gwat':
    # GWAT: list of annotation files
    json_annotation_files = [
        'KLF14-B6NTAC-36.1a PAT 96-16 B1 - 2016-02-10 15.32.31.json',
        'KLF14-B6NTAC-36.1b PAT 97-16 B1 - 2016-02-10 17.15.16.json',
        'KLF14-B6NTAC-36.1c PAT 98-16 B1 - 2016-02-10 18.32.40.json',
        'KLF14-B6NTAC 36.1d PAT 99-16 B1 - 2016-02-11 11.29.55.json',
        'KLF14-B6NTAC 36.1e PAT 100-16 B1 - 2016-02-11 12.51.11.json',
        'KLF14-B6NTAC 36.1f PAT 101-16 B1 - 2016-02-11 14.57.03.json',
        'KLF14-B6NTAC 36.1g PAT 102-16 B1 - 2016-02-11 16.12.01.json',
        'KLF14-B6NTAC 36.1h PAT 103-16 B1 - 2016-02-12 09.51.08.json',
        # 'KLF14-B6NTAC 36.1i PAT 104-16 B1 - 2016-02-12 11.37.56.json',
        'KLF14-B6NTAC 36.1j PAT 105-16 B1 - 2016-02-12 14.08.19.json',
        'KLF14-B6NTAC 37.1a PAT 106-16 B1 - 2016-02-12 15.33.02.json',
        'KLF14-B6NTAC-37.1b PAT 107-16 B1 - 2016-02-15 11.25.20.json',
        'KLF14-B6NTAC-37.1c PAT 108-16 B1 - 2016-02-15 12.33.10.json',
        'KLF14-B6NTAC-37.1d PAT 109-16 B1 - 2016-02-15 15.03.44.json',
        'KLF14-B6NTAC-37.1e PAT 110-16 B1 - 2016-02-15 16.16.06.json',
        'KLF14-B6NTAC-37.1g PAT 112-16 B1 - 2016-02-16 12.02.07.json',
        'KLF14-B6NTAC-37.1h PAT 113-16 B1 - 2016-02-16 14.53.02.json',
        'KLF14-B6NTAC-38.1e PAT 94-16 B1 - 2016-02-10 11.35.53.json',
        'KLF14-B6NTAC-38.1f PAT 95-16 B1 - 2016-02-10 14.16.55.json',
        'KLF14-B6NTAC-MAT-16.2a  211-16 B1 - 2016-02-17 11.21.54.json',
        'KLF14-B6NTAC-MAT-16.2b  212-16 B1 - 2016-02-17 12.33.18.json',
        'KLF14-B6NTAC-MAT-16.2c  213-16 B1 - 2016-02-17 14.01.06.json',
        'KLF14-B6NTAC-MAT-16.2d  214-16 B1 - 2016-02-17 15.43.57.json',
        'KLF14-B6NTAC-MAT-16.2e  215-16 B1 - 2016-02-17 17.14.16.json',
        'KLF14-B6NTAC-MAT-16.2f  216-16 B1 - 2016-02-18 10.05.52.json',
        # 'KLF14-B6NTAC-MAT-17.1a  44-16 B1 - 2016-02-01 09.19.20.json',
        'KLF14-B6NTAC-MAT-17.1b  45-16 B1 - 2016-02-01 12.05.15.json',
        'KLF14-B6NTAC-MAT-17.1c  46-16 B1 - 2016-02-01 13.01.30.json',
        'KLF14-B6NTAC-MAT-17.1d  47-16 B1 - 2016-02-01 15.11.42.json',
        'KLF14-B6NTAC-MAT-17.1e  48-16 B1 - 2016-02-01 16.01.09.json',
        'KLF14-B6NTAC-MAT-17.1f  49-16 B1 - 2016-02-01 17.12.31.json',
        'KLF14-B6NTAC-MAT-17.2a  64-16 B1 - 2016-02-04 08.57.34.json',
        'KLF14-B6NTAC-MAT-17.2b  65-16 B1 - 2016-02-04 10.06.00.json',
        'KLF14-B6NTAC-MAT-17.2c  66-16 B1 - 2016-02-04 11.14.28.json',
        'KLF14-B6NTAC-MAT-17.2d  67-16 B1 - 2016-02-04 12.20.20.json',
        'KLF14-B6NTAC-MAT-17.2f  68-16 B1 - 2016-02-04 14.01.40.json',
        'KLF14-B6NTAC-MAT-17.2g  69-16 B1 - 2016-02-04 15.52.52.json',
        'KLF14-B6NTAC-MAT-18.1a  50-16 B1 - 2016-02-02 08.49.06.json',
        'KLF14-B6NTAC-MAT-18.1b  51-16 B1 - 2016-02-02 09.46.31.json',
        'KLF14-B6NTAC-MAT-18.1c  52-16 B1 - 2016-02-02 11.24.31.json',
        'KLF14-B6NTAC-MAT-18.1d  53-16 B1 - 2016-02-02 14.11.37.json',
        # 'KLF14-B6NTAC-MAT-18.1e  54-16 B1 - 2016-02-02 15.06.05.json',
        'KLF14-B6NTAC-MAT-18.2a  57-16 B1 - 2016-02-03 08.54.27.json',
        'KLF14-B6NTAC-MAT-18.2b  58-16 B1 - 2016-02-03 09.58.06.json',
        'KLF14-B6NTAC-MAT-18.2c  59-16 B1 - 2016-02-03 11.41.32.json',
        'KLF14-B6NTAC-MAT-18.2d  60-16 B1 - 2016-02-03 12.56.49.json',
        'KLF14-B6NTAC-MAT-18.2e  61-16 B1 - 2016-02-03 14.02.25.json',
        'KLF14-B6NTAC-MAT-18.2f  62-16 B1 - 2016-02-03 15.00.17.json',
        'KLF14-B6NTAC-MAT-18.2g  63-16 B1 - 2016-02-03 16.40.37.json',
        'KLF14-B6NTAC-MAT-18.3b  223-16 B1 - 2016-02-25 16.53.42.json',
        'KLF14-B6NTAC-MAT-18.3c  218-16 B1 - 2016-02-18 12.51.46.json',
        'KLF14-B6NTAC-MAT-18.3d  224-16 B1 - 2016-02-26 10.48.56.json',
        'KLF14-B6NTAC-MAT-19.1a  56-16 B1 - 2016-02-02 16.57.46.json',
        'KLF14-B6NTAC-MAT-19.2b  219-16 B1 - 2016-02-18 14.21.50.json',
        'KLF14-B6NTAC-MAT-19.2c  220-16 B1 - 2016-02-18 16.40.48.json',
        'KLF14-B6NTAC-MAT-19.2e  221-16 B1 - 2016-02-25 13.15.27.json',
        'KLF14-B6NTAC-MAT-19.2f  217-16 B1 - 2016-02-18 11.23.22.json',
        'KLF14-B6NTAC-MAT-19.2g  222-16 B1 - 2016-02-25 14.51.57.json',
        'KLF14-B6NTAC-PAT-36.3a  409-16 B1 - 2016-03-15 09.24.54.json',
        'KLF14-B6NTAC-PAT-36.3b  412-16 B1 - 2016-03-15 14.11.47.json',
        'KLF14-B6NTAC-PAT-36.3d  416-16 B1 - 2016-03-16 14.22.04.json',
        # 'KLF14-B6NTAC-PAT-37.2a  406-16 B1 - 2016-03-14 11.46.47.json',
        'KLF14-B6NTAC-PAT-37.2b  410-16 B1 - 2016-03-15 11.12.01.json',
        'KLF14-B6NTAC-PAT-37.2c  407-16 B1 - 2016-03-14 12.54.55.json',
        'KLF14-B6NTAC-PAT-37.2d  411-16 B1 - 2016-03-15 12.01.13.json',
        'KLF14-B6NTAC-PAT-37.2e  408-16 B1 - 2016-03-14 16.06.43.json',
        'KLF14-B6NTAC-PAT-37.2f  405-16 B1 - 2016-03-14 09.49.45.json',
        'KLF14-B6NTAC-PAT-37.2g  415-16 B1 - 2016-03-16 11.04.45.json',
        'KLF14-B6NTAC-PAT-37.2h  418-16 B1 - 2016-03-16 16.42.16.json',
        'KLF14-B6NTAC-PAT-37.3a  413-16 B1 - 2016-03-15 15.31.26.json',
        'KLF14-B6NTAC-PAT-37.3c  414-16 B1 - 2016-03-15 16.49.22.json',
        'KLF14-B6NTAC-PAT-37.4a  417-16 B1 - 2016-03-16 15.25.38.json',
        'KLF14-B6NTAC-PAT-37.4b  419-16 B1 - 2016-03-17 09.10.42.json',
        'KLF14-B6NTAC-PAT-38.1a  90-16 B1 - 2016-02-04 17.27.42.json',
        'KLF14-B6NTAC-PAT-39.1h  453-16 B1 - 2016-03-17 11.15.50.json',
        'KLF14-B6NTAC-PAT-39.2d  454-16 B1 - 2016-03-17 12.16.06.json'
    ]

# modify filenames to select the particular segmentation we want (e.g. the automatic ones, or the manually refined ones)
json_annotation_files = [x.replace('.json', '_exp_0097_corrected.json') for x in json_annotation_files]
json_annotation_files = [os.path.join(annotations_dir, x) for x in json_annotation_files]

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# make sure that in the boxplots PAT comes before MAT
metainfo['sex'] = metainfo['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
metainfo['ko_parent'] = metainfo['ko_parent'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
metainfo['genotype'] = metainfo['genotype'].astype(pd.api.types.CategoricalDtype(categories=['KLF14-KO:WT', 'KLF14-KO:Het'], ordered=True))

quantiles = np.linspace(0, 1, 11)
quantiles = quantiles[1:-1]

# load or compute area quantiles
filename_quantiles = os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_area_quantiles_' + depot + '.npz')
if os.path.isfile(filename_quantiles):

    aux = np.load(filename_quantiles)
    area_mean_all = aux['area_mean_all']
    area_q_all = aux['area_q_all']
    id_all = aux['id_all']
    ko_all = aux['ko_all']
    genotype_all = aux['genotype_all']
    sex_all = aux['sex_all']

else:

    area_mean_all = []
    area_q_all = []
    id_all = []
    ko_all = []
    genotype_all = []
    sex_all = []
    bw_all = []
    gwat_all = []
    sc_all = []
    for i_file, json_file in enumerate(json_annotation_files):

        print('File ' + str(i_file) + '/' + str(len(json_annotation_files)-1) + ': ' + os.path.basename(json_file))

        if not os.path.isfile(json_file):
            print('Missing file')
            continue

        # ndpi file that corresponds to this .json file
        ndpi_file = json_file.replace('_exp_0097_corrected.json', '.ndpi')
        ndpi_file = ndpi_file.replace(annotations_dir, ndpi_dir)

        # open full resolution histology slide
        im = openslide.OpenSlide(ndpi_file)

        # pixel size
        assert (im.properties['tiff.ResolutionUnit'] == 'centimeter')
        xres = 1e-2 / float(im.properties['tiff.XResolution'])  # m
        yres = 1e-2 / float(im.properties['tiff.YResolution'])  # m

        # create dataframe for this image
        df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(json_file),
                                                              values=[i_file,], values_tag='i_file',
                                                              tags_to_keep=['id', 'ko_parent', 'genotype', 'sex',
                                                                            'BW', 'gWAT', 'SC'])

        # mouse ID as a string
        id = df_common['id'].values[0]
        ko = df_common['ko_parent'].values[0]
        genotype = df_common['genotype'].values[0]
        sex = df_common['sex'].values[0]
        bw = df_common['BW'].values[0]
        gwat = df_common['gWAT'].values[0]
        sc = df_common['SC'].values[0]

        # read contours from AIDA annotations
        contours = cytometer.data.aida_get_contours(os.path.join(annotations_dir, json_file), layer_name='White adipocyte.*')

        # compute area of each contour
        areas = [Polygon(c).area * xres * yres for c in contours]  # (um^2)

        # compute average area of all contours
        area_mean = np.mean(areas)

        # compute HD quantiles
        area_q = scipy.stats.mstats.hdquantiles(areas, prob=quantiles, axis=0)

        # append to totals
        area_mean_all.append(area_mean)
        area_q_all.append(area_q)
        id_all.append(id)
        ko_all.append(ko)
        genotype_all.append(genotype)
        sex_all.append(sex)
        bw_all.append(bw)
        gwat_all.append(gwat)
        sc_all.append(sc)

    # reorder from largest to smallest final area value
    area_mean_all = np.array(area_mean_all)
    area_q_all = np.array(area_q_all)
    id_all = np.array(id_all)
    ko_all = np.array(ko_all)
    genotype_all = np.array(genotype_all)
    sex_all = np.array(sex_all)
    bw_all = np.array(bw_all)
    gwat_all = np.array(gwat_all)
    sc_all = np.array(sc_all)

    idx = np.argsort(area_q_all[:, -1])
    idx = idx[::-1]  # sort from larger to smaller
    area_mean_all = area_mean_all[idx]
    area_q_all = area_q_all[idx, :]
    id_all = id_all[idx]
    ko_all = ko_all[idx]
    genotype_all = genotype_all[idx]
    sex_all = sex_all[idx]
    bw_all = bw_all[idx]
    gwat_all = gwat_all[idx]
    sc_all = sc_all[idx]

    np.savez_compressed(filename_quantiles, area_mean_all=area_mean_all, area_q_all=area_q_all, id_all=id_all,
                        ko_all=ko_all, genotype_all=genotype_all, sex_all=sex_all,
                        bw_all=bw_all, gwat_all=gwat_all, sc_all=sc_all)

if DEBUG:
    plt.clf()

    for i in range(len(area_q_all)):

        # plot
        if ko_all[i] == 'PAT':
            color = 'g'
        elif ko_all[i] == 'MAT':
            color = 'r'
        else:
            raise ValueError('Unknown ko value: ' + ko)

        if sex_all[i] == 'f':
            plt.subplot(121)
            plt.plot(quantiles, area_q_all[i] * 1e12 * 1e-3, color=color)
        elif sex_all[i] == 'm':
            plt.subplot(122)
            plt.plot(quantiles, area_q_all[i] * 1e12 * 1e-3, color=color)
        else:
            raise ValueError('Unknown sex value: ' + sex)


    legend_f = [i + ' ' + j.replace('KLF14-KO:', '') for i, j
                in zip(id_all[sex_all == 'f'], genotype_all[sex_all == 'f'])]
    legend_m = [i + ' ' + j.replace('KLF14-KO:', '') for i, j
                in zip(id_all[sex_all == 'm'], genotype_all[sex_all == 'm'])]
    plt.subplot(121)
    plt.title('Female', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.xlabel('Quantile', fontsize=14)
    plt.ylabel('Area ($10^{3}\ \mu m^2$)', fontsize=14)
    plt.legend(legend_f, fontsize=12)
    plt.subplot(122)
    plt.title('Male', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.xlabel('Quantile', fontsize=14)
    plt.legend(legend_m, fontsize=12)

# DEBUG:
# area_q_all = np.vstack((area_q_all, area_q_all))
# id_all = np.hstack((id_all, id_all))
# ko_all = np.hstack((ko_all, ko_all))
# genotype_all = np.hstack((genotype_all, genotype_all))
# sex_all = np.hstack((sex_all, sex_all))

# compute variability of area values for each quantile
area_q_f_pat = area_q_all[(sex_all == 'f') * (ko_all == 'PAT'), :]
area_q_m_pat = area_q_all[(sex_all == 'm') * (ko_all == 'PAT'), :]
area_q_f_mat = area_q_all[(sex_all == 'f') * (ko_all == 'MAT'), :]
area_q_m_mat = area_q_all[(sex_all == 'm') * (ko_all == 'MAT'), :]
area_interval_f_pat = scipy.stats.mstats.hdquantiles(area_q_f_pat, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
area_interval_m_pat = scipy.stats.mstats.hdquantiles(area_q_m_pat, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
area_interval_f_mat = scipy.stats.mstats.hdquantiles(area_q_f_mat, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
area_interval_m_mat = scipy.stats.mstats.hdquantiles(area_q_m_mat, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)

area_q_f_pat_wt = area_q_all[(sex_all == 'f') * (ko_all == 'PAT') * (genotype_all == 'KLF14-KO:WT'), :]
area_q_m_pat_wt = area_q_all[(sex_all == 'm') * (ko_all == 'PAT') * (genotype_all == 'KLF14-KO:WT'), :]
area_q_f_mat_wt = area_q_all[(sex_all == 'f') * (ko_all == 'MAT') * (genotype_all == 'KLF14-KO:WT'), :]
area_q_m_mat_wt = area_q_all[(sex_all == 'm') * (ko_all == 'MAT') * (genotype_all == 'KLF14-KO:WT'), :]
area_interval_f_pat_wt = scipy.stats.mstats.hdquantiles(area_q_f_pat_wt, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
area_interval_m_pat_wt = scipy.stats.mstats.hdquantiles(area_q_m_pat_wt, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
area_interval_f_mat_wt = scipy.stats.mstats.hdquantiles(area_q_f_mat_wt, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
area_interval_m_mat_wt = scipy.stats.mstats.hdquantiles(area_q_m_mat_wt, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)

area_q_f_pat_het = area_q_all[(sex_all == 'f') * (ko_all == 'PAT') * (genotype_all == 'KLF14-KO:Het'), :]
area_q_m_pat_het = area_q_all[(sex_all == 'm') * (ko_all == 'PAT') * (genotype_all == 'KLF14-KO:Het'), :]
area_q_f_mat_het = area_q_all[(sex_all == 'f') * (ko_all == 'MAT') * (genotype_all == 'KLF14-KO:Het'), :]
area_q_m_mat_het = area_q_all[(sex_all == 'm') * (ko_all == 'MAT') * (genotype_all == 'KLF14-KO:Het'), :]
area_interval_f_pat_het = scipy.stats.mstats.hdquantiles(area_q_f_pat_het, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
area_interval_m_pat_het = scipy.stats.mstats.hdquantiles(area_q_m_pat_het, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
area_interval_f_mat_het = scipy.stats.mstats.hdquantiles(area_q_f_mat_het, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
area_interval_m_mat_het = scipy.stats.mstats.hdquantiles(area_q_m_mat_het, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)

n_f_pat_wt = area_q_f_pat_wt.shape[0]
n_m_pat_wt = area_q_m_pat_wt.shape[0]
n_f_mat_wt = area_q_f_mat_wt.shape[0]
n_m_mat_wt = area_q_m_mat_wt.shape[0]
n_f_pat_het = area_q_f_pat_het.shape[0]
n_m_pat_het = area_q_m_pat_het.shape[0]
n_f_mat_het = area_q_f_mat_het.shape[0]
n_m_mat_het = area_q_m_mat_het.shape[0]

if DEBUG:
    plt.clf()

    plt.subplot(121)
    plt.plot(quantiles * 100, area_interval_f_pat_wt[2, :] * 1e12 * 1e-3, 'C0', linewidth=3, label=str(n_f_pat_wt) + ' Female PAT WT')
    plt.fill_between(quantiles * 100, area_interval_f_pat_wt[1, :] * 1e12 * 1e-3, area_interval_f_pat_wt[3, :] * 1e12 * 1e-3,
                     facecolor='C0', alpha=0.3)
    # plt.plot(quantiles, area_interval_f_pat_wt[0, :] * 1e12 * 1e-3, 'C0:', linewidth=2, label='Female PAT WT 95%-interval')
    # plt.plot(quantiles, area_interval_f_pat_wt[4, :] * 1e12 * 1e-3, 'C0:', linewidth=2)

    plt.plot(quantiles * 100, area_interval_f_pat_het[2, :] * 1e12 * 1e-3, 'C1', linewidth=3, label=str(n_f_pat_het) + ' Female PAT Het')
    plt.fill_between(quantiles * 100, area_interval_f_pat_het[1, :] * 1e12 * 1e-3, area_interval_f_pat_het[3, :] * 1e12 * 1e-3,
                     facecolor='C1', alpha=0.3)

    # plt.title('Inguinal subcutaneous', fontsize=16)
    plt.xlabel('Cell population quantile (%)', fontsize=14)
    plt.ylabel('Area ($\cdot 10^3 \mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='upper left', prop={'size': 12})
    plt.ylim(0, 15)
    plt.tight_layout()

    plt.subplot(122)
    plt.plot(quantiles * 100, area_interval_f_mat_wt[2, :] * 1e12 * 1e-3, 'C2', linewidth=3, label=str(n_f_mat_wt) + ' Female MAT WT')
    plt.fill_between(quantiles * 100, area_interval_f_mat_wt[1, :] * 1e12 * 1e-3, area_interval_f_mat_wt[3, :] * 1e12 * 1e-3,
                     facecolor='C2', alpha=0.3)

    plt.plot(quantiles * 100, area_interval_f_mat_het[2, :] * 1e12 * 1e-3, 'C3', linewidth=3, label=str(n_f_mat_het) + ' Female MAT Het')
    plt.fill_between(quantiles * 100, area_interval_f_mat_het[1, :] * 1e12 * 1e-3, area_interval_f_mat_het[3, :] * 1e12 * 1e-3,
                     facecolor='C3', alpha=0.3)

    # plt.title('Inguinal subcutaneous', fontsize=16)
    plt.xlabel('Cell population quantile (%)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='upper left', prop={'size': 12})
    plt.ylim(0, 15)
    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'exp_0099_' + depot + '_cell_area_female_pat_vs_mat_bands.svg'))
    plt.savefig(os.path.join(figures_dir, 'exp_0099_' + depot + '_cell_area_female_pat_vs_mat_bands.png'))

if DEBUG:
    plt.clf()

    plt.subplot(121)
    plt.plot(quantiles * 100, area_interval_m_pat_wt[2, :] * 1e12 * 1e-3, 'C0', linewidth=3, label=str(n_m_pat_wt) + ' Male PAT WT')
    plt.fill_between(quantiles * 100, area_interval_m_pat_wt[1, :] * 1e12 * 1e-3, area_interval_m_pat_wt[3, :] * 1e12 * 1e-3,
                     facecolor='C0', alpha=0.3)
    # plt.plot(quantiles, area_interval_f_pat_wt[0, :] * 1e12 * 1e-3, 'C0:', linewidth=2, label='Female PAT WT 95%-interval')
    # plt.plot(quantiles, area_interval_f_pat_wt[4, :] * 1e12 * 1e-3, 'C0:', linewidth=2)

    plt.plot(quantiles * 100, area_interval_m_pat_het[2, :] * 1e12 * 1e-3, 'C1', linewidth=3, label=str(n_m_pat_het) + ' Male PAT Het')
    plt.fill_between(quantiles * 100, area_interval_m_pat_het[1, :] * 1e12 * 1e-3, area_interval_m_pat_het[3, :] * 1e12 * 1e-3,
                     facecolor='C1', alpha=0.3)

    # plt.title('Inguinal subcutaneous', fontsize=16)
    plt.xlabel('Cell population quantile (%)', fontsize=14)
    plt.ylabel('Area ($\cdot 10^3 \mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='upper left', prop={'size': 12})
    plt.ylim(0, 16)
    plt.tight_layout()

    plt.subplot(122)
    plt.plot(quantiles * 100, area_interval_m_mat_wt[2, :] * 1e12 * 1e-3, 'C2', linewidth=3, label=str(n_m_mat_wt) + ' Male MAT WT')
    plt.fill_between(quantiles * 100, area_interval_m_mat_wt[1, :] * 1e12 * 1e-3, area_interval_m_mat_wt[3, :] * 1e12 * 1e-3,
                     facecolor='C2', alpha=0.3)

    plt.plot(quantiles * 100, area_interval_m_mat_het[2, :] * 1e12 * 1e-3, 'C3', linewidth=3, label=str(n_m_mat_het) + ' Male MAT Het')
    plt.fill_between(quantiles * 100, area_interval_m_mat_het[1, :] * 1e12 * 1e-3, area_interval_m_mat_het[3, :] * 1e12 * 1e-3,
                     facecolor='C3', alpha=0.3)

    # plt.title('Inguinal subcutaneous', fontsize=16)
    plt.xlabel('Cell population quantile (%)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='upper left', prop={'size': 12})
    plt.ylim(0, 16)
    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'exp_0099_' + depot + '_cell_area_male_pat_vs_mat_bands.svg'))
    plt.savefig(os.path.join(figures_dir, 'exp_0099_' + depot + '_cell_area_male_pat_vs_mat_bands.png'))

filename_pvals = os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_pvals_' + depot + '.npz')
if os.path.isfile(filename_pvals):

    aux = np.load(filename_pvals)
    pval_perc_f_pat2mat = aux['pval_perc_f_pat2mat']
    pval_perc_m_pat2mat = aux['pval_perc_m_pat2mat']
    pval_perc_f_pat_wt2het = aux['pval_perc_f_pat_wt2het']
    pval_perc_f_mat_wt2het = aux['pval_perc_f_mat_wt2het']
    pval_perc_m_pat_wt2het = aux['pval_perc_m_pat_wt2het']
    pval_perc_m_mat_wt2het = aux['pval_perc_m_mat_wt2het']
    permutation_sample_size = aux['permutation_sample_size']

else:

    # test whether the median values are different enough between two groups
    func = lambda x, y: np.abs(scipy.stats.mstats.hdquantiles(x, prob=0.5, axis=0).data[0]
                               - scipy.stats.mstats.hdquantiles(y, prob=0.5, axis=0).data[0])
    # func = lambda x, y: np.abs(np.mean(x) - np.mean(y))

    ## PAT vs. MAT

    # test whether the median values are different enough between PAT vs. MAT
    pval_perc_f_pat2mat = np.zeros(shape=(len(quantiles),))
    for i, q in enumerate(quantiles):
        pval_perc_f_pat2mat[i] = permutation_test(x=area_q_f_pat[:, i], y=area_q_f_mat[:, i],
                                                  func=func, seed=None,
                                                  method='approximate', num_rounds=math.factorial(permutation_sample_size))

    pval_perc_m_pat2mat = np.zeros(shape=(len(quantiles),))
    for i, q in enumerate(quantiles):
        pval_perc_m_pat2mat[i] = permutation_test(x=area_q_m_pat[:, i], y=area_q_m_mat[:, i],
                                                  func=func, seed=None,
                                                  method='approximate', num_rounds=math.factorial(permutation_sample_size))

    ## WT vs. Het

    # PAT Females
    pval_perc_f_pat_wt2het = np.zeros(shape=(len(quantiles),))
    for i, q in enumerate(quantiles):
        pval_perc_f_pat_wt2het[i] = permutation_test(x=area_q_f_pat_wt[:, i], y=area_q_f_pat_het[:, i],
                                                     func=func, seed=None,
                                                     method='approximate',
                                                     num_rounds=math.factorial(permutation_sample_size))

    # MAT Females
    pval_perc_f_mat_wt2het = np.zeros(shape=(len(quantiles),))
    for i, q in enumerate(quantiles):
        pval_perc_f_mat_wt2het[i] = permutation_test(x=area_q_f_mat_wt[:, i], y=area_q_f_mat_het[:, i],
                                                     func=func, seed=None,
                                                     method='approximate',
                                                     num_rounds=math.factorial(permutation_sample_size))

    # PAT Males
    pval_perc_m_pat_wt2het = np.zeros(shape=(len(quantiles),))
    for i, q in enumerate(quantiles):
        pval_perc_m_pat_wt2het[i] = permutation_test(x=area_q_m_pat_wt[:, i], y=area_q_m_pat_het[:, i],
                                                     func=func, seed=None,
                                                     method='approximate',
                                                     num_rounds=math.factorial(permutation_sample_size))

    # MAT Males
    pval_perc_m_mat_wt2het = np.zeros(shape=(len(quantiles),))
    for i, q in enumerate(quantiles):
        pval_perc_m_mat_wt2het[i] = permutation_test(x=area_q_m_mat_wt[:, i], y=area_q_m_mat_het[:, i],
                                                     func=func, seed=None,
                                                     method='approximate',
                                                     num_rounds=math.factorial(permutation_sample_size))

    np.savez_compressed(filename_pvals, permutation_sample_size=permutation_sample_size,
                        pval_perc_f_pat2mat=pval_perc_f_pat2mat, pval_perc_m_pat2mat=pval_perc_m_pat2mat,
                        pval_perc_f_pat_wt2het=pval_perc_f_pat_wt2het, pval_perc_f_mat_wt2het=pval_perc_f_mat_wt2het,
                        pval_perc_m_pat_wt2het=pval_perc_m_pat_wt2het, pval_perc_m_mat_wt2het=pval_perc_m_mat_wt2het)


# data has been loaded or computed

np.set_printoptions(precision=2)
print('PAT vs. MAT before multitest correction')
print('Female:')
print(pval_perc_f_pat2mat)
print('Male:')
print(pval_perc_m_pat2mat)
np.set_printoptions(precision=8)

# multitest correction using Hochberg a.k.a. Simes-Hochberg method
_, pval_perc_f_pat2mat, _, _ = multipletests(pval_perc_f_pat2mat, method='simes-hochberg', alpha=0.05, returnsorted=False)
_, pval_perc_m_pat2mat, _, _ = multipletests(pval_perc_m_pat2mat, method='simes-hochberg', alpha=0.05, returnsorted=False)

np.set_printoptions(precision=2)
print('PAT vs. MAT with multitest correction')
print('Female:')
print(pval_perc_f_pat2mat)
print('Male:')
print(pval_perc_m_pat2mat)
np.set_printoptions(precision=8)

np.set_printoptions(precision=2)
print('WT vs. Het before multitest correction')
print('Female:')
print(pval_perc_f_pat_wt2het)
print(pval_perc_f_mat_wt2het)
print('Male:')
print(pval_perc_m_pat_wt2het)
print(pval_perc_m_mat_wt2het)
np.set_printoptions(precision=8)

# multitest correction using Hochberg a.k.a. Simes-Hochberg method
_, pval_perc_f_pat_wt2het, _, _ = multipletests(pval_perc_f_pat_wt2het, method='simes-hochberg', alpha=0.05, returnsorted=False)
_, pval_perc_f_mat_wt2het, _, _ = multipletests(pval_perc_f_mat_wt2het, method='simes-hochberg', alpha=0.05, returnsorted=False)
_, pval_perc_m_pat_wt2het, _, _ = multipletests(pval_perc_m_pat_wt2het, method='simes-hochberg', alpha=0.05, returnsorted=False)
_, pval_perc_m_mat_wt2het, _, _ = multipletests(pval_perc_m_mat_wt2het, method='simes-hochberg', alpha=0.05, returnsorted=False)

np.set_printoptions(precision=2)
print('WT vs. Het with multitest correction')
print('Female:')
print(pval_perc_f_pat_wt2het)
print(pval_perc_f_mat_wt2het)
print('Male:')
print(pval_perc_m_pat_wt2het)
print(pval_perc_m_mat_wt2het)
np.set_printoptions(precision=8)

# # plot the median difference and the population quantiles at which the difference is significant
# if DEBUG:
#     plt.clf()
#     idx = pval_perc_f_pat2mat < 0.05
#     delta_a_f_pat2mat = (area_interval_f_mat[1, :] - area_interval_f_pat[1, :]) / area_interval_f_pat[1, :]
#     if np.any(idx):
#         plt.stem(quantiles[idx], 100 * delta_a_f_pat2mat[idx],
#                  markerfmt='.', linefmt='C6-', basefmt='C6',
#                  label='p-val$_{\mathrm{PAT}}$ < 0.05')
#
#     idx = pval_perc_m_pat2mat < 0.05
#     delta_a_m_pat2mat = (area_interval_m_mat[1, :] - area_interval_m_pat[1, :]) / area_interval_m_pat[1, :]
#     if np.any(idx):
#         plt.stem(quantiles[idx], 100 * delta_a_m_pat2mat[idx],
#                  markerfmt='.', linefmt='C7-', basefmt='C7', bottom=250,
#                  label='p-val$_{\mathrm{MAT}}$ < 0.05')
#
#     plt.plot(quantiles, 100 * delta_a_f_pat2mat, 'C6', linewidth=3, label='Female PAT to MAT')
#     plt.plot(quantiles, 100 * delta_a_m_pat2mat, 'C7', linewidth=3, label='Male PAT to MAT')
#
#     plt.xlabel('Cell population quantile', fontsize=14)
#     plt.ylabel('Area change (%)', fontsize=14)
#     plt.tick_params(axis='both', which='major', labelsize=14)
#     plt.legend(loc='lower right', prop={'size': 12})
#     plt.tight_layout()
#
#     plt.savefig(os.path.join(figures_dir, 'exp_0099_' + depot + '_cell_area_change_pat_2_mat.svg'))
#     plt.savefig(os.path.join(figures_dir, 'exp_0099_' + depot + '_cell_area_change_pat_2_mat.png'))

########################################################################################################################
## Linear models of body weight (BW), fat depots weight (SC and gWAT), and categorical variables (sex, ko, genotype)
########################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import pandas as pd
import statsmodels.api as sm

# directories
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
annotations_dir = os.path.join(home, 'Software/AIDA/dist/data/annotations')
ndpi_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
metainfo_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')

DEBUG = False

quantiles = np.linspace(0, 1, 11)
quantiles = quantiles[1:-1]

# load metainfo file
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# make sure that in the boxplots PAT comes before MAT
metainfo['sex'] = metainfo['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
metainfo['ko_parent'] = metainfo['ko_parent'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
metainfo['genotype'] = metainfo['genotype'].astype(pd.api.types.CategoricalDtype(categories=['KLF14-KO:WT', 'KLF14-KO:Het'], ordered=True))

# subgroups
idx_f_pat_wt = (metainfo.sex == 'f') * (metainfo.ko_parent == 'PAT') * (metainfo.genotype == 'KLF14-KO:WT') * ~np.isnan(metainfo.BW)
idx_f_pat_het = (metainfo.sex == 'f') * (metainfo.ko_parent == 'PAT') * (metainfo.genotype == 'KLF14-KO:Het') * ~np.isnan(metainfo.BW)
idx_f_mat_wt = (metainfo.sex == 'f') * (metainfo.ko_parent == 'MAT') * (metainfo.genotype == 'KLF14-KO:WT') * ~np.isnan(metainfo.BW)
idx_f_mat_het = (metainfo.sex == 'f') * (metainfo.ko_parent == 'MAT') * (metainfo.genotype == 'KLF14-KO:Het') * ~np.isnan(metainfo.BW)
idx_m_pat_wt = (metainfo.sex == 'm') * (metainfo.ko_parent == 'PAT') * (metainfo.genotype == 'KLF14-KO:WT') * ~np.isnan(metainfo.BW)
idx_m_pat_het = (metainfo.sex == 'm') * (metainfo.ko_parent == 'PAT') * (metainfo.genotype == 'KLF14-KO:Het') * ~np.isnan(metainfo.BW)
idx_m_mat_wt = (metainfo.sex == 'm') * (metainfo.ko_parent == 'MAT') * (metainfo.genotype == 'KLF14-KO:WT') * ~np.isnan(metainfo.BW)
idx_m_mat_het = (metainfo.sex == 'm') * (metainfo.ko_parent == 'MAT') * (metainfo.genotype == 'KLF14-KO:Het') * ~np.isnan(metainfo.BW)

# body weight
bw_f_pat_wt = metainfo.BW[idx_f_pat_wt]
bw_f_pat_het = metainfo.BW[idx_f_pat_het]
bw_f_mat_wt = metainfo.BW[idx_f_mat_wt]
bw_f_mat_het = metainfo.BW[idx_f_mat_het]
bw_m_pat_wt = metainfo.BW[idx_m_pat_wt]
bw_m_pat_het = metainfo.BW[idx_m_pat_het]
bw_m_mat_wt = metainfo.BW[idx_m_mat_wt]
bw_m_mat_het = metainfo.BW[idx_m_mat_het]

# SQWAT depot weight
sq_f_pat_wt = metainfo.SC[idx_f_pat_wt]
sq_f_pat_het = metainfo.SC[idx_f_pat_het]
sq_f_mat_wt = metainfo.SC[idx_f_mat_wt]
sq_f_mat_het = metainfo.SC[idx_f_mat_het]
sq_m_pat_wt = metainfo.SC[idx_m_pat_wt]
sq_m_pat_het = metainfo.SC[idx_m_pat_het]
sq_m_mat_wt = metainfo.SC[idx_m_mat_wt]
sq_m_mat_het = metainfo.SC[idx_m_mat_het]

# GWAT depot weight
g_f_pat_wt = metainfo.gWAT[idx_f_pat_wt]
g_f_pat_het = metainfo.gWAT[idx_f_pat_het]
g_f_mat_wt = metainfo.gWAT[idx_f_mat_wt]
g_f_mat_het = metainfo.gWAT[idx_f_mat_het]
g_m_pat_wt = metainfo.gWAT[idx_m_pat_wt]
g_m_pat_het = metainfo.gWAT[idx_m_pat_het]
g_m_mat_wt = metainfo.gWAT[idx_m_mat_wt]
g_m_mat_het = metainfo.gWAT[idx_m_mat_het]

if DEBUG:
    plt.clf()
    plt.subplot(131)
    plt.boxplot(
        (bw_f_pat_wt, bw_f_pat_het, bw_f_mat_wt, bw_f_mat_het, bw_m_pat_wt, bw_m_pat_het, bw_m_mat_wt, bw_m_mat_het),
        labels=('f_PAT_WT', 'f_PAT_Het', 'f_MAT_WT', 'f_MAT_Het', 'm_PAT_WT', 'm_PAT_Het', 'm_MAT_WT', 'm_MAT_Het'),
        notch=False
    )
    plt.xticks(rotation=45)
    plt.title('Body')
    plt.ylabel('Weight (g)', fontsize=14)
    plt.subplot(132)
    plt.boxplot(
        (sq_f_pat_wt, sq_f_pat_het, sq_f_mat_wt, sq_f_mat_het, sq_m_pat_wt, sq_m_pat_het, sq_m_mat_wt, sq_m_mat_het),
        labels=('f_PAT_WT', 'f_PAT_Het', 'f_MAT_WT', 'f_MAT_Het', 'm_PAT_WT', 'm_PAT_Het', 'm_MAT_WT', 'm_MAT_Het'),
        notch=False
    )
    plt.xticks(rotation=45)
    plt.title('SQWAT')
    plt.subplot(133)
    plt.boxplot(
        (g_f_pat_wt, g_f_pat_het, g_f_mat_wt, g_f_mat_het, g_m_pat_wt, g_m_pat_het, g_m_mat_wt, g_m_mat_het),
        labels=('f_PAT_WT', 'f_PAT_Het', 'f_MAT_WT', 'f_MAT_Het', 'm_PAT_WT', 'm_PAT_Het', 'm_MAT_WT', 'm_MAT_Het'),
        notch=False
    )
    plt.xticks(rotation=45)
    plt.title('GWAT')
    plt.tight_layout()

if DEBUG:
    plt.clf()
    plt.scatter(np.concatenate((bw_f_pat_wt, bw_m_pat_wt)), np.concatenate((sq_f_pat_wt, sq_m_pat_wt)))
    plt.scatter(np.concatenate((bw_f_mat_wt, bw_m_mat_wt)), np.concatenate((sq_f_mat_wt, sq_m_mat_wt)))
    plt.tight_layout()

# Model BW ~ C(sex) + C(ko_parent) + C(genotype) + SC * gWAT
idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]
model = sm.formula.ols('BW ~ C(sex) + C(ko_parent) + C(genotype) + SC * gWAT', data=metainfo, subset=idx_not_nan).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     BW   R-squared:                       0.740
# Model:                            OLS   Adj. R-squared:                  0.718
# Method:                 Least Squares   F-statistic:                     32.76
# Date:                Wed, 19 Feb 2020   Prob (F-statistic):           2.32e-18
# Time:                        11:22:19   Log-Likelihood:                -210.84
# No. Observations:                  76   AIC:                             435.7
# Df Residuals:                      69   BIC:                             452.0
# Df Model:                           6
# Covariance Type:            nonrobust
# ===============================================================================================
#                                   coef    std err          t      P>|t|      [0.025      0.975]
# -----------------------------------------------------------------------------------------------
# Intercept                      19.0376      2.217      8.587      0.000      14.615      23.460
# C(sex)[T.m]                    10.3729      1.017     10.198      0.000       8.344      12.402
# C(ko_parent)[T.MAT]                    2.6057      1.023      2.548      0.013       0.565       4.646
# C(genotype)[T.KLF14-KO:Het]    -1.1439      0.941     -1.216      0.228      -3.021       0.733
# SC                              6.2866      4.062      1.548      0.126      -1.817      14.391
# gWAT                            9.4847      2.434      3.897      0.000       4.630      14.340
# SC:gWAT                        -6.8830      3.465     -1.986      0.051     -13.796       0.030
# ==============================================================================
# Omnibus:                        2.321   Durbin-Watson:                   1.296
# Prob(Omnibus):                  0.313   Jarque-Bera (JB):                1.921
# Skew:                           0.389   Prob(JB):                        0.383
# Kurtosis:                       3.047   Cond. No.                         23.5
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

# list of point with high influence (large residuals and leverage)
idx_influence = [65, 52, 64, 32, 72, 75, 0]
idx_no_influence = list(set(range(metainfo.shape[0])) - set(idx_influence))
print(metainfo.loc[idx_influence, ['id', 'ko_parent', 'sex', 'genotype', 'BW', 'SC', 'gWAT']])

#        id   ko sex      genotype     BW    SC  gWAT
# 65  37.2e  PAT   f  KLF14-KO:Het  21.18  1.62  0.72
# 52  36.3d  PAT   m  KLF14-KO:Het  40.77  1.38  1.78
# 64  37.2d  PAT   f   KLF14-KO:WT  20.02  0.72  0.12
# 32  18.3c  MAT   m   KLF14-KO:WT  37.83  1.24  1.68
# 72  37.4b  PAT   m   KLF14-KO:WT  50.54  0.87  1.11
# 75  38.1f  PAT   m   KLF14-KO:WT  38.98  0.49  0.98
# 0   16.2a  MAT   f   KLF14-KO:WT  40.80  0.38  1.10

# linear model removing the high influence points
# Ordinary least squares linear model
model = sm.formula.ols('BW ~ C(sex) + C(ko_parent) + C(genotype) + SC * gWAT', data=metainfo, subset=idx_no_influence).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     BW   R-squared:                       0.797
# Model:                            OLS   Adj. R-squared:                  0.777
# Method:                 Least Squares   F-statistic:                     40.51
# Date:                Fri, 14 Feb 2020   Prob (F-statistic):           1.20e-19
# Time:                        17:07:13   Log-Likelihood:                -179.50
# No. Observations:                  69   AIC:                             373.0
# Df Residuals:                      62   BIC:                             388.6
# Df Model:                           6
# Covariance Type:            nonrobust
# ==============================================================================================
#                                  coef    std err          t      P>|t|      [0.025      0.975]
# ----------------------------------------------------------------------------------------------
# Intercept                     17.6196      2.188      8.051      0.000      13.245      21.994
# C(sex)[T.m]                    9.2063      0.946      9.728      0.000       7.315      11.098
# C(ko_parent)[T.PAT]                  -3.5029      0.911     -3.845      0.000      -5.324      -1.682
# C(genotype)[T.KLF14-KO:WT]     0.7111      0.833      0.853      0.397      -0.955       2.377
# SC                            23.8520      6.429      3.710      0.000      11.000      36.704
# gWAT                          11.6718      2.399      4.865      0.000       6.875      16.468
# SC:gWAT                      -20.0493      5.320     -3.769      0.000     -30.683      -9.415
# ==============================================================================
# Omnibus:                        2.530   Durbin-Watson:                   1.505
# Prob(Omnibus):                  0.282   Jarque-Bera (JB):                1.543
# Skew:                           0.041   Prob(JB):                        0.462
# Kurtosis:                       2.272   Cond. No.                         38.3
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# further refinement of the model, to decrease AIC and BIC, and improve R^2

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

# list of point with high influence (large residuals and leverage)
idx_influence = [65, 52, 64, 32, 72, 75, 0]

# list of data points to use in the model
idx_for_model = (set(range(metainfo.shape[0])) - set(idx_influence)) & set(idx_not_nan)
idx_for_model = list(idx_for_model)
model = sm.formula.ols('BW ~ C(sex) + C(ko_parent) + C(genotype) : (SC * gWAT)', data=metainfo, subset=idx_for_model).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     BW   R-squared:                       0.798
# Model:                            OLS   Adj. R-squared:                  0.771
# Method:                 Least Squares   F-statistic:                     29.69
# Date:                Wed, 19 Feb 2020   Prob (F-statistic):           3.92e-18
# Time:                        11:26:41   Log-Likelihood:                -179.24
# No. Observations:                  69   AIC:                             376.5
# Df Residuals:                      60   BIC:                             396.6
# Df Model:                           8
# Covariance Type:            nonrobust
# =====================================================================================================
#                                         coef    std err          t      P>|t|      [0.025      0.975]
# -----------------------------------------------------------------------------------------------------
# Intercept                            14.3003      2.322      6.159      0.000       9.656      18.945
# C(sex)[T.m]                           9.2769      0.969      9.577      0.000       7.339      11.215
# C(ko_parent)[T.MAT]                          3.4809      0.947      3.677      0.001       1.588       5.374
# C(genotype)[KLF14-KO:WT]:SC          24.5670      7.179      3.422      0.001      10.206      38.928
# C(genotype)[KLF14-KO:Het]:SC         24.3363      7.999      3.042      0.003       8.336      40.337
# C(genotype)[KLF14-KO:WT]:gWAT        11.7082      2.688      4.355      0.000       6.331      17.086
# C(genotype)[KLF14-KO:Het]:gWAT       11.8047      2.592      4.554      0.000       6.620      16.990
# C(genotype)[KLF14-KO:WT]:SC:gWAT    -19.8530      5.828     -3.407      0.001     -31.510      -8.196
# C(genotype)[KLF14-KO:Het]:SC:gWAT   -21.0835      6.575     -3.207      0.002     -34.236      -7.931
# ==============================================================================
# Omnibus:                        1.026   Durbin-Watson:                   1.515
# Prob(Omnibus):                  0.599   Jarque-Bera (JB):                0.925
# Skew:                          -0.009   Prob(JB):                        0.630
# Kurtosis:                       2.433   Cond. No.                         45.9
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


# range of weight values
print(np.min(metainfo.BW))
print(np.max(metainfo.BW))

print(np.min(metainfo.SC))
print(np.max(metainfo.SC))

print(np.min(metainfo.gWAT))
print(np.max(metainfo.gWAT))

########################################################################################################################
## Load SQWAT and gWAT quantile data computed in a previous section
########################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import pandas as pd
import statsmodels.api as sm

# directories
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
annotations_dir = os.path.join(home, 'Software/AIDA/dist/data/annotations')
ndpi_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
metainfo_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')

DEBUG = False

quantiles = np.linspace(0, 1, 11)
quantiles = quantiles[1:-1]

# load metainfo file
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# make sure that in the boxplots PAT comes before MAT
metainfo['sex'] = metainfo['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
metainfo['ko_parent'] = metainfo['ko_parent'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
metainfo['genotype'] = metainfo['genotype'].astype(pd.api.types.CategoricalDtype(categories=['KLF14-KO:WT', 'KLF14-KO:Het'], ordered=True))

# load SQWAT data
depot = 'sqwat'
filename_quantiles = os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_area_quantiles_' + depot + '.npz')

aux = np.load(filename_quantiles)
area_mean_sqwat = aux['area_mean_all']
area_q_sqwat = aux['area_q_all']
id_sqwat = aux['id_all']
ko_sqwat = aux['ko_all']
genotype_sqwat = aux['genotype_all']
sex_sqwat = aux['sex_all']

# load gWAT data
depot = 'gwat'
filename_quantiles = os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_area_quantiles_' + depot + '.npz')

aux = np.load(filename_quantiles)
area_mean_gwat = aux['area_mean_all']
area_q_gwat = aux['area_q_all']
id_gwat = aux['id_all']
ko_gwat = aux['ko_all']
genotype_gwat = aux['genotype_all']
sex_gwat = aux['sex_all']

# volume sphere with same radius as given circle
def vol_sphere(area_circle):
    return (4 / 3 / np.sqrt(np.pi)) * np.power(area_circle, 3/2)

# add a new column to the metainfo frame with the mean cell volume for SQWAT
metainfo_idx = [np.where(metainfo['id'] == x)[0][0] for x in id_sqwat]
metainfo['SC_vol_mean'] = np.NaN
metainfo.loc[metainfo_idx, 'SC_vol_mean'] = vol_sphere(area_mean_sqwat)  # m^3
metainfo['SC_vol_mean_1e12'] = metainfo['SC_vol_mean'] * 1e12

# add a new column to the metainfo frame with the mean cell volume for gWAT
metainfo_idx = [np.where(metainfo['id'] == x)[0][0] for x in id_gwat]
metainfo['gWAT_vol_mean'] = np.NaN
metainfo.loc[metainfo_idx, 'gWAT_vol_mean'] = vol_sphere(area_mean_gwat)  # m^3
metainfo['gWAT_vol_mean_1e12'] = metainfo['gWAT_vol_mean'] * 1e12

# add a new column to the metainfo frame with the median cell volume for SQWAT
metainfo_idx = [np.where(metainfo['id'] == x)[0][0] for x in id_sqwat]
metainfo['SC_vol_median'] = np.NaN
metainfo.loc[metainfo_idx, 'SC_vol_median'] = vol_sphere(area_q_sqwat[:, 4])  # m^3
metainfo['SC_vol_median_1e12'] = metainfo['SC_vol_median'] * 1e12

# add a new column to the metainfo frame with the median cell volume for gWAT
metainfo_idx = [np.where(metainfo['id'] == x)[0][0] for x in id_gwat]
metainfo['gWAT_vol_median'] = np.NaN
metainfo.loc[metainfo_idx, 'gWAT_vol_median'] = vol_sphere(area_q_gwat[:, 4])  # m^3
metainfo['gWAT_vol_median_1e12'] = metainfo['gWAT_vol_median'] * 1e12

# add new column with fat depot weights normalised by body weight
metainfo['SC_BW'] = metainfo['SC'] / metainfo['BW']
metainfo['gWAT_BW'] = metainfo['gWAT'] / metainfo['BW']

# add new column with estimates number of cells
fat_density_SC = 0.9038  # g / cm^3
fat_density_gWAT = 0.9029  # g / cm^3
metainfo['SC_rho_N'] = metainfo['SC'] / (fat_density_SC * 1e6 * metainfo['SC_vol_mean'])
metainfo['gWAT_rho_N'] = metainfo['gWAT'] / (fat_density_gWAT * 1e6 * metainfo['gWAT_vol_mean'])

if DEBUG:
    plt.clf()
    plt.scatter(metainfo['gWAT_vol_mean'], metainfo['gWAT_vol_median'])
    plt.clf()
    plt.scatter(metainfo['SC_vol_mean'], metainfo['SC_vol_median'])

if DEBUG:
    # compare SQWAT to gWAT
    plt.clf()
    plt.scatter(metainfo['SC_vol_median'], metainfo['gWAT_vol_median'])
    plt.scatter(metainfo['SC_vol_mean'], metainfo['gWAT_vol_median'])

    # BW vs. SQWAT
    plt.clf()
    plt.scatter(metainfo['SC_vol_median'], metainfo['BW'])
    plt.scatter(metainfo['SC_vol_mean'], metainfo['BW'])

    # BW vs. gWAT
    plt.clf()
    plt.scatter(metainfo['gWAT_vol_median'], metainfo['BW'])
    plt.scatter(metainfo['gWAT_vol_mean'], metainfo['BW'])

if DEBUG:
    print(np.min(metainfo['SC_vol_mean']) * 1e12)  # cm^3
    print(np.max(metainfo['SC_vol_mean']) * 1e12)  # cm^3
    print(np.min(metainfo['gWAT_vol_mean']) * 1e12)  # cm^3
    print(np.max(metainfo['gWAT_vol_mean']) * 1e12)  # cm^3

    print(np.min(metainfo['SC_rho_N']))
    print(np.max(metainfo['SC_rho_N']))
    print(np.min(metainfo['gWAT_rho_N']))
    print(np.max(metainfo['gWAT_rho_N']))

########################################################################################################################
### Model BW ~ SC + gWAT
########################################################################################################################

model = sm.formula.ols('BW ~ SC + gWAT', data=metainfo).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     BW   R-squared:                       0.283
# Model:                            OLS   Adj. R-squared:                  0.263
# Method:                 Least Squares   F-statistic:                     14.37
# Date:                Wed, 19 Feb 2020   Prob (F-statistic):           5.46e-06
# Time:                        16:23:02   Log-Likelihood:                -249.44
# No. Observations:                  76   AIC:                             504.9
# Df Residuals:                      73   BIC:                             511.9
# Df Model:                           2
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     22.8628      2.127     10.749      0.000      18.624      27.102
# SC            -0.8182      2.520     -0.325      0.746      -5.841       4.204
# gWAT          11.5932      2.478      4.679      0.000       6.655      16.531
# ==============================================================================
# Omnibus:                        1.243   Durbin-Watson:                   1.056
# Prob(Omnibus):                  0.537   Jarque-Bera (JB):                1.109
# Skew:                           0.292   Prob(JB):                        0.574
# Kurtosis:                       2.898   Cond. No.                         6.66
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

# list of point with high influence (large residuals and leverage)
idx_influence = [65, 64, 4, 63, 35, 72, 5, 45, 62]
idx_no_influence = list(set(range(metainfo.shape[0])) - set(idx_influence))
print(metainfo.loc[idx_influence, ['id', 'ko_parent', 'sex', 'genotype', 'BW', 'SC', 'gWAT']])

# Refine ordinary least squares linear model by removing influence points
model = sm.formula.ols('BW ~ SC + gWAT',
                       data=metainfo, subset=idx_no_influence).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     BW   R-squared:                       0.453
# Model:                            OLS   Adj. R-squared:                  0.436
# Method:                 Least Squares   F-statistic:                     26.53
# Date:                Wed, 19 Feb 2020   Prob (F-statistic):           4.06e-09
# Time:                        16:24:40   Log-Likelihood:                -203.63
# No. Observations:                  67   AIC:                             413.3
# Df Residuals:                      64   BIC:                             419.9
# Df Model:                           2
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     21.1608      1.813     11.670      0.000      17.538      24.783
# SC            -2.8056      2.503     -1.121      0.267      -7.806       2.195
# gWAT          14.4122      2.376      6.066      0.000       9.665      19.159
# ==============================================================================
# Omnibus:                        1.761   Durbin-Watson:                   1.521
# Prob(Omnibus):                  0.415   Jarque-Bera (JB):                1.705
# Skew:                           0.305   Prob(JB):                        0.426
# Kurtosis:                       2.512   Cond. No.                         7.98
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

########################################################################################################################
### Model BW ~ SC + gWAT * sex
#
# Fitting the (SC + gWAT) * sex model shows no significant interaction between sex and SC
########################################################################################################################

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

model = sm.formula.ols('BW ~ SC + gWAT * C(sex)', data=metainfo, subset=idx_not_nan).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     BW   R-squared:                       0.717
# Model:                            OLS   Adj. R-squared:                  0.702
# Method:                 Least Squares   F-statistic:                     45.08
# Date:                Wed, 19 Feb 2020   Prob (F-statistic):           8.60e-19
# Time:                        17:20:47   Log-Likelihood:                -214.02
# No. Observations:                  76   AIC:                             438.0
# Df Residuals:                      71   BIC:                             449.7
# Df Model:                           4
# Covariance Type:            nonrobust
# ====================================================================================
#                        coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------------
# Intercept           20.8264      1.621     12.844      0.000      17.593      24.060
# C(sex)[T.m]         17.1459      3.107      5.519      0.000      10.951      23.341
# SC                  -2.5526      1.619     -1.577      0.119      -5.781       0.676
# gWAT                 9.5160      1.931      4.927      0.000       5.665      13.367
# gWAT:C(sex)[T.m]    -6.6193      2.972     -2.227      0.029     -12.546      -0.693
# ==============================================================================
# Omnibus:                        2.780   Durbin-Watson:                   1.257
# Prob(Omnibus):                  0.249   Jarque-Bera (JB):                2.029
# Skew:                           0.348   Prob(JB):                        0.363
# Kurtosis:                       3.395   Cond. No.                         16.4
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

# list of point with high influence (large residuals and leverage)
idx_influence = [49, 63, 35, 72, 0, 20]

# list of data points to use in the model
idx_for_model = (set(range(metainfo.shape[0])) - set(idx_influence)) & set(idx_not_nan)
idx_for_model = list(idx_for_model)

# Refine ordinary least squares linear model by removing influence points
model = sm.formula.ols('BW ~ SC + gWAT * C(sex)',
                       data=metainfo, subset=idx_for_model).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     BW   R-squared:                       0.813
# Model:                            OLS   Adj. R-squared:                  0.802
# Method:                 Least Squares   F-statistic:                     70.75
# Date:                Wed, 19 Feb 2020   Prob (F-statistic):           5.70e-23
# Time:                        17:22:59   Log-Likelihood:                -180.32
# No. Observations:                  70   AIC:                             370.6
# Df Residuals:                      65   BIC:                             381.9
# Df Model:                           4
# Covariance Type:            nonrobust
# ====================================================================================
#                        coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------------
# Intercept           20.3360      1.308     15.548      0.000      17.724      22.948
# C(sex)[T.m]         20.2885      2.725      7.445      0.000      14.846      25.731
# SC                  -2.1036      1.317     -1.597      0.115      -4.734       0.527
# gWAT                 9.8118      1.643      5.971      0.000       6.530      13.093
# gWAT:C(sex)[T.m]    -9.6857      2.599     -3.726      0.000     -14.876      -4.495
# ==============================================================================
# Omnibus:                        0.328   Durbin-Watson:                   1.060
# Prob(Omnibus):                  0.849   Jarque-Bera (JB):                0.504
# Skew:                          -0.034   Prob(JB):                        0.777
# Kurtosis:                       2.590   Cond. No.                         17.4
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

########################################################################################################################
### Model BW ~ (SC + gWAT * C(sex)) + (SC * C(ko_parent) + gWAT)
########################################################################################################################

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

model = sm.formula.ols('BW ~ (SC + gWAT * C(sex)) + (SC * C(ko_parent) + gWAT)', data=metainfo, subset=idx_not_nan).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     BW   R-squared:                       0.767
# Model:                            OLS   Adj. R-squared:                  0.747
# Method:                 Least Squares   F-statistic:                     37.83
# Date:                Thu, 20 Feb 2020   Prob (F-statistic):           5.90e-20
# Time:                        01:00:26   Log-Likelihood:                -206.72
# No. Observations:                  76   AIC:                             427.4
# Df Residuals:                      69   BIC:                             443.8
# Df Model:                           6
# Covariance Type:            nonrobust
# ====================================================================================
#                        coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------------
# Intercept           17.3729      1.760      9.872      0.000      13.862      20.884
# C(sex)[T.m]         17.5361      2.869      6.113      0.000      11.814      23.259
# C(ko_parent)[T.MAT]         6.4310      1.705      3.771      0.000       3.029       9.833
# SC                   1.6510      1.863      0.886      0.379      -2.065       5.367
# SC:C(ko_parent)[T.MAT]     -7.3225      2.664     -2.749      0.008     -12.636      -2.009
# gWAT                 9.0049      1.834      4.910      0.000       5.346      12.664
# gWAT:C(sex)[T.m]    -6.9600      2.746     -2.535      0.014     -12.438      -1.482
# ==============================================================================
# Omnibus:                        4.784   Durbin-Watson:                   1.548
# Prob(Omnibus):                  0.091   Jarque-Bera (JB):                3.992
# Skew:                           0.525   Prob(JB):                        0.136
# Kurtosis:                       3.395   Cond. No.                         17.4
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

# list of point with high influence (large residuals and leverage)
idx_influence = [65, 49, 72, 35]

# list of data points to use in the model
idx_for_model = (set(range(metainfo.shape[0])) - set(idx_influence)) & set(idx_not_nan)
idx_for_model = list(idx_for_model)

model = sm.formula.ols('BW ~ (SC + gWAT * C(sex)) + (SC * C(ko_parent) + gWAT)', data=metainfo, subset=idx_for_model).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     BW   R-squared:                       0.809
# Model:                            OLS   Adj. R-squared:                  0.791
# Method:                 Least Squares   F-statistic:                     45.76
# Date:                Thu, 20 Feb 2020   Prob (F-statistic):           1.77e-21
# Time:                        01:01:22   Log-Likelihood:                -185.96
# No. Observations:                  72   AIC:                             385.9
# Df Residuals:                      65   BIC:                             401.9
# Df Model:                           6
# Covariance Type:            nonrobust
# ====================================================================================
#                        coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------------
# Intercept           17.0769      1.555     10.983      0.000      13.972      20.182
# C(sex)[T.m]         19.9881      2.795      7.152      0.000      14.406      25.570
# C(ko_parent)[T.MAT]         6.3089      1.566      4.027      0.000       3.180       9.437
# SC                   3.0020      1.911      1.571      0.121      -0.815       6.819
# SC:C(ko_parent)[T.MAT]     -5.8590      2.496     -2.347      0.022     -10.844      -0.874
# gWAT                 8.8744      1.665      5.331      0.000       5.550      12.199
# gWAT:C(sex)[T.m]   -10.1070      2.649     -3.815      0.000     -15.398      -4.816
# ==============================================================================
# Omnibus:                        1.846   Durbin-Watson:                   1.456
# Prob(Omnibus):                  0.397   Jarque-Bera (JB):                1.591
# Skew:                           0.223   Prob(JB):                        0.451
# Kurtosis:                       2.425   Cond. No.                         18.5
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

########################################################################################################################
### Model BW ~ (SC + gWAT * C(sex)) + (SC * C(ko_parent) + gWAT) + C(sex) : C(genotype)
########################################################################################################################

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

model = sm.formula.ols('BW ~ (SC + gWAT * C(sex)) + SC * C(ko_parent) + C(sex) : C(genotype)', data=metainfo, subset=idx_not_nan).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     BW   R-squared:                       0.781
# Model:                            OLS   Adj. R-squared:                  0.755
# Method:                 Least Squares   F-statistic:                     29.84
# Date:                Thu, 20 Feb 2020   Prob (F-statistic):           2.98e-19
# Time:                        01:14:16   Log-Likelihood:                -204.36
# No. Observations:                  76   AIC:                             426.7
# Df Residuals:                      67   BIC:                             447.7
# Df Model:                           8
# Covariance Type:            nonrobust
# =========================================================================================================
#                                             coef    std err          t      P>|t|      [0.025      0.975]
# ---------------------------------------------------------------------------------------------------------
# Intercept                                18.4278      1.819     10.129      0.000      14.796      22.059
# C(sex)[T.m]                              16.2815      2.940      5.538      0.000      10.413      22.150
# C(ko_parent)[T.MAT]                              7.1224      1.712      4.160      0.000       3.705      10.540
# C(sex)[f]:C(genotype)[T.KLF14-KO:Het]    -2.5284      1.263     -2.002      0.049      -5.050      -0.007
# C(sex)[m]:C(genotype)[T.KLF14-KO:Het]    -0.6830      1.252     -0.545      0.587      -3.182       1.816
# SC                                        2.1720      1.862      1.167      0.248      -1.544       5.888
# SC:C(ko_parent)[T.MAT]                          -8.5476      2.687     -3.182      0.002     -13.910      -3.185
# gWAT                                      8.8175      1.808      4.876      0.000       5.208      12.427
# gWAT:C(sex)[T.m]                         -6.6075      2.710     -2.438      0.017     -12.017      -1.197
# ==============================================================================
# Omnibus:                        2.805   Durbin-Watson:                   1.413
# Prob(Omnibus):                  0.246   Jarque-Bera (JB):                2.106
# Skew:                           0.380   Prob(JB):                        0.349
# Kurtosis:                       3.298   Cond. No.                         18.0
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

# list of point with high influence (large residuals and leverage)
idx_influence = [72, 65, 49, 35]

# list of data points to use in the model
idx_for_model = (set(range(metainfo.shape[0])) - set(idx_influence)) & set(idx_not_nan)
idx_for_model = list(idx_for_model)

model = sm.formula.ols('BW ~ (SC + gWAT * C(sex)) + SC * C(ko_parent) + C(sex) : C(genotype)', data=metainfo, subset=idx_for_model).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     BW   R-squared:                       0.823
# Model:                            OLS   Adj. R-squared:                  0.800
# Method:                 Least Squares   F-statistic:                     36.61
# Date:                Thu, 20 Feb 2020   Prob (F-statistic):           7.33e-21
# Time:                        01:19:09   Log-Likelihood:                -183.15
# No. Observations:                  72   AIC:                             384.3
# Df Residuals:                      63   BIC:                             404.8
# Df Model:                           8
# Covariance Type:            nonrobust
# =========================================================================================================
#                                             coef    std err          t      P>|t|      [0.025      0.975]
# ---------------------------------------------------------------------------------------------------------
# Intercept                                18.2000      1.599     11.380      0.000      15.004      21.396
# C(sex)[T.m]                              18.5612      2.802      6.623      0.000      12.961      24.161
# C(ko_parent)[T.MAT]                              6.7051      1.562      4.294      0.000       3.585       9.826
# C(sex)[f]:C(genotype)[T.KLF14-KO:Het]    -2.4983      1.114     -2.242      0.029      -4.725      -0.271
# C(sex)[m]:C(genotype)[T.KLF14-KO:Het]     0.3268      1.147      0.285      0.777      -1.965       2.619
# SC                                        2.9757      1.906      1.561      0.123      -0.833       6.784
# SC:C(ko_parent)[T.MAT]                          -6.6264      2.484     -2.668      0.010     -11.590      -1.663
# gWAT                                      8.9535      1.631      5.491      0.000       5.695      12.212
# gWAT:C(sex)[T.m]                        -10.0162      2.625     -3.816      0.000     -15.261      -4.771
# ==============================================================================
# Omnibus:                        6.052   Durbin-Watson:                   1.299
# Prob(Omnibus):                  0.049   Jarque-Bera (JB):                2.566
# Skew:                           0.101   Prob(JB):                        0.277
# Kurtosis:                       2.097   Cond. No.                         19.1
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

########################################################################################################################
### Model SC_BW ~ SC_vol_mean_1e12 * C(ko_parent) * C(sex) * C(genotype)
########################################################################################################################

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

model = sm.formula.ols('SC_BW ~ SC_vol_mean_1e12 * C(ko_parent) * C(sex) * C(genotype)', data=metainfo, subset=idx_not_nan).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                  SC_BW   R-squared:                       0.388
# Model:                            OLS   Adj. R-squared:                  0.230
# Method:                 Least Squares   F-statistic:                     2.451
# Date:                Thu, 20 Feb 2020   Prob (F-statistic):            0.00761
# Time:                        11:22:30   Log-Likelihood:                 241.89
# No. Observations:                  74   AIC:                            -451.8
# Df Residuals:                      58   BIC:                            -414.9
# Df Model:                          15
# Covariance Type:            nonrobust
# ====================================================================================================================================
#                                                                        coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                            0.0040      0.008      0.490      0.626      -0.012       0.020
# C(ko_parent)[T.MAT]                                                         0.0156      0.011      1.430      0.158      -0.006       0.037
# C(sex)[T.m]                                                         -0.0042      0.016     -0.263      0.794      -0.036       0.027
# C(genotype)[T.KLF14-KO:Het]                                         -0.0058      0.012     -0.486      0.629      -0.030       0.018
# C(ko_parent)[T.MAT]:C(sex)[T.m]                                            -0.0202      0.025     -0.804      0.425      -0.070       0.030
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]                            -0.0086      0.016     -0.555      0.581      -0.040       0.022
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                              0.0163      0.023      0.713      0.479      -0.029       0.062
# C(ko_parent)[T.MAT]:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                 0.0351      0.034      1.044      0.301      -0.032       0.102
# SC_vol_mean                                                          0.0916      0.056      1.626      0.109      -0.021       0.204
# SC_vol_mean:C(ko_parent)[T.MAT]                                            -0.1107      0.065     -1.695      0.095      -0.242       0.020
# SC_vol_mean:C(sex)[T.m]                                             -0.0239      0.077     -0.312      0.756      -0.178       0.130
# SC_vol_mean:C(genotype)[T.KLF14-KO:Het]                              0.1840      0.099      1.865      0.067      -0.014       0.381
# SC_vol_mean:C(ko_parent)[T.MAT]:C(sex)[T.m]                                 0.1107      0.107      1.040      0.303      -0.102       0.324
# SC_vol_mean:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]                -0.1416      0.110     -1.290      0.202      -0.361       0.078
# SC_vol_mean:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                 -0.2032      0.122     -1.664      0.102      -0.448       0.041
# SC_vol_mean:C(ko_parent)[T.MAT]:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]     0.0134      0.160      0.084      0.933      -0.306       0.333
# ==============================================================================
# Omnibus:                       45.980   Durbin-Watson:                   1.451
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):              173.644
# Skew:                           1.884   Prob(JB):                     1.97e-38
# Kurtosis:                       9.490   Cond. No.                         318.
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

# list of point with high influence (large residuals and leverage)
idx_influence = [65, 35, 49]

# list of data points to use in the model
idx_for_model = (set(range(metainfo.shape[0])) - set(idx_influence)) & set(idx_not_nan)
idx_for_model = list(idx_for_model)

model = sm.formula.ols('SC_BW ~ SC_vol_mean_1e12 * C(ko_parent) * C(sex) * C(genotype)', data=metainfo, subset=idx_for_model).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                  SC_BW   R-squared:                       0.473
# Model:                            OLS   Adj. R-squared:                  0.329
# Method:                 Least Squares   F-statistic:                     3.293
# Date:                Thu, 20 Feb 2020   Prob (F-statistic):           0.000620
# Time:                        11:24:56   Log-Likelihood:                 255.25
# No. Observations:                  71   AIC:                            -478.5
# Df Residuals:                      55   BIC:                            -442.3
# Df Model:                          15
# Covariance Type:            nonrobust
# ====================================================================================================================================
#                                                                        coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                            0.0040      0.006      0.674      0.503      -0.008       0.016
# C(ko_parent)[T.MAT]                                                         0.0083      0.008      1.018      0.313      -0.008       0.025
# C(sex)[T.m]                                                         -0.0042      0.011     -0.362      0.719      -0.027       0.019
# C(genotype)[T.KLF14-KO:Het]                                         -0.0052      0.009     -0.597      0.553      -0.023       0.012
# C(ko_parent)[T.MAT]:C(sex)[T.m]                                            -0.0129      0.018     -0.704      0.485      -0.050       0.024
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]                            -0.0020      0.011     -0.173      0.863      -0.025       0.021
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                              0.0293      0.020      1.475      0.146      -0.011       0.069
# C(ko_parent)[T.MAT]:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                 0.0148      0.027      0.553      0.583      -0.039       0.069
# SC_vol_mean                                                          0.0916      0.041      2.240      0.029       0.010       0.174
# SC_vol_mean:C(ko_parent)[T.MAT]                                            -0.0879      0.048     -1.836      0.072      -0.184       0.008
# SC_vol_mean:C(sex)[T.m]                                             -0.0239      0.056     -0.430      0.669      -0.136       0.088
# SC_vol_mean:C(genotype)[T.KLF14-KO:Het]                              0.1190      0.072      1.646      0.106      -0.026       0.264
# SC_vol_mean:C(ko_parent)[T.MAT]:C(sex)[T.m]                                 0.0879      0.078      1.133      0.262      -0.068       0.243
# SC_vol_mean:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]                -0.0994      0.081     -1.234      0.222      -0.261       0.062
# SC_vol_mean:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                 -0.1821      0.096     -1.899      0.063      -0.374       0.010
# SC_vol_mean:C(ko_parent)[T.MAT]:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]     0.0151      0.122      0.124      0.901      -0.229       0.259
# ==============================================================================
# Omnibus:                       12.050   Durbin-Watson:                   1.441
# Prob(Omnibus):                  0.002   Jarque-Bera (JB):               12.787
# Skew:                           0.868   Prob(JB):                      0.00167
# Kurtosis:                       4.144   Cond. No.                         324.
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

########################################################################################################################
### Model gWAT_BW ~ gWAT_vol_mean_1e12 * C(ko_parent) * C(sex) * C(genotype)
########################################################################################################################

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

model = sm.formula.ols('gWAT_BW ~ gWAT_vol_mean_1e12 * C(ko_parent) * C(sex) * C(genotype)', data=metainfo, subset=idx_not_nan).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                gWAT_BW   R-squared:                       0.274
# Model:                            OLS   Adj. R-squared:                  0.072
# Method:                 Least Squares   F-statistic:                     1.358
# Date:                Thu, 20 Feb 2020   Prob (F-statistic):              0.202
# Time:                        11:27:58   Log-Likelihood:                 228.64
# No. Observations:                  70   AIC:                            -425.3
# Df Residuals:                      54   BIC:                            -389.3
# Df Model:                          15
# Covariance Type:            nonrobust
# ======================================================================================================================================
#                                                                          coef    std err          t      P>|t|      [0.025      0.975]
# --------------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                              0.0101      0.008      1.306      0.197      -0.005       0.026
# C(ko_parent)[T.MAT]                                                           0.0241      0.012      2.087      0.042       0.001       0.047
# C(sex)[T.m]                                                            0.0103      0.025      0.410      0.683      -0.040       0.060
# C(genotype)[T.KLF14-KO:Het]                                            0.0169      0.013      1.333      0.188      -0.009       0.042
# C(ko_parent)[T.MAT]:C(sex)[T.m]                                               0.0275      0.040      0.683      0.498      -0.053       0.108
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]                              -0.0461      0.018     -2.534      0.014      -0.083      -0.010
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                               -0.0598      0.047     -1.282      0.205      -0.153       0.034
# C(ko_parent)[T.MAT]:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                   0.0550      0.063      0.867      0.390      -0.072       0.182
# gWAT_vol_mean                                                          0.0907      0.040      2.263      0.028       0.010       0.171
# gWAT_vol_mean:C(ko_parent)[T.MAT]                                            -0.0991      0.047     -2.090      0.041      -0.194      -0.004
# gWAT_vol_mean:C(sex)[T.m]                                             -0.0705      0.069     -1.016      0.314      -0.209       0.069
# gWAT_vol_mean:C(genotype)[T.KLF14-KO:Het]                             -0.0608      0.059     -1.035      0.305      -0.179       0.057
# gWAT_vol_mean:C(ko_parent)[T.MAT]:C(sex)[T.m]                                -0.0281      0.102     -0.277      0.783      -0.232       0.176
# gWAT_vol_mean:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]                 0.1487      0.074      1.999      0.051      -0.000       0.298
# gWAT_vol_mean:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                  0.1493      0.117      1.273      0.209      -0.086       0.385
# gWAT_vol_mean:C(ko_parent)[T.MAT]:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]    -0.1482      0.159     -0.930      0.357      -0.468       0.171
# ==============================================================================
# Omnibus:                        8.962   Durbin-Watson:                   1.486
# Prob(Omnibus):                  0.011   Jarque-Bera (JB):                9.826
# Skew:                           0.609   Prob(JB):                      0.00735
# Kurtosis:                       4.373   Cond. No.                         282.
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

# list of point with high influence (large residuals and leverage)
idx_influence = [63, 62, 35, 53]

# list of data points to use in the model
idx_for_model = (set(range(metainfo.shape[0])) - set(idx_influence)) & set(idx_not_nan)
idx_for_model = list(idx_for_model)

model = sm.formula.ols('gWAT_BW ~ gWAT_vol_mean_1e12 * C(ko_parent) * C(sex) * C(genotype)', data=metainfo, subset=idx_for_model).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                gWAT_BW   R-squared:                       0.495
# Model:                            OLS   Adj. R-squared:                  0.343
# Method:                 Least Squares   F-statistic:                     3.262
# Date:                Thu, 20 Feb 2020   Prob (F-statistic):           0.000844
# Time:                        11:29:52   Log-Likelihood:                 236.10
# No. Observations:                  66   AIC:                            -440.2
# Df Residuals:                      50   BIC:                            -405.2
# Df Model:                          15
# Covariance Type:            nonrobust
# ======================================================================================================================================
#                                                                          coef    std err          t      P>|t|      [0.025      0.975]
# --------------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                             -0.0009      0.006     -0.147      0.884      -0.014       0.012
# C(ko_parent)[T.MAT]                                                           0.0216      0.010      2.251      0.029       0.002       0.041
# C(sex)[T.m]                                                            0.0213      0.019      1.140      0.260      -0.016       0.059
# C(genotype)[T.KLF14-KO:Het]                                            0.0105      0.015      0.692      0.492      -0.020       0.041
# C(ko_parent)[T.MAT]:C(sex)[T.m]                                               0.0299      0.030      0.995      0.324      -0.030       0.090
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]                              -0.0262      0.018     -1.430      0.159      -0.063       0.011
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                               -0.0534      0.037     -1.463      0.150      -0.127       0.020
# C(ko_parent)[T.MAT]:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                   0.0351      0.049      0.722      0.473      -0.062       0.133
# gWAT_vol_mean                                                          0.1322      0.031      4.213      0.000       0.069       0.195
# gWAT_vol_mean:C(ko_parent)[T.MAT]                                            -0.1084      0.037     -2.891      0.006      -0.184      -0.033
# gWAT_vol_mean:C(sex)[T.m]                                             -0.1119      0.052     -2.141      0.037      -0.217      -0.007
# gWAT_vol_mean:C(genotype)[T.KLF14-KO:Het]                             -0.0368      0.061     -0.605      0.548      -0.159       0.086
# gWAT_vol_mean:C(ko_parent)[T.MAT]:C(sex)[T.m]                                -0.0189      0.076     -0.247      0.806      -0.172       0.134
# gWAT_vol_mean:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]                 0.0925      0.070      1.319      0.193      -0.048       0.233
# gWAT_vol_mean:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                  0.1253      0.097      1.296      0.201      -0.069       0.319
# gWAT_vol_mean:C(ko_parent)[T.MAT]:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]    -0.0919      0.126     -0.732      0.468      -0.344       0.160
# ==============================================================================
# Omnibus:                        1.926   Durbin-Watson:                   1.904
# Prob(Omnibus):                  0.382   Jarque-Bera (JB):                1.345
# Skew:                           0.336   Prob(JB):                        0.510
# Kurtosis:                       3.191   Cond. No.                         307.
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

########################################################################################################################
### Model SC ~ SC_vol_mean_1e12 * C(ko_parent) * C(sex) * C(genotype)
########################################################################################################################

formula = 'SC ~ SC_vol_mean_1e12 * C(ko_parent) * C(sex) * C(genotype)'

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

model = sm.formula.ols(formula, data=metainfo, subset=idx_not_nan).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     SC   R-squared:                       0.516
# Model:                            OLS   Adj. R-squared:                  0.390
# Method:                 Least Squares   F-statistic:                     4.116
# Date:                Thu, 20 Feb 2020   Prob (F-statistic):           4.64e-05
# Time:                        11:40:05   Log-Likelihood:                -2.3226
# No. Observations:                  74   AIC:                             36.65
# Df Residuals:                      58   BIC:                             73.51
# Df Model:                          15
# Covariance Type:            nonrobust
# ====================================================================================================================================
#                                                                        coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                            0.0268      0.223      0.120      0.905      -0.419       0.473
# C(ko_parent)[T.MAT]                                                         0.4371      0.296      1.475      0.145      -0.156       1.030
# C(sex)[T.m]                                                         -0.2833      0.429     -0.661      0.511      -1.142       0.575
# C(genotype)[T.KLF14-KO:Het]                                         -0.1929      0.326     -0.592      0.556      -0.845       0.459
# C(ko_parent)[T.MAT]:C(sex)[T.m]                                            -0.4068      0.680     -0.598      0.552      -1.769       0.955
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]                            -0.1811      0.421     -0.431      0.668      -1.023       0.661
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                              0.6555      0.619      1.060      0.294      -0.583       1.894
# C(ko_parent)[T.MAT]:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                 1.0524      0.912      1.154      0.253      -0.773       2.878
# SC_vol_mean                                                          3.2500      1.528      2.126      0.038       0.190       6.310
# SC_vol_mean:C(ko_parent)[T.MAT]                                            -3.1775      1.772     -1.793      0.078      -6.724       0.369
# SC_vol_mean:C(sex)[T.m]                                              0.4043      2.083      0.194      0.847      -3.765       4.574
# SC_vol_mean:C(genotype)[T.KLF14-KO:Het]                              4.8356      2.676      1.807      0.076      -0.520      10.191
# SC_vol_mean:C(ko_parent)[T.MAT]:C(sex)[T.m]                                 2.3689      2.888      0.820      0.415      -3.412       8.150
# SC_vol_mean:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]                -3.8944      2.975     -1.309      0.196      -9.849       2.060
# SC_vol_mean:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                 -5.9087      3.312     -1.784      0.080     -12.538       0.721
# SC_vol_mean:C(ko_parent)[T.MAT]:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]    -0.3243      4.328     -0.075      0.941      -8.989       8.340
# ==============================================================================
# Omnibus:                       12.378   Durbin-Watson:                   1.277
# Prob(Omnibus):                  0.002   Jarque-Bera (JB):               12.995
# Skew:                           0.907   Prob(JB):                      0.00151
# Kurtosis:                       3.962   Cond. No.                         318.
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

# list of point with high influence (large residuals and leverage)
idx_influence = [65, 49, 16]

# list of data points to use in the model
idx_for_model = (set(range(metainfo.shape[0])) - set(idx_influence)) & set(idx_not_nan)
idx_for_model = list(idx_for_model)

model = sm.formula.ols(formula, data=metainfo, subset=idx_for_model).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     SC   R-squared:                       0.592
# Model:                            OLS   Adj. R-squared:                  0.481
# Method:                 Least Squares   F-statistic:                     5.318
# Date:                Thu, 20 Feb 2020   Prob (F-statistic):           2.12e-06
# Time:                        11:42:52   Log-Likelihood:                 7.5157
# No. Observations:                  71   AIC:                             16.97
# Df Residuals:                      55   BIC:                             53.17
# Df Model:                          15
# Covariance Type:            nonrobust
# ====================================================================================================================================
#                                                                        coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                            0.0268      0.195      0.137      0.891      -0.365       0.418
# C(ko_parent)[T.MAT]                                                         0.4371      0.260      1.683      0.098      -0.083       0.958
# C(sex)[T.m]                                                         -0.2833      0.376     -0.754      0.454      -1.037       0.470
# C(genotype)[T.KLF14-KO:Het]                                         -0.1814      0.286     -0.635      0.528      -0.754       0.391
# C(ko_parent)[T.MAT]:C(sex)[T.m]                                            -0.4068      0.597     -0.682      0.498      -1.603       0.789
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]                            -0.1926      0.369     -0.522      0.604      -0.932       0.547
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                              1.1441      0.650      1.760      0.084      -0.159       2.447
# C(ko_parent)[T.MAT]:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                 1.0067      0.895      1.125      0.266      -0.787       2.801
# SC_vol_mean                                                          3.2500      1.340      2.425      0.019       0.564       5.936
# SC_vol_mean:C(ko_parent)[T.MAT]                                            -3.1775      1.554     -2.045      0.046      -6.291      -0.064
# SC_vol_mean:C(sex)[T.m]                                              0.4043      1.827      0.221      0.826      -3.256       4.065
# SC_vol_mean:C(genotype)[T.KLF14-KO:Het]                              3.6627      2.369      1.546      0.128      -1.085       8.411
# SC_vol_mean:C(ko_parent)[T.MAT]:C(sex)[T.m]                                 2.3689      2.532      0.935      0.354      -2.706       7.444
# SC_vol_mean:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]                -2.7215      2.629     -1.035      0.305      -7.991       2.548
# SC_vol_mean:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                 -6.3447      3.142     -2.019      0.048     -12.641      -0.048
# SC_vol_mean:C(ko_parent)[T.MAT]:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]    -1.6498      4.047     -0.408      0.685      -9.759       6.460
# ==============================================================================
# Omnibus:                       10.882   Durbin-Watson:                   1.409
# Prob(Omnibus):                  0.004   Jarque-Bera (JB):               11.004
# Skew:                           0.837   Prob(JB):                      0.00408
# Kurtosis:                       3.957   Cond. No.                         319.
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

########################################################################################################################
### Model gWAT ~ gWAT_vol_mean_1e12 * C(ko_parent) * C(sex) * C(genotype)
########################################################################################################################

formula = 'gWAT ~ gWAT_vol_mean_1e12 * C(ko_parent) * C(sex) * C(genotype)'

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

model = sm.formula.ols(formula, data=metainfo, subset=idx_not_nan).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                   gWAT   R-squared:                       0.465
# Model:                            OLS   Adj. R-squared:                  0.317
# Method:                 Least Squares   F-statistic:                     3.132
# Date:                Thu, 20 Feb 2020   Prob (F-statistic):            0.00105
# Time:                        11:47:14   Log-Likelihood:                -8.4409
# No. Observations:                  70   AIC:                             48.88
# Df Residuals:                      54   BIC:                             84.86
# Df Model:                          15
# Covariance Type:            nonrobust
# ======================================================================================================================================
#                                                                          coef    std err          t      P>|t|      [0.025      0.975]
# --------------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                              0.1167      0.229      0.510      0.612      -0.342       0.576
# C(ko_parent)[T.MAT]                                                           0.5163      0.341      1.513      0.136      -0.168       1.201
# C(sex)[T.m]                                                            0.6129      0.739      0.829      0.411      -0.870       2.095
# C(genotype)[T.KLF14-KO:Het]                                            0.3916      0.376      1.042      0.302      -0.362       1.145
# C(ko_parent)[T.MAT]:C(sex)[T.m]                                               1.3474      1.189      1.133      0.262      -1.037       3.731
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]                              -1.0333      0.538     -1.920      0.060      -2.112       0.046
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                               -1.8740      1.380     -1.358      0.180      -4.641       0.893
# C(ko_parent)[T.MAT]:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                   1.3060      1.877      0.696      0.489      -2.457       5.069
# gWAT_vol_mean                                                          3.4100      1.185      2.878      0.006       1.035       5.785
# gWAT_vol_mean:C(ko_parent)[T.MAT]                                            -2.3242      1.403     -1.657      0.103      -5.137       0.489
# gWAT_vol_mean:C(sex)[T.m]                                             -2.4867      2.050     -1.213      0.230      -6.597       1.624
# gWAT_vol_mean:C(genotype)[T.KLF14-KO:Het]                             -1.8693      1.738     -1.075      0.287      -5.354       1.616
# gWAT_vol_mean:C(ko_parent)[T.MAT]:C(sex)[T.m]                                -2.2568      3.006     -0.751      0.456      -8.284       3.771
# gWAT_vol_mean:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]                 3.6775      2.200      1.672      0.100      -0.733       8.088
# gWAT_vol_mean:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                  4.9107      3.470      1.415      0.163      -2.046      11.867
# gWAT_vol_mean:C(ko_parent)[T.MAT]:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]    -3.4917      4.713     -0.741      0.462     -12.941       5.957
# ==============================================================================
# Omnibus:                        6.645   Durbin-Watson:                   1.825
# Prob(Omnibus):                  0.036   Jarque-Bera (JB):                5.842
# Skew:                           0.641   Prob(JB):                       0.0539
# Kurtosis:                       3.600   Cond. No.                         282.
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

# list of point with high influence (large residuals and leverage)
idx_influence = [63, 49, 52, 62, 35, 54]

# list of data points to use in the model
idx_for_model = (set(range(metainfo.shape[0])) - set(idx_influence)) & set(idx_not_nan)
idx_for_model = list(idx_for_model)

model = sm.formula.ols(formula, data=metainfo, subset=idx_for_model).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                   gWAT   R-squared:                       0.681
# Model:                            OLS   Adj. R-squared:                  0.581
# Method:                 Least Squares   F-statistic:                     6.830
# Date:                Thu, 20 Feb 2020   Prob (F-statistic):           1.46e-07
# Time:                        11:48:18   Log-Likelihood:                 13.190
# No. Observations:                  64   AIC:                             5.619
# Df Residuals:                      48   BIC:                             40.16
# Df Model:                          15
# Covariance Type:            nonrobust
# ======================================================================================================================================
#                                                                          coef    std err          t      P>|t|      [0.025      0.975]
# --------------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                             -0.1588      0.186     -0.854      0.397      -0.532       0.215
# C(ko_parent)[T.MAT]                                                           0.4728      0.281      1.681      0.099      -0.093       1.038
# C(sex)[T.m]                                                            0.8884      0.547      1.624      0.111      -0.211       1.988
# C(genotype)[T.KLF14-KO:Het]                                            0.2912      0.366      0.796      0.430      -0.444       1.026
# C(ko_parent)[T.MAT]:C(sex)[T.m]                                               1.3909      0.879      1.582      0.120      -0.377       3.159
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]                              -0.6139      0.473     -1.299      0.200      -1.564       0.336
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                               -0.8940      1.310     -0.683      0.498      -3.527       1.739
# C(ko_parent)[T.MAT]:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                   0.0070      1.610      0.004      0.997      -3.229       3.243
# gWAT_vol_mean                                                          4.4445      0.918      4.842      0.000       2.599       6.290
# gWAT_vol_mean:C(ko_parent)[T.MAT]                                            -2.5967      1.097     -2.367      0.022      -4.802      -0.391
# gWAT_vol_mean:C(sex)[T.m]                                             -3.5212      1.530     -2.302      0.026      -6.597      -0.445
# gWAT_vol_mean:C(genotype)[T.KLF14-KO:Het]                             -1.3656      1.527     -0.894      0.376      -4.436       1.705
# gWAT_vol_mean:C(ko_parent)[T.MAT]:C(sex)[T.m]                                -1.9842      2.233     -0.889      0.379      -6.474       2.506
# gWAT_vol_mean:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]                 2.4118      1.834      1.315      0.195      -1.275       6.099
# gWAT_vol_mean:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                  2.3745      3.136      0.757      0.453      -3.931       8.680
# gWAT_vol_mean:C(ko_parent)[T.MAT]:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]    -0.1935      3.916     -0.049      0.961      -8.068       7.681
# ==============================================================================
# Omnibus:                        2.710   Durbin-Watson:                   1.710
# Prob(Omnibus):                  0.258   Jarque-Bera (JB):                2.207
# Skew:                           0.453   Prob(JB):                        0.332
# Kurtosis:                       3.071   Cond. No.                         316.
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

########################################################################################################################
### Model SC_vol_mean ~ C(sex) * C(ko_parent) * C(genotype)
########################################################################################################################

formula = 'SC_vol_mean ~ C(sex) * C(ko_parent) * C(genotype)'

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

model = sm.formula.ols(formula, data=metainfo, subset=idx_not_nan).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:            SC_vol_mean   R-squared:                       0.404
# Model:                            OLS   Adj. R-squared:                  0.341
# Method:                 Least Squares   F-statistic:                     6.404
# Date:                Fri, 21 Feb 2020   Prob (F-statistic):           9.26e-06
# Time:                        16:10:24   Log-Likelihood:                 2135.2
# No. Observations:                  74   AIC:                            -4254.
# Df Residuals:                      66   BIC:                            -4236.
# Df Model:                           7
# Covariance Type:            nonrobust
# ===============================================================================================================================
#                                                                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                     1.28e-13   2.85e-14      4.490      0.000    7.11e-14    1.85e-13
# C(sex)[T.m]                                                  1.231e-13   3.72e-14      3.311      0.002    4.88e-14    1.97e-13
# C(ko_parent)[T.MAT]                                          6.585e-14   3.72e-14      1.772      0.081   -8.36e-15     1.4e-13
# C(genotype)[T.KLF14-KO:Het]                                 -2.858e-14    3.8e-14     -0.752      0.455   -1.04e-13    4.73e-14
# C(sex)[T.m]:C(ko_parent)[T.MAT]                             -4.697e-14   5.02e-14     -0.936      0.353   -1.47e-13    5.32e-14
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                      4.829e-14   5.22e-14      0.925      0.358   -5.59e-14    1.53e-13
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]              1.541e-15   5.08e-14      0.030      0.976   -9.99e-14    1.03e-13
# C(sex)[T.m]:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het] -7.957e-14   7.07e-14     -1.125      0.265   -2.21e-13    6.16e-14
# ==============================================================================
# Omnibus:                        2.630   Durbin-Watson:                   1.580
# Prob(Omnibus):                  0.268   Jarque-Bera (JB):                2.575
# Skew:                           0.437   Prob(JB):                        0.276
# Kurtosis:                       2.731   Cond. No.                         19.2
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

# list of point with high influence (large residuals and leverage)
idx_influence = [7, 20, 6, 26]

# list of data points to use in the model
idx_for_model = (set(range(metainfo.shape[0])) - set(idx_influence)) & set(idx_not_nan)
idx_for_model = list(idx_for_model)

model = sm.formula.ols(formula, data=metainfo, subset=idx_for_model).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:            SC_vol_mean   R-squared:                       0.563
# Model:                            OLS   Adj. R-squared:                  0.513
# Method:                 Least Squares   F-statistic:                     11.39
# Date:                Fri, 21 Feb 2020   Prob (F-statistic):           3.43e-09
# Time:                        16:11:16   Log-Likelihood:                 2034.0
# No. Observations:                  70   AIC:                            -4052.
# Df Residuals:                      62   BIC:                            -4034.
# Df Model:                           7
# Covariance Type:            nonrobust
# ===============================================================================================================================
#                                                                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                     1.28e-13   2.34e-14      5.482      0.000    8.13e-14    1.75e-13
# C(sex)[T.m]                                                  1.231e-13   3.04e-14      4.042      0.000    6.22e-14    1.84e-13
# C(ko_parent)[T.MAT]                                          2.746e-14    3.2e-14      0.859      0.394   -3.65e-14    9.14e-14
# C(genotype)[T.KLF14-KO:Het]                                 -2.858e-14   3.11e-14     -0.918      0.362   -9.08e-14    3.37e-14
# C(sex)[T.m]:C(ko_parent)[T.MAT]                              -8.58e-15   4.23e-14     -0.203      0.840   -9.31e-14    7.59e-14
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                      4.829e-14   4.28e-14      1.129      0.263   -3.72e-14    1.34e-13
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]             -4.573e-15   4.39e-14     -0.104      0.917   -9.22e-14    8.31e-14
# C(sex)[T.m]:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het] -7.346e-14   5.95e-14     -1.234      0.222   -1.92e-13    4.56e-14
# ==============================================================================
# Omnibus:                        6.511   Durbin-Watson:                   1.528
# Prob(Omnibus):                  0.039   Jarque-Bera (JB):                2.547
# Skew:                          -0.006   Prob(JB):                        0.280
# Kurtosis:                       2.066   Cond. No.                         19.2
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

print(model.pvalues)

# ANOVA table
aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)

########################################################################################################################
### Model gWAT_vol_mean ~ C(sex) * C(ko_parent) * C(genotype)
########################################################################################################################

formula = 'gWAT_vol_mean ~ C(sex) * C(ko_parent) * C(genotype)'

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

model = sm.formula.ols(formula, data=metainfo, subset=idx_not_nan).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:          gWAT_vol_mean   R-squared:                       0.596
# Model:                            OLS   Adj. R-squared:                  0.551
# Method:                 Least Squares   F-statistic:                     13.08
# Date:                Fri, 21 Feb 2020   Prob (F-statistic):           3.30e-10
# Time:                        16:56:33   Log-Likelihood:                 2009.7
# No. Observations:                  70   AIC:                            -4003.
# Df Residuals:                      62   BIC:                            -3986.
# Df Model:                           7
# Covariance Type:            nonrobust
# ===============================================================================================================================
#                                                                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                    1.724e-13   2.91e-14      5.920      0.000    1.14e-13    2.31e-13
# C(sex)[T.m]                                                  2.433e-13   4.12e-14      5.909      0.000    1.61e-13    3.26e-13
# C(ko_parent)[T.MAT]                                           1.38e-13   4.01e-14      3.440      0.001    5.78e-14    2.18e-13
# C(genotype)[T.KLF14-KO:Het]                                  4.539e-14   4.24e-14      1.070      0.289   -3.94e-14     1.3e-13
# C(sex)[T.m]:C(ko_parent)[T.MAT]                              -1.23e-13   5.75e-14     -2.138      0.036   -2.38e-13   -8.02e-15
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                     -1.171e-14   6.11e-14     -0.192      0.849   -1.34e-13    1.11e-13
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]             -1.132e-13   5.84e-14     -1.938      0.057    -2.3e-13    3.57e-15
# C(sex)[T.m]:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]  4.928e-14   8.39e-14      0.587      0.559   -1.19e-13    2.17e-13
# ==============================================================================
# Omnibus:                        1.120   Durbin-Watson:                   1.766
# Prob(Omnibus):                  0.571   Jarque-Bera (JB):                1.193
# Skew:                           0.260   Prob(JB):                        0.551
# Kurtosis:                       2.628   Cond. No.                         17.8
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

# list of point with high influence (large residuals and leverage)
idx_influence = [37, 7, 0]

# list of data points to use in the model
idx_for_model = (set(range(metainfo.shape[0])) - set(idx_influence)) & set(idx_not_nan)
idx_for_model = list(idx_for_model)

model = sm.formula.ols(formula, data=metainfo, subset=idx_for_model).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:          gWAT_vol_mean   R-squared:                       0.689
# Model:                            OLS   Adj. R-squared:                  0.652
# Method:                 Least Squares   F-statistic:                     18.66
# Date:                Fri, 21 Feb 2020   Prob (F-statistic):           7.43e-13
# Time:                        16:57:05   Log-Likelihood:                 1933.1
# No. Observations:                  67   AIC:                            -3850.
# Df Residuals:                      59   BIC:                            -3833.
# Df Model:                           7
# Covariance Type:            nonrobust
# ===============================================================================================================================
#                                                                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                    1.724e-13   2.54e-14      6.798      0.000    1.22e-13    2.23e-13
# C(sex)[T.m]                                                  2.433e-13   3.59e-14      6.785      0.000    1.72e-13    3.15e-13
# C(ko_parent)[T.MAT]                                              9e-14    3.7e-14      2.435      0.018     1.6e-14    1.64e-13
# C(genotype)[T.KLF14-KO:Het]                                  4.539e-14    3.7e-14      1.228      0.224   -2.86e-14    1.19e-13
# C(sex)[T.m]:C(ko_parent)[T.MAT]                             -7.491e-14   5.15e-14     -1.455      0.151   -1.78e-13    2.81e-14
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                     -1.171e-14   5.32e-14     -0.220      0.827   -1.18e-13    9.48e-14
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]             -8.848e-14    5.3e-14     -1.668      0.101   -1.95e-13    1.76e-14
# C(sex)[T.m]:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]  2.457e-14   7.46e-14      0.329      0.743   -1.25e-13    1.74e-13
# ==============================================================================
# Omnibus:                        3.101   Durbin-Watson:                   1.777
# Prob(Omnibus):                  0.212   Jarque-Bera (JB):                1.925
# Skew:                           0.177   Prob(JB):                        0.382
# Kurtosis:                       2.249   Cond. No.                         17.9
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

print(model.pvalues)

# ANOVA table
aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)

########################################################################################################################
### Model SC_rho_N ~ C(sex) * C(ko_parent) * C(genotype)
########################################################################################################################

formula = 'SC_rho_N ~ C(sex) * C(ko_parent) * C(genotype)'

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

model = sm.formula.ols(formula, data=metainfo, subset=idx_not_nan).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:               SC_rho_N   R-squared:                       0.282
# Model:                            OLS   Adj. R-squared:                  0.206
# Method:                 Least Squares   F-statistic:                     3.708
# Date:                Fri, 21 Feb 2020   Prob (F-statistic):            0.00193
# Time:                        17:53:06   Log-Likelihood:                -1184.4
# No. Observations:                  74   AIC:                             2385.
# Df Residuals:                      66   BIC:                             2403.
# Df Model:                           7
# Covariance Type:            nonrobust
# ===============================================================================================================================
#                                                                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                    4.263e+06   8.65e+05      4.930      0.000    2.54e+06    5.99e+06
# C(sex)[T.m]                                                 -1.426e+06   1.13e+06     -1.265      0.210   -3.68e+06    8.25e+05
# C(ko_parent)[T.MAT]                                         -9.607e+05   1.13e+06     -0.852      0.397   -3.21e+06    1.29e+06
# C(genotype)[T.KLF14-KO:Het]                                  2.268e+06   1.15e+06      1.967      0.053   -3.37e+04    4.57e+06
# C(sex)[T.m]:C(ko_parent)[T.MAT]                              3.072e+05   1.52e+06      0.202      0.841   -2.73e+06    3.35e+06
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                     -1.371e+06   1.58e+06     -0.866      0.390   -4.53e+06    1.79e+06
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]             -3.702e+06   1.54e+06     -2.402      0.019   -6.78e+06   -6.25e+05
# C(sex)[T.m]:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]  4.165e+06   2.14e+06      1.942      0.056   -1.17e+05    8.45e+06
# ==============================================================================
# Omnibus:                       32.335   Durbin-Watson:                   1.792
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):               70.647
# Skew:                           1.499   Prob(JB):                     4.56e-16
# Kurtosis:                       6.731   Cond. No.                         19.2
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

# list of point with high influence (large residuals and leverage)
idx_influence = [65, 53, 35, 31, 62, 42]

# list of data points to use in the model
idx_for_model = (set(range(metainfo.shape[0])) - set(idx_influence)) & set(idx_not_nan)
idx_for_model = list(idx_for_model)

model = sm.formula.ols(formula, data=metainfo, subset=idx_for_model).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:               SC_rho_N   R-squared:                       0.347
# Model:                            OLS   Adj. R-squared:                  0.271
# Method:                 Least Squares   F-statistic:                     4.562
# Date:                Fri, 21 Feb 2020   Prob (F-statistic):           0.000390
# Time:                        17:53:50   Log-Likelihood:                -1048.4
# No. Observations:                  68   AIC:                             2113.
# Df Residuals:                      60   BIC:                             2131.
# Df Model:                           7
# Covariance Type:            nonrobust
# ===============================================================================================================================
#                                                                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                     3.55e+06   5.22e+05      6.802      0.000    2.51e+06    4.59e+06
# C(sex)[T.m]                                                 -7.126e+05    6.6e+05     -1.079      0.285   -2.03e+06    6.08e+05
# C(ko_parent)[T.MAT]                                          -9.44e+05   6.74e+05     -1.401      0.166   -2.29e+06    4.04e+05
# C(genotype)[T.KLF14-KO:Het]                                  1.494e+06   7.38e+05      2.024      0.047    1.72e+04    2.97e+06
# C(sex)[T.m]:C(ko_parent)[T.MAT]                              2.905e+05   8.84e+05      0.329      0.743   -1.48e+06    2.06e+06
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                     -5.965e+05   9.55e+05     -0.625      0.535   -2.51e+06    1.31e+06
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]             -2.231e+06   9.43e+05     -2.366      0.021   -4.12e+06   -3.45e+05
# C(sex)[T.m]:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]  2.021e+06   1.27e+06      1.597      0.116   -5.11e+05    4.55e+06
# ==============================================================================
# Omnibus:                       21.123   Durbin-Watson:                   1.610
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):               35.834
# Skew:                           1.105   Prob(JB):                     1.65e-08
# Kurtosis:                       5.786   Cond. No.                         20.1
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


print(model.pvalues)

# ANOVA table
aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)

########################################################################################################################
### Model gWAT_rho_N ~ C(sex) * C(ko_parent) * C(genotype)
########################################################################################################################

formula = 'gWAT_rho_N ~ C(sex) * C(ko_parent) * C(genotype)'

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

model = sm.formula.ols(formula, data=metainfo, subset=idx_not_nan).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:             gWAT_rho_N   R-squared:                       0.135
# Model:                            OLS   Adj. R-squared:                  0.038
# Method:                 Least Squares   F-statistic:                     1.388
# Date:                Fri, 21 Feb 2020   Prob (F-statistic):              0.226
# Time:                        17:55:14   Log-Likelihood:                -1121.8
# No. Observations:                  70   AIC:                             2260.
# Df Residuals:                      62   BIC:                             2278.
# Df Model:                           7
# Covariance Type:            nonrobust
# ===============================================================================================================================
#                                                                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                     4.73e+06   7.82e+05      6.049      0.000    3.17e+06    6.29e+06
# C(sex)[T.m]                                                 -1.716e+06   1.11e+06     -1.551      0.126   -3.93e+06    4.95e+05
# C(ko_parent)[T.MAT]                                         -7.536e+05   1.08e+06     -0.699      0.487   -2.91e+06     1.4e+06
# C(genotype)[T.KLF14-KO:Het]                                  4.282e+05   1.14e+06      0.376      0.708   -1.85e+06    2.71e+06
# C(sex)[T.m]:C(ko_parent)[T.MAT]                              4.602e+05   1.54e+06      0.298      0.767   -2.63e+06    3.55e+06
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                     -9.345e+05   1.64e+06     -0.569      0.571   -4.22e+06    2.35e+06
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]             -1.323e+06   1.57e+06     -0.843      0.402   -4.46e+06    1.81e+06
# C(sex)[T.m]:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]  2.516e+06   2.25e+06      1.116      0.269   -1.99e+06    7.02e+06
# ==============================================================================
# Omnibus:                       66.663   Durbin-Watson:                   1.398
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):              432.785
# Skew:                           2.837   Prob(JB):                     1.05e-94
# Kurtosis:                      13.779   Cond. No.                         17.8
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

# list of point with high influence (large residuals and leverage)
idx_influence = [63, 62, 35]

# list of data points to use in the model
idx_for_model = (set(range(metainfo.shape[0])) - set(idx_influence)) & set(idx_not_nan)
idx_for_model = list(idx_for_model)

model = sm.formula.ols(formula, data=metainfo, subset=idx_for_model).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:             gWAT_rho_N   R-squared:                       0.109
# Model:                            OLS   Adj. R-squared:                  0.004
# Method:                 Least Squares   F-statistic:                     1.034
# Date:                Fri, 21 Feb 2020   Prob (F-statistic):              0.418
# Time:                        17:56:47   Log-Likelihood:                -1022.2
# No. Observations:                  67   AIC:                             2060.
# Df Residuals:                      59   BIC:                             2078.
# Df Model:                           7
# Covariance Type:            nonrobust
# ===============================================================================================================================
#                                                                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                    3.616e+06   3.85e+05      9.382      0.000    2.84e+06    4.39e+06
# C(sex)[T.m]                                                 -6.021e+05    5.3e+05     -1.137      0.260   -1.66e+06    4.58e+05
# C(ko_parent)[T.MAT]                                         -3.454e+05    5.3e+05     -0.652      0.517   -1.41e+06    7.15e+05
# C(genotype)[T.KLF14-KO:Het]                                 -2.249e+04   5.64e+05     -0.040      0.968   -1.15e+06    1.11e+06
# C(sex)[T.m]:C(ko_parent)[T.MAT]                              5.197e+04   7.38e+05      0.070      0.944   -1.42e+06    1.53e+06
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                     -4.838e+05   7.88e+05     -0.614      0.541   -2.06e+06    1.09e+06
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]             -1.668e+05   7.63e+05     -0.219      0.828   -1.69e+06    1.36e+06
# C(sex)[T.m]:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]   1.36e+06   1.07e+06      1.269      0.209   -7.84e+05     3.5e+06
# ==============================================================================
# Omnibus:                        5.089   Durbin-Watson:                   1.635
# Prob(Omnibus):                  0.079   Jarque-Bera (JB):                4.244
# Skew:                           0.583   Prob(JB):                        0.120
# Kurtosis:                       3.402   Cond. No.                         18.4
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

print(model.pvalues)

# ANOVA table
aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)
