# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import cytometer.utils

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

os.environ['KERAS_BACKEND'] = 'tensorflow'

import openslide
import numpy as np
from statistics import mode
import matplotlib.pyplot as plt
import cv2
from random import randint, seed
import tifffile
import glob
from cytometer.utils import rough_foreground_mask
from cytometer.data import append_paths_to_aida_json_file
import PIL
from pysto.imgproc import block_split, block_stack, imfuse
import tensorflow as tf
import keras
from skimage.measure import find_contours

# limit GPU memory used
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
figures_dir = os.path.join(root_data_dir, 'figures')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')
annotations_dir = os.path.join(home, 'Software/AIDA/dist/data/annotations')

saved_contour_model_basename = 'klf14_b6ntac_exp_0034_cnn_contour'
saved_dmap_model_basename = 'klf14_b6ntac_exp_0035_cnn_dmap'
# saved_quality_model_basename = 'klf14_b6ntac_exp_0040_cnn_qualitynet_thresholded_sigmoid_masked_segmentation'
# saved_quality_model_basename = 'klf14_b6ntac_exp_0041_cnn_qualitynet_thresholded_sigmoid_pm_1_masked_segmentation'
saved_quality_model_basename = 'klf14_b6ntac_exp_0042_cnn_qualitynet_thresholded_sigmoid_pm_1_band_masked_segmentation'

contour_model_name = saved_contour_model_basename + '*.h5'
dmap_model_name = saved_dmap_model_basename + '*.h5'
quality_model_name = saved_quality_model_basename + '*.h5'

# full resolution image window and network expected receptive field parameters
fullres_box_size = np.array([1001, 1001])
receptive_field = np.array([131, 131])

# rough_foreground_mask() parameters
downsample_factor = 8.0
dilation_size = 25
component_size_threshold = 1e5

# block_split() parameters in downsampled image
block_len = np.ceil((fullres_box_size - receptive_field) / downsample_factor)
block_overlap = np.ceil((receptive_field - 1) / 2 / downsample_factor).astype(np.int)

files_list = glob.glob(os.path.join(data_dir, 'KLF14*.ndpi'))

# trained models for all folds
contour_model_files = sorted(glob.glob(os.path.join(saved_models_dir, contour_model_name)))
dmap_model_files = sorted(glob.glob(os.path.join(saved_models_dir, dmap_model_name)))
quality_model_files = sorted(glob.glob(os.path.join(saved_models_dir, quality_model_name)))

# select the models that correspond to current fold
fold_i = 0
contour_model_file = contour_model_files[fold_i]
dmap_model_file = dmap_model_files[fold_i]
quality_model_file = quality_model_files[fold_i]

# load models
contour_model = keras.models.load_model(contour_model_file)
dmap_model = keras.models.load_model(dmap_model_file)
quality_model = keras.models.load_model(quality_model_file)

# "KLF14-B6NTAC-MAT-18.2b  58-16 B3 - 2016-02-03 11.01.43.ndpi"
# file_i = 10; file = files_list[file_i]
# "KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38.ndpi"
# file_i = 331; file = files_list[file_i]
for file_i, file in enumerate(files_list):

    print('File ' + str(file_i) + '/' + str(len(files_list)) + ': ' + file)

    # name of the annotations file
    annotations_file = os.path.basename(file)
    annotations_file = os.path.splitext(annotations_file)[0]
    annotations_file = os.path.join(annotations_dir, annotations_file + '.json')

    # rough segmentation of the tissue in the image
    lores_istissue0, im_downsampled = rough_foreground_mask(file, downsample_factor=downsample_factor, dilation_size=dilation_size,
                                                            component_size_threshold=component_size_threshold, return_im=True)
    lores_istissue0 = (lores_istissue0 > 0).astype(np.uint8)

    if DEBUG:
        plt.clf()
        plt.subplot(121)
        plt.imshow(im_downsampled)
        plt.subplot(122)
        plt.imshow(lores_istissue0)

    # segmentation copy, to keep track of what's left to do
    lores_istissue = lores_istissue0.copy()

    # # save segmentation as a tiff file (with ZLIB compression)
    # outfilename = os.path.basename(file)
    # outfilename = os.path.splitext(outfilename)[0] + '_seg'
    # outfilename = os.path.join(seg_dir, outfilename + '.tif')
    # tifffile.imsave(outfilename, seg,
    #                 compress=9,
    #                 resolution=(int(im.properties["tiff.XResolution"]) / downsample_factor,
    #                             int(im.properties["tiff.YResolution"]) / downsample_factor,
    #                             im.properties["tiff.ResolutionUnit"].upper()))

    # open full resolution histology slide
    im = openslide.OpenSlide(file)

    # keep extracting histology windows until we have finished
    step = 0
    while np.count_nonzero(lores_istissue) > 0:

        step += 1

        # get indices for the next histology window to process
        (first_row, last_row, first_col, last_col), \
        (lores_first_row, lores_last_row, lores_first_col, lores_last_col) = \
            cytometer.utils.get_next_roi_to_process(lores_istissue, downsample_factor=downsample_factor,
                                                    max_window_size=[1000, 1000],
                                                    border=np.round((receptive_field-1)/2))

        # # DEBUG
        # first_row = int(3190 * downsample_factor)
        # last_row = first_row + 1001
        # first_col = int(3205 * downsample_factor)
        # last_col = first_col + 1001
        # lores_first_row = 3190
        # lores_last_row = lores_first_row + int(np.round(1001 / downsample_factor))
        # lores_first_col = 3205
        # lores_last_col = lores_first_col + int(np.round(1001 / downsample_factor))

        # load window from full resolution slide
        tile = im.read_region(location=(first_col, first_row), level=0,
                              size=(last_col - first_col, last_row - first_row))
        tile = np.array(tile)
        tile = tile[:, :, 0:3]
        tile = np.reshape(tile, (1,) + tile.shape)
        tile = tile.astype(np.float32)
        tile /= 255

        # interpolate coarse tissue segmentation to full resolution
        istissue_tile = lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col]
        istissue_tile = PIL.Image.fromarray(istissue_tile)
        istissue_tile = istissue_tile.resize((last_col - first_col, last_row - first_row), resample=PIL.Image.NEAREST)
        istissue_tile = np.array(istissue_tile)
        istissue_tile = np.reshape(istissue_tile, newshape=(1,) + istissue_tile.shape)
        istissue_tile = (istissue_tile != 0).astype(np.uint8)

        # segment histology
        labels, labels_info = cytometer.utils.segmentation_pipeline(tile,
                                                                    contour_model, dmap_model, quality_model,
                                                                    quality_model_type='-1_1_band',
                                                                    mask=istissue_tile,
                                                                    smallest_cell_area=804)

        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(tile[0, :, :, :])
            plt.title('Histology', fontsize=16)
            plt.subplot(222)
            plt.imshow(tile[0, :, :, :])
            plt.contour(labels[0, :, :, 0], levels=labels_info['label'], colors='blue', linewidths=1)
            plt.title('Full segmentation', fontsize=16)
            plt.subplot(223)
            plt.boxplot(labels_info['quality'])
            plt.tick_params(labelbottom=False, bottom=False)
            plt.title('Quality values', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.subplot(224)
            aux = cytometer.utils.paint_labels(labels, labels_info['label'], labels_info['quality'] >= 0.9)
            plt.imshow(tile[0, :, :, :])
            plt.contour(aux[0, :, :, 0] * labels[0, :, :, 0],
                        levels=labels_info['label'], colors='blue', linewidths=1)
            plt.title('Labels with quality >= 0.9', fontsize=16)

        # list of cells that are on the edges
        edge_labels = cytometer.utils.edge_labels(labels[0, :, :, 0])

        # list of cells that are not on the edges
        non_edge_labels = np.setdiff1d(labels_info['label'], edge_labels)

        # list of cells that are OK'ed by quality network
        good_labels = labels_info['label'][labels_info['quality'] >= 0.9]

        # remove edge cells from the list of good cells
        good_labels = np.setdiff1d(good_labels, edge_labels)

        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(tile[0, :, :, :])
            plt.title('Histology', fontsize=16)
            plt.subplot(222)
            plt.imshow(tile[0, :, :, :])
            plt.contour(labels[0, :, :, 0], levels=labels_info['label'], colors='blue', linewidths=1)
            plt.title('Full segmentation', fontsize=16)
            plt.subplot(223)
            plt.boxplot(labels_info['quality'])
            plt.tick_params(labelbottom=False, bottom=False)
            plt.title('Quality values', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.subplot(224)
            aux = cytometer.utils.paint_labels(labels, labels_info['label'], np.isin(labels_info['label'], good_labels))
            plt.imshow(tile[0, :, :, :])
            plt.contour(aux[0, :, :, 0] * labels[0, :, :, 0],
                        levels=labels_info['label'], colors='blue', linewidths=1)
            plt.title('Labels with quality >= 0.9', fontsize=16)

        # # DEBUG
        # # remove ROI from segmentation
        # seg[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0
        #
        # if DEBUG:
        #     plt.clf()
        #     plt.subplot(121)
        #     plt.imshow(seg)
        #     plt.plot([lores_first_col, lores_last_col, lores_last_col, lores_first_col, lores_first_col],
        #              [lores_last_row, lores_last_row, lores_first_row, lores_first_row, lores_last_row], 'r')
        #     plt.xlim(700, 1500)
        #     plt.ylim(650, 300)
        #     plt.subplot(122)
        #     plt.imshow(imfuse(lores_istissue0, seg))
        #     plt.plot([lores_first_col, lores_last_col, lores_last_col, lores_first_col, lores_first_col],
        #              [lores_last_row, lores_last_row, lores_first_row, lores_first_row, lores_last_row], 'r')
        #     plt.xlim(700, 1500)
        #     plt.ylim(650, 300)
        #
        # if SAVE_FIGS:
        #     plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0043_get_next_roi_to_process_' +
        #                              str(step).zfill(2) + '.png'))

        # mark all edge cells as "to do"
        todo_labels = np.isin(labels[0, :, :, 0], edge_labels) * istissue_tile

        # downsample "to do"
        lores_todo_labels = PIL.Image.fromarray(todo_labels[0, :, :])
        lores_todo_labels = lores_todo_labels.resize((lores_last_col - lores_first_col,
                                                      lores_last_row - lores_first_row),
                                                     resample=PIL.Image.NEAREST)
        lores_todo_labels = np.array(lores_todo_labels)

        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col])
            plt.title('Low res tissue mask', fontsize=16)
            plt.subplot(222)
            plt.imshow(istissue_tile[0, :, :])
            plt.title('Full res tissue mask', fontsize=16)
            plt.subplot(223)
            plt.imshow(todo_labels[0, :, :])
            plt.title('Full res left over tissue', fontsize=16)
            plt.subplot(224)
            plt.imshow(lores_todo_labels)
            plt.title('Low res left over tissue', fontsize=16)

        # update the tissue segmentation mask with the current window
        lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = lores_todo_labels

        if DEBUG:
            plt.clf()
            plt.imshow(tile[0, :, :, :])
            plt.title('Histology', fontsize=16)

        # convert labels to contours (points) using marching squares
        # http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.find_contours
        contours = []
        for lab in good_labels:
            # load (row, col) coordinates
            aux = find_contours(labels[0, :, :, 0] == lab, 0.5,
                                fully_connected='low', positive_orientation='low')[0]
            # convert to (x, y) coordinates
            aux = aux[:, [1, 0]]
            if DEBUG:
                plt.plot(aux[:, 0], aux[:, 1])
            # add window offset
            aux[:, 0] = aux[:, 0] + first_col
            aux[:, 1] = aux[:, 1] + first_row
            # add to the list of contours
            contours.append(aux)

        # add segmented contours to annotations file
        append_paths_to_aida_json_file(annotations_file, contours)

