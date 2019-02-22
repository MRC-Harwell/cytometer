# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import cytometer.utils

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

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
from pysto.imgproc import block_split, block_stack, imfuse
import tensorflow as tf
import keras

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

saved_contour_model_basename = 'klf14_b6ntac_exp_0034_cnn_contour'  # contour
saved_dmap_model_basename = 'klf14_b6ntac_exp_0035_cnn_dmap'  # dmap

contour_model_name = saved_contour_model_basename + '*.h5'
dmap_model_name = saved_dmap_model_basename + '*.h5'

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

# select the models that correspond to current fold
fold_i = 0
contour_model_file = contour_model_files[fold_i]
dmap_model_file = dmap_model_files[fold_i]

# load models
contour_model = keras.models.load_model(contour_model_file)
dmap_model = keras.models.load_model(dmap_model_file)

# file_i = 10; file = files_list[file_i]
# "KLF14-B6NTAC-MAT-18.2b  58-16 B3 - 2016-02-03 11.01.43.ndpi"
for file_i, file in enumerate(files_list):

    print('File ' + str(file_i) + '/' + str(len(files_list)) + ': ' + file)

    # rough segmentation of the tissue in the image
    seg0, im_downsampled = rough_foreground_mask(file, downsample_factor=downsample_factor, dilation_size=dilation_size,
                                                 component_size_threshold=component_size_threshold, return_im=True)

    if DEBUG:
        plt.clf()
        plt.subplot(121)
        plt.imshow(im_downsampled)
        plt.subplot(122)
        plt.imshow(seg0)

    # segmentation copy, to keep track of what's left to do
    seg = seg0.copy()

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
    while np.count_nonzero(seg) > 0:

        step += 1

        # get indices for the next histology window to process
        (first_row, last_row, first_col, last_col), \
        (lores_first_row, lores_last_row, lores_first_col, lores_last_col) = \
            cytometer.utils.get_next_roi_to_process(seg, downsample_factor=downsample_factor,
                                                    max_window_size=[1000, 1000],
                                                    border=np.round((receptive_field-1)/2))

        # load window from full resolution slide
        tile = im.read_region(location=(first_col, first_row), level=0,
                              size=(last_col - first_col, last_row - first_row))
        tile = np.array(tile)
        tile = tile[:, :, 0:3]

        # set input layer to size of images
        contour_model = cytometer.models.change_input_size(contour_model, batch_shape=(None,) + tile.shape)
        dmap_model = cytometer.models.change_input_size(dmap_model, batch_shape=(None,) + tile.shape)

        # run histology image through network
        contour_pred = contour_model.predict(tile.reshape((1,) + tile.shape))
        dmap_pred = dmap_model.predict(tile.reshape((1,) + tile.shape))

        # cell segmentation
        labels, labels_borders \
            = cytometer.utils.segment_dmap_contour(dmap_pred[0, :, :, 0],
                                                   contour=contour_pred[0, :, :, 0],
                                                   border_dilation=0)

        # remove ROI from segmentation
        seg[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0

        if DEBUG:
            plt.clf()
            plt.subplot(121)
            plt.imshow(seg)
            plt.plot([lores_first_col, lores_last_col, lores_last_col, lores_first_col, lores_first_col],
                     [lores_last_row, lores_last_row, lores_first_row, lores_first_row, lores_last_row], 'r')
            plt.xlim(700, 1500)
            plt.ylim(650, 300)
            plt.subplot(122)
            plt.imshow(imfuse(seg0, seg))
            plt.plot([lores_first_col, lores_last_col, lores_last_col, lores_first_col, lores_first_col],
                     [lores_last_row, lores_last_row, lores_first_row, lores_first_row, lores_last_row], 'r')
            plt.xlim(700, 1500)
            plt.ylim(650, 300)

        if SAVE_FIGS:
            plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0043_get_next_roi_to_process_' +
                                     str(step).zfill(2) + '.png'))
