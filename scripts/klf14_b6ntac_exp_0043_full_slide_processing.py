# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import cytometer.utils

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2'

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


DEBUG = False
SAVE_FIGS = False

root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
data_dir = os.path.join(home, root_data_dir, 'Maz Yon')
training_dir = os.path.join(home, root_data_dir, 'klf14_b6ntac_training')
seg_dir = os.path.join(home, root_data_dir, 'klf14_b6ntac_seg')
figures_dir = os.path.join(root_data_dir, 'figures')

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

files_list = glob.glob(os.path.join(data_dir, '*.ndpi'))

# file_i = 10; file = files_list[file_i]
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

    # keep extracting histology windows until we have finished
    while np.count_nonzero(seg) > 0:

        # get indices for the next histology window to process
        (first_row, last_row, first_col, last_col), \
        (lores_first_row, lores_last_row, lores_first_col, lores_last_col) = \
            cytometer.utils.get_next_roi_to_process(seg, downsample_factor=downsample_factor,
                                                    max_window_size=[1000, 1000],
                                                    border=np.round((receptive_field-1)/2))

        # remove ROI from segmentation
        seg[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0

        if DEBUG:
            plt.clf()
            plt.subplot(121)
            plt.imshow(im_downsampled)
            plt.subplot(122)
            plt.imshow(imfuse(seg0, seg))
            plt.plot([lores_first_col, lores_last_col, lores_last_col, lores_first_col, lores_first_col],
                     [lores_last_row, lores_last_row, lores_first_row, lores_first_row, lores_last_row], 'r')
