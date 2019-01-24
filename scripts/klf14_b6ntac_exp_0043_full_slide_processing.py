# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os

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


DEBUG = False
SAVE_FIGS = False

root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
data_dir = os.path.join(home, root_data_dir, 'Maz Yon')
training_dir = os.path.join(home, root_data_dir, 'klf14_b6ntac_training')
seg_dir = os.path.join(home, root_data_dir, 'klf14_b6ntac_seg')
figures_dir = os.path.join(root_data_dir, 'figures')

# full resolution image window and network expected receptive field parameters
fullres_box_size = np.array([1001, 1001])
receptive_field = 131

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
    seg, im_downsampled = rough_foreground_mask(file, downsample_factor=downsample_factor, dilation_size=dilation_size,
                                                component_size_threshold=component_size_threshold, return_im=True)

    # # save segmentation as a tiff file (with ZLIB compression)
    # outfilename = os.path.basename(file)
    # outfilename = os.path.splitext(outfilename)[0] + '_seg'
    # outfilename = os.path.join(seg_dir, outfilename + '.tif')
    # tifffile.imsave(outfilename, seg,
    #                 compress=9,
    #                 resolution=(int(im.properties["tiff.XResolution"]) / downsample_factor,
    #                             int(im.properties["tiff.YResolution"]) / downsample_factor,
    #                             im.properties["tiff.ResolutionUnit"].upper()))

    # number of blocks. We want a number of blocks that will produce approximately blocks of size
    # 1001x1001 in the full resolution image
    nblocks = np.floor(np.array(seg.shape) / block_len).astype(np.int)

    # split downsampled segmentation into overlapping blocks
    block_slices, blocks, _ = block_split(seg, nblocks=nblocks, pad_width=block_overlap,
                                          mode='constant', constant_values=0)

    if DEBUG:
        # copy of blocks to display selected blocks
        blocks_filled = blocks.copy()

    # blocks that contain some tissue
    has_tissue = np.zeros(shape=(len(blocks)), dtype=np.bool)
    for block_i in range(len(blocks)):
        has_tissue[block_i] = np.any(blocks[block_i])
        if DEBUG and has_tissue[block_i]:
            blocks_filled[block_i].fill(1)

    if DEBUG:
        # reassemble blocks
        seg_filled, _ = block_stack(blocks_filled, block_slices, pad_width=block_overlap)

        plt.clf()

        plt.subplot(221)
        plt.imshow(im_downsampled)
        plt.title('Histology')

        plt.subplot(222)
        plt.imshow(seg)
        plt.title('Rough tissue segmentation')

        plt.subplot(223)
        plt.imshow(seg_filled)
        plt.title('Blocks to be processed by pipeline')

        plt.subplot(224)
        plt.imshow(imfuse(seg, seg_filled * 255))
        plt.title('Overlap of segmentation and blocks')

        if SAVE_FIGS:
            plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0043_blocks_for_pipeline.png'))

    print('Selected blocks: ' + "{:.1f}".format(100 * np.count_nonzero(has_tissue) / len(blocks_filled)) + '%')
