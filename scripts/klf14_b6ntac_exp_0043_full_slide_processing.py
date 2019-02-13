# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'

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

    if DEBUG:
        plt.clf()
        plt.subplot(121)
        plt.imshow(im_downsampled)
        plt.subplot(122)
        plt.imshow(seg)

    # # save segmentation as a tiff file (with ZLIB compression)
    # outfilename = os.path.basename(file)
    # outfilename = os.path.splitext(outfilename)[0] + '_seg'
    # outfilename = os.path.join(seg_dir, outfilename + '.tif')
    # tifffile.imsave(outfilename, seg,
    #                 compress=9,
    #                 resolution=(int(im.properties["tiff.XResolution"]) / downsample_factor,
    #                             int(im.properties["tiff.YResolution"]) / downsample_factor,
    #                             im.properties["tiff.ResolutionUnit"].upper()))


# downsample_factor=8.0
# max_window_size=(1000, 1000)
# receptive_field=(130, 130)
def foo(seg, downsample_factor=1.0, max_window_size=(1000, 1000), receptive_field=(130, 130)):
    """

    :param seg:
    :param max_window_size: this is the largest (row, col) image size that be can pass to the neural network (usually,
    due to GPU memory limitations).
    :param receptive_field:
    :return:
    """

    # approximate measures in the downsampled image
    lores_max_window_size = (max_window_size[0] / downsample_factor, max_window_size[1] / downsample_factor)
    lores_receptive_field = (receptive_field[0] / downsample_factor, receptive_field[1] / downsample_factor)

    # maximum size of the processing window without the overlapping edges
    # max_window_size_no_borders = (max_window_size[0] - receptive_field[0], max_window_size[1] - receptive_field[1])

    # for convenience, receptive_field/2
    lores_receptive_field_2 = (np.ceil(lores_receptive_field[0] / 2.0), np.ceil(lores_receptive_field[1] / 2.0))

    # rows with segmentation pixels
    seg_rows = np.any(seg, axis=1)

    # place the vertical side of a window without borders at the top
    first_row = np.where(seg_rows)[0]
    if len(first_row) == 0:
        return
    first_row = first_row[0]

    # add a border on top of the window to account for receptive field
    first_row = np.int(np.max([0, first_row - lores_receptive_field_2[0]]))

    # bottom of the window (we follow the python convention that e.g. 3:5 = [3, 4], i.e. last_row not included in window)
    last_row = first_row + max_window_size[0]
    last_row = np.min([len(seg_rows), last_row])
    last_row = np.int(last_row)

    # columns with segmentation pixels, within the vertical range of the window, not the whole image
    seg_cols = np.any(seg[first_row:last_row, :], axis=0)

    # place the horizontal side of a window without borders at the left
    first_col = np.where(seg_cols)[0]
    if len(first_col) == 0:
        return
    first_col = first_col[0]

    # add a border on top of the window to account for receptive field
    first_col = np.int(np.max([0, first_col - receptive_field_2[1]]))

    # right side of the window (we follow the same python convention as for rows above
    last_col = first_col + max_window_size[1]
    last_col = np.min([len(seg_cols), last_col])
    last_col = np.int(last_col)

    # Note: at this point, the window with borders = seg[first_row:last_row, first_col:last_col]

    # scale


    ## OLD CODE

    # number of blocks. We want a number of blocks that will produce approximately blocks of size
    # 1001x1001 in the full resolution image
    nblocks = np.floor(np.array(seg.shape) / block_len).astype(np.int)

    # split downsampled segmentation into overlapping blocks
    block_slices, blocks, _ = block_split(seg, nblocks=nblocks, pad_width=block_overlap,
                                          mode='constant', constant_values=0)

    # copy of blocks to display selected blocks
    blocks_filled = blocks.copy()

    # blocks that contain some tissue
    has_tissue = np.zeros(shape=(len(blocks)), dtype=np.bool)
    for block_i in range(len(blocks)):
        has_tissue[block_i] = np.any(blocks[block_i])
        if has_tissue[block_i]:
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

    # convert slice coordinates to full resolution
    fullres_block_slices = []
    for slice_i, (slice_row, slice_col) in enumerate(block_slices):
        # row-slice
        fullres_block_slices.append([
            slice(
                int(slice_row.start * downsample_factor),
                int(slice_row.stop * downsample_factor),
                slice_row.step),
            slice(
                int(slice_col.start * downsample_factor),
                int(slice_col.stop * downsample_factor),
                slice_col.step)])
