"""
Compare the number of blocks with a uniform tiling vs. the adaptive block algorithm
"""

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])

import matplotlib.pyplot as plt
import glob
import cytometer.data
import openslide
import numpy as np
from shapely.geometry import Polygon

DEBUG = False

histology_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
area2quantile_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v7')
annotations_dir = os.path.join(home, 'Data/cytometer_data/aida_data_Klf14/annotations')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')

# file with area->quantile map precomputed from all automatically segmented slides in klf14_b6ntac_exp_0098_full_slide_size_analysis_v7.py
filename_area2quantile = os.path.join(area2quantile_dir, 'klf14_b6ntac_exp_0098_filename_area2quantile.npz')

# suffixes of annotation filenames
auto_filename_suffix = '_exp_0097_no_overlap.json'
corrected_filename_suffix = '_exp_0097_corrected.json'

# list of annotations
auto_annotation_files_list = os.path.join(annotations_dir, 'KLF14*' + auto_filename_suffix)
auto_annotation_files_list = glob.glob(auto_annotation_files_list)
corrected_annotation_files_list = os.path.join(annotations_dir, 'KLF14*' + corrected_filename_suffix)
corrected_annotation_files_list = glob.glob(corrected_annotation_files_list)

# full resolution image window and network expected receptive field parameters
# fullres_box_size = np.array([2751, 2751])  # this is different in different images
receptive_field = np.array([131, 131])
receptive_field_len = receptive_field[0]

# largest cell area that we need to account for
max_cell_area = 100e3  # pixel^2
max_cell_radius = np.ceil(np.sqrt(max_cell_area / np.pi)).astype(np.int)  # pixels, assuming the cell is a circle

# rough mask parameters
downsample_factor = 8.0

# overlap between blocks in uniform tiling:
# each two overlapping blocks can be seen as having 3 parts:
#    *********************.....OOO
#                      OOO.....*********************
#
# where *: pixel in rest of the block
#       .: pixel in a region big enough to accommodate diameter of largest cell (radius R)
#       O: pixel in a region with size ERF / 2 (ERF: Effective Receptive Field)
#
# Thus, if a block has size L, the start of the 2nd block (we use this to generate the grid of uniform tiles) is
#
#       L - ERF - 2*R

########################################################################################################################
## Colourmap for AIDA
########################################################################################################################

if os.path.isfile(filename_area2quantile):
    with np.load(filename_area2quantile, allow_pickle=True) as aux:
        f_area2quantile_f = aux['f_area2quantile_f'].item()
        f_area2quantile_m = aux['f_area2quantile_m'].item()
else:
    raise FileNotFoundError('Cannot find file with area->quantile map precomputed from all automatically segmented' +
                            ' slides in klf14_b6ntac_exp_0098_full_slide_size_analysis_v7.py')

# # load AIDA's colourmap
# cm = cytometer.data.aida_colourmap()

########################################################################################################################
## Process files for segmentation refinement
########################################################################################################################

# init lists of block sum area for each file
area_all_adaptive_images = []
area_all_uniform_images = []

for annotation_file in auto_annotation_files_list:

    print('File: ' + os.path.basename(annotation_file))

    # name of the original .ndpi file
    histo_file = os.path.basename(annotation_file).replace(auto_filename_suffix, '.ndpi')
    histo_file = histo_file.replace(corrected_filename_suffix, '.ndpi')
    histo_file = os.path.join(histology_dir, histo_file)

    im = openslide.OpenSlide(histo_file)
    xres = 1e-2 / float(im.properties['tiff.XResolution'])
    yres = 1e-2 / float(im.properties['tiff.YResolution'])

    # load blocks from annotation file
    # each block is given as the 4-points of a recontagle, with the first point repeated at the end to close the
    # rectangle
    blocks = cytometer.data.aida_get_contours(annotation_file, layer_name='Blocks.*')

    # initialise total blocks area for the adaptive algorithm blocks
    area_all_adaptive_blocks = []
    height_all_adaptive_blocks = []
    width_all_adaptive_blocks = []

    if DEBUG:
        from matplotlib.patches import Rectangle
        cmap = plt.cm.tab20b

        fig = plt.figure()
        plt.clf()
        plt.imshow(lores_istissue0)
        colors = list(np.remainder(range(len(x0_tiles) * len(y0_tiles)), 20) / 19)
        plt.axis('off')
        plt.tight_layout()

    for block in blocks:

        # we could compute the area of the block as the product of the rectangle's sides, but if we use Polygon().area
        # the code is more compact
        pol = Polygon(block)
        area_all_adaptive_blocks.append(pol.area)

        # get height and width of each block
        # remember that x0, xend are in python format, i.e. [0, 3] means that the rectangle goes from 0 to 2 inclusive
        height_all_adaptive_blocks.append(pol.bounds[3] - pol.bounds[1])
        width_all_adaptive_blocks.append(pol.bounds[2] - pol.bounds[0])

        if DEBUG:
            # block coordinates in the low resolution rough mask
            x0_tile_lores = np.int(pol.bounds[0] / downsample_factor)
            y0_tile_lores = np.int(pol.bounds[1] / downsample_factor)
            xend_tile_lores = np.int(pol.bounds[2] / downsample_factor)
            yend_tile_lores = np.int(pol.bounds[3] / downsample_factor)

            # plot the block from the uniform tiling
            rect = Rectangle((x0_tile_lores, y0_tile_lores),
                             xend_tile_lores - x0_tile_lores,
                             yend_tile_lores - y0_tile_lores,
                             fill=True, color=cmap(colors.pop()), alpha=0.5)
            fig.axes[0].add_patch(rect)

    # sum the areas of all blocks to get a total "processed area"
    area_all_adaptive_images.append(np.sum(area_all_adaptive_blocks))

    # we are going to use the largest block dimensions in the image as the block size for the uniform tiling
    block_len = np.max(height_all_adaptive_blocks + width_all_adaptive_blocks).astype(np.int)

    # uniform tiling
    # start of second tile = spacing between tiles' first pixels
    block_spacing = np.int(block_len - receptive_field_len - 2 * max_cell_radius)
    if block_spacing <= 0:
        raise RuntimeError('Block spacing is <= 0 in ' + annotation_file)
    x0_tiles = np.array(range(0, im.dimensions[0], block_spacing))
    y0_tiles = np.array(range(0, im.dimensions[1], block_spacing))
    if DEBUG:
        print('Number of uniform tiles = ' + str(len(x0_tiles) * len(y0_tiles)))

    # block ends in python notation: x0=0, xend=3 means that the window covers 0, 1, 2
    xend_tiles = x0_tiles + block_len
    xend_tiles[xend_tiles > im.dimensions[0]] = im.dimensions[0]
    yend_tiles = y0_tiles + block_len
    yend_tiles[yend_tiles > im.dimensions[1]] = im.dimensions[1]

    # name of file to save rough mask, current mask, and time steps
    rough_mask_file = os.path.basename(histo_file)
    rough_mask_file = rough_mask_file.replace('.ndpi', '_rough_mask.npz')
    rough_mask_file = os.path.join(annotations_dir, rough_mask_file)

    # load rough mask
    with np.load(rough_mask_file) as aux:
        lores_istissue0 = aux['lores_istissue0']

    if DEBUG:
        from matplotlib.patches import Rectangle
        cmap = plt.cm.tab20b

        fig = plt.figure()
        plt.clf()
        plt.imshow(lores_istissue0)
        colors = list(np.remainder(range(len(x0_tiles) * len(y0_tiles)), 20) / 19)
        plt.axis('off')
        plt.tight_layout()

    # initialise total blocks area for the uniform algorithm blocks
    area_all_uniform_blocks = []

    # loop uniform tiling blocks
    for x0_tile, xend_tile in zip(x0_tiles, xend_tiles):
        for y0_tile, yend_tile in zip(y0_tiles, yend_tiles):

            # block coordinates in the low resolution rough mask
            x0_tile_lores = np.int(x0_tile / downsample_factor)
            y0_tile_lores = np.int(y0_tile / downsample_factor)
            xend_tile_lores = np.int(xend_tile / downsample_factor)
            yend_tile_lores = np.int(yend_tile / downsample_factor)

            # check which uniform blocks contain pixels to process
            tile_lores_istissue0 = lores_istissue0[y0_tile_lores:yend_tile_lores, x0_tile_lores:xend_tile_lores]
            if np.any(tile_lores_istissue0):
                area_all_uniform_blocks.append((xend_tile - x0_tile) * (yend_tile - y0_tile))

                if DEBUG:
                    # plot the block from the uniform tiling
                    rect = Rectangle((x0_tile_lores, y0_tile_lores),
                                     xend_tile_lores - x0_tile_lores,
                                     yend_tile_lores - y0_tile_lores,
                                     fill=True, color=cmap(colors.pop()), alpha=0.5)
                    fig.axes[0].add_patch(rect)

    # sum the areas of all blocks to get a total "processed area"
    area_all_uniform_images.append(np.sum(area_all_uniform_blocks))

# plot uniform vs. adaptive areas
plt.clf()
plt.scatter(np.array(area_all_uniform_images) * 1e-9, np.array(area_all_adaptive_images) * 1e-9)
plt.plot([0, 5], [0, 5], 'C1', linewidth=2)
plt.xlabel('Processed area with uniform tiling ($\cdot 10^9$ pixel$^2$)', fontsize=14)
plt.ylabel('Processed area with adaptive tiling\n($\cdot 10^9$ pixel$^2$)', fontsize=14)
plt.tick_params(labelsize=14)
plt.tight_layout()

plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0105_uniform_vs_adaptive_tiling.png'),
            bbox_inches='tight')

print('Processed area adaptive / uniform ratio: '
      + str(np.mean(np.array(area_all_adaptive_images) / np.array(area_all_uniform_images)))
      + ' ± '
      + str(np.std(1 - np.array(area_all_adaptive_images) / np.array(area_all_uniform_images))))
print('Processed area reduced by: '
      + str(1 - np.mean(np.array(area_all_adaptive_images) / np.array(area_all_uniform_images))))
