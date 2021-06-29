"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0002_paper_figures'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])

DEBUG = False
SAVE_FIGS = False

import matplotlib.pyplot as plt
import cytometer.data
import cytometer.stats
import shapely
import scipy
# import scipy.stats as stats
import numpy as np
# import sklearn.neighbors, sklearn.model_selection
import pandas as pd
from PIL import Image, ImageDraw
import openslide

# directories
root_data_dir = os.path.join(home, 'Data/cytometer_data/arl15del2')
annotations_dir = os.path.join(home, 'Data/cytometer_data/aida_data_Arl15_del2/annotations')
histo_dir = os.path.join(home, 'scan_srv2_cox/Ying Bai/For Ramon')
dataframe_dir = os.path.join(home, 'GoogleDrive/Research/20210628_arl15del2_paper')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20210628_arl15del2_paper/figures')
metainfo_dir = os.path.join(home, 'Data/cytometer_data/arl15del2')
area2quantile_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v8')

# list of annotation files: This files have already been filtered
json_annotation_files = [
    'APL15-DEL2-EM1-B6N 31.1a 696-19 Gwat 1 - 2019-08-19 12.44.49_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 31.1a 696-19 Gwat 3 - 2019-08-19 12.57.53_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 31.1a 696-19 Gwat 6 - 2019-08-19 13.16.35_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 31.1a 696-19 Iwat 1 - 2019-08-19 13.54.09_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 31.1a 696-19 Iwat 3 - 2019-08-19 15.29.54_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 31.1a 696-19 Iwat 6 - 2019-08-19 15.52.25_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 32.2b 695-19 Gwat 1 - 2019-08-16 13.04.34_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 32.2b 695-19 Gwat 3 - 2019-08-16 13.18.24_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 32.2b 695-19 Gwat 6 - 2019-08-16 13.37.00_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 32.2b 695-19 Iwat 1 - 2019-08-19 12.01.13_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 32.2b 695-19 Iwat 3 - 2019-08-19 12.13.50_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 32.2b 695-19 Iwat 6 - 2019-08-19 12.31.59_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 34.1b 693-19 Gwat 1 - 2019-08-15 11.11.40_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 34.1b 693-19 Gwat 3 - 2019-08-15 11.24.16_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 34.1b 693-19 Gwat 6 - 2019-08-15 11.45.59_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 34.1b 693-19 Iwat 1 - 2019-08-15 11.55.14_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 34.1b 693-19 Iwat 3 - 2019-08-15 13.31.37_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 34.1b 693-19 Iwat 6 - 2019-08-15 12.22.16_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 35.2b 701-19 Gwat 1 - 2019-08-22 09.47.48_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 35.2b 701-19 Gwat 3 - 2019-08-22 10.59.34_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 35.2b 701-19 Gwat 6 - 2019-08-22 10.15.02_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 35.2b 701-19 Iwat 1 - 2019-08-22 11.12.17_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 35.2b 701-19 Iwat 3 - 2019-08-22 11.24.19_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 35.2b 701-19 Iwat 6 - 2019-08-22 11.41.44_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 35.2c 694-19 Gwat 1 - 2019-08-15 13.41.17_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 35.2c 694-19 Gwat 3 - 2019-08-15 13.48.15_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 35.2c 694-19 Gwat 6 - 2019-08-15 14.03.44_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 35.2c 694-19 Iwat 1 - 2019-08-16 11.13.41_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 35.2c 694-19 Iwat 3 - 2019-08-16 11.25.11_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 35.2c 694-19 Iwat 6 - 2019-08-16 11.42.18_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 37.1b 697-19 Gwat 1 - 2019-08-20 11.40.02_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 37.1b 697-19 Gwat 3 - 2019-08-20 12.44.02_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 37.1b 697-19 Gwat 6 - 2019-08-20 13.03.14_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 37.1b 697-19 Iwat 1 - 2019-08-20 14.08.10_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 37.1b 697-19 Iwat 3 - 2019-08-20 14.22.06_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 37.1b 697-19 Iwat 6 - 2019-08-20 14.41.48_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 38.1c 700-19 Gwat 1 - 2019-08-21 11.59.38_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 38.1c 700-19 Gwat 3 - 2019-08-21 16.23.02_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 38.1c 700-19 Gwat 6 - 2019-08-21 16.43.02_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 38.1c 700-19 Iwat 1 - 2019-08-22 08.36.58_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 38.1c 700-19 Iwat 3 - 2019-08-22 08.49.27_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 38.1c 700-19 Iwat 6 - 2019-08-22 09.08.29_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 38.1d 698-19 Gwat 1 - 2019-08-20 15.05.02_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 38.1d 698-19 Gwat 3 - 2019-08-20 15.18.30_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 38.1d 698-19 Gwat 6 - 2019-08-20 15.37.51_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 38.1d 698-19 Iwat 1 - 2019-08-21 08.59.51_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 38.1d 698-19 Iwat 3 - 2019-08-21 09.13.35_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 38.1d 698-19 Iwat 6 - 2019-08-21 09.32.24_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 38.1e 699-19 Gwat 1 - 2019-08-21 09.44.01_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 38.1e 699-19 Gwat 3 - 2019-08-21 09.57.22_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 38.1e 699-19 Gwat 6 - 2019-08-21 10.15.39_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 38.1e 699-19 Iwat 1 - 2019-08-21 11.16.47_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 38.1e 699-19 Iwat 3 - 2019-08-21 11.29.58_exp_0004_auto_aggregated.json',
    'APL15-DEL2-EM1-B6N 38.1e 699-19 Iwat 6 - 2019-08-21 11.49.53_exp_0004_auto_aggregated.json',
    'APL15-DEL2_EM1_B6N 34.1e 692-19 Gwat 1 - 2019-08-14 16.48.21_exp_0004_auto_aggregated.json',
    'APL15-DEL2_EM1_B6N 34.1e 692-19 Gwat 3 - 2019-08-14 17.09.59_exp_0004_auto_aggregated.json',
    'APL15-DEL2_EM1_B6N 34.1e 692-19 Gwat 6 - 2019-08-14 17.28.37_exp_0004_auto_aggregated.json',
    'APL15-DEL2_EM1_B6N 34.1e 692-19 Iwat 1 - 2019-08-15 09.49.41_exp_0004_auto_aggregated.json',
    'APL15-DEL2_EM1_B6N 34.1e 692-19 Iwat 3 - 2019-08-15 10.02.15_exp_0004_auto_aggregated.json',
    'APL15-DEL2_EM1_B6N 34.1e 692-19 Iwat 6 - 2019-08-15 10.20.19_exp_0004_auto_aggregated.json',
    'APL15-DEL2_EM1_B6N 35.1a 690-19 Gwat 1 - 2019-08-14 12.06.19_exp_0004_auto_aggregated.json',
    'APL15-DEL2_EM1_B6N 35.1a 690-19 Gwat 3 - 2019-08-14 12.22.40_exp_0004_auto_aggregated.json',
    'APL15-DEL2_EM1_B6N 35.1a 690-19 Gwat 6 - 2019-08-14 12.42.00_exp_0004_auto_aggregated.json',
    'APL15-DEL2_EM1_B6N 35.1a 690-19 Iwat 1 - 2019-08-14 12.53.04_exp_0004_auto_aggregated.json',
    'APL15-DEL2_EM1_B6N 35.1a 690-19 Iwat 3 - 2019-08-14 13.04.25_exp_0004_auto_aggregated.json',
    'APL15-DEL2_EM1_B6N 35.1a 690-19 Iwat 6 - 2019-08-14 13.22.36_exp_0004_auto_aggregated.json',
    'APL15-DEL2_EM1_B6N 38.1a 691-19 Gwat 1 - 2019-08-14 13.48.53_exp_0004_auto_aggregated.json',
    'APL15-DEL2_EM1_B6N 38.1a 691-19 Gwat 3 - 2019-08-14 14.02.42_exp_0004_auto_aggregated.json',
    'APL15-DEL2_EM1_B6N 38.1a 691-19 Gwat 6 - 2019-08-14 14.24.10_exp_0004_auto_aggregated.json',
    'APL15-DEL2_EM1_B6N 38.1a 691-19 Iwat 1 - 2019-08-14 15.34.14_exp_0004_auto_aggregated.json',
    'APL15-DEL2_EM1_B6N 38.1a 691-19 Iwat 3 - 2019-08-14 15.44.04_exp_0004_auto_aggregated.json',
    'APL15-DEL2_EM1_B6N 38.1a 691-19 Iwat 6 - 2019-08-14 15.59.53_exp_0004_auto_aggregated.json',
    'ARL15-DEL2-EM1-B6N-29.1E Gwat 689-19 1 - 2019-08-08 11.51.43_exp_0004_auto_aggregated.json',
    'ARL15-DEL2-EM1-B6N-29.1E Gwat 689-19 3 - 2019-08-08 12.04.20_exp_0004_auto_aggregated.json',
    'ARL15-DEL2-EM1-B6N-29.1E Gwat 689-19 6 - 2019-08-08 12.24.13_exp_0004_auto_aggregated.json',
    'ARL15-DEL2-EM1-B6N-29.1E Iwat 689-19 1 - 2019-08-08 12.33.17_exp_0004_auto_aggregated.json',
    'ARL15-DEL2-EM1-B6N-29.1E Iwat 689-19 3 - 2019-08-12 16.08.24_exp_0004_auto_aggregated.json',
    'ARL15-DEL2-EM1-B6N-29.1E Iwat 689-19 6 - 2019-08-12 16.30.33_exp_0004_auto_aggregated.json',
    'ARL15-DEL2-EM1-B6N-32.1f Gwat 688-19 1 - 2019-08-08 10.07.15_exp_0004_auto_aggregated.json',
    'ARL15-DEL2-EM1-B6N-32.1f Gwat 688-19 3 - 2019-08-08 10.23.58_exp_0004_auto_aggregated.json',
    'ARL15-DEL2-EM1-B6N-32.1f Gwat 688-19 6 - 2019-08-08 10.46.56_exp_0004_auto_aggregated.json',
    'ARL15-DEL2-EM1-B6N-32.1f Iwat 688-19 1 - 2019-08-08 11.13.12_exp_0004_auto_aggregated.json',
    'ARL15-DEL2-EM1-B6N-32.1f Iwat 688-19 3 - 2019-08-08 11.31.23_exp_0004_auto_aggregated.json',
    'ARL15-DEL2-EM1-B6N-32.1f Iwat 688-19 6 - 2019-08-08 11.43.11_exp_0004_auto_aggregated.json'
]

########################################################################################################################
## Heatmaps of whole slides
########################################################################################################################

# # parameters to constrain cell areas
# min_cell_area = 0  # pixel; we want all small objects
# max_cell_area = 200e3  # pixel
# xres_ref = 0.4538234626730202
# yres_ref = 0.4537822752643282
# min_cell_area_um2 = min_cell_area * xres_ref * yres_ref
# max_cell_area_um2 = max_cell_area * xres_ref * yres_ref

# # CSV file with metainformation of all mice
# metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
# metainfo = pd.read_csv(metainfo_csv_file)

# # make sure that in the boxplots PAT comes before MAT
# metainfo['sex'] = metainfo['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
# metainfo['ko_parent'] = metainfo['ko_parent'].astype(
#     pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
# metainfo['genotype'] = metainfo['genotype'].astype(
#     pd.api.types.CategoricalDtype(categories=['KLF14-KO:WT', 'KLF14-KO:Het'], ordered=True))

# load area to quantile maps computed in exp_0106
filename_area2quantile = os.path.join(area2quantile_dir, 'klf14_b6ntac_exp_0106_filename_area2quantile_v8.npz')
with np.load(filename_area2quantile, allow_pickle=True) as aux:
    f_area2quantile_f = aux['f_area2quantile_f'].item()
    f_area2quantile_m = aux['f_area2quantile_m'].item()

# load AIDA's colourmap
cm = cytometer.data.aida_colourmap()

# rough_foreground_mask() parameters used in arl15del2_b6ntac_exp_0001_full_slide_pipeline_v8_no_correction
downsample_factor = 8.0

## compute whole slide heatmaps

# loop annotations files to compute the heatmaps
for i_file, json_file in enumerate(json_annotation_files):

    print('File ' + str(i_file) + '/' + str(len(json_annotation_files) - 1) + ': ' + json_file)

    # name of corresponding .ndpi file
    histo_file = json_file.replace('_exp_0004_auto_aggregated.json', '.ndpi')
    kernel_file = os.path.basename(histo_file).replace('.ndpi', '')

    # add path to file
    json_file = os.path.join(annotations_dir, json_file)
    histo_file = os.path.join(histo_dir, os.path.basename(histo_file))

    # open full resolution histology slide
    im = openslide.OpenSlide(histo_file)

    # pixel size
    assert (im.properties['tiff.ResolutionUnit'] == 'centimeter')
    xres = 1e-2 / float(im.properties['tiff.XResolution']) * 1e6  # um^2
    yres = 1e-2 / float(im.properties['tiff.YResolution']) * 1e6  # um^2

    # name of file where rough mask was saved to
    coarse_mask_file = os.path.basename(histo_file)
    coarse_mask_file = coarse_mask_file.replace('.ndpi', '_coarse_mask.npz')
    coarse_mask_file = os.path.join(annotations_dir, coarse_mask_file)

    # load coarse tissue mask
    with np.load(coarse_mask_file) as aux:
        print('Completed: ' + str(aux['perc_completed_all'][-1]) + ' %')
        lores_istissue0 = aux['lores_istissue0']
        im_downsampled = aux['im_downsampled']

    if DEBUG:
        plt.clf()
        plt.subplot(211)
        plt.imshow(im_downsampled)
        plt.subplot(212)
        plt.imshow(lores_istissue0)

    # load list of contours in segmentations
    json_file_corrected = os.path.join(annotations_dir, json_file)

    # list of items (there's a contour in each item)
    contours_corrected, props = cytometer.data.aida_get_contours(json_file_corrected, layer_name='White adipocyte.*', return_props=True)

    # init array for interpolated quantiles
    quantiles_grid = np.zeros(shape=lores_istissue0.shape, dtype=np.float32)

    # init array for mask where there are segmentations
    areas_mask = Image.new("1", lores_istissue0.shape[::-1], "black")
    draw = ImageDraw.Draw(areas_mask)

    # init lists for contour centroids and areas
    areas_all = []
    centroids_all = []
    centroids_down_all = []

    # loop items (one contour per item)
    for c in contours_corrected:

        # convert to downsampled coordinates
        c = np.array(c)
        c_down = c / downsample_factor

        if DEBUG:
            plt.fill(c_down[:, 0], c_down[:, 1], fill=False, color='r')

        # compute cell area
        area = shapely.geometry.Polygon(c).area * xres * yres  # (um^2)
        areas_all.append(area)

        # compute centroid of contour
        centroid = np.mean(c, axis=0)
        centroids_all.append(centroid)
        centroids_down_all.append(centroid / downsample_factor)

        # add object described by contour to areas_mask
        draw.polygon(list(c_down.flatten()), outline="white", fill="white")

    # convert mask from RGBA to binary
    areas_mask = np.array(areas_mask, dtype=np.bool)

    areas_all = np.array(areas_all)

    # interpolate scattered area data to regular grid
    grid_row, grid_col = np.mgrid[0:areas_mask.shape[0], 0:areas_mask.shape[1]]
    quantiles_grid = scipy.interpolate.griddata(centroids_down_all, areas_all, (grid_col, grid_row), method='linear', fill_value=0)
    quantiles_grid[~areas_mask] = 0

    if DEBUG:
        plt.clf()
        plt.subplot(211)
        plt.imshow(im_downsampled)
        plt.axis('off')
        plt.subplot(212)
        plt.imshow(quantiles_grid)

    # # get metainfo for this slide
    # df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(histo_file),
    #                                                       values=[i_file,], values_tag='i',
    #                                                       tags_to_keep=['id', 'ko_parent', 'sex'])
    #
    # # mouse ID as a string
    # id = df_common['id'].values[0]
    # sex = df_common['sex'].values[0]
    # ko = df_common['ko_parent'].values[0]

    # convert area values to quantiles
    quantiles_grid = f_area2quantile_m(quantiles_grid)
    # if sex == 'f':
    #     quantiles_grid = f_area2quantile_f(quantiles_grid)
    # elif sex == 'm':
    #     quantiles_grid = f_area2quantile_m(quantiles_grid)
    # else:
    #     raise ValueError('Wrong sex value')

    # make background white in the plot
    quantiles_grid[~areas_mask] = np.nan

    if DEBUG:
        plt.clf()
        plt.subplot(211)
        plt.imshow(im_downsampled)
        plt.axis('off')
        plt.subplot(212)
        # plt.imshow(areas_grid, vmin=0.0, vmax=1.0, cmap='gnuplot2')
        plt.imshow(quantiles_grid, vmin=0.0, vmax=1.0, cmap=cm)
        cbar = plt.colorbar(shrink=1.0)
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.set_ylabel('Cell area quantile', rotation=90, fontsize=14)
        plt.axis('off')
        plt.tight_layout()

    if DEBUG:
        plt.clf()
        plt.hist(areas_all, bins=50, density=True, histtype='step')

    # plot cell areas for paper with quantile colourmap
    plt.clf()
    plt.imshow(Image.fromarray((cm(quantiles_grid) * 255).astype(np.uint8)), vmin=0.0, vmax=255.0, cmap=cm)
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, kernel_file + '_exp_0002_cell_size_heatmap_quantile_colourmap.png'),
                bbox_inches='tight')


## crop heatmaps for paper

# for i_file, json_file in enumerate(json_annotation_files):
#     # i_file += 1;  json_file = json_annotation_files[i_file]  # hack: for manual looping, one by one
#
#     print('File ' + str(i_file) + '/' + str(len(json_annotation_files) - 1) + ': ' + json_file)
#
#     # name of heatmap
#     kernel_file = json_file.replace('.json', '')
#     heatmap_file = os.path.join(figures_dir, kernel_file + '_exp_0110_cell_size_heatmap.png')
#     heatmap_large_file = os.path.join(figures_dir, kernel_file + '_exp_0110_cell_size_heatmap_large.png')
#     cropped_heatmap_file = os.path.join(figures_dir, kernel_file + '_exp_0110_cell_size_heatmap.png')
#
#     # load heatmaps. We have to load two of them, because first I saved them using the regular savefig(), which reduces
#     # the image resolution and modifies the borders. I defined the crop boxes on those image, so in order to avoid having
#     # to redo the crop boxes, now we have to figure out where those low resolution crop boxes map in the full resolution
#     # image
#     heatmap = Image.open(heatmap_file)
#     heatmap_large = Image.open(heatmap_large_file)
#
#     # convert cropping coordinates from smaller PNG to full size PNG
#     x0, xend, y0, yend = np.array(get_crop_box(i_file))
#     heatmap_array = np.array(Image.open(heatmap_file))  # hack: because passing heatmap sometimes doesn't work, due to some bug
#     cropped_heatmap = heatmap_array[y0:yend+1, x0:xend+1]
#
#     if DEBUG:
#         plt.clf()
#         plt.subplot(211)
#         plt.imshow(heatmap)
#         plt.plot([x0, xend, xend, x0, x0], [yend, yend, y0, y0, yend], 'r')
#         plt.subplot(212)
#         plt.imshow(cropped_heatmap)
#
#     # when saving a figure to PNG with savefig() and our options above, it adds a white border of 9 pixels on each side.
#     # We need to remove that border before coordinate interpolation
#     x0, xend = np.interp((x0 - 9, xend - 9), (0, heatmap.size[0] - 1 - 18), (0, heatmap_large.size[0] - 1))
#     y0, yend = np.interp((y0 - 9, yend - 9), (0, heatmap.size[1] - 1 - 18), (0, heatmap_large.size[1] - 1))
#     x0, xend, y0, yend = np.round([x0, xend, y0, yend]).astype(np.int)
#
#     # open histology image at 16x downsampled size
#     histo_file = json_file.replace('.json', '.ndpi')
#     histo_file = os.path.join(histo_dir, histo_file)
#     im = openslide.OpenSlide(histo_file)
#     im_large = im.read_region(location=(0, 0), level=im.get_best_level_for_downsample(16), size=heatmap_large.size)
#
#     # crop image and heatmap
#     cropped_im_large = np.array(im_large)[y0:yend+1, x0:xend+1, 0:3]
#     cropped_heatmap_large = np.array(Image.open(heatmap_large_file))[y0:yend+1, x0:xend+1] # hack for same bug as above
#
#     if DEBUG:
#         plt.clf()
#         plt.subplot(211)
#         plt.imshow(heatmap_large)
#         plt.plot([x0, xend, xend, x0, x0], [yend, yend, y0, y0, yend], 'r')
#         plt.subplot(212)
#         plt.imshow(cropped_heatmap_large)
#
#     # print cropped histology
#     plt.clf()
#     plt.gcf().set_size_inches([6.4, 4.8])
#     plt.imshow(cropped_im_large, interpolation='none')
#     plt.axis('off')
#     plt.tight_layout()
#
#     plt.savefig(os.path.join(figures_dir, kernel_file + '_exp_0110_cell_size_histo_large_cropped.png'),
#                 bbox_inches='tight')
#     plt.savefig(os.path.join(figures_dir, kernel_file + '_exp_0110_cell_size_histo_large_cropped.jpg'),
#                 bbox_inches='tight')
#
#     # print cropped heatmap
#     plt.clf()
#     plt.gcf().set_size_inches([6.4, 4.8])
#     plt.imshow(cropped_heatmap_large, interpolation='none')
#     plt.axis('off')
#     plt.tight_layout()
#
#     plt.savefig(os.path.join(figures_dir, kernel_file + '_exp_0110_cell_size_heatmap_large_cropped.png'),
#                 bbox_inches='tight')
#     plt.savefig(os.path.join(figures_dir, kernel_file + '_exp_0110_cell_size_heatmap_large_cropped.jpg'),
#                 bbox_inches='tight')

## plot colourmaps and cell area density functions

# # colourmap plot
# a = np.array([[0,1]])
# plt.figure(figsize=(9, 1.5))
# img = plt.imshow(a, cmap=cm)
# plt.gca().set_visible(False)
# cax = plt.axes([0.1, 0.2, 0.8, 0.6])
# cbar = plt.colorbar(orientation='horizontal', cax=cax)
# for q in np.linspace(0, 1, 11):
#     plt.plot([q, q], [0, 0.5], 'k', linewidth=2, zorder=1)
# cbar.ax.tick_params(labelsize=14)
# plt.title('Cell area colourmap with decile divisions', rotation=0, fontsize=14)
# plt.xlabel('Quantile', fontsize=14)
# plt.tight_layout()
#
# plt.savefig(os.path.join(figures_dir, 'exp_0110_aida_colourmap.png'), bbox_inches='tight')
