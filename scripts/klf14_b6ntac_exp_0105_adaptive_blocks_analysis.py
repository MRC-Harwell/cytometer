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
area2quantile_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v8')
annotations_dir = os.path.join(home, 'Data/cytometer_data/aida_data_Klf14_v8/annotations')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')

# file with area->quantile map precomputed from all automatically segmented slides in klf14_b6ntac_exp_0098_full_slide_size_analysis_v7.py
filename_area2quantile = os.path.join(area2quantile_dir, 'klf14_b6ntac_exp_0098_filename_area2quantile.npz')

# suffixes of annotation filenames
auto_filename_suffix = '_exp_0106_auto.json'
corrected_filename_suffix = '_exp_0106_corrected.json'

# list of annotations
# SQWAT: list of annotation files (same list used for phenotyping analysis in klf14_b6ntac_exp_0110_paper_figures_v8.py
json_annotation_files_dict = {}
json_annotation_files_dict['sqwat'] = [
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

# GWAT: list of annotation files
json_annotation_files_dict['gwat'] = [
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

# get all the files in a single list
auto_annotation_files_list = json_annotation_files_dict['gwat'] + json_annotation_files_dict['sqwat']
auto_annotation_files_list = [x.replace('.json', auto_filename_suffix) for x in auto_annotation_files_list]
auto_annotation_files_list = [os.path.join(annotations_dir, x) for x in auto_annotation_files_list]

# full resolution image window and network expected receptive field parameters
# fullres_box_size = np.array([2751, 2751])  # this is different in different images
receptive_field = np.array([131, 131])
receptive_field_len = receptive_field[0]

# largest cell area that we need to account for
max_cell_area = 100e3  # pixel^2
max_cell_radius = np.ceil(np.sqrt(max_cell_area / np.pi)).astype(np.int)  # pixels, assuming the cell is a circle

# rough mask parameters
downsample_factor = 16.0

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
    # xres = 1e-2 / float(im.properties['tiff.XResolution'])
    # yres = 1e-2 / float(im.properties['tiff.YResolution'])

    # load blocks from annotation file
    # each block is given as the 4-points of a rectagle, with the first point repeated at the end to close the
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

    # # we are going to use the largest block dimensions in the image as the block size for the uniform tiling
    # block_len = np.max(height_all_adaptive_blocks + width_all_adaptive_blocks).astype(np.int)

    # we fix the maximum block length
    block_len = 2751

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
    coarse_mask_file = os.path.basename(histo_file)
    coarse_mask_file = coarse_mask_file.replace('.ndpi', '_coarse_mask.npz')
    coarse_mask_file = os.path.join(annotations_dir, coarse_mask_file)

    # load rough mask
    with np.load(coarse_mask_file) as aux:
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
      + str(100 - 100 * np.mean(np.array(area_all_adaptive_images) / np.array(area_all_uniform_images))) + ' %')
