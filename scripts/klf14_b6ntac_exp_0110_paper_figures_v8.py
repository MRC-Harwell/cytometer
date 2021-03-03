"""
Generate figures for the DeepCytometer paper for v8 of the pipeline.

Here's the analysis for automatically segmented cells. Hand traced cells are analysed in
klf14_b6ntac_exp_0109_pipeline_v8_validation.py.

This script partly deprecates klf14_b6ntac_exp_0099_paper_figures_v7.py:
* Some figures have been updated to have v8 of the pipeline in the paper.

"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0110_paper_figures'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])

# post-processing parameters
min_area = 203 / 2  # (pix^2) smaller objects are rejected
max_area = 44879 * 3  # (pix^2) larger objects are rejected

xres_ref = 0.4538234626730202
yres_ref = 0.4537822752643282
min_area_um2 = min_area * xres_ref * yres_ref
max_area_um2 = max_area * xres_ref * yres_ref

# json_annotation_files_dict here needs to have the same files as in
# klf14_b6ntac_exp_0098_full_slide_size_analysis_v7.py

# SQWAT: list of annotation files
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

########################################################################################################################
## Compute cell populations from automatically segmented images in two depots: SQWAT and GWAT:
##   Cell area histograms
##   HD quantiles of cell areas
##   Cell count estimates per depot
## The results are saved, so in later sections, it's possible to just read them for further analysis.
## GENERATES DATA LATER USED IN THIS PYTHON SCRIPT
########################################################################################################################

import matplotlib.pyplot as plt
import cytometer.data
import cytometer.stats
import shapely
import scipy.stats as stats
import openslide
import numpy as np
import sklearn.neighbors, sklearn.model_selection
import pandas as pd
import PIL
# import time

# directories
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
annotations_dir = os.path.join(home, 'Data/cytometer_data/aida_data_Klf14_v8/annotations')
ndpi_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
dataframe_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
metainfo_dir = os.path.join(home, 'Data/cytometer_data/klf14')

DEBUG = False
SAVEFIG = False

# 0.05, 0.1 , 0.15, 0.2, ..., 0.9 , 0.95
quantiles = np.linspace(0, 1, 21)

# bins for the cell population histograms
area_bin_edges = np.linspace(min_area_um2, max_area_um2, 201)
area_bin_centers = (area_bin_edges[0:-1] + area_bin_edges[1:]) / 2.0

# data file with extra info for the dataframe (quantiles and histograms bins)
dataframe_areas_extra_filename = os.path.join(dataframe_dir, 'klf14_b6ntac_exp_0110_dataframe_areas_extra.npz')

# if the file doesn't exist, save it
if not os.path.isfile(dataframe_areas_extra_filename):
    np.savez(dataframe_areas_extra_filename, quantiles=quantiles, area_bin_edges=area_bin_edges,
             area_bin_centers=area_bin_centers)

for method in ['auto', 'corrected']:

    # dataframe with histograms and smoothed histograms of cell populations in each slide
    dataframe_areas_filename = os.path.join(dataframe_dir, 'klf14_b6ntac_exp_0110_dataframe_areas_' + method + '.pkl')

    if os.path.isfile(dataframe_areas_filename):

        # load dataframe with cell population quantiles and histograms
        df_all = pd.read_pickle(dataframe_areas_filename)

    else:

        # CSV file with metainformation of all mice
        metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
        metainfo = pd.read_csv(metainfo_csv_file)

        # make sure that in the boxplots PAT comes before MAT
        metainfo['sex'] = metainfo['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
        metainfo['ko_parent'] = metainfo['ko_parent'].astype(
            pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
        metainfo['genotype'] = metainfo['genotype'].astype(
            pd.api.types.CategoricalDtype(categories=['KLF14-KO:WT', 'KLF14-KO:Het'], ordered=True))

        # mean mouse body weight (female and male)
        mean_bw_f = metainfo[metainfo['sex'] == 'f']['BW'].mean()
        mean_bw_m = metainfo[metainfo['sex'] == 'm']['BW'].mean()

        # dataframe to keep all results, one row per annotations file
        df_all = pd.DataFrame()

        for depot in ['sqwat', 'gwat']:

            # list of annotation files for this depot
            json_annotation_files = json_annotation_files_dict[depot]

            # modify filenames to select the particular segmentation we want (e.g. the automatic ones, or the manually refined ones)
            json_annotation_files = [x.replace('.json', '_exp_0106_' + method + '_aggregated.json') for x in
                                     json_annotation_files]
            json_annotation_files = [os.path.join(annotations_dir, x) for x in json_annotation_files]

            # process annotation files and coarse masks
            filename_coarse_mask_area = os.path.join(figures_dir,
                                                     'klf14_b6ntac_exp_0110_coarse_mask_area_' + depot + '.npz')
            for i_file, json_file in enumerate(json_annotation_files):

                print('File ' + str(i_file) + '/' + str(len(json_annotation_files)-1) + ': '
                      + os.path.basename(json_file))

                if not os.path.isfile(json_file):
                    print('Missing annotations file')
                    continue

                # open full resolution histology slide
                ndpi_file = json_file.replace('_exp_0106_' + method + '_aggregated.json', '.ndpi')
                ndpi_file = os.path.join(ndpi_dir, os.path.basename(ndpi_file))
                im = openslide.OpenSlide(ndpi_file)

                # pixel size
                assert (im.properties['tiff.ResolutionUnit'] == 'centimeter')
                xres = float(im.properties['openslide.mpp-x'])  # um/pixel
                yres = float(im.properties['openslide.mpp-y'])  # um/pixel

                # load mask
                coarse_mask_file = json_file.replace('_exp_0106_' + method + '_aggregated.json', '_coarse_mask.npz')
                coarse_mask_file = os.path.join(annotations_dir, coarse_mask_file)

                with np.load(coarse_mask_file) as aux:
                    lores_istissue0 = aux['lores_istissue0']

                    if DEBUG:
                        foo = aux['im_downsampled']
                        foo = PIL.Image.fromarray(foo)
                        foo = foo.resize(tuple((np.round(np.array(foo.size[0:2]) / 4)).astype(np.int)))
                        plt.imshow(foo)
                        plt.title(os.path.basename(ndpi_file))

                # create dataframe for this image
                if depot == 'sqwat':
                    df = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(json_file),
                                                                   values=[depot, ], values_tag='depot',
                                                                   tags_to_keep=['id', 'ko_parent', 'sex', 'genotype',
                                                                                 'BW', 'SC'])
                    df.rename(columns={'SC': 'DW'}, inplace=True)
                elif depot == 'gwat':
                    df = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(json_file),
                                                                   values=[depot, ], values_tag='depot',
                                                                   tags_to_keep=['id', 'ko_parent', 'sex', 'genotype',
                                                                                 'BW', 'gWAT'])
                    df.rename(columns={'gWAT': 'DW'}, inplace=True)
                else:
                    raise RuntimeError('Unknown depot type')

                # compute scaling factor between downsampled mask and original image
                size_orig = np.array(im.dimensions)  # width, height
                size_downsampled = np.array(lores_istissue0.shape)[::-1]  # width, height
                downsample_factor = size_orig / size_downsampled  # width, height

                # compute area of coarse mask
                coarse_mask_area = 1e-6 * np.count_nonzero(lores_istissue0) * (xres * downsample_factor[0]) * (yres * downsample_factor[1])  # mm^2

                # correct area because most slides contain two slices, but some don't
                id = df['id'].values[0]
                if depot == 'sqwat' and not (id in ['16.2d', '17.1e', '17.2g', '16.2e', '18.1f', '37.4a', '37.2e']):
                    # two slices in the slide, so slice area is approx. one half
                    coarse_mask_area /= 2
                elif depot == 'gwat' and not (id in ['36.1d', '16.2a', '16.2b', '16.2c', '16.2d', '16.2e', '17.1b', '17.1d',
                                                     '17.1e', '17.1f', '17.2c', '17.2d', '17.2f', '17.2g', '18.1b', '18.1c',
                                                     '18.1d', '18.2a', '18.2c', '18.2d', '18.2f', '18.2g', '18.3c', '19.1a',
                                                     '19.2e', '19.2f', '19.2g', '36.3d', '37.2e', '37.2f', '37.2g', '37.2h',
                                                     '37.3a', '37.4a', '37.4b', '39.2d']):
                    # two slices in the slide, so slice area is approx. one half
                    coarse_mask_area /= 2

                df['coarse_mask_area_mm3'] = coarse_mask_area

                # load contours and their confidence measure from annotation file
                cells, props = cytometer.data.aida_get_contours(json_file, layer_name='White adipocyte.*', return_props=True)

                # compute areas of the cells (um^2)
                areas = np.array([shapely.geometry.Polygon(cell).area for cell in cells]) * xres * yres  # um^2

                # smooth out histogram
                kde = sklearn.neighbors.KernelDensity(bandwidth=100, kernel='gaussian').fit(areas.reshape(-1, 1))
                log_dens = kde.score_samples(area_bin_centers.reshape(-1, 1))
                pdf = np.exp(log_dens)

                # compute mode
                df['area_smoothed_mode'] = area_bin_centers[np.argmax(pdf)]

                # cell volume estimate from cell areas
                volumes = 4 / (3 * np.sqrt(np.pi)) * (areas ** (3/2)) * 1e-18 / 1e-6  # cm^3

                # cell count estimate from cell volumes and depot weight
                if depot == 'gwat':
                    rho = 0.9029  # g/cm^3
                elif depot == 'sqwat':
                    rho = 0.9038  # g/cm^3
                else:
                    raise ValueError('Invalid depot name')
                df['kN'] = df['DW'] / rho / volumes.mean()

                # compute areas at population quantiles
                areas_at_quantiles = stats.mstats.hdquantiles(areas, prob=quantiles, axis=0)
                df['area_at_quantiles'] = [areas_at_quantiles]

                # compute stderr of the areas at population quantiles
                # Note: We are using my modified hdquantiles_sd() function, which is 530x faster than the current scipy
                # implementation (my modification has been merged into scipy:main, but the scipy package I'm using is
                # slightly older)
                stderr_at_quantiles = cytometer.stats.hdquantiles_sd(areas, prob=quantiles, axis=0)
                df['stderr_at_quantiles'] = [stderr_at_quantiles]

                # compute histograms with area binning
                histo, _ = np.histogram(areas, bins=area_bin_edges, density=True)
                df['histo'] = [histo]

                # smoothed histogram
                df['smoothed_histo'] = [pdf]

                if DEBUG:
                    plt.clf()
                    plt.plot(1e-3 * area_bin_centers, df['histo'][0], label='Areas')
                    plt.plot(1e-3 * area_bin_centers, df['smoothed_histo'], label='Kernel')
                    plt.plot([df['area_smoothed_mode'] * 1e-3, df['area_smoothed_mode'] * 1e-3],
                             [0, df['smoothed_histo'].max()], 'k', label='Mode')
                    plt.legend()
                    plt.xlabel('Area ($10^3 \cdot \mu m^2$)', fontsize=14)

                # add results to total dataframe
                df_all = pd.concat([df_all, df], ignore_index=True)

        # save dataframe with data from both depots for current method (auto or corrected)
        df_all.to_pickle(dataframe_areas_filename)

########################################################################################################################
## Import packages and auxiliary functions common to all analysis sections
## USED IN PAPER
########################################################################################################################

import pickle
from toolz import interleave
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import seaborn as sns
# import openslide
import cytometer.data
import cytometer.stats
# import shapely

# directories
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
hand_traced_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training_v2')
annotations_dir = os.path.join(home, 'Data/cytometer_data/aida_data_Klf14_v8/annotations')
ndpi_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
dataframe_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
metainfo_dir = os.path.join(home, 'Data/cytometer_data/klf14')
saved_models_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v8')

DEBUG = False

method = 'corrected'

# k-folds file with hand traced filenames
saved_kfolds_filename = 'klf14_b6ntac_exp_0094_generate_extra_training_images.pickle'

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# make sure that in the boxplots PAT comes before MAT
metainfo['sex'] = metainfo['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
metainfo['ko_parent'] = metainfo['ko_parent'].astype(
    pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
metainfo['genotype'] = metainfo['genotype'].astype(
    pd.api.types.CategoricalDtype(categories=['KLF14-KO:WT', 'KLF14-KO:Het'], ordered=True))

# remove BW=NaNs
metainfo = metainfo[~np.isnan(metainfo['BW'])]
metainfo = metainfo.reset_index()

# load dataframe with cell population quantiles and histograms
dataframe_areas_filename = os.path.join(dataframe_dir, 'klf14_b6ntac_exp_0110_dataframe_areas_' + method + '.pkl')
df_all = pd.read_pickle(dataframe_areas_filename)
df_all = df_all[~np.isnan(df_all['BW'])]
df_all = df_all.reset_index()

df_all['sex'] = df_all['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
df_all['ko_parent'] = df_all['ko_parent'].astype(
    pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
df_all['genotype'] = df_all['genotype'].astype(
    pd.api.types.CategoricalDtype(categories=['KLF14-KO:WT', 'KLF14-KO:Het'], ordered=True))

# load extra info needed for the histograms
dataframe_areas_extra_filename = os.path.join(dataframe_dir, 'klf14_b6ntac_exp_0110_dataframe_areas_extra.npz')
with np.load(dataframe_areas_extra_filename) as aux:
    quantiles = aux['quantiles']
    area_bin_edges = aux['area_bin_edges']
    area_bin_centers = aux['area_bin_centers']

## auxiliary functions
def table_of_hand_traced_regions(file_svg_list):
    """
    Open SVG files in a list, and count the number of different types of regions (Cells, Other, Background, Windows,
    Windows with cells) and create a table with them for the paper
    :param file_svg_list: list of filenames
    :return: pd.Dataframe
    """

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
        background_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Background',
                                                                      add_offset_from_filename=False,
                                                                      minimum_npoints=3)
        contours = cell_contours + other_contours + brown_contours + background_contours

        # make a list with the type of cell each contour is classified as
        contour_type = [np.zeros(shape=(len(cell_contours),), dtype=np.uint8),  # 0: white-adipocyte
                        np.ones(shape=(len(other_contours),), dtype=np.uint8),  # 1: other types of tissue
                        np.ones(shape=(len(brown_contours),), dtype=np.uint8),
                        # 1: brown cells (treated as "other" tissue)
                        np.zeros(shape=(len(background_contours),), dtype=np.uint8)]  # 0: background
        contour_type = np.concatenate(contour_type)

        print('Cells: ' + str(len(cell_contours)) + '. Other: ' + str(len(other_contours))
              + '. Brown: ' + str(len(brown_contours)) + '. Background: ' + str(len(background_contours)))

        # create dataframe for this image
        df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_svg),
                                                              values=[i, ], values_tag='i',
                                                              tags_to_keep=['id', 'ko_parent', 'sex'])

        # mouse ID as a string
        id = df_common['id'].values[0]
        sex = df_common['sex'].values[0]
        ko = df_common['ko_parent'].values[0]

        # row to add to the table
        df = pd.DataFrame(
            [(sex, ko,
              len(cell_contours), len(other_contours) + len(brown_contours), len(background_contours), 1,
              int(len(cell_contours) > 0))],
            columns=['Sex', 'Genotype', 'Cells', 'Other', 'Background', 'Windows', 'Windows with cells'], index=[id])

        if id in table.index:

            num_cols = ['Cells', 'Other', 'Background', 'Windows', 'Windows with cells']
            table.loc[id, num_cols] = (table.loc[id, num_cols] + df.loc[id, num_cols])

        else:

            table = table.append(df, sort=False, ignore_index=False, verify_integrity=True)

    # alphabetical order by mouse IDs
    table = table.sort_index()

    return table


########################################################################################################################
## For "Data" section in the paper:
## Summary tables of hand traced datasets used for
#   * DeepCytometer training/validation
#   * Cell population studies
# USED IN PAPER
########################################################################################################################

## DeepCytometer training/validation

# original dataset used in pipelines up to v6 + extra "other" tissue images
kfold_filename = os.path.join(saved_models_dir, saved_kfolds_filename)
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

# loop the folds to get the ndpi files that correspond to testing of each fold,
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

# create a table with the number of different hand traced regions per animal
table = table_of_hand_traced_regions(file_svg_list)

# save dataframe
if SAVEFIG:
    table.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_training_hand_traced_objects_count.csv'))

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

print('f PAT: ' + str(np.sum(table.loc[idx_f & idx_pat, 'Cells'])))
print('f MAT: ' + str(np.sum(table.loc[idx_f & idx_mat, 'Cells'])))
print('m PAT: ' + str(np.sum(table.loc[idx_m & idx_pat, 'Cells'])))
print('m MAT: ' + str(np.sum(table.loc[idx_m & idx_mat, 'Cells'])))

## Cell population studies

# list of hand traced contours
# The list contains 126 XCF (Gimp format) files with the contours that were used for training DeepCytometer,
# plus 5 files (131 in total) with extra contours for 2 mice where the cell population was not well
# represented.
hand_file_svg_list = [
    'KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00_row_010512_col_006912.svg',
    'KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00_row_012848_col_016240.svg',
    'KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00_row_016812_col_017484.svg',
    'KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00_row_019228_col_015060.svg',
    'KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00_row_029472_col_015520.svg',
    'KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_row_005348_col_019844.svg',
    'KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_row_006652_col_061724.svg',
    'KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_row_006900_col_071980.svg',
    'KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_row_010732_col_016692.svg',
    'KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_row_012828_col_018388.svg',
    'KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_row_013600_col_022880.svg',
    'KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_row_014768_col_022576.svg',
    'KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_row_014980_col_027052.svg',
    'KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_row_027388_col_018468.svg',
    'KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_row_028864_col_024512.svg',
    'KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_row_041392_col_026032.svg',
    'KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38_row_009588_col_028676.svg',
    'KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38_row_011680_col_013984.svg',
    'KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38_row_015856_col_012416.svg',
    'KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38_row_018720_col_031152.svg',
    'KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38_row_021796_col_055852.svg',
    'KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06_row_011852_col_071620.svg',
    'KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06_row_013300_col_055476.svg',
    'KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06_row_014320_col_007600.svg',
    'KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06_row_015200_col_021536.svg',
    'KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06_row_020256_col_002880.svg',
    'KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06_row_021136_col_010880.svg',
    'KLF14-B6NTAC-37.1c PAT 108-16 C1 - 2016-02-15 14.49.45_row_001292_col_004348.svg',
    'KLF14-B6NTAC-37.1c PAT 108-16 C1 - 2016-02-15 14.49.45_row_005600_col_004224.svg',
    'KLF14-B6NTAC-37.1c PAT 108-16 C1 - 2016-02-15 14.49.45_row_007216_col_008896.svg',
    'KLF14-B6NTAC-37.1c PAT 108-16 C1 - 2016-02-15 14.49.45_row_007372_col_008556.svg',
    'KLF14-B6NTAC-37.1c PAT 108-16 C1 - 2016-02-15 14.49.45_row_011904_col_005280.svg',
    'KLF14-B6NTAC-37.1d PAT 109-16 C1 - 2016-02-15 15.19.08_row_010048_col_001856.svg',
    'KLF14-B6NTAC-37.1d PAT 109-16 C1 - 2016-02-15 15.19.08_row_012172_col_049588.svg',
    'KLF14-B6NTAC-37.1d PAT 109-16 C1 - 2016-02-15 15.19.08_row_013232_col_009008.svg',
    'KLF14-B6NTAC-37.1d PAT 109-16 C1 - 2016-02-15 15.19.08_row_016068_col_007276.svg',
    'KLF14-B6NTAC-37.1d PAT 109-16 C1 - 2016-02-15 15.19.08_row_019680_col_016480.svg',
    'KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46_row_004124_col_012524.svg',
    'KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46_row_004384_col_005456.svg',
    'KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46_row_006040_col_005272.svg',
    'KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46_row_006640_col_008848.svg',
    'KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46_row_008532_col_009804.svg',
    'KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46_row_013952_col_002624.svg',
    'KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46_row_017044_col_031228.svg',
    'KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46_row_021804_col_035412.svg',
    'KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04_row_010716_col_008924.svg',
    'KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04_row_016832_col_016944.svg',
    'KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04_row_018784_col_010912.svg',
    'KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04_row_024528_col_014688.svg',
    'KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04_row_026108_col_068956.svg',
    'KLF14-B6NTAC-MAT-17.2c  66-16 C1 - 2016-02-04 11.46.39_row_009840_col_008736.svg',
    'KLF14-B6NTAC-MAT-17.2c  66-16 C1 - 2016-02-04 11.46.39_row_017792_col_017504.svg',
    'KLF14-B6NTAC-MAT-17.2c  66-16 C1 - 2016-02-04 11.46.39_row_020032_col_018640.svg',
    'KLF14-B6NTAC-MAT-17.2c  66-16 C1 - 2016-02-04 11.46.39_row_030820_col_022204.svg',
    'KLF14-B6NTAC-MAT-17.2f  68-16 C1 - 2016-02-04 15.05.54_row_007500_col_050372.svg',
    'KLF14-B6NTAC-MAT-17.2f  68-16 C1 - 2016-02-04 15.05.54_row_008000_col_003680.svg',
    'KLF14-B6NTAC-MAT-17.2f  68-16 C1 - 2016-02-04 15.05.54_row_013348_col_019316.svg',
    'KLF14-B6NTAC-MAT-17.2f  68-16 C1 - 2016-02-04 15.05.54_row_019168_col_019600.svg',
    'KLF14-B6NTAC-MAT-17.2f  68-16 C1 - 2016-02-04 15.05.54_row_022960_col_007808.svg',
    'KLF14-B6NTAC-MAT-17.2f  68-16 C1 - 2016-02-04 15.05.54_row_026132_col_012148.svg',
    'KLF14-B6NTAC-MAT-17.2f  68-16 C1 - 2016-02-04 15.05.54_row_027968_col_011200.svg',
    'KLF14-B6NTAC-MAT-18.1a  50-16 C1 - 2016-02-02 09.12.41_row_003584_col_017280.svg',
    'KLF14-B6NTAC-MAT-18.1a  50-16 C1 - 2016-02-02 09.12.41_row_012908_col_010212.svg',
    'KLF14-B6NTAC-MAT-18.1a  50-16 C1 - 2016-02-02 09.12.41_row_013984_col_012576.svg',
    'KLF14-B6NTAC-MAT-18.1a  50-16 C1 - 2016-02-02 09.12.41_row_014448_col_019088.svg',
    'KLF14-B6NTAC-MAT-18.1a  50-16 C1 - 2016-02-02 09.12.41_row_015200_col_015920.svg',
    'KLF14-B6NTAC-MAT-18.1a  50-16 C1 - 2016-02-02 09.12.41_row_028156_col_018596.svg',
    'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_001920_col_014048.svg',
    'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_005344_col_019360.svg',
    'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_009236_col_018316.svg',
    'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_012680_col_023936.svg',
    'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_013256_col_007952.svg',
    'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_014800_col_020976.svg',
    'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_016756_col_063692.svg',
    'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_017360_col_024712.svg',
    'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_020824_col_018688.svg',
    'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_024128_col_010112.svg',
    'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_024836_col_055124.svg',
    'KLF14-B6NTAC-MAT-18.2b  58-16 C1 - 2016-02-03 11.10.52_row_005424_col_006896.svg',
    'KLF14-B6NTAC-MAT-18.2b  58-16 C1 - 2016-02-03 11.10.52_row_006268_col_013820.svg',
    'KLF14-B6NTAC-MAT-18.2b  58-16 C1 - 2016-02-03 11.10.52_row_013820_col_057052.svg',
    'KLF14-B6NTAC-MAT-18.2b  58-16 C1 - 2016-02-03 11.10.52_row_014272_col_008064.svg',
    'KLF14-B6NTAC-MAT-18.2b  58-16 C1 - 2016-02-03 11.10.52_row_017808_col_012400.svg',
    'KLF14-B6NTAC-MAT-18.2d  60-16 C1 - 2016-02-03 13.13.57_row_007296_col_010640.svg',
    'KLF14-B6NTAC-MAT-18.2d  60-16 C1 - 2016-02-03 13.13.57_row_013856_col_014128.svg',
    'KLF14-B6NTAC-MAT-18.2d  60-16 C1 - 2016-02-03 13.13.57_row_018380_col_063068.svg',
    'KLF14-B6NTAC-MAT-18.2d  60-16 C1 - 2016-02-03 13.13.57_row_020448_col_013824.svg',
    'KLF14-B6NTAC-MAT-18.2d  60-16 C1 - 2016-02-03 13.13.57_row_024076_col_020404.svg',
    'KLF14-B6NTAC-MAT-18.2g  63-16 C1 - 2016-02-03 16.58.52_row_010128_col_013536.svg',
    'KLF14-B6NTAC-MAT-18.2g  63-16 C1 - 2016-02-03 16.58.52_row_015776_col_010976.svg',
    'KLF14-B6NTAC-MAT-18.2g  63-16 C1 - 2016-02-03 16.58.52_row_015984_col_026832.svg',
    'KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52_row_005428_col_058372.svg',
    'KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52_row_012404_col_054316.svg',
    'KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52_row_013604_col_024644.svg',
    'KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52_row_014628_col_069148.svg',
    'KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52_row_018384_col_014688.svg',
    'KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52_row_019340_col_017348.svg',
    'KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52_row_020128_col_010096.svg',
    'KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52_row_022000_col_015568.svg',
    'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_row_006880_col_017808.svg',
    'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_row_008212_col_015364.svg',
    'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_row_011004_col_005988.svg',
    'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_row_015004_col_010364.svg',
    'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_row_018992_col_005952.svg',
    'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_row_019556_col_057972.svg',
    'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_row_021812_col_022916.svg',
    'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_row_022208_col_018128.svg',
    'KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11_row_010084_col_058476.svg',
    'KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11_row_012208_col_007472.svg',
    'KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11_row_013680_col_019152.svg',
    'KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11_row_016260_col_058300.svg',
    'KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11_row_019220_col_061724.svg',
    'KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11_row_020048_col_028896.svg',
    'KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11_row_021012_col_057844.svg',
    'KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11_row_023236_col_011084.svg',
    'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_006124_col_082236.svg',
    'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_007436_col_019092.svg',
    'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_009296_col_029664.svg',
    'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_015872_col_019456.svg',
    'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_016556_col_010292.svg',
    'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_023100_col_009220.svg',
    'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_023728_col_011904.svg',
    'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_031860_col_033476.svg',
    'KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32_row_004256_col_017552.svg',
    'KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32_row_005424_col_010432.svg',
    'KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32_row_006412_col_012484.svg',
    'KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32_row_012144_col_007056.svg',
    'KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32_row_013012_col_019820.svg',
    'KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32_row_031172_col_025996.svg',
    'KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32_row_034628_col_040116.svg',
    'KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32_row_035948_col_041492.svg'
]

# get v2 of the hand traced contours
hand_file_svg_list = [os.path.join(hand_traced_dir, x) for x in hand_file_svg_list]

# create a table with the number of different hand traced regions per animal
hand_traced_table = table_of_hand_traced_regions(hand_file_svg_list)

# save dataframe
if SAVEFIG:
    hand_traced_table.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_cellpop_v2_hand_traced_objects_count.csv'))

# total number of sampled windows
print('Total number of windows = ' + str(np.sum(hand_traced_table['Windows'])))
print('Total number of windows with cells = ' + str(np.sum(hand_traced_table['Windows with cells'])))

# total number of "Other" and background areas
print('Total number of Other areas = ' + str(np.sum(hand_traced_table['Other'])))
print('Total number of Background areas = ' + str(np.sum(hand_traced_table['Background'])))

# aggregate by sex and genotype
idx_f = hand_traced_table['Sex'] == 'f'
idx_m = hand_traced_table['Sex'] == 'm'
idx_pat = hand_traced_table['Genotype'] == 'PAT'
idx_mat = hand_traced_table['Genotype'] == 'MAT'

print('f PAT: ' + str(np.sum(hand_traced_table.loc[idx_f * idx_pat, 'Cells'])))
print('f MAT: ' + str(np.sum(hand_traced_table.loc[idx_f * idx_mat, 'Cells'])))
print('m PAT: ' + str(np.sum(hand_traced_table.loc[idx_m * idx_pat, 'Cells'])))
print('m MAT: ' + str(np.sum(hand_traced_table.loc[idx_m * idx_mat, 'Cells'])))

########################################################################################################################
## Whole animal studies (cull age, body weight, depot weight)
## USED IN PAPER
########################################################################################################################

## some data preparations

print('Min cull age: ' + str(metainfo['cull_age'].min()) + ' days')
print('Max cull age: ' + str(metainfo['cull_age'].max()) + ' days')

# we need numerical instead of categorical values for logistic regression
metainfo['ko_parent_num'] = (metainfo['ko_parent'] == 'MAT').astype(np.float32)
metainfo['genotype_num'] = (metainfo['genotype'] == 'KLF14-KO:Het').astype(np.float32)

# scale cull_age to avoid large condition numbers
metainfo['cull_age__'] = (metainfo['cull_age'] - np.mean(metainfo['cull_age'])) / np.std(metainfo['cull_age'])

# for convenience create two dataframes (female and male) with the data for the current depot
metainfo_f = metainfo[metainfo['sex'] == 'f']
metainfo_m = metainfo[metainfo['sex'] == 'm']

## effect of sex on body weight
########################################################################################################################

bw_model = sm.RLM.from_formula('BW ~ C(sex)', data=metainfo, subset=metainfo['ko_parent']=='PAT', M=sm.robust.norms.HuberT()).fit()
print(bw_model.summary())
print(bw_model.pvalues)
print('Males are ' + str(bw_model.params['C(sex)[T.m]'] / bw_model.params['Intercept'] * 100)
      + ' % larger than females')


## effect of cull age on body weight
########################################################################################################################

bw_model_f = sm.OLS.from_formula('BW ~ cull_age', data=metainfo_f).fit()
print(bw_model_f.summary())
bw_model_m = sm.OLS.from_formula('BW ~ cull_age', data=metainfo_m).fit()
print(bw_model_m.summary())

# logistic regression of parent ~ cull_age
cull_model_f = smf.logit('ko_parent_num ~ cull_age', data=metainfo_f).fit()
print(cull_model_f.summary())
cull_model_m = smf.logit('ko_parent_num ~ cull_age', data=metainfo_m).fit()
print(cull_model_m.summary())

# logistic regression of genotype ~ cull_age
cull_model_f = smf.logit('genotype_num ~ cull_age', data=metainfo_f).fit()
print(cull_model_f.summary())
cull_model_m = smf.logit('genotype_num ~ cull_age', data=metainfo_m).fit()
print(cull_model_m.summary())

# does cull_age make a difference in the BW ~ parent model?
bw_null_model_f = sm.OLS.from_formula('BW ~ C(ko_parent)', data=metainfo_f).fit()
bw_null_model_m = sm.OLS.from_formula('BW ~ C(ko_parent)', data=metainfo_m).fit()
bw_model_f = sm.OLS.from_formula('BW ~ C(ko_parent) * cull_age__', data=metainfo_f).fit()
bw_model_m = sm.OLS.from_formula('BW ~ C(ko_parent) * cull_age__', data=metainfo_m).fit()

print(bw_null_model_f.summary())
print(bw_null_model_m.summary())
print(bw_model_f.summary())
print(bw_model_m.summary())

print('Female')
null_model = bw_null_model_f
alt_model = bw_model_f
lr, pval = cytometer.stats.lrtest(null_model.llf, alt_model.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
print('p-val: ' + pval_text)
print('AIC_null=' + '{0:.2f}'.format(null_model.aic) + ', AIC_alt=' + '{0:.2f}'.format(alt_model.aic)
      + ', ΔAIC=' + '{0:.2f}'.format(alt_model.aic - null_model.aic))

print('Male')
null_model = bw_null_model_m
alt_model = bw_model_m
lr, pval = cytometer.stats.lrtest(null_model.llf, alt_model.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.3g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
print('p-val: ' + pval_text)
print('AIC_null=' + '{0:.2f}'.format(null_model.aic) + ', AIC_alt=' + '{0:.2f}'.format(alt_model.aic)
      + ', ΔAIC=' + '{0:.2f}'.format(alt_model.aic - null_model.aic))

print(bw_null_model_f.summary())
print(bw_model_f.summary())

# extract coefficients, errors and p-values from models
# 'BW ~ C(ko_parent) * cull_age__'
df_coeff_f, df_ci_lo_f, df_ci_hi_f, df_pval_f = \
    cytometer.stats.models_coeff_ci_pval([bw_model_f])
df_coeff_m, df_ci_lo_m, df_ci_hi_m, df_pval_m = \
    cytometer.stats.models_coeff_ci_pval([bw_model_m])

# multitest correction using Benjamini-Yekuteli
_, df_corrected_pval_f, _, _ = multipletests(df_pval_f.values.flatten(), method='fdr_by', alpha=0.05, returnsorted=False)
df_corrected_pval_f = pd.DataFrame(df_corrected_pval_f.reshape(df_pval_f.shape), columns=df_pval_f.columns)
_, df_corrected_pval_m, _, _ = multipletests(df_pval_m.values.flatten(), method='fdr_by', alpha=0.05, returnsorted=False)
df_corrected_pval_m = pd.DataFrame(df_corrected_pval_m.reshape(df_pval_m.shape), columns=df_pval_m.columns)

# convert p-values to asterisks
df_asterisk_f = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval_f, brackets=False), columns=df_coeff_f.columns)
df_asterisk_m = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval_m, brackets=False), columns=df_coeff_m.columns)
df_corrected_asterisk_f = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_corrected_pval_f, brackets=False), columns=df_coeff_f.columns)
df_corrected_asterisk_m = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_corrected_pval_m, brackets=False), columns=df_coeff_m.columns)

if SAVEFIG:
    # plot body weight vs. age of culling
    plt.clf()
    # 'BW ~ C(ko_parent) * cull_age__'
    cytometer.stats.plot_linear_regression(bw_model_f, metainfo_f,
                                           ind_var='cull_age__', other_vars={'ko_parent': 'PAT'}, dep_var='BW',
                                           sx=np.std(metainfo['cull_age']), tx=np.mean(metainfo['cull_age']),
                                           c='C2', marker='o', line_label='f PAT')
    cytometer.stats.plot_linear_regression(bw_model_f, metainfo_f,
                                           ind_var='cull_age__', other_vars={'ko_parent': 'MAT'}, dep_var='BW',
                                           sx=np.std(metainfo['cull_age']), tx=np.mean(metainfo['cull_age']),
                                           c='C3', marker='o', line_label='f MAT')
    cytometer.stats.plot_linear_regression(bw_model_m, metainfo_m,
                                           ind_var='cull_age__', other_vars={'ko_parent': 'PAT'}, dep_var='BW',
                                           sx=np.std(metainfo['cull_age']), tx=np.mean(metainfo['cull_age']),
                                           c='C4', marker='o', line_label='m PAT')
    cytometer.stats.plot_linear_regression(bw_model_m, metainfo_m,
                                           ind_var='cull_age__', other_vars={'ko_parent': 'MAT'}, dep_var='BW',
                                           sx=np.std(metainfo['cull_age']), tx=np.mean(metainfo['cull_age']),
                                           c='C5', marker='o', line_label='m MAT')

    plt.xlabel('Cull age (days)', fontsize=14)
    plt.ylabel('Body weight (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_bw_vs_cull_age.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_bw_vs_cull_age.svg'))

## effect of parent and genotype on body weight
########################################################################################################################

# robust model BW ~ parent * genotype for female/male to account for outliers
bw_model_f = sm.RLM.from_formula('BW ~ C(ko_parent) * C(genotype)', data=metainfo_f, M=sm.robust.norms.HuberT()).fit()
bw_model_m = sm.RLM.from_formula('BW ~ C(ko_parent) * C(genotype)', data=metainfo_m, M=sm.robust.norms.HuberT()).fit()

print(bw_model_f.summary())
print(bw_model_m.summary())

# refit removing the parent variable
bw_model_f = sm.RLM.from_formula('BW ~ C(genotype)', data=metainfo_f, M=sm.robust.norms.HuberT()).fit()
bw_model_m = sm.RLM.from_formula('BW ~ C(genotype)', data=metainfo_m, M=sm.robust.norms.HuberT()).fit()

print(bw_model_f.summary())
print(bw_model_m.summary())

if SAVEFIG:
    plt.clf()
    plt.gcf().set_size_inches([5.48, 4.8 ])

    ax = sns.swarmplot(x='sex', y='BW', hue='genotype', data=metainfo, dodge=True, palette=['C2', 'C3'])
    plt.xlabel('')
    plt.ylabel('Body weight (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.xticks([0, 1], labels=['Female', 'Male'])
    ax.get_legend().set_title('')
    ax.legend(['WT', 'Het'], loc='lower right', fontsize=12)

    plt.plot([-0.2, -0.2, 0.2, 0.2], [42, 44, 44, 42], 'k', lw=1.5)
    pval_text = '$p$=' + '{0:.2f}'.format(bw_model_f.pvalues['C(genotype)[T.KLF14-KO:Het]']) + \
                ' ' + cytometer.stats.pval_to_asterisk(bw_model_f.pvalues['C(genotype)[T.KLF14-KO:Het]'])
    plt.text(0, 44.5, pval_text, ha='center', va='bottom', fontsize=14)
    plt.plot([0.8, 0.8, 1.2, 1.2], [52, 54, 54, 52], 'k', lw=1.5)
    pval_text = '$p$=' + '{0:.2f}'.format(bw_model_m.pvalues['C(genotype)[T.KLF14-KO:Het]']) + \
                ' ' + cytometer.stats.pval_to_asterisk(bw_model_m.pvalues['C(genotype)[T.KLF14-KO:Het]'])
    plt.text(1, 54.5, pval_text, ha='center', va='bottom', fontsize=14)
    plt.ylim(18, 58)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_swarm_bw_genotype.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_swarm_bw_genotype.svg'))

# refit removing the genotype variable
bw_model_f = sm.RLM.from_formula('BW ~ C(ko_parent)', data=metainfo_f, M=sm.robust.norms.HuberT()).fit()
bw_model_m = sm.RLM.from_formula('BW ~ C(ko_parent)', data=metainfo_m, M=sm.robust.norms.HuberT()).fit()

print(bw_model_f.summary())
print(bw_model_f.pvalues)
print('MAT females are ' + str(bw_model_f.params['C(ko_parent)[T.MAT]'] / bw_model_f.params['Intercept'] * 100)
      + ' % larger than PATs')

print(bw_model_m.summary())

if SAVEFIG:
    plt.clf()
    plt.gcf().set_size_inches([5.48, 4.8 ])

    ax = sns.swarmplot(x='sex', y='BW', hue='ko_parent', data=metainfo, dodge=True)
    plt.xlabel('')
    plt.ylabel('Body weight (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.xticks([0, 1], labels=['Female', 'Male'])
    ax.get_legend().set_title('')
    ax.legend(loc='lower right', fontsize=12)

    plt.plot([-0.2, -0.2, 0.2, 0.2], [42, 44, 44, 42], 'k', lw=1.5)
    pval_text = '$p$=' + '{0:.4f}'.format(bw_model_f.pvalues['C(ko_parent)[T.MAT]']) + \
                ' ' + cytometer.stats.pval_to_asterisk(bw_model_f.pvalues['C(ko_parent)[T.MAT]'])
    plt.text(0, 44.5, pval_text, ha='center', va='bottom', fontsize=14)
    plt.plot([0.8, 0.8, 1.2, 1.2], [52, 54, 54, 52], 'k', lw=1.5)
    pval_text = '$p$=' + '{0:.2f}'.format(bw_model_m.pvalues['C(ko_parent)[T.MAT]']) + \
                ' ' + cytometer.stats.pval_to_asterisk(bw_model_m.pvalues['C(ko_parent)[T.MAT]'])
    plt.text(1, 54.5, pval_text, ha='center', va='bottom', fontsize=14)
    plt.ylim(18, 58)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_swarm_bw_parent.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_swarm_bw_parent.svg'))

## effect of genotype, parent and body weight on depot weight
########################################################################################################################

# scale BW to avoid large condition numbers
BW_mean = metainfo['BW'].mean()
metainfo['BW__'] = metainfo['BW'] / BW_mean

# update the sub-dataframes we created for convenience
metainfo_f = metainfo[metainfo['sex'] == 'f']
metainfo_m = metainfo[metainfo['sex'] == 'm']

## double-check that no parent effect is found on DW if BW is not used
gwat_model_f = sm.OLS.from_formula('gWAT ~ C(ko_parent)', data=metainfo_f).fit()
gwat_model_m = sm.OLS.from_formula('gWAT ~ C(ko_parent)', data=metainfo_m).fit()
sqwat_model_f = sm.OLS.from_formula('SC ~ C(ko_parent)', data=metainfo_f).fit()
sqwat_model_m = sm.OLS.from_formula('SC ~ C(ko_parent)', data=metainfo_m).fit()

# extract coefficients, errors and p-values from models
model_names = ['gwat_model_f',
         'sqwat_model_f',
         'gwat_model_m',
         'sqwat_model_m']
df_coeff, df_ci_lo, df_ci_hi, df_pval = \
    cytometer.stats.models_coeff_ci_pval(
        [gwat_model_f,
         sqwat_model_f,
         gwat_model_m,
         sqwat_model_m],
    model_names=model_names)

# multitest correction using Benjamini-Yekuteli
_, df_corrected_pval, _, _ = multipletests(df_pval.values.flatten(), method='fdr_by', alpha=0.05, returnsorted=False)
df_corrected_pval = pd.DataFrame(df_corrected_pval.reshape(df_pval.shape), columns=df_pval.columns, index=model_names)

# convert p-values to asterisks
df_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval, brackets=False), columns=df_coeff.columns,
                           index=model_names)
df_corrected_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_corrected_pval, brackets=False),
                                     columns=df_coeff.columns, index=model_names)

## depot ~ BW * genotype models

# models for Likelihood Ratio Test, to check whether genotype variable has an effect
gwat_null_model_f = sm.OLS.from_formula('gWAT ~ BW__', data=metainfo_f).fit()
gwat_null_model_m = sm.OLS.from_formula('gWAT ~ BW__', data=metainfo_m).fit()
sqwat_null_model_f = sm.OLS.from_formula('SC ~ BW__', data=metainfo_f).fit()
sqwat_null_model_m = sm.OLS.from_formula('SC ~ BW__', data=metainfo_m).fit()

gwat_model_f = sm.OLS.from_formula('gWAT ~ BW__ * C(genotype)', data=metainfo_f).fit()
gwat_model_m = sm.OLS.from_formula('gWAT ~ BW__ * C(genotype)', data=metainfo_m).fit()
sqwat_model_f = sm.OLS.from_formula('SC ~ BW__ * C(genotype)', data=metainfo_f).fit()
sqwat_model_m = sm.OLS.from_formula('SC ~ BW__ * C(genotype)', data=metainfo_m).fit()

# Likelihood ratio tests of the genotype variable
print('Likelihood Ratio Tests: Genotype')

print('Female')
lr, pval = cytometer.stats.lrtest(gwat_null_model_f.llf, gwat_model_f.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
print('Gonadal: ' + pval_text)
print('Gonadal: AIC_null=' + '{0:.2f}'.format(gwat_null_model_f.aic) + ', AIC_alt=' + '{0:.2f}'.format(gwat_model_f.aic))

lr, pval = cytometer.stats.lrtest(sqwat_null_model_f.llf, sqwat_model_f.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
print('Subcutaneous: ' + pval_text)
print('Subcutaneous: AIC_null=' + '{0:.2f}'.format(sqwat_null_model_f.aic) + ', AIC_alt=' + '{0:.2f}'.format(sqwat_model_f.aic))

print('Male')
lr, pval = cytometer.stats.lrtest(gwat_null_model_m.llf, gwat_model_m.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
print('Gonadal: ' + pval_text)
print('Gonadal: AIC_null=' + '{0:.2f}'.format(gwat_null_model_m.aic) + ', AIC_alt=' + '{0:.2f}'.format(gwat_model_m.aic))

lr, pval = cytometer.stats.lrtest(sqwat_null_model_m.llf, sqwat_model_m.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
print('Subcutaneous: ' + pval_text)
print('Subcutaneous: AIC_null=' + '{0:.2f}'.format(sqwat_null_model_m.aic) + ', AIC_alt=' + '{0:.2f}'.format(sqwat_model_m.aic))

## fit robust linear models DW ~ BW__, stratified by sex and genotype
# female WT and Het
gwat_model_f_wt = sm.RLM.from_formula('gWAT ~ BW__', data=metainfo_f, subset=metainfo_f['genotype'] == 'KLF14-KO:WT', M=sm.robust.norms.HuberT()).fit()
gwat_model_f_het = sm.RLM.from_formula('gWAT ~ BW__', data=metainfo_f, subset=metainfo_f['genotype'] == 'KLF14-KO:Het', M=sm.robust.norms.HuberT()).fit()
sqwat_model_f_wt = sm.RLM.from_formula('SC ~ BW__', data=metainfo_f, subset=metainfo_f['genotype'] == 'KLF14-KO:WT', M=sm.robust.norms.HuberT()).fit()
sqwat_model_f_het = sm.RLM.from_formula('SC ~ BW__', data=metainfo_f, subset=metainfo_f['genotype'] == 'KLF14-KO:Het', M=sm.robust.norms.HuberT()).fit()

# male PAT and MAT
gwat_model_m_wt = sm.RLM.from_formula('gWAT ~ BW__', data=metainfo_m, subset=metainfo_m['genotype'] == 'KLF14-KO:WT', M=sm.robust.norms.HuberT()).fit()
gwat_model_m_het = sm.RLM.from_formula('gWAT ~ BW__', data=metainfo_m, subset=metainfo_m['genotype'] == 'KLF14-KO:Het', M=sm.robust.norms.HuberT()).fit()
sqwat_model_m_wt = sm.RLM.from_formula('SC ~ BW__', data=metainfo_m, subset=metainfo_m['genotype'] == 'KLF14-KO:WT', M=sm.robust.norms.HuberT()).fit()
sqwat_model_m_het = sm.RLM.from_formula('SC ~ BW__', data=metainfo_m, subset=metainfo_m['genotype'] == 'KLF14-KO:Het', M=sm.robust.norms.HuberT()).fit()

# extract coefficients, errors and p-values from models
model_names = ['gwat_model_f_wt', 'gwat_model_f_het',
         'sqwat_model_f_wt', 'sqwat_model_f_het',
         'gwat_model_m_wt', 'gwat_model_m_het',
         'sqwat_model_m_wt', 'sqwat_model_m_het']
df_coeff, df_ci_lo, df_ci_hi, df_pval = \
    cytometer.stats.models_coeff_ci_pval(
        [gwat_model_f_wt, gwat_model_f_het,
         sqwat_model_f_wt, sqwat_model_f_het,
         gwat_model_m_wt, gwat_model_m_het,
         sqwat_model_m_wt, sqwat_model_m_het],
    model_names=model_names)

# multitest correction using Benjamini-Yekuteli
_, df_corrected_pval, _, _ = multipletests(df_pval.values.flatten(), method='fdr_by', alpha=0.05, returnsorted=False)
df_corrected_pval = pd.DataFrame(df_corrected_pval.reshape(df_pval.shape), columns=df_pval.columns, index=model_names)

# convert p-values to asterisks
df_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval, brackets=False), columns=df_coeff.columns,
                           index=model_names)
df_corrected_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_corrected_pval, brackets=False),
                                     columns=df_coeff.columns, index=model_names)

if SAVEFIG:
    df_concat = pd.concat([df_coeff, df_pval, df_asterisk, df_corrected_pval, df_corrected_asterisk],
                          axis=1)
    idx = list(interleave(np.array_split(range(df_concat.shape[1]), 5)))
    df_concat = df_concat.iloc[:, idx]
    df_concat.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_depot_weight_models_coeffs_pvals_genotype.csv'))

if SAVEFIG:
    plt.clf()
    plt.subplot(221)
    sex = 'f'
    cytometer.stats.plot_linear_regression(gwat_null_model_f, metainfo_f, 'BW__',
                                           other_vars={'sex':sex}, sx=BW_mean, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(gwat_model_f_wt, metainfo_f, 'BW__',
                                           other_vars={'sex':sex, 'genotype':'KLF14-KO:WT'},
                                           dep_var='gWAT', sx=BW_mean, c='C2', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(gwat_model_f_het, metainfo_f, 'BW__',
                                           other_vars={'sex':sex, 'genotype':'KLF14-KO:Het'},
                                           dep_var='gWAT', sx=BW_mean, c='C3', marker='+',
                                           line_label='Het')
    plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    plt.ylim(0, 2.1)
    plt.tick_params(labelsize=14)
    plt.title('Female', fontsize=14)
    plt.ylabel('Gonadal\ndepot weight (g)', fontsize=14)

    plt.subplot(222)
    sex = 'm'
    cytometer.stats.plot_linear_regression(gwat_null_model_m, metainfo_m, 'BW__',
                                           other_vars={'sex':sex}, sx=BW_mean, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(gwat_model_m_wt, metainfo_m, 'BW__',
                                           other_vars={'sex':sex, 'genotype':'KLF14-KO:WT'},
                                           dep_var='gWAT', sx=BW_mean, c='C2', marker='x',
                                           line_label='KLF14-KO:WT')
    cytometer.stats.plot_linear_regression(gwat_model_m_het, metainfo_m, 'BW__',
                                           other_vars={'sex':sex, 'genotype':'KLF14-KO:Het'},
                                           dep_var='gWAT', sx=BW_mean, c='C3', marker='+',
                                           line_label='KLF14-KO:Het')
    plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    plt.ylim(0, 2.1)
    plt.tick_params(labelsize=14)
    plt.title('Male', fontsize=14)

    plt.subplot(223)
    sex = 'f'
    cytometer.stats.plot_linear_regression(sqwat_null_model_f, metainfo_f, 'BW__',
                                           other_vars={'sex':sex}, sx=BW_mean, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(sqwat_model_f_wt, metainfo_f, 'BW__',
                                           other_vars={'sex':sex, 'genotype':'KLF14-KO:WT'},
                                           dep_var='SC', sx=BW_mean, c='C2', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(sqwat_model_f_het, metainfo_f, 'BW__',
                                           other_vars={'sex':sex, 'genotype':'KLF14-KO:Het'},
                                           dep_var='SC', sx=BW_mean, c='C3', marker='+',
                                           line_label='Het')
    plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    plt.tick_params(labelsize=14)
    plt.ylim(0, 2.1)
    plt.xlabel('Body weight (g)', fontsize=14)
    plt.ylabel('Subcutaneous\ndepot weight (g)', fontsize=14)
    plt.legend(loc='upper right')

    plt.subplot(224)
    sex = 'm'
    cytometer.stats.plot_linear_regression(sqwat_null_model_m, metainfo_m, 'BW__',
                                           other_vars={'sex':sex}, sx=BW_mean, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(sqwat_model_m_wt, metainfo_m, 'BW__',
                                           other_vars={'sex':sex, 'genotype':'KLF14-KO:WT'},
                                           dep_var='SC', sx=BW_mean, c='C2', marker='x',
                                           line_label='KLF14-KO:WT')
    cytometer.stats.plot_linear_regression(sqwat_model_m_het, metainfo_m, 'BW__',
                                           other_vars={'sex':sex, 'genotype':'KLF14-KO:Het'},
                                           dep_var='SC', sx=BW_mean, c='C3', marker='+',
                                           line_label='KLF14-KO:Het')
    plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    plt.ylim(0, 2.1)
    plt.tick_params(labelsize=14)
    plt.xlabel('Body weight (g)', fontsize=14)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_depot_linear_model_genotype.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_depot_linear_model_genotype.svg'))


## depot ~ BW * parent models

# models for Likelihood Ratio Test, to check whether parent variable has an effect
gwat_null_model_f = sm.OLS.from_formula('gWAT ~ BW__', data=metainfo_f).fit()
gwat_null_model_m = sm.OLS.from_formula('gWAT ~ BW__', data=metainfo_m).fit()
sqwat_null_model_f = sm.OLS.from_formula('SC ~ BW__', data=metainfo_f).fit()
sqwat_null_model_m = sm.OLS.from_formula('SC ~ BW__', data=metainfo_m).fit()

gwat_model_f = sm.OLS.from_formula('gWAT ~ BW__ * C(ko_parent)', data=metainfo_f).fit()
gwat_model_m = sm.OLS.from_formula('gWAT ~ BW__ * C(ko_parent)', data=metainfo_m).fit()
sqwat_model_f = sm.OLS.from_formula('SC ~ BW__ * C(ko_parent)', data=metainfo_f).fit()
sqwat_model_m = sm.OLS.from_formula('SC ~ BW__ * C(ko_parent)', data=metainfo_m).fit()

# Likelihood ratio tests of the parent variable
print('Likelihood Ratio Tests: Parent')

print('Female')
lr, pval = cytometer.stats.lrtest(gwat_null_model_f.llf, gwat_model_f.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
print('Gonadal: ' + pval_text)
print('Gonadal: AIC_null=' + '{0:.2f}'.format(gwat_null_model_f.aic) + ', AIC_alt=' + '{0:.2f}'.format(gwat_model_f.aic))

lr, pval = cytometer.stats.lrtest(sqwat_null_model_f.llf, sqwat_model_f.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
print('Subcutaneous: ' + pval_text)
print('Subcutaneous: AIC_null=' + '{0:.2f}'.format(sqwat_null_model_f.aic) + ', AIC_alt=' + '{0:.2f}'.format(sqwat_model_f.aic))

print('Male')
lr, pval = cytometer.stats.lrtest(gwat_null_model_m.llf, gwat_model_m.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
print('Gonadal: ' + pval_text)
print('Gonadal: AIC_null=' + '{0:.2f}'.format(gwat_null_model_m.aic) + ', AIC_alt=' + '{0:.2f}'.format(gwat_model_m.aic))

lr, pval = cytometer.stats.lrtest(sqwat_null_model_m.llf, sqwat_model_m.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
print('Subcutaneous: ' + pval_text)
print('Subcutaneous: AIC_null=' + '{0:.2f}'.format(sqwat_null_model_m.aic) + ', AIC_alt=' + '{0:.2f}'.format(sqwat_model_m.aic))

## fit robust linear models DW ~ BW__, stratified by sex and parent
# female PAT and MAT
gwat_model_f_pat = sm.RLM.from_formula('gWAT ~ BW__', data=metainfo_f, subset=metainfo_f['ko_parent']=='PAT', M=sm.robust.norms.HuberT()).fit()
gwat_model_f_mat = sm.RLM.from_formula('gWAT ~ BW__', data=metainfo_f, subset=metainfo_f['ko_parent']=='MAT', M=sm.robust.norms.HuberT()).fit()
sqwat_model_f_pat = sm.RLM.from_formula('SC ~ BW__', data=metainfo_f, subset=metainfo_f['ko_parent']=='PAT', M=sm.robust.norms.HuberT()).fit()
sqwat_model_f_mat = sm.RLM.from_formula('SC ~ BW__', data=metainfo_f, subset=metainfo_f['ko_parent']=='MAT', M=sm.robust.norms.HuberT()).fit()

# male PAT and MAT
gwat_model_m_pat = sm.RLM.from_formula('gWAT ~ BW__', data=metainfo_m, subset=metainfo_m['ko_parent']=='PAT', M=sm.robust.norms.HuberT()).fit()
gwat_model_m_mat = sm.RLM.from_formula('gWAT ~ BW__', data=metainfo_m, subset=metainfo_m['ko_parent']=='MAT', M=sm.robust.norms.HuberT()).fit()
sqwat_model_m_pat = sm.RLM.from_formula('SC ~ BW__', data=metainfo_m, subset=metainfo_m['ko_parent']=='PAT', M=sm.robust.norms.HuberT()).fit()
sqwat_model_m_mat = sm.RLM.from_formula('SC ~ BW__', data=metainfo_m, subset=metainfo_m['ko_parent']=='MAT', M=sm.robust.norms.HuberT()).fit()

# extract coefficients, errors and p-values from models
model_names = ['gwat_model_f_pat', 'gwat_model_f_mat',
         'sqwat_model_f_pat', 'sqwat_model_f_mat',
         'gwat_model_m_pat', 'gwat_model_m_mat',
         'sqwat_model_m_pat', 'sqwat_model_m_mat']
df_coeff, df_ci_lo, df_ci_hi, df_pval = \
    cytometer.stats.models_coeff_ci_pval(
        [gwat_model_f_pat, gwat_model_f_mat,
         sqwat_model_f_pat, sqwat_model_f_mat,
         gwat_model_m_pat, gwat_model_m_mat,
         sqwat_model_m_pat, sqwat_model_m_mat],
    model_names=model_names)

# multitest correction using Benjamini-Yekuteli
_, df_corrected_pval, _, _ = multipletests(df_pval.values.flatten(), method='fdr_by', alpha=0.05, returnsorted=False)
df_corrected_pval = pd.DataFrame(df_corrected_pval.reshape(df_pval.shape), columns=df_pval.columns, index=model_names)

# convert p-values to asterisks
df_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval, brackets=False), columns=df_coeff.columns,
                           index=model_names)
df_corrected_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_corrected_pval, brackets=False),
                                     columns=df_coeff.columns, index=model_names)

if SAVEFIG:
    df_concat = pd.concat([df_coeff, df_pval, df_asterisk, df_corrected_pval, df_corrected_asterisk],
                          axis=1)
    idx = list(interleave(np.array_split(range(df_concat.shape[1]), 5)))
    df_concat = df_concat.iloc[:, idx]
    df_concat.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_depot_weight_models_coeffs_pvals_parent.csv'))

if SAVEFIG:
    plt.clf()
    plt.subplot(221)
    sex = 'f'
    cytometer.stats.plot_linear_regression(gwat_null_model_f, metainfo_f, 'BW__',
                                           other_vars={'sex':sex}, sx=BW_mean, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(gwat_model_f_pat, metainfo_f, 'BW__',
                                           other_vars={'sex':sex, 'ko_parent':'PAT'},
                                           dep_var='gWAT', sx=BW_mean, c='C0', marker='x',
                                           line_label='PAT')
    cytometer.stats.plot_linear_regression(gwat_model_f_mat, metainfo_f, 'BW__',
                                           other_vars={'sex':sex, 'ko_parent':'MAT'},
                                           dep_var='gWAT', sx=BW_mean, c='C1', marker='+',
                                           line_label='MAT')
    plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    plt.ylim(0, 2.1)
    plt.tick_params(labelsize=14)
    plt.title('Female', fontsize=14)
    plt.ylabel('Gonadal\ndepot weight (g)', fontsize=14)
    plt.legend(loc='lower right')

    plt.subplot(222)
    sex = 'm'
    cytometer.stats.plot_linear_regression(gwat_null_model_m, metainfo_m, 'BW__',
                                           other_vars={'sex':sex}, sx=BW_mean, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(gwat_model_m_pat, metainfo_m, 'BW__',
                                           other_vars={'sex':sex, 'ko_parent':'PAT'},
                                           dep_var='gWAT', sx=BW_mean, c='C0', marker='x',
                                           line_label='PAT')
    cytometer.stats.plot_linear_regression(gwat_model_m_mat, metainfo_m, 'BW__',
                                           other_vars={'sex':sex, 'ko_parent':'MAT'},
                                           dep_var='gWAT', sx=BW_mean, c='C1', marker='+',
                                           line_label='MAT')
    plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    plt.ylim(0, 2.1)
    plt.tick_params(labelsize=14)
    plt.title('Male', fontsize=14)

    plt.subplot(223)
    sex = 'f'
    cytometer.stats.plot_linear_regression(sqwat_null_model_f, metainfo_f, 'BW__',
                                           other_vars={'sex':sex}, sx=BW_mean, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(sqwat_model_f_pat, metainfo_f, 'BW__',
                                           other_vars={'sex':sex, 'ko_parent':'PAT'},
                                           dep_var='SC', sx=BW_mean, c='C0', marker='x',
                                           line_label='PAT')
    cytometer.stats.plot_linear_regression(sqwat_model_f_mat, metainfo_f, 'BW__',
                                           other_vars={'sex':sex, 'ko_parent':'MAT'},
                                           dep_var='SC', sx=BW_mean, c='C1', marker='+',
                                           line_label='MAT')
    plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    plt.tick_params(labelsize=14)
    plt.ylim(0, 2.1)
    plt.xlabel('Body weight (g)', fontsize=14)
    plt.ylabel('Subcutaneous\ndepot weight (g)', fontsize=14)

    plt.subplot(224)
    sex = 'm'
    cytometer.stats.plot_linear_regression(sqwat_null_model_m, metainfo_m, 'BW__',
                                           other_vars={'sex':sex}, sx=BW_mean, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(sqwat_model_m_pat, metainfo_m, 'BW__',
                                           other_vars={'sex':sex, 'ko_parent':'PAT'},
                                           dep_var='SC', sx=BW_mean, c='C0', marker='x',
                                           line_label='PAT')
    cytometer.stats.plot_linear_regression(sqwat_model_m_mat, metainfo_m, 'BW__',
                                           other_vars={'sex':sex, 'ko_parent':'MAT'},
                                           dep_var='SC', sx=BW_mean, c='C1', marker='+',
                                           line_label='MAT')
    plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    plt.ylim(0, 2.1)
    plt.tick_params(labelsize=14)
    plt.xlabel('Body weight (g)', fontsize=14)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_depot_linear_model_parent.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_depot_linear_model_parent.svg'))


########################################################################################################################
## Analyse cell populations from automatically segmented images in two depots: SQWAT and GWAT:
########################################################################################################################

## smoothed histograms
## USED IN THE PAPER
########################################################################################################################

depot = 'gwat'
# depot = 'sqwat'

if SAVEFIG:
    plt.clf()

    # f PAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'PAT')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())

    plt.subplot(221)
    plt.plot(area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.text(0.9, 0.9, 'female PAT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xticks([0, 10, 20])
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # f MAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())

    plt.subplot(222)
    plt.plot(area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.text(0.9, 0.9, 'female MAT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xticks([0, 10, 20])
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # m PAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'PAT')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())

    plt.subplot(223)
    plt.plot(area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.text(0.9, 0.9, 'male PAT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xticks([0, 10, 20])
    plt.xlim(-1.2, max_area_um2 * 1e-3)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)

    # m MAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'MAT')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())

    plt.subplot(224)
    plt.plot(area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.xticks([0, 10, 20])
    plt.text(0.9, 0.9, 'male MAT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)

    if depot == 'gwat':
        plt.suptitle('Gonadal', fontsize=14)
    elif depot == 'sqwat':
        plt.suptitle('Subcutaneous', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_smoothed_histo_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_smoothed_histo_' + depot + '.svg'))

if SAVEFIG:
    plt.clf()

    # f PAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'PAT')]
    histo = np.array(df['smoothed_histo'].tolist())
    histo_beg = stats.mstats.hdquantiles(histo, prob=0.025, axis=0)
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)
    histo_end = stats.mstats.hdquantiles(histo, prob=0.975, axis=0)

    plt.subplot(221)
    hist_max = histo_end.max()
    plt.fill_between(area_bin_centers * 1e-3, histo_beg[0,] / hist_max, histo_end[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.fill_between(area_bin_centers * 1e-3, histo_q1[0,] / hist_max, histo_q3[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.plot(area_bin_centers * 1e-3, histo_q2[0,] / hist_max, 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.text(0.9, 0.9, 'female PAT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # f MAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')]
    histo = np.array(df['smoothed_histo'].tolist())
    histo_beg = stats.mstats.hdquantiles(histo, prob=0.025, axis=0)
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)
    histo_end = stats.mstats.hdquantiles(histo, prob=0.975, axis=0)

    plt.subplot(222)
    hist_max = histo_end.max()
    plt.fill_between(area_bin_centers * 1e-3, histo_beg[0,] / hist_max, histo_end[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.fill_between(area_bin_centers * 1e-3, histo_q1[0,] / hist_max, histo_q3[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.plot(area_bin_centers * 1e-3, histo_q2[0,] / hist_max, 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.text(0.9, 0.9, 'female MAT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # m PAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'PAT')]
    histo = np.array(df['smoothed_histo'].tolist())
    histo_beg = stats.mstats.hdquantiles(histo, prob=0.025, axis=0)
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)
    histo_end = stats.mstats.hdquantiles(histo, prob=0.975, axis=0)

    plt.subplot(223)
    hist_max = histo_end.max()
    plt.fill_between(area_bin_centers * 1e-3, histo_beg[0,] / hist_max, histo_end[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.fill_between(area_bin_centers * 1e-3, histo_q1[0,] / hist_max, histo_q3[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.plot(area_bin_centers * 1e-3, histo_q2[0,] / hist_max, 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)
    plt.text(0.9, 0.9, 'male PAT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # m MAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'MAT')]
    histo = np.array(df['smoothed_histo'].tolist())
    histo_beg = stats.mstats.hdquantiles(histo, prob=0.025, axis=0)
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)
    histo_end = stats.mstats.hdquantiles(histo, prob=0.975, axis=0)

    plt.subplot(224)
    hist_max = histo_end.max()
    plt.fill_between(area_bin_centers * 1e-3, histo_beg[0,] / hist_max, histo_end[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.fill_between(area_bin_centers * 1e-3, histo_q1[0,] / hist_max, histo_q3[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.plot(area_bin_centers * 1e-3, histo_q2[0,] / hist_max, 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)
    plt.text(0.9, 0.9, 'male MAT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    if depot == 'gwat':
        plt.suptitle('Gonadal', fontsize=14)
    elif depot == 'sqwat':
        plt.suptitle('Subcutaneous', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

## numerical quartiles and CIs associated to the histograms

idx_q1 = np.where(quantiles == 0.25)[0][0]
idx_q2 = np.where(quantiles == 0.50)[0][0]
idx_q3 = np.where(quantiles == 0.75)[0][0]

# f PAT
df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'PAT')]
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())
stderr_at_quantiles = np.array(df['stderr_at_quantiles'].to_list())
stderr_at_quantiles[:, [0, -1]] = np.nan  ## first and last values are artifacts of saving to the CSV file

# inverse-variance method to combine quantiles and sdterr values from multiple mice
areas_at_quantiles_hat, stderr_at_quantiles_hat = \
    cytometer.stats.inverse_variance_method(areas_at_quantiles, stderr_at_quantiles)

# compute combined value and CIs in 10^3 um^2 units
alpha = 0.05
q1_hat, q2_hat, q3_hat = areas_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]] * 1e-3
k = stats.norm.ppf(1 - alpha/2, loc=0, scale=1)  # multiplier for CI length (~1.96 for 95% CI)
q1_ci_lo, q2_ci_lo, q3_ci_lo = [q1_hat, q2_hat, q3_hat] - k * stderr_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_hi, q2_ci_hi, q3_ci_hi = [q1_hat, q2_hat, q3_hat] + k * stderr_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]] * 1e-3

print('f PAT')
print('\t' + '{0:.2f}'.format(q1_hat) + ' (' + '{0:.2f}'.format(q1_ci_lo) + ', ' + '{0:.2f}'.format(q1_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q2_hat) + ' (' + '{0:.2f}'.format(q2_ci_lo) + ', ' + '{0:.2f}'.format(q2_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q3_hat) + ' (' + '{0:.2f}'.format(q3_ci_lo) + ', ' + '{0:.2f}'.format(q3_ci_hi) + ')')

if SAVEFIG:
    plt.subplot(221)
    plt.plot([q1_hat, q1_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q2_hat, q2_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q3_hat, q3_hat], [0, 1], 'k', linewidth=1)

# f MAT
df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')]
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())
stderr_at_quantiles = np.array(df['stderr_at_quantiles'].to_list())
stderr_at_quantiles[:, [0, -1]] = np.nan  ## first and last values are artifacts of saving to the CSV file

# inverse-variance method to combine quantiles and sdterr values from multiple mice
areas_at_quantiles_hat, stderr_at_quantiles_hat = \
    cytometer.stats.inverse_variance_method(areas_at_quantiles, stderr_at_quantiles)

# compute combined value and CIs in 10^3 um^2 units
alpha = 0.05
q1_hat, q2_hat, q3_hat = areas_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]] * 1e-3
k = stats.norm.ppf(1 - alpha/2, loc=0, scale=1)  # multiplier for CI length (~1.96 for 95% CI)
q1_ci_lo, q2_ci_lo, q3_ci_lo = [q1_hat, q2_hat, q3_hat] - k * stderr_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_hi, q2_ci_hi, q3_ci_hi = [q1_hat, q2_hat, q3_hat] + k * stderr_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]] * 1e-3

print('f MAT')
print('\t' + '{0:.2f}'.format(q1_hat) + ' (' + '{0:.2f}'.format(q1_ci_lo) + ', ' + '{0:.2f}'.format(q1_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q2_hat) + ' (' + '{0:.2f}'.format(q2_ci_lo) + ', ' + '{0:.2f}'.format(q2_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q3_hat) + ' (' + '{0:.2f}'.format(q3_ci_lo) + ', ' + '{0:.2f}'.format(q3_ci_hi) + ')')

if SAVEFIG:
    plt.subplot(222)
    plt.plot([q1_hat, q1_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q2_hat, q2_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q3_hat, q3_hat], [0, 1], 'k', linewidth=1)

# m PAT
df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'PAT')]
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())
stderr_at_quantiles = np.array(df['stderr_at_quantiles'].to_list())
stderr_at_quantiles[:, [0, -1]] = np.nan  ## first and last values are artifacts of saving to the CSV file

# inverse-variance method to combine quantiles and sdterr values from multiple mice
areas_at_quantiles_hat, stderr_at_quantiles_hat = \
    cytometer.stats.inverse_variance_method(areas_at_quantiles, stderr_at_quantiles)

# compute combined value and CIs in 10^3 um^2 units
alpha = 0.05
q1_hat, q2_hat, q3_hat = areas_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]] * 1e-3
k = stats.norm.ppf(1 - alpha/2, loc=0, scale=1)  # multiplier for CI length (~1.96 for 95% CI)
q1_ci_lo, q2_ci_lo, q3_ci_lo = [q1_hat, q2_hat, q3_hat] - k * stderr_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_hi, q2_ci_hi, q3_ci_hi = [q1_hat, q2_hat, q3_hat] + k * stderr_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]] * 1e-3

print('m PAT')
print('\t' + '{0:.2f}'.format(q1_hat) + ' (' + '{0:.2f}'.format(q1_ci_lo) + ', ' + '{0:.2f}'.format(q1_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q2_hat) + ' (' + '{0:.2f}'.format(q2_ci_lo) + ', ' + '{0:.2f}'.format(q2_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q3_hat) + ' (' + '{0:.2f}'.format(q3_ci_lo) + ', ' + '{0:.2f}'.format(q3_ci_hi) + ')')

if SAVEFIG:
    plt.subplot(223)
    plt.plot([q1_hat, q1_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q2_hat, q2_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q3_hat, q3_hat], [0, 1], 'k', linewidth=1)

# m MAT
df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'MAT')]
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())
stderr_at_quantiles = np.array(df['stderr_at_quantiles'].to_list())
stderr_at_quantiles[:, [0, -1]] = np.nan  ## first and last values are artifacts of saving to the CSV file

# inverse-variance method to combine quantiles and sdterr values from multiple mice
areas_at_quantiles_hat, stderr_at_quantiles_hat = \
    cytometer.stats.inverse_variance_method(areas_at_quantiles, stderr_at_quantiles)

# compute combined value and CIs in 10^3 um^2 units
alpha = 0.05
q1_hat, q2_hat, q3_hat = areas_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]] * 1e-3
k = stats.norm.ppf(1 - alpha/2, loc=0, scale=1)  # multiplier for CI length (~1.96 for 95% CI)
q1_ci_lo, q2_ci_lo, q3_ci_lo = [q1_hat, q2_hat, q3_hat] - k * stderr_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_hi, q2_ci_hi, q3_ci_hi = [q1_hat, q2_hat, q3_hat] + k * stderr_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]] * 1e-3

print('m MAT')
print('\t' + '{0:.2f}'.format(q1_hat) + ' (' + '{0:.2f}'.format(q1_ci_lo) + ', ' + '{0:.2f}'.format(q1_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q2_hat) + ' (' + '{0:.2f}'.format(q2_ci_lo) + ', ' + '{0:.2f}'.format(q2_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q3_hat) + ' (' + '{0:.2f}'.format(q3_ci_lo) + ', ' + '{0:.2f}'.format(q3_ci_hi) + ')')

if SAVEFIG:
    plt.subplot(224)
    plt.plot([q1_hat, q1_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q2_hat, q2_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q3_hat, q3_hat], [0, 1], 'k', linewidth=1)

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_smoothed_histo_quartiles_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_smoothed_histo_quartiles_' + depot + '.svg'))

## one data point per animal
## linear regression analysis of quantile_area ~ DW * genotype
## USED IN PAPER
########################################################################################################################

## (only mode, 25%-, 50%- and 75%-quantiles for illustration purposes and debugging)

# 0.05, 0.1 , 0.15, 0.2, ..., 0.9 , 0.95
quantiles = np.linspace(0, 1, 21)  #

# indices of the quantiles we are going to model
i_quantiles = [5, 10, 15]  # Q1, Q2, Q3

# for convenience
df_all_f = df_all[df_all['sex'] == 'f']
df_all_m = df_all[df_all['sex'] == 'm']

depot = 'gwat'
# depot = 'sqwat'

# fit linear models to area quantiles
q_models_f_wt = []
q_models_f_het = []
q_models_m_wt = []
q_models_m_het = []
q_models_f_null = []
q_models_m_null = []
q_models_f = []
q_models_m = []
for i_q in i_quantiles:

    # choose one area_at_quantile value as the output of the linear model
    df_all['area_at_quantile'] = np.array(df_all['area_at_quantiles'].to_list())[:, i_q]

    # fit WT/Het linear models
    idx = (df_all['sex'] == 'f') & (df_all['depot'] == depot) & (df_all['genotype'] == 'KLF14-KO:WT')
    q_model_f_wt = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
    idx = (df_all['sex'] == 'f') & (df_all['depot'] == depot) & (df_all['genotype'] == 'KLF14-KO:Het')
    q_model_f_het = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
    idx = (df_all['sex'] == 'm') & (df_all['depot'] == depot) & (df_all['genotype'] == 'KLF14-KO:WT')
    q_model_m_wt = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
    idx = (df_all['sex'] == 'm') & (df_all['depot'] == depot) & (df_all['genotype'] == 'KLF14-KO:Het')
    q_model_m_het = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()

    # fit null models
    idx = (df_all['sex'] == 'f') & (df_all['depot'] == depot)
    q_model_f_null = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
    idx = (df_all['sex'] == 'm') & (df_all['depot'] == depot)
    q_model_m_null = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()

    # fit models with effect variable
    idx = (df_all['sex'] == 'f') & (df_all['depot'] == depot)
    q_model_f = sm.OLS.from_formula('area_at_quantile ~ DW * C(genotype)', data=df_all, subset=idx).fit()
    idx = (df_all['sex'] == 'm') & (df_all['depot'] == depot)
    q_model_m = sm.OLS.from_formula('area_at_quantile ~ DW * C(genotype)', data=df_all, subset=idx).fit()

    q_models_f_wt.append(q_model_f_wt)
    q_models_f_het.append(q_model_f_het)
    q_models_m_wt.append(q_model_m_wt)
    q_models_m_het.append(q_model_m_het)
    q_models_f_null.append(q_model_f_null)
    q_models_m_null.append(q_model_m_null)
    q_models_f.append(q_model_f)
    q_models_m.append(q_model_m)

    if DEBUG:
        print(q_model_f_wt.summary())
        print(q_model_f_het.summary())
        print(q_model_m_wt.summary())
        print(q_model_m_het.summary())
        print(q_model_f_null.summary())
        print(q_model_m_null.summary())
        print(q_model_f.summary())
        print(q_model_m.summary())

# extract coefficients, errors and p-values from PAT and MAT models
model_names = []
for model_name in ['model_f_wt', 'model_f_het', 'model_m_wt', 'model_m_het']:
    for i_q in i_quantiles:
        model_names.append('q_' + '{0:.0f}'.format(quantiles[i_q] * 100) + '_' + model_name)
df_coeff, df_ci_lo, df_ci_hi, df_pval = \
    cytometer.stats.models_coeff_ci_pval(
        q_models_f_wt + q_models_f_het + q_models_m_wt + q_models_m_het,
    model_names=model_names)

# multitest correction using Benjamini-Yekuteli
_, df_corrected_pval, _, _ = multipletests(df_pval.values.flatten(), method='fdr_by', alpha=0.05, returnsorted=False)
df_corrected_pval = pd.DataFrame(df_corrected_pval.reshape(df_pval.shape), columns=df_pval.columns, index=model_names)

# convert p-values to asterisks
df_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval, brackets=False), columns=df_coeff.columns,
                           index=model_names)
df_corrected_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_corrected_pval, brackets=False),
                                     columns=df_coeff.columns, index=model_names)

if SAVEFIG:
    df_concat = pd.concat([df_coeff, df_pval, df_asterisk, df_corrected_pval, df_corrected_asterisk],
                          axis=1)
    idx = list(interleave(np.array_split(range(df_concat.shape[1]), 5)))
    df_concat = df_concat.iloc[:, idx]
    df_concat.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_area_at_quartiles_genotype_models_coeffs_pvals_' + depot + '.csv'))

# plot
if SAVEFIG:

    plt.clf()
    plt.gcf().set_size_inches([6.4, 7.6])

    plt.subplot(321)
    # Q1 Female
    i = 0  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'f'
    df = df_all_f[df_all_f['depot'] == depot].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(q_models_f_null[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(q_models_f_wt[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(q_models_f_het[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:Het'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
                                           line_label='Het')
    plt.tick_params(labelsize=14)
    plt.ylabel('Area$_{\mathrm{Q1}}$ ($10^3\ \mu m^2$)', fontsize=14)
    plt.title('Female', fontsize=14)
    if depot == 'gwat':
        plt.legend(loc='best', fontsize=12)
    if depot == 'gwat':
        plt.ylim(0.9, 4.3)
    elif depot == 'sqwat':
        plt.ylim(0.5, 3)

    plt.subplot(322)
    # Q1 Male
    i = 0  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'm'
    df = df_all_m[df_all_m['depot'] == depot].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(q_models_m_null[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(q_models_m_wt[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(q_models_m_het[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:Het'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
                                           line_label='Het')
    plt.tick_params(labelsize=14)
    plt.title('Male', fontsize=14)
    if depot == 'gwat':
        plt.ylim(0.9, 4.3)
    elif depot == 'sqwat':
        plt.ylim(0.5, 3)

    plt.subplot(323)
    # Q2 Female
    i = 1  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'f'
    df = df_all_f[df_all_f['depot'] == depot].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(q_models_f_null[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(q_models_f_wt[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(q_models_f_het[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:Het'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
                                           line_label='Het')
    plt.tick_params(labelsize=14)
    plt.ylabel('Area$_{\mathrm{Q2}}$ ($10^3\ \mu m^2$)', fontsize=14)
    if depot == 'gwat':
        plt.ylim(1.4, 8.5)
    elif depot == 'sqwat':
        plt.ylim(0.8, 6)

    plt.subplot(324)
    # Q2 Male
    i = 1  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'm'
    df = df_all_m[df_all_m['depot'] == depot].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(q_models_m_null[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(q_models_m_wt[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(q_models_m_het[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:Het'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
                                           line_label='Het')
    plt.tick_params(labelsize=14)
    if depot == 'gwat':
        plt.ylim(1.4, 8.5)
    elif depot == 'sqwat':
        plt.ylim(0.8, 6)

    plt.subplot(325)
    # Q3 Female
    i = 2  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'f'
    df = df_all_f[df_all_f['depot'] == depot].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(q_models_f_null[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(q_models_f_wt[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(q_models_f_het[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:Het'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
                                           line_label='Het')
    plt.tick_params(labelsize=14)
    plt.ylabel('Area$_{\mathrm{Q3}}$ ($10^3\ \mu m^2$)', fontsize=14)
    plt.xlabel('Depot weight (g)', fontsize=14)
    if depot == 'gwat':
        plt.ylim(1, 14)
        # plt.text(0.02, 0.98, pval_text, transform=plt.gca().transAxes, va='top', fontsize=12)
    elif depot == 'sqwat':
        plt.ylim(1, 10.5)

    plt.subplot(326)
    # Q3 Male
    i = 2  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'm'
    df = df_all_m[df_all_m['depot'] == depot].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(q_models_m_null[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(q_models_m_wt[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(q_models_m_het[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:Het'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
                                           line_label='Het')
    plt.tick_params(labelsize=14)
    plt.xlabel('Depot weight (g)', fontsize=14)
    if depot == 'gwat':
        plt.ylim(1, 14)
    elif depot == 'sqwat':
        plt.ylim(1, 10.5)

    depot_title = depot.replace('gwat', 'Gonadal').replace('sqwat', 'Subcutaneous')
    plt.suptitle(depot_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_area_at_quartile_genotype_models_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_area_at_quartile_genotype_models_' + depot + '.svg'))

# Likelihood ratio tests of the genotype variable
print('Likelihood Ratio Test')

print('Female')
for i, i_q in enumerate(i_quantiles):
    lr, pval = cytometer.stats.lrtest(q_models_f_null[i].llf, q_models_f[i].llf)
    pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    print('q=' + str(quantiles[i_q]) + ', ' + depot + ': ' + pval_text)
    print('AIC_null=' + '{0:.2f}'.format(q_models_f_null[i].aic) + ', AIC_alt=' + '{0:.2f}'.format(q_models_f[i].aic))

print('Male')
for i, i_q in enumerate(i_quantiles):
    lr, pval = cytometer.stats.lrtest(q_models_m_null[i].llf, q_models_m[i].llf)
    pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    print('q=' + str(quantiles[i_q]) + ', ' + depot + ': ' + pval_text)
    print('AIC_null=' + '{0:.2f}'.format(q_models_m_null[i].aic) + ', AIC_alt=' + '{0:.2f}'.format(q_models_m[i].aic))

## one data point per animal
## linear regression analysis of quantile_area ~ DW * ko_parent
## USED IN PAPER
########################################################################################################################

## (only mode, 25%-, 50%- and 75%-quantiles for illustration purposes and debugging)

# 0.05, 0.1 , 0.15, 0.2, ..., 0.9 , 0.95
quantiles = np.linspace(0, 1, 21)  #

# indices of the quantiles we are going to model
i_quantiles = [5, 10, 15]  # Q1, Q2, Q3

# for convenience
df_all_f = df_all[df_all['sex'] == 'f']
df_all_m = df_all[df_all['sex'] == 'm']

depot = 'gwat'
# depot = 'sqwat'

# fit linear models to area quantiles
q_models_f_pat = []
q_models_f_mat = []
q_models_m_pat = []
q_models_m_mat = []
q_models_f_null = []
q_models_m_null = []
q_models_f = []
q_models_m = []
for i_q in i_quantiles:

    # choose one area_at_quantile value as the output of the linear model
    df_all['area_at_quantile'] = np.array(df_all['area_at_quantiles'].to_list())[:, i_q]

    # fit WT/Het linear models
    idx = (df_all['sex'] == 'f') & (df_all['depot'] == depot) & (df_all['ko_parent'] == 'PAT')
    q_model_f_pat = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
    idx = (df_all['sex'] == 'f') & (df_all['depot'] == depot) & (df_all['ko_parent'] == 'MAT')
    q_model_f_mat = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
    idx = (df_all['sex'] == 'm') & (df_all['depot'] == depot) & (df_all['ko_parent'] == 'PAT')
    q_model_m_pat = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
    idx = (df_all['sex'] == 'm') & (df_all['depot'] == depot) & (df_all['ko_parent'] == 'MAT')
    q_model_m_mat = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()

    # fit null models
    idx = (df_all['sex'] == 'f') & (df_all['depot'] == depot)
    q_model_f_null = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
    idx = (df_all['sex'] == 'm') & (df_all['depot'] == depot)
    q_model_m_null = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()

    # fit models with parent variable
    idx = (df_all['sex'] == 'f') & (df_all['depot'] == depot)
    q_model_f = sm.OLS.from_formula('area_at_quantile ~ DW * C(ko_parent)', data=df_all, subset=idx).fit()
    idx = (df_all['sex'] == 'm') & (df_all['depot'] == depot)
    q_model_m = sm.OLS.from_formula('area_at_quantile ~ DW * C(ko_parent)', data=df_all, subset=idx).fit()

    q_models_f_pat.append(q_model_f_pat)
    q_models_f_mat.append(q_model_f_mat)
    q_models_m_pat.append(q_model_m_pat)
    q_models_m_mat.append(q_model_m_mat)
    q_models_f_null.append(q_model_f_null)
    q_models_m_null.append(q_model_m_null)
    q_models_f.append(q_model_f)
    q_models_m.append(q_model_m)

    if DEBUG:
        print(q_model_f_pat.summary())
        print(q_model_f_mat.summary())
        print(q_model_m_pat.summary())
        print(q_model_m_mat.summary())
        print(q_model_f_null.summary())
        print(q_model_m_null.summary())
        print(q_model_f.summary())
        print(q_model_m.summary())

# extract coefficients, errors and p-values from PAT and MAT models
model_names = []
for model_name in ['model_f_pat', 'model_f_mat', 'model_m_pat', 'model_m_mat']:
    for i_q in i_quantiles:
        model_names.append('q_' + '{0:.0f}'.format(quantiles[i_q] * 100) + '_' + model_name)
df_coeff, df_ci_lo, df_ci_hi, df_pval = \
    cytometer.stats.models_coeff_ci_pval(
        q_models_f_pat + q_models_f_mat + q_models_m_pat + q_models_m_mat,
    model_names=model_names)

# multitest correction using Benjamini-Yekuteli
_, df_corrected_pval, _, _ = multipletests(df_pval.values.flatten(), method='fdr_by', alpha=0.05, returnsorted=False)
df_corrected_pval = pd.DataFrame(df_corrected_pval.reshape(df_pval.shape), columns=df_pval.columns, index=model_names)

# convert p-values to asterisks
df_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval, brackets=False), columns=df_coeff.columns,
                           index=model_names)
df_corrected_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_corrected_pval, brackets=False),
                                     columns=df_coeff.columns, index=model_names)

if SAVEFIG:
    df_concat = pd.concat([df_coeff, df_pval, df_asterisk, df_corrected_pval, df_corrected_asterisk],
                          axis=1)
    idx = list(interleave(np.array_split(range(df_concat.shape[1]), 5)))
    df_concat = df_concat.iloc[:, idx]
    df_concat.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_area_at_quartiles_parent_models_coeffs_pvals_' + depot + '.csv'))

# plot
if SAVEFIG:

    plt.clf()
    plt.gcf().set_size_inches([6.4, 7.6])

    plt.subplot(321)
    # Q1 Female
    i = 0  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'f'
    df = df_all_f[df_all_f['depot'] == depot].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(q_models_f_null[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(q_models_f_pat[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C0', marker='x',
                                           line_label='PAT')
    cytometer.stats.plot_linear_regression(q_models_f_mat[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C1', marker='+',
                                           line_label='MAT')
    plt.tick_params(labelsize=14)
    plt.ylabel('Area$_{\mathrm{Q1}}$ ($10^3\ \mu m^2$)', fontsize=14)
    plt.title('Female', fontsize=14)
    if depot == 'gwat':
        plt.legend(loc='best', fontsize=12)
    if depot == 'gwat':
        plt.ylim(0.9, 4.3)
    elif depot == 'sqwat':
        plt.ylim(0.5, 3)

    plt.subplot(322)
    # Q1 Male
    i = 0  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'm'
    df = df_all_m[df_all_m['depot'] == depot].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(q_models_m_null[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(q_models_m_pat[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C0', marker='x',
                                           line_label='PAT')
    cytometer.stats.plot_linear_regression(q_models_m_mat[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C1', marker='+',
                                           line_label='MAT')
    plt.tick_params(labelsize=14)
    plt.title('Male', fontsize=14)
    if depot == 'gwat':
        plt.ylim(0.9, 4.3)
    elif depot == 'sqwat':
        plt.ylim(0.5, 3)

    plt.subplot(323)
    # Q2 Female
    i = 1  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'f'
    df = df_all_f[df_all_f['depot'] == depot].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(q_models_f_null[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(q_models_f_pat[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C0', marker='x',
                                           line_label='PAT')
    cytometer.stats.plot_linear_regression(q_models_f_mat[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C1', marker='+',
                                           line_label='MAT')
    plt.tick_params(labelsize=14)
    plt.ylabel('Area$_{\mathrm{Q2}}$ ($10^3\ \mu m^2$)', fontsize=14)
    if depot == 'gwat':
        plt.ylim(1.4, 8.5)
    elif depot == 'sqwat':
        plt.ylim(0.8, 6)

    plt.subplot(324)
    # Q2 Male
    i = 1  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'm'
    df = df_all_m[df_all_m['depot'] == depot].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(q_models_m_null[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(q_models_m_pat[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C0', marker='x',
                                           line_label='PAT')
    cytometer.stats.plot_linear_regression(q_models_m_mat[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C1', marker='+',
                                           line_label='MAT')
    plt.tick_params(labelsize=14)
    if depot == 'gwat':
        plt.ylim(1.4, 8.5)
    elif depot == 'sqwat':
        plt.ylim(0.8, 6)

    plt.subplot(325)
    # Q3 Female
    i = 2  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'f'
    df = df_all_f[df_all_f['depot'] == depot].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(q_models_f_null[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(q_models_f_pat[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C0', marker='x',
                                           line_label='PAT')
    cytometer.stats.plot_linear_regression(q_models_f_mat[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C1', marker='+',
                                           line_label='MAT')
    plt.tick_params(labelsize=14)
    plt.ylabel('Area$_{\mathrm{Q3}}$ ($10^3\ \mu m^2$)', fontsize=14)
    plt.xlabel('Depot weight (g)', fontsize=14)
    if depot == 'gwat':
        plt.ylim(1, 14)
        # plt.text(0.02, 0.98, pval_text, transform=plt.gca().transAxes, va='top', fontsize=12)
    elif depot == 'sqwat':
        plt.ylim(1, 10.5)

    plt.subplot(326)
    # Q3 Male
    i = 2  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'm'
    df = df_all_m[df_all_m['depot'] == depot].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(q_models_m_null[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(q_models_m_pat[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C0', marker='x',
                                           line_label='PAT')
    cytometer.stats.plot_linear_regression(q_models_m_mat[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C1', marker='+',
                                           line_label='MAT')
    plt.tick_params(labelsize=14)
    plt.xlabel('Depot weight (g)', fontsize=14)
    if depot == 'gwat':
        plt.ylim(1, 14)
    elif depot == 'sqwat':
        plt.ylim(1, 10.5)

    depot_title = depot.replace('gwat', 'Gonadal').replace('sqwat', 'Subcutaneous')
    plt.suptitle(depot_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_area_at_quartile_parent_models_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_area_at_quartile_parent_models_' + depot + '.svg'))

# Likelihood ratio tests of the parent variable
print('Likelihood Ratio Test')

print('Female')
for i, i_q in enumerate(i_quantiles):
    lr, pval = cytometer.stats.lrtest(q_models_f_null[i].llf, q_models_f[i].llf)
    pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    print('q=' + str(quantiles[i_q]) + ', ' + depot + ': ' + pval_text)
    print('AIC_null=' + '{0:.2f}'.format(q_models_f_null[i].aic) + ', AIC_alt=' + '{0:.2f}'.format(q_models_f[i].aic))

print('Male')
for i, i_q in enumerate(i_quantiles):
    lr, pval = cytometer.stats.lrtest(q_models_m_null[i].llf, q_models_m[i].llf)
    pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    print('q=' + str(quantiles[i_q]) + ', ' + depot + ': ' + pval_text)
    print('AIC_null=' + '{0:.2f}'.format(q_models_m_null[i].aic) + ', AIC_alt=' + '{0:.2f}'.format(q_models_m[i].aic))

## one data point per animal
## linear regression analysis of kN ~ DW * genotype
## USED IN PAPER
########################################################################################################################

# fit models kN ~ DW
idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'gwat') & (df_all['genotype'] == 'KLF14-KO:WT')
gwat_model_f_wt = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'gwat') & (df_all['genotype'] == 'KLF14-KO:Het')
gwat_model_f_het = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'gwat') & (df_all['genotype'] == 'KLF14-KO:WT')
gwat_model_m_wt = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'gwat') & (df_all['genotype'] == 'KLF14-KO:Het')
gwat_model_m_het = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()

idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'sqwat') & (df_all['genotype'] == 'KLF14-KO:WT')
sqwat_model_f_wt = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'sqwat') & (df_all['genotype'] == 'KLF14-KO:Het')
sqwat_model_f_het = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'sqwat') & (df_all['genotype'] == 'KLF14-KO:WT')
sqwat_model_m_wt = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'sqwat') & (df_all['genotype'] == 'KLF14-KO:Het')
sqwat_model_m_het = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()

# fit null models and models with effect variable
idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'gwat')
gwat_model_f_null = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
gwat_model_f = sm.OLS.from_formula('kN ~ DW * C(genotype)', data=df_all, subset=idx).fit()
idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'gwat')
gwat_model_m_null = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
gwat_model_m = sm.OLS.from_formula('kN ~ DW * C(genotype)', data=df_all, subset=idx).fit()

idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'sqwat')
sqwat_model_f_null = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
sqwat_model_f = sm.OLS.from_formula('kN ~ DW * C(genotype)', data=df_all, subset=idx).fit()
idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'sqwat')
sqwat_model_m_null = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
sqwat_model_m = sm.OLS.from_formula('kN ~ DW * C(genotype)', data=df_all, subset=idx).fit()

# Likelihood Ratio Tests
print('Genotype effect')
print('Female')
lr, pval = cytometer.stats.lrtest(gwat_model_f_null.llf, gwat_model_f.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(
    pval)
print('\t' + 'gwat: ' + pval_text)
lr, pval = cytometer.stats.lrtest(sqwat_model_f_null.llf, sqwat_model_f.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(
    pval)
print('\t' + 'sqwat: ' + pval_text)

print('Male')
lr, pval = cytometer.stats.lrtest(gwat_model_m_null.llf, gwat_model_m.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(
    pval)
print('\t' + 'gwat: ' + pval_text)
lr, pval = cytometer.stats.lrtest(sqwat_model_m_null.llf, sqwat_model_m.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(
    pval)
print('\t' + 'sqwat: ' + pval_text)

# extract coefficients, errors and p-values from PAT and MAT models
model_names = ['gwat_model_f_wt', 'gwat_model_f_het', 'sqwat_model_f_wt', 'sqwat_model_f_het',
               'gwat_model_m_wt', 'gwat_model_m_het', 'sqwat_model_m_wt', 'sqwat_model_m_het']
df_coeff, df_ci_lo, df_ci_hi, df_pval = \
    cytometer.stats.models_coeff_ci_pval(
        [gwat_model_f_wt, gwat_model_f_het, sqwat_model_f_wt, sqwat_model_f_het,
         gwat_model_m_wt, gwat_model_m_het, sqwat_model_m_wt, sqwat_model_m_het],
        model_names=model_names)

# multitest correction using Benjamini-Yekuteli
_, df_corrected_pval, _, _ = multipletests(df_pval.values.flatten(), method='fdr_by', alpha=0.05, returnsorted=False)
df_corrected_pval = pd.DataFrame(df_corrected_pval.reshape(df_pval.shape), columns=df_pval.columns, index=model_names)

# convert p-values to asterisks
df_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval, brackets=False), columns=df_coeff.columns,
                           index=model_names)
df_corrected_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_corrected_pval, brackets=False),
                                     columns=df_coeff.columns, index=model_names)

if SAVEFIG:
    df_concat = pd.concat([df_coeff, df_pval, df_asterisk, df_corrected_pval, df_corrected_asterisk],
                          axis=1)
    idx = list(interleave(np.array_split(range(df_concat.shape[1]), 5)))
    df_concat = df_concat.iloc[:, idx]
    df_concat.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_kN_genotype_models_coeffs_pvals.csv'))

# plot
if SAVEFIG:

    plt.clf()

    plt.subplot(221)
    # Female
    depot = 'gwat'; sex = 'f'
    idx = (df_all['depot'] == depot) & (df_all['sex'] == sex)
    df = df_all[idx]
    cytometer.stats.plot_linear_regression(gwat_model_f_null, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-6, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(gwat_model_f_wt, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:WT'},
                                           dep_var='kN', sy=1e-6, c='C2', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(gwat_model_f_het, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:Het'},
                                           dep_var='kN', sy=1e-6, c='C3', marker='x',
                                           line_label='Het')
    plt.tick_params(labelsize=14)
    plt.ylabel('Gonadal\nk N ($10^6$ cells)', fontsize=14)
    plt.title('Female', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.ylim(-1, 16)

    plt.subplot(222)
    # Male
    depot = 'gwat'; sex = 'm'
    idx = (df_all['depot'] == depot) & (df_all['sex'] == sex)
    df = df_all[idx]
    cytometer.stats.plot_linear_regression(gwat_model_m_null, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-6, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(gwat_model_m_wt, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:WT'},
                                           dep_var='kN', sy=1e-6, c='C2', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(gwat_model_m_het, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:Het'},
                                           dep_var='kN', sy=1e-6, c='C3', marker='x',
                                           line_label='Het')
    plt.tick_params(labelsize=14)
    plt.title('Male', fontsize=14)
    plt.ylim(-1, 10)

    plt.subplot(223)
    depot = 'sqwat'; sex = 'f'
    idx = (df_all['depot'] == depot) & (df_all['sex'] == sex)
    df = df_all[idx]
    cytometer.stats.plot_linear_regression(sqwat_model_f_null, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-6, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(sqwat_model_f_wt, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:WT'},
                                           dep_var='kN', sy=1e-6, c='C2', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(sqwat_model_f_het, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:Het'},
                                           dep_var='kN', sy=1e-6, c='C3', marker='x',
                                           line_label='Het')
    plt.tick_params(labelsize=14)
    plt.ylabel('Subcutaneous\nk N ($10^6$ cells)', fontsize=14)
    plt.xlabel('Depot weight (g)', fontsize=14)
    plt.ylim(-1, 16)

    plt.subplot(224)
    # Male
    depot = 'sqwat'; sex = 'm'
    idx = (df_all['depot'] == depot) & (df_all['sex'] == sex)
    df = df_all[idx]
    cytometer.stats.plot_linear_regression(sqwat_model_m_null, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-6, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(sqwat_model_m_wt, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:WT'},
                                           dep_var='kN', sy=1e-6, c='C2', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(sqwat_model_m_het, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:Het'},
                                           dep_var='kN', sy=1e-6, c='C3', marker='x',
                                           line_label='Het')
    plt.tick_params(labelsize=14)
    plt.xlabel('Depot weight (g)', fontsize=14)
    plt.ylim(-1, 10)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_kN_linear_genotype_model.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_kN_linear_genotype_model.svg'))

## one data point per animal
## linear regression analysis of kN ~ DW * parent
## USED IN PAPER
########################################################################################################################

# fit models kN ~ DW
idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'gwat') & (df_all['ko_parent'] == 'PAT')
gwat_model_f_pat = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'gwat') & (df_all['ko_parent'] == 'MAT')
gwat_model_f_mat = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'gwat') & (df_all['ko_parent'] == 'PAT')
gwat_model_m_pat = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'gwat') & (df_all['ko_parent'] == 'MAT')
gwat_model_m_mat = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()

idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'sqwat') & (df_all['ko_parent'] == 'PAT')
sqwat_model_f_pat = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'sqwat') & (df_all['ko_parent'] == 'MAT')
sqwat_model_f_mat = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'sqwat') & (df_all['ko_parent'] == 'PAT')
sqwat_model_m_pat = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'sqwat') & (df_all['ko_parent'] == 'MAT')
sqwat_model_m_mat = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()

# fit null models and models with effect variable
idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'gwat')
gwat_model_f_null = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
gwat_model_f = sm.OLS.from_formula('kN ~ DW * C(ko_parent)', data=df_all, subset=idx).fit()
idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'gwat')
gwat_model_m_null = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
gwat_model_m = sm.OLS.from_formula('kN ~ DW * C(ko_parent)', data=df_all, subset=idx).fit()

idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'sqwat')
sqwat_model_f_null = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
sqwat_model_f = sm.OLS.from_formula('kN ~ DW * C(ko_parent)', data=df_all, subset=idx).fit()
idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'sqwat')
sqwat_model_m_null = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
sqwat_model_m = sm.OLS.from_formula('kN ~ DW * C(ko_parent)', data=df_all, subset=idx).fit()

# Likelihood Ratio Tests
print('Parent effect')
print('Female')
lr, pval = cytometer.stats.lrtest(gwat_model_f_null.llf, gwat_model_f.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(
    pval)
print('\t' + 'gwat: ' + pval_text)
lr, pval = cytometer.stats.lrtest(sqwat_model_f_null.llf, sqwat_model_f.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(
    pval)
print('\t' + 'sqwat: ' + pval_text)

print('Male')
lr, pval = cytometer.stats.lrtest(gwat_model_m_null.llf, gwat_model_m.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(
    pval)
print('\t' + 'gwat: ' + pval_text)
lr, pval = cytometer.stats.lrtest(sqwat_model_m_null.llf, sqwat_model_m.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(
    pval)
print('\t' + 'sqwat: ' + pval_text)

# extract coefficients, errors and p-values from PAT and MAT models
model_names = ['gwat_model_f_pat', 'gwat_model_f_mat', 'sqwat_model_f_pat', 'sqwat_model_f_mat',
               'gwat_model_m_pat', 'gwat_model_m_mat', 'sqwat_model_m_pat', 'sqwat_model_m_mat']
df_coeff, df_ci_lo, df_ci_hi, df_pval = \
    cytometer.stats.models_coeff_ci_pval(
        [gwat_model_f_pat, gwat_model_f_mat, sqwat_model_f_pat, sqwat_model_f_mat,
         gwat_model_m_pat, gwat_model_m_mat, sqwat_model_m_pat, sqwat_model_m_mat],
        model_names=model_names)

# multitest correction using Benjamini-Yekuteli
_, df_corrected_pval, _, _ = multipletests(df_pval.values.flatten(), method='fdr_by', alpha=0.05, returnsorted=False)
df_corrected_pval = pd.DataFrame(df_corrected_pval.reshape(df_pval.shape), columns=df_pval.columns, index=model_names)

# convert p-values to asterisks
df_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval, brackets=False), columns=df_coeff.columns,
                           index=model_names)
df_corrected_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_corrected_pval, brackets=False),
                                     columns=df_coeff.columns, index=model_names)

if SAVEFIG:
    df_concat = pd.concat([df_coeff, df_pval, df_asterisk, df_corrected_pval, df_corrected_asterisk],
                          axis=1)
    idx = list(interleave(np.array_split(range(df_concat.shape[1]), 5)))
    df_concat = df_concat.iloc[:, idx]
    df_concat.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_kN_parent_models_coeffs_pvals.csv'))

# plot
if SAVEFIG:

    plt.clf()

    plt.subplot(221)
    # Female
    depot = 'gwat'; sex = 'f'
    idx = (df_all['depot'] == depot) & (df_all['sex'] == sex)
    df = df_all[idx]
    cytometer.stats.plot_linear_regression(gwat_model_f_null, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-6, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(gwat_model_f_pat, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
                                           dep_var='kN', sy=1e-6, c='C0', marker='x',
                                           line_label='PAT')
    cytometer.stats.plot_linear_regression(gwat_model_f_mat, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
                                           dep_var='kN', sy=1e-6, c='C1', marker='x',
                                           line_label='MAT')
    plt.tick_params(labelsize=14)
    plt.ylabel('Gonadal\nk N ($10^6$ cells)', fontsize=14)
    plt.title('Female', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.ylim(-1, 16)

    plt.subplot(222)
    # Male
    depot = 'gwat'; sex = 'm'
    idx = (df_all['depot'] == depot) & (df_all['sex'] == sex)
    df = df_all[idx]
    cytometer.stats.plot_linear_regression(gwat_model_m_null, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-6, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(gwat_model_m_pat, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
                                           dep_var='kN', sy=1e-6, c='C0', marker='x',
                                           line_label='PAT')
    cytometer.stats.plot_linear_regression(gwat_model_m_mat, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
                                           dep_var='kN', sy=1e-6, c='C1', marker='x',
                                           line_label='MAT')
    plt.tick_params(labelsize=14)
    plt.title('Male', fontsize=14)
    plt.ylim(-1, 10)

    plt.subplot(223)
    depot = 'sqwat'; sex = 'f'
    idx = (df_all['depot'] == depot) & (df_all['sex'] == sex)
    df = df_all[idx]
    cytometer.stats.plot_linear_regression(sqwat_model_f_null, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-6, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(sqwat_model_f_pat, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
                                           dep_var='kN', sy=1e-6, c='C0', marker='x',
                                           line_label='PAT')
    cytometer.stats.plot_linear_regression(sqwat_model_f_mat, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
                                           dep_var='kN', sy=1e-6, c='C1', marker='x',
                                           line_label='MAT')
    plt.tick_params(labelsize=14)
    plt.ylabel('Subcutaneous\nk N ($10^6$ cells)', fontsize=14)
    plt.xlabel('Depot weight (g)', fontsize=14)
    plt.ylim(-1, 16)

    plt.subplot(224)
    # Male
    depot = 'sqwat'; sex = 'm'
    idx = (df_all['depot'] == depot) & (df_all['sex'] == sex)
    df = df_all[idx]
    cytometer.stats.plot_linear_regression(sqwat_model_m_null, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex}, sy=1e-6, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(sqwat_model_m_pat, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
                                           dep_var='kN', sy=1e-6, c='C0', marker='x',
                                           line_label='PAT')
    cytometer.stats.plot_linear_regression(sqwat_model_m_mat, df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
                                           dep_var='kN', sy=1e-6, c='C1', marker='x',
                                           line_label='MAT')
    plt.tick_params(labelsize=14)
    plt.xlabel('Depot weight (g)', fontsize=14)
    plt.ylim(-1, 10)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_kN_linear_parent_model.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_kN_linear_parent_model.svg'))
