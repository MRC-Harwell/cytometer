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
## The results are saved, so in later sections, it's possible to just read them for further analysis.
## GENERATES DATA USED IN FOLLOWING SECTIONS
########################################################################################################################

import matplotlib.pyplot as plt
import cytometer.data
import shapely
import scipy.stats as stats
import openslide
import numpy as np
# import scipy.stats
import sklearn.neighbors, sklearn.model_selection
import pandas as pd
import PIL

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

                # compute areas at population quantiles
                areas_at_quantiles = stats.mstats.hdquantiles(areas, prob=quantiles, axis=0)
                df['area_at_quantiles'] = [areas_at_quantiles]

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

from toolz import interleave
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import openslide
import cytometer.data
import cytometer.stats
import shapely

# directories
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
annotations_dir = os.path.join(home, 'Data/cytometer_data/aida_data_Klf14_v8/annotations')
ndpi_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
dataframe_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
metainfo_dir = os.path.join(home, 'Data/cytometer_data/klf14')

DEBUG = False

method = 'corrected'

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

# def plot_linear_regression_BW(model, df, sex=None, ko_parent=None, genotype=None, style=None, sy=1.0):
#     if np.std(df['BW'] / df['BW__']) > 1e-8:
#         raise ValueError('BW / BW__ is not the same for all rows')
#     BW__factor = (df['BW'] / df['BW__']).mean()
#     BW__lim = np.array([df['BW__'].min(), df['BW__'].max()])
#     X = pd.DataFrame(data={'BW__': BW__lim, 'sex': [sex, sex], 'ko_parent': [ko_parent, ko_parent],
#                            'genotype': [genotype, genotype]})
#     y_pred = model.predict(X)
#     plt.plot(BW__lim * BW__factor, y_pred * sy, style)
#
# def plot_linear_regression_DW(model, df, sex=None, ko_parent=None, genotype=None, style=None, sx=1.0, sy=1.0, label=None):
#     DW_BW = df['DW'] / df['BW']
#     DW_BW_lim = np.array([DW_BW.min(), DW_BW.max()])
#     X = pd.DataFrame(data={'DW_BW': DW_BW_lim, 'sex': [sex, sex], 'ko_parent': [ko_parent, ko_parent],
#                            'genotype': [genotype, genotype]})
#     y_pred = model.predict(X)
#     plt.plot(DW_BW_lim * sx, y_pred * sy, style, label=label)

def read_contours_compute_areas(metainfo, json_annotation_files_dict, depot, method='corrected'):

    # dataframe to keep all results, one row per cell
    df_all = pd.DataFrame()

    # list of annotation files for this depot
    json_annotation_files = json_annotation_files_dict[depot]

    # modify filenames to select the particular segmentation we want (e.g. the automatic ones, or the manually refined ones)
    json_annotation_files = [x.replace('.json', '_exp_0106_' + method + '_aggregated.json') for x in
                             json_annotation_files]
    json_annotation_files = [os.path.join(annotations_dir, x) for x in json_annotation_files]

    for i_file, json_file in enumerate(json_annotation_files):

        print('File ' + str(i_file) + '/' + str(len(json_annotation_files) - 1) + ': '
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

        # load contours and their confidence measure from annotation file
        cells, props = cytometer.data.aida_get_contours(json_file, layer_name='White adipocyte.*',
                                                        return_props=True)

        # compute areas of the cells (um^2)
        areas = np.array([shapely.geometry.Polygon(cell).area for cell in cells]) * xres * yres  # (um^2)
        df = df.reindex(df.index.repeat(len(areas)))
        df.loc[:, 'area'] = areas

        # add results to total dataframe
        df_all = pd.concat([df_all, df], ignore_index=True)

    return df_all


## Analyse cell populations from automatically segmented images in two depots: SQWAT and GWAT:
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
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)

    plt.subplot(221)
    plt.fill_between(area_bin_centers * 1e-3, histo_q1[0,] / histo_q3.max(), histo_q3[0,] / histo_q3.max(),
                     alpha=0.5, color='C0')
    plt.plot(area_bin_centers * 1e-3, histo_q2[0,] / histo_q3.max(), 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.text(0.9, 0.9, 'female PAT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # f MAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)

    plt.subplot(222)
    plt.fill_between(area_bin_centers * 1e-3, histo_q1[0,] / histo_q3.max(), histo_q3[0,] / histo_q3.max(),
                     alpha=0.5, color='C0')
    plt.plot(area_bin_centers * 1e-3, histo_q2[0,] / histo_q3.max(), 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.text(0.9, 0.9, 'female MAT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # m PAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'PAT')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)

    plt.subplot(223)
    plt.fill_between(area_bin_centers * 1e-3, histo_q1[0,] / histo_q3.max(), histo_q3[0,] / histo_q3.max(),
                     alpha=0.5, color='C0')
    plt.plot(area_bin_centers * 1e-3, histo_q2[0,] / histo_q3.max(), 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)
    plt.text(0.9, 0.9, 'male PAT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # m MAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'MAT')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)

    plt.subplot(224)
    plt.fill_between(area_bin_centers * 1e-3, histo_q1[0,] / histo_q3.max(), histo_q3[0,] / histo_q3.max(),
                     alpha=0.5, color='C0')
    plt.plot(area_bin_centers * 1e-3, histo_q2[0,] / histo_q3.max(), 'C0', linewidth=2)
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

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_smoothed_histo_quartiles_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_smoothed_histo_quartiles_' + depot + '.svg'))

## numerical quartiles and CIs associated to the histograms

idx_q1 = np.where(quantiles == 0.25)[0][0]
idx_q2 = np.where(quantiles == 0.50)[0][0]
idx_q3 = np.where(quantiles == 0.75)[0][0]

# f PAT
df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'PAT')]
df = df.reset_index()
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())

# compute mean value and CIs in 10^3 um^2 units
q1_mean, q2_mean, q3_mean = areas_at_quantiles.mean(axis=0)[[idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_lo, q2_ci_lo, q3_ci_lo = stats.mstats.hdquantiles(areas_at_quantiles, prob=0.025, axis=0).data[0, [idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_hi, q2_ci_hi, q3_ci_hi = stats.mstats.hdquantiles(areas_at_quantiles, prob=0.975, axis=0).data[0, [idx_q1, idx_q2, idx_q3]] * 1e-3

print('f PAT')
print('\t' + '{0:.2f}'.format(q1_mean) + ' (' + '{0:.2f}'.format(q1_ci_lo) + ', ' + '{0:.2f}'.format(q1_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q2_mean) + ' (' + '{0:.2f}'.format(q2_ci_lo) + ', ' + '{0:.2f}'.format(q2_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q3_mean) + ' (' + '{0:.2f}'.format(q3_ci_lo) + ', ' + '{0:.2f}'.format(q3_ci_hi) + ')')

# f MAT
df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')]
df = df.reset_index()
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())

# compute mean value and CIs in 10^3 um^2 units
q1_mean, q2_mean, q3_mean = areas_at_quantiles.mean(axis=0)[[idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_lo, q2_ci_lo, q3_ci_lo = stats.mstats.hdquantiles(areas_at_quantiles, prob=0.025, axis=0).data[0, [idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_hi, q2_ci_hi, q3_ci_hi = stats.mstats.hdquantiles(areas_at_quantiles, prob=0.975, axis=0).data[0, [idx_q1, idx_q2, idx_q3]] * 1e-3

print('f MAT')
print('\t' + '{0:.2f}'.format(q1_mean) + ' (' + '{0:.2f}'.format(q1_ci_lo) + ', ' + '{0:.2f}'.format(q1_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q2_mean) + ' (' + '{0:.2f}'.format(q2_ci_lo) + ', ' + '{0:.2f}'.format(q2_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q3_mean) + ' (' + '{0:.2f}'.format(q3_ci_lo) + ', ' + '{0:.2f}'.format(q3_ci_hi) + ')')

# m PAT
df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'PAT')]
df = df.reset_index()
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())

# compute mean value and CIs in 10^3 um^2 units
q1_mean, q2_mean, q3_mean = areas_at_quantiles.mean(axis=0)[[idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_lo, q2_ci_lo, q3_ci_lo = stats.mstats.hdquantiles(areas_at_quantiles, prob=0.025, axis=0).data[0, [idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_hi, q2_ci_hi, q3_ci_hi = stats.mstats.hdquantiles(areas_at_quantiles, prob=0.975, axis=0).data[0, [idx_q1, idx_q2, idx_q3]] * 1e-3

print('m PAT')
print('\t' + '{0:.2f}'.format(q1_mean) + ' (' + '{0:.2f}'.format(q1_ci_lo) + ', ' + '{0:.2f}'.format(q1_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q2_mean) + ' (' + '{0:.2f}'.format(q2_ci_lo) + ', ' + '{0:.2f}'.format(q2_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q3_mean) + ' (' + '{0:.2f}'.format(q3_ci_lo) + ', ' + '{0:.2f}'.format(q3_ci_hi) + ')')

# m MAT
df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'MAT')]
df = df.reset_index()
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())

# compute mean value and CIs in 10^3 um^2 units
q1_mean, q2_mean, q3_mean = areas_at_quantiles.mean(axis=0)[[idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_lo, q2_ci_lo, q3_ci_lo = stats.mstats.hdquantiles(areas_at_quantiles, prob=0.025, axis=0).data[0, [idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_hi, q2_ci_hi, q3_ci_hi = stats.mstats.hdquantiles(areas_at_quantiles, prob=0.975, axis=0).data[0, [idx_q1, idx_q2, idx_q3]] * 1e-3

print('m MAT')
print('\t' + '{0:.2f}'.format(q1_mean) + ' (' + '{0:.2f}'.format(q1_ci_lo) + ', ' + '{0:.2f}'.format(q1_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q2_mean) + ' (' + '{0:.2f}'.format(q2_ci_lo) + ', ' + '{0:.2f}'.format(q2_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q3_mean) + ' (' + '{0:.2f}'.format(q3_ci_lo) + ', ' + '{0:.2f}'.format(q3_ci_hi) + ')')

## Plots of body weight vs cull age, and swarm plots of body weight
## USED IN PAPER
########################################################################################################################

# scale cull_age to avoid large condition numbers
metainfo['cull_age__'] = (metainfo['cull_age'] - np.mean(metainfo['cull_age'])) / np.std(metainfo['cull_age'])

# for convenience create two dataframes (female and male) with the data for the current depot
metainfo_f = metainfo[metainfo['sex'] == 'f'].reset_index()
metainfo_m = metainfo[metainfo['sex'] == 'm'].reset_index()

# ANCOVA-like comparison of means between PAT and MAT
bw_model_f = sm.RLM.from_formula('BW ~ C(ko_parent) * cull_age__', data=metainfo_f, M=sm.robust.norms.HuberT()).fit()
bw_model_m = sm.RLM.from_formula('BW ~ C(ko_parent) * cull_age__', data=metainfo_m, M=sm.robust.norms.HuberT()).fit()

print(bw_model_f.summary())
print(bw_model_m.summary())

# extract coefficients, errors and p-values from models
df_coeff_f, df_ci_lo_f, df_ci_hi_f, df_pval_f = \
    cytometer.stats.models_coeff_ci_pval([bw_model_f], extra_hypotheses='Intercept + C(ko_parent)[T.MAT], cull_age__ + C(ko_parent)[T.MAT]:cull_age__')
df_coeff_m, df_ci_lo_m, df_ci_hi_m, df_pval_m = \
    cytometer.stats.models_coeff_ci_pval([bw_model_m], extra_hypotheses='Intercept + C(ko_parent)[T.MAT], cull_age__ + C(ko_parent)[T.MAT]:cull_age__')

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
    # save a table for the summary of findings spreadsheet: "summary_of_WAT_findings"
    cols = ['Intercept', 'Intercept+C(ko_parent)[T.MAT]', 'C(ko_parent)[T.MAT]',
            'cull_age__', 'cull_age__+C(ko_parent)[T.MAT]:cull_age__', 'C(ko_parent)[T.MAT]:cull_age__']

    df_concat = pd.DataFrame()
    for col in cols:
        df_concat = pd.concat([df_concat, df_coeff_f[col], df_pval_f[col], df_asterisk_f[col],
                               df_corrected_pval_f[col], df_corrected_asterisk_f[col]], axis=1)
    df_concat.to_csv(os.path.join(figures_dir, 'foo.csv'))

    df_concat = pd.DataFrame()
    for col in cols:
        df_concat = pd.concat([df_concat, df_coeff_m[col], df_pval_m[col], df_asterisk_m[col],
                               df_corrected_pval_m[col], df_corrected_asterisk_m[col]], axis=1)
    df_concat.to_csv(os.path.join(figures_dir, 'foo.csv'))



if SAVEFIG:
    # plot body weight vs. age of culling
    plt.clf()
    plt.scatter(metainfo_f[metainfo_f['ko_parent'] == 'PAT']['cull_age'],
                metainfo_f[metainfo_f['ko_parent'] == 'PAT']['BW'], c='C2')
    plt.scatter(metainfo_f[metainfo_f['ko_parent'] == 'MAT']['cull_age'],
                metainfo_f[metainfo_f['ko_parent'] == 'MAT']['BW'], c='C3')
    plt.scatter(metainfo_m[metainfo_f['ko_parent'] == 'PAT']['cull_age'],
                metainfo_m[metainfo_f['ko_parent'] == 'PAT']['BW'], c='C4')
    plt.scatter(metainfo_m[metainfo_f['ko_parent'] == 'MAT']['cull_age'],
                metainfo_m[metainfo_f['ko_parent'] == 'MAT']['BW'], c='C5')
    plt.xlabel('Cull age (days)', fontsize=14)
    plt.ylabel('Body weight (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

    cull_age__lim = np.array([metainfo['cull_age__'].min(), metainfo['cull_age__'].max()])
    cull_age_lim = np.array([metainfo['cull_age'].min(), metainfo['cull_age'].max()])
    X = pd.DataFrame(data={'cull_age__': cull_age__lim, 'ko_parent': ['PAT', 'PAT']})
    y_pred = bw_model_f.predict(X)
    plt.plot(cull_age_lim, y_pred, 'C2', linewidth=2, label='f PAT')
    X = pd.DataFrame(data={'cull_age__': cull_age__lim, 'ko_parent': ['MAT', 'MAT']})
    y_pred = bw_model_f.predict(X)
    plt.plot(cull_age_lim, y_pred, 'C3', linewidth=2, label='f MAT')
    pval_cull_age = bw_model_f.pvalues['cull_age__']
    pval_mat = bw_model_f.pvalues['C(ko_parent)[T.MAT]']
    pval_text = '$p_{cull\ age}$=' + '{0:.2f}'.format(pval_cull_age) + ' ' + cytometer.stats.pval_to_asterisk(pval_cull_age) + \
                '\n' + \
                '$p_{MAT}$=' + '{0:.3f}'.format(pval_mat) + ' ' + cytometer.stats.pval_to_asterisk(pval_mat)
    plt.text(142.75, 26.3, pval_text, va='top', fontsize=12)

    X = pd.DataFrame(data={'cull_age__': cull_age__lim, 'ko_parent': ['PAT', 'PAT']})
    y_pred = bw_model_m.predict(X)
    plt.plot(cull_age_lim, y_pred, 'C4', linewidth=2, label='m PAT')
    X = pd.DataFrame(data={'cull_age__': cull_age__lim, 'ko_parent': ['MAT', 'MAT']})
    y_pred = bw_model_m.predict(X)
    plt.plot(cull_age_lim, y_pred, 'C5', linewidth=2, label='m MAT')
    pval_cull_age = bw_model_m.pvalues['cull_age__']
    pval_mat = bw_model_m.pvalues['C(ko_parent)[T.MAT]']
    pval_text = '$p_{cull\ age}$=' + '{0:.2f}'.format(pval_cull_age) + ' ' + cytometer.stats.pval_to_asterisk(pval_cull_age) + \
                '\n' + \
                '$p_{MAT}$=' + '{0:.2f}'.format(pval_mat) + ' ' + cytometer.stats.pval_to_asterisk(pval_mat)
    plt.text(142.75, 44, pval_text, va='bottom', fontsize=12)

    plt.legend(fontsize=12)

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_bw_vs_cull_age.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_bw_vs_cull_age.svg'))


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
    pval_text = '$p$=' + '{0:.3f}'.format(bw_model_f.pvalues['C(ko_parent)[T.MAT]']) + \
                ' ' + cytometer.stats.pval_to_asterisk(bw_model_f.pvalues['C(ko_parent)[T.MAT]'])
    plt.text(0, 44.5, pval_text, ha='center', va='bottom', fontsize=14)
    plt.plot([0.8, 0.8, 1.2, 1.2], [52, 54, 54, 52], 'k', lw=1.5)
    pval_text = '$p$=' + '{0:.2f}'.format(bw_model_m.pvalues['C(ko_parent)[T.MAT]']) + \
                ' ' + cytometer.stats.pval_to_asterisk(bw_model_m.pvalues['C(ko_parent)[T.MAT]'])
    plt.text(1, 54.5, pval_text, ha='center', va='bottom', fontsize=14)
    plt.ylim(18, 58)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_swarm_bw.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_swarm_bw.svg'))

## sex effect on mouse BW

sex_model = sm.RLM.from_formula('BW ~ C(sex)', data=metainfo, M=sm.robust.norms.HuberT()).fit()
print(sex_model.summary())

pval_text = 'p=' + '{0:.3e}'.format(sex_model.pvalues['C(sex)[T.m]']) + \
            ' ' + cytometer.stats.pval_to_asterisk(sex_model.pvalues['C(sex)[T.m]'])
print(pval_text)

## depot ~ BW * parent models

# scale BW to avoid large condition numbers
BW_mean = metainfo['BW'].mean()
metainfo['BW__'] = metainfo['BW'] / BW_mean

# for convenience
metainfo_f = metainfo[metainfo['sex'] == 'f']
metainfo_m = metainfo[metainfo['sex'] == 'm']

# models of depot weight ~ BW * ko_parent

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
print('Likelihood Ratio Test')

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
    df_concat.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_depot_weight_models_coeffs_pvals.csv'))

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

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_depot_linear_model.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_depot_linear_model.svg'))


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

    q_models_f_pat.append(q_model_f_pat)
    q_models_f_mat.append(q_model_f_mat)
    q_models_m_pat.append(q_model_m_pat)
    q_models_m_mat.append(q_model_m_mat)
    q_models_f_null.append(q_model_f_null)
    q_models_m_null.append(q_model_m_null)

    if DEBUG:
        print(q_model_f_pat.summary())
        print(q_model_f_mat.summary())
        print(q_model_m_pat.summary())
        print(q_model_m_mat.summary())
        print(q_model_f_null.summary())
        print(q_model_m_null.summary())

# extract coefficients, errors and p-values from models
model_names = []
for model_name in ['model_f_pat', 'model_f_mat', 'model_f_null', 'model_m_pat', 'model_m_mat', 'model_m_null']:
    for i_q in i_quantiles:
        model_names.append('q_' + '{0:.0f}'.format(quantiles[i_q] * 100) + '_' + model_name)
df_coeff, df_ci_lo, df_ci_hi, df_pval = \
    cytometer.stats.models_coeff_ci_pval(
        q_models_f_pat + q_models_f_mat + q_models_f_null + q_models_m_pat + q_models_m_mat + q_models_m_null,
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
    df_concat.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_area_at_quartiles_models_coeffs_pvals_' + depot + '.csv'))

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

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_area_at_quartile_models_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_area_at_quartile_models_' + depot + '.svg'))

####### OLD CODE

# 0.1 , 0.2, ..., 0.9
deciles_idx = list(range(2, 20, 2))
deciles = quantiles[deciles_idx]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# fit linear models for each decile
decile_models_f = [sm.RLM.from_formula('area_q_' + '{0:03d}'.format(int(d*100)) + ' ~ DW_BW * C(ko_parent)', data=df_f, M=sm.robust.norms.HuberT()).fit()
                   for d in deciles]
decile_models_m = [sm.RLM.from_formula('area_q_' + '{0:03d}'.format(int(d*100)) + ' ~ DW_BW * C(ko_parent)', data=df_m, M=sm.robust.norms.HuberT()).fit()
                   for d in deciles]

print(decile_models_f[4].summary())
print(decile_models_m[4].summary())

# extract coefficients, errors and p-values from quartile models
df_coeff_f, df_ci_lo_f, df_ci_hi_f, df_pval_f = \
    cytometer.stats.models_coeff_ci_pval(decile_models_f, extra_hypotheses='Intercept + C(ko_parent)[T.MAT], DW_BW + DW_BW:C(ko_parent)[T.MAT]')
df_coeff_m, df_ci_lo_m, df_ci_hi_m, df_pval_m = \
    cytometer.stats.models_coeff_ci_pval(decile_models_m, extra_hypotheses='Intercept + C(ko_parent)[T.MAT], DW_BW + DW_BW:C(ko_parent)[T.MAT]')

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
    # save a table for the summary of findings spreadsheet: "summary_of_WAT_findings"
    cols = ['Intercept', 'Intercept+C(ko_parent)[T.MAT]', 'C(ko_parent)[T.MAT]',
            'DW_BW', 'DW_BW+DW_BW:C(ko_parent)[T.MAT]', 'DW_BW:C(ko_parent)[T.MAT]']

    df_concat = pd.DataFrame()
    for col in cols:
        df_concat = pd.concat([df_concat, df_coeff_f[col], df_pval_f[col], df_asterisk_f[col],
                               df_corrected_pval_f[col], df_corrected_asterisk_f[col]], axis=1)
    df_concat.to_csv(os.path.join(figures_dir, 'foo.csv'))

    df_concat = pd.DataFrame()
    for col in cols:
        df_concat = pd.concat([df_concat, df_coeff_m[col], df_pval_m[col], df_asterisk_m[col],
                               df_corrected_pval_m[col], df_corrected_asterisk_m[col]], axis=1)
    df_concat.to_csv(os.path.join(figures_dir, 'foo.csv'))

if SAVEFIG:
    plt.clf()
    plt.gcf().set_size_inches([7.2, 9.5])
    q = np.array(deciles) * 100

    plt.subplot(4,2,1)
    if depot == 'gwat':
        ylim = (-1, 34)
    elif depot == 'sqwat':
        ylim = (-1, 20)
    h1, h2 = cytometer.stats.cytometer.stats.plot_model_coeff_compare2(q, df_coeff_f['Intercept'] * 1e-3,
                              df_ci_lo_f['Intercept'] * 1e-3,
                              df_ci_hi_f['Intercept'] * 1e-3,
                              df_pval_f['Intercept'],
                              df_coeff_f['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
                              df_ci_lo_f['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
                              df_ci_hi_f['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
                              df_pval_f['Intercept+C(ko_parent)[T.MAT]'],
                              df_corrected_pval_1=df_corrected_pval_f['Intercept'],
                              df_corrected_pval_2=df_corrected_pval_f['Intercept+C(ko_parent)[T.MAT]'],
                              ylim=ylim, color_1='C0', color_2='C1', label_1='PAT', label_2='MAT')
    plt.title('Female', fontsize=14)
    plt.ylabel(r'$\beta_{0}\ (10^3\ \mu m^2)$', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.ylim(ylim)

    plt.subplot(4,2,2)
    if depot == 'gwat':
        ylim = (-1, 34)
    elif depot == 'sqwat':
        ylim = (-1, 20)
    h1, h2 = cytometer.stats.cytometer.stats.plot_model_coeff_compare2(q, df_coeff_m['Intercept'] * 1e-3,
                              df_ci_lo_m['Intercept'] * 1e-3,
                              df_ci_hi_m['Intercept'] * 1e-3,
                              df_pval_m['Intercept'],
                              df_coeff_m['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
                              df_ci_lo_m['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
                              df_ci_hi_m['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
                              df_pval_m['Intercept+C(ko_parent)[T.MAT]'],
                              df_corrected_pval_1=df_corrected_pval_m['Intercept'],
                              df_corrected_pval_2=df_corrected_pval_m['Intercept+C(ko_parent)[T.MAT]'],
                              ylim=ylim, color_1='C0', color_2='C1', label_1='PAT', label_2='MAT')
    plt.title('Male', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.ylim(ylim)

    plt.subplot(4,2,3)
    if depot == 'gwat':
        ylim = (-5, 12)
    elif depot == 'sqwat':
        ylim = (-1, 7)
    cytometer.stats.plot_model_coeff(q, df_coeff_f['C(ko_parent)[T.MAT]'] * 1e-3,
                     df_ci_lo_f['C(ko_parent)[T.MAT]'] * 1e-3,
                     df_ci_hi_f['C(ko_parent)[T.MAT]'] * 1e-3,
                     df_pval_f['C(ko_parent)[T.MAT]'],
                     df_corrected_pval=df_corrected_pval_f['C(ko_parent)[T.MAT]'],
                     ylim=ylim, color='k', label='PAT$\mapsto$MAT')
    plt.ylabel(r'$\Delta\beta_{0}\ (10^3\ \mu m^2)$', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.ylim(ylim)

    plt.subplot(4,2,4)
    if depot == 'gwat':
        ylim = (-5, 12)
    elif depot == 'sqwat':
        ylim = (-1, 7)
    cytometer.stats.plot_model_coeff(q, df_coeff_m['C(ko_parent)[T.MAT]'] * 1e-3,
                     df_ci_lo_m['C(ko_parent)[T.MAT]'] * 1e-3,
                     df_ci_hi_m['C(ko_parent)[T.MAT]'] * 1e-3,
                     df_pval_m['C(ko_parent)[T.MAT]'],
                     df_corrected_pval=df_corrected_pval_m['C(ko_parent)[T.MAT]'],
                     ylim=ylim, color='k', label='PAT$\mapsto$MAT')
    plt.tick_params(labelsize=14)
    plt.ylim(ylim)

    plt.subplot(4,2,5)
    if depot == 'gwat':
        ylim = (-2.5, 3.5)
    elif depot == 'sqwat':
        ylim = (-2, 3.5)
    h1, h2 = cytometer.stats.cytometer.stats.plot_model_coeff_compare2(q, df_coeff_f['DW_BW'] * 1e-5,
                              df_ci_lo_f['DW_BW'] * 1e-5,
                              df_ci_hi_f['DW_BW'] * 1e-5,
                              df_pval_f['DW_BW'],
                              df_coeff_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                              df_ci_lo_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                              df_ci_hi_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                              df_pval_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'],
                              df_corrected_pval_1=df_corrected_pval_f['DW_BW'],
                              df_corrected_pval_2=df_corrected_pval_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'],
                              ylim=ylim, color_1='C0', color_2='C1', label_1='PAT', label_2='MAT')
    plt.ylabel(r'$\beta_{DW/BW}\ (10^5\ \mu m^2)$', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.ylim(ylim)

    plt.subplot(4,2,6)
    if depot == 'gwat':
        ylim = (-2.5, 3.5)
    elif depot == 'sqwat':
        ylim = (-2, 3.5)
    h1, h2 = cytometer.stats.cytometer.stats.plot_model_coeff_compare2(q, df_coeff_m['DW_BW'] * 1e-5,
                              df_ci_lo_m['DW_BW'] * 1e-5,
                              df_ci_hi_m['DW_BW'] * 1e-5,
                              df_pval_m['DW_BW'],
                              df_coeff_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                              df_ci_lo_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                              df_ci_hi_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                              df_pval_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'],
                              df_corrected_pval_1=df_corrected_pval_m['DW_BW'],
                              df_corrected_pval_2=df_corrected_pval_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'],
                              ylim=ylim, color_1='C0', color_2='C1', label_1='PAT', label_2='MAT')
    plt.tick_params(labelsize=14)
    plt.ylim(ylim)

    plt.subplot(4,2,7)
    if depot == 'gwat':
        ylim = (-4, 3)
    elif depot == 'sqwat':
        ylim = (-4, 2.4)
    cytometer.stats.plot_model_coeff(q, df_coeff_f['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                     df_ci_lo_f['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                     df_ci_hi_f['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                     df_pval_f['DW_BW:C(ko_parent)[T.MAT]'],
                     df_corrected_pval=df_corrected_pval_f['DW_BW:C(ko_parent)[T.MAT]'],
                     ylim=ylim, color='k', label='PAT$\mapsto$MAT')
    plt.xlabel('Quantile (%)', fontsize=14)
    plt.ylabel(r'$\Delta \beta_{DW/BW}\ (10^5\ \mu m^2)$', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.ylim(ylim)

    plt.subplot(4,2,8)
    if depot == 'gwat':
        ylim = (-4, 3)
    elif depot == 'sqwat':
        ylim = (-4, 2.4)
    cytometer.stats.plot_model_coeff(q, df_coeff_m['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                     df_ci_lo_m['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                     df_ci_hi_m['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                     df_pval_m['DW_BW:C(ko_parent)[T.MAT]'],
                     df_corrected_pval=df_corrected_pval_m['DW_BW:C(ko_parent)[T.MAT]'],
                     ylim=ylim, color='k', label='PAT$\mapsto$MAT')
    plt.xlabel('Quantile (%)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.ylim(ylim)

    plt.tight_layout()

    if depot == 'gwat':
        plt.suptitle('Gonadal', fontsize=14)
    elif depot == 'sqwat':
        plt.suptitle('Subcutaneous', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_decile_dw_bw_linreg_coeffs_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_decile_dw_bw_linreg_coeffs_' + depot + '.svg'))

########################################################################################################################
## one data point per cell
## quantile regression analysis of area ~ DW/BW * ko_parent
## USED IN PAPER
########################################################################################################################

# 0.05, 0.1 , 0.15, 0.2, ..., 0.9 , 0.95
quantiles = np.linspace(0, 1, 21)

# 1.32020052, 1.33581401, ..., 4.42728541, 4.4428989
log10_area_bin_edges = np.linspace(np.log10(min_area_um2), np.log10(max_area_um2), 201)
log10_area_bin_centers = (log10_area_bin_edges[0:-1] + log10_area_bin_edges[1:]) / 2.0

method = 'corrected'

depot = 'gwat'
# depot = 'sqwat'

# compute or load dataframe with one row per cell
dataframe_cells_filename = os.path.join(dataframe_dir, 'klf14_b6ntac_exp_0110_dataframe_individual_cells_' + method + '_' + depot + '.csv')
if not os.path.isfile(dataframe_cells_filename):
    # create dataframe with one row per cell
    df_all = read_contours_compute_areas(metainfo, json_annotation_files_dict, depot, method='corrected')

    # save for later use
    df_all.to_csv(dataframe_cells_filename, index=False)
else:
    # load dataframe with cell population quantiles and histograms
    df_all = pd.read_csv(dataframe_cells_filename)

# remove DW=NaNs
df_all = df_all[~np.isnan(df_all['DW'])]
df_all = df_all.reset_index()

df_all['sex'] = df_all['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
df_all['ko_parent'] = df_all['ko_parent'].astype(
    pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
df_all['genotype'] = df_all['genotype'].astype(
    pd.api.types.CategoricalDtype(categories=['KLF14-KO:WT', 'KLF14-KO:Het'], ordered=True))

# for convenience of boxplots
df_all['area_m'] = df_all['area'] * 1e-3

# count number of cells per mouse, and assign that number to each cell, so that we can use it to compute weights for
# the quantile regression later
_, idx, cell_count = np.unique(df_all['id'], return_counts=True, return_inverse=True)
df_all['cell_count'] = cell_count[idx]

# normalise DW by BW
df_all['DW_BW'] = df_all['DW'] / df_all['BW']

# stratify by sex
df_f = df_all[df_all['sex'] == 'f']
df_m = df_all[df_all['sex'] == 'm']

# compute quantile regression models
model_f = smf.quantreg('area ~ DW_BW * C(ko_parent)', data=df_f, weights=1/df_f['cell_count']**2)
q25_model_f = model_f.fit(q=0.25)
q50_model_f = model_f.fit(q=0.50)
q75_model_f = model_f.fit(q=0.75)
print(q25_model_f.summary())
print(q50_model_f.summary())
print(q75_model_f.summary())

model_m = smf.quantreg('area ~ DW_BW * C(ko_parent)', data=df_m, weights=1/df_m['cell_count']**2)
q25_model_m = model_m.fit(q=0.25)
q50_model_m = model_m.fit(q=0.50)
q75_model_m = model_m.fit(q=0.75)
print(q25_model_m.summary())
print(q50_model_m.summary())
print(q75_model_m.summary())

# extract coefficients, errors and p-values from quartile models
df_coeff_f, df_ci_lo_f, df_ci_hi_f, df_pval_f = \
    cytometer.stats.models_coeff_ci_pval([q25_model_f, q50_model_f, q75_model_f], extra_hypotheses='Intercept + C(ko_parent)[T.MAT], DW_BW + DW_BW:C(ko_parent)[T.MAT]')
df_coeff_m, df_ci_lo_m, df_ci_hi_m, df_pval_m = \
    cytometer.stats.models_coeff_ci_pval([q25_model_m, q50_model_m, q75_model_m], extra_hypotheses='Intercept + C(ko_parent)[T.MAT], DW_BW + DW_BW:C(ko_parent)[T.MAT]')

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
    # save a table for the summary of findings spreadsheet: "summary_of_WAT_findings"
    cols = ['Intercept', 'Intercept+C(ko_parent)[T.MAT]', 'C(ko_parent)[T.MAT]',
            'DW_BW', 'DW_BW+DW_BW:C(ko_parent)[T.MAT]', 'DW_BW:C(ko_parent)[T.MAT]']

    df_concat = pd.DataFrame()
    for col in cols:
        df_concat = pd.concat([df_concat, df_coeff_f[col], df_pval_f[col], df_asterisk_f[col],
                               df_corrected_pval_f[col], df_corrected_asterisk_f[col]], axis=1)
    df_concat.to_csv(os.path.join(figures_dir, 'foo.csv'))

    df_concat = pd.DataFrame()
    for col in cols:
        df_concat = pd.concat([df_concat, df_coeff_m[col], df_pval_m[col], df_asterisk_m[col],
                               df_corrected_pval_m[col], df_corrected_asterisk_m[col]], axis=1)
    df_concat.to_csv(os.path.join(figures_dir, 'foo.csv'))

# plot
if DEBUG:
    plt.clf()
    plt.gcf().set_size_inches([9.6, 9.6])

    # Female PAT
    ax = plt.subplot(221)
    df = df_f[df_f['ko_parent'] == 'PAT']
    ids = list(df.groupby(by='id').groups.keys())
    positions = [df[df['id'] == x]['DW_BW'].iloc[0] * 1e3 for x in ids]
    widths = (np.max(positions) - np.min(positions)) / 100
    flierprops = dict(marker='.', markersize=1)
    df.boxplot(ax=ax, column='area_m', by='id', positions=positions, widths=widths, flierprops=flierprops)

    plot_linear_regression_DW(q25_model_f, df_f, ko_parent='PAT', style='C2', sx=1e3, sy=1e-3, label='Q1')
    plot_linear_regression_DW(q50_model_f, df_f, ko_parent='PAT', style='C3', sx=1e3, sy=1e-3, label='Q2')
    plot_linear_regression_DW(q75_model_f, df_f, ko_parent='PAT', style='C4', sx=1e3, sy=1e-3, label='Q3')

    plt.xlim(4, 60)
    xticks = [5, 10, 20, 30, 40, 50, 60]
    plt.tick_params(labelsize=14)
    plt.xticks(xticks, labels=['{0:.0f}'.format(x) for x in xticks])
    plt.xlabel('')
    plt.ylabel('Area ($10^3\ \mu m^2$)', fontsize=14)
    plt.title('Female PAT')
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Female MAT
    ax = plt.subplot(222)
    df = df_f[df_f['ko_parent'] == 'MAT']
    ids = list(df.groupby(by='id').groups.keys())
    positions = [df[df['id'] == x]['DW_BW'].iloc[0] * 1e3 for x in ids]
    widths = (np.max(positions) - np.min(positions)) / 100
    flierprops = dict(marker='.', markersize=1)
    boxprops = dict(color='C1')
    df.boxplot(ax=ax, column='area_m', by='id', positions=positions, widths=widths, flierprops=flierprops)

    plot_linear_regression_DW(q25_model_f, df_f, ko_parent='MAT', style='C2--', sx=1e3, sy=1e-3, label='Q1')
    plot_linear_regression_DW(q50_model_f, df_f, ko_parent='MAT', style='C3--', sx=1e3, sy=1e-3, label='Q2')
    plot_linear_regression_DW(q75_model_f, df_f, ko_parent='MAT', style='C4--', sx=1e3, sy=1e-3, label='Q3')

    plt.xlim(4, 60)
    xticks = [5, 10, 20, 30, 40, 50, 60]
    plt.tick_params(labelsize=14)
    plt.xticks(xticks, labels=['{0:.0f}'.format(x) for x in xticks])
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Female MAT')
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Male PAT
    ax = plt.subplot(223)
    df = df_m[df_m['ko_parent'] == 'PAT']
    ids = list(df.groupby(by='id').groups.keys())
    positions = [df[df['id'] == x]['DW_BW'].iloc[0] * 1e3 for x in ids]
    widths = (np.max(positions) - np.min(positions)) / 100
    flierprops = dict(marker='.', markersize=1)
    df.boxplot(ax=ax, column='area_m', by='id', positions=positions, widths=widths, flierprops=flierprops)

    plot_linear_regression_DW(q25_model_m, df_m, ko_parent='PAT', style='C2', sx=1e3, sy=1e-3, label='Q1')
    plot_linear_regression_DW(q50_model_m, df_m, ko_parent='PAT', style='C3', sx=1e3, sy=1e-3, label='Q2')
    plot_linear_regression_DW(q75_model_m, df_m, ko_parent='PAT', style='C4', sx=1e3, sy=1e-3, label='Q3')

    plt.xlim(4, 60)
    xticks = [5, 10, 20, 30, 40, 50, 60]
    plt.tick_params(labelsize=14)
    plt.xticks(xticks, labels=['{0:.0f}'.format(x) for x in xticks])
    plt.xlabel('DW / BW (mg / g)', fontsize=14)
    plt.ylabel('Area ($10^3\ \mu m^2$)', fontsize=14)
    plt.title('Male PAT')
    plt.tight_layout()

    # Male MAT
    ax = plt.subplot(224)
    df = df_m[df_m['ko_parent'] == 'MAT']
    ids = list(df.groupby(by='id').groups.keys())
    positions = [df[df['id'] == x]['DW_BW'].iloc[0] * 1e3 for x in ids]
    widths = (np.max(positions) - np.min(positions)) / 100
    flierprops = dict(marker='.', markersize=1)
    df.boxplot(ax=ax, column='area_m', by='id', positions=positions, widths=widths, flierprops=flierprops)

    plot_linear_regression_DW(q25_model_m, df_m, ko_parent='MAT', style='C2--', sx=1e3, sy=1e-3, label='Q1')
    plot_linear_regression_DW(q50_model_m, df_m, ko_parent='MAT', style='C3--', sx=1e3, sy=1e-3, label='Q2')
    plot_linear_regression_DW(q75_model_m, df_m, ko_parent='MAT', style='C4--', sx=1e3, sy=1e-3, label='Q3')

    plt.xlim(4, 60)
    xticks = [5, 10, 20, 30, 40, 50, 60]
    plt.tick_params(labelsize=14)
    plt.xticks(xticks, labels=['{0:.0f}'.format(x) for x in xticks])
    plt.xlabel('DW / BW (mg / g)', fontsize=14)
    plt.ylabel('')
    plt.title('Male MAT')
    plt.tight_layout()

    if depot == 'gwat':
        plt.suptitle('Gonadal', fontsize=14)
    elif depot == 'sqwat':
        plt.suptitle('Subcutaneous', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_quartile_dw_bw_quantreg_boxplots_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_quartile_dw_bw_quantreg_boxplots_' + depot + '.svg'))


# 0.1 , 0.2, ..., 0.9
deciles_idx = list(range(2, 20, 2))
deciles = quantiles[deciles_idx]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# fit linear models for each decile
model_f = smf.quantreg('area ~ DW_BW * C(ko_parent)', data=df_f, weights=1/df_f['cell_count']**2)
decile_models_f = [model_f.fit(q=d) for d in deciles]
model_m = smf.quantreg('area ~ DW_BW * C(ko_parent)', data=df_m, weights=1/df_m['cell_count']**2)
decile_models_m = [model_m.fit(q=d) for d in deciles]

print(decile_models_f[4].summary())
print(decile_models_m[4].summary())

# extract coefficients, errors and p-values from quantile models
df_coeff_f, df_ci_lo_f, df_ci_hi_f, df_pval_f = \
    cytometer.stats.models_coeff_ci_pval(decile_models_f, extra_hypotheses='Intercept + C(ko_parent)[T.MAT], DW_BW + DW_BW:C(ko_parent)[T.MAT]')
df_coeff_m, df_ci_lo_m, df_ci_hi_m, df_pval_m = \
    cytometer.stats.models_coeff_ci_pval(decile_models_m, extra_hypotheses='Intercept + C(ko_parent)[T.MAT], DW_BW + DW_BW:C(ko_parent)[T.MAT]')

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
    # save a table for the summary of findings spreadsheet: "summary_of_WAT_findings"
    cols = ['Intercept', 'Intercept+C(ko_parent)[T.MAT]', 'C(ko_parent)[T.MAT]',
            'DW_BW', 'DW_BW+DW_BW:C(ko_parent)[T.MAT]', 'DW_BW:C(ko_parent)[T.MAT]']

    df_concat = pd.DataFrame()
    for col in cols:
        df_concat = pd.concat([df_concat, df_coeff_f[col], df_pval_f[col], df_asterisk_f[col],
                               df_corrected_pval_f[col], df_corrected_asterisk_f[col]], axis=1)
    df_concat.to_csv(os.path.join(figures_dir, 'foo.csv'))

    df_concat = pd.DataFrame()
    for col in cols:
        df_concat = pd.concat([df_concat, df_coeff_m[col], df_pval_m[col], df_asterisk_m[col],
                               df_corrected_pval_m[col], df_corrected_asterisk_m[col]], axis=1)
    df_concat.to_csv(os.path.join(figures_dir, 'foo.csv'))

if DEBUG:
    plt.clf()
    plt.gcf().set_size_inches([7.2, 9.5])
    q = np.array(deciles) * 100

    plt.subplot(4,2,1)
    if depot == 'gwat':
        ylim = (-1, 28)
    elif depot == 'sqwat':
        ylim = (-1, 15)
    h1, h2 = cytometer.stats.cytometer.stats.plot_model_coeff_compare2(q, df_coeff_f['Intercept'] * 1e-3,
                              df_ci_lo_f['Intercept'] * 1e-3,
                              df_ci_hi_f['Intercept'] * 1e-3,
                              df_pval_f['Intercept'],
                              df_coeff_f['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
                              df_ci_lo_f['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
                              df_ci_hi_f['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
                              df_pval_f['Intercept+C(ko_parent)[T.MAT]'],
                              df_corrected_pval_1=df_corrected_pval_f['Intercept'],
                              df_corrected_pval_2=df_corrected_pval_f['Intercept+C(ko_parent)[T.MAT]'],
                              ylim=ylim, color_1='C0', color_2='C1', label_1='PAT', label_2='MAT')
    plt.title('Female', fontsize=14)
    plt.ylabel(r'$\beta_{0}\ (10^3\ \mu m^2)$', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.ylim(ylim)

    plt.subplot(4,2,2)
    if depot == 'gwat':
        ylim = (-1, 28)
    elif depot == 'sqwat':
        ylim = (-1, 15)
    h1, h2 = cytometer.stats.cytometer.stats.plot_model_coeff_compare2(q, df_coeff_m['Intercept'] * 1e-3,
                              df_ci_lo_m['Intercept'] * 1e-3,
                              df_ci_hi_m['Intercept'] * 1e-3,
                              df_pval_m['Intercept'],
                              df_coeff_m['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
                              df_ci_lo_m['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
                              df_ci_hi_m['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
                              df_pval_m['Intercept+C(ko_parent)[T.MAT]'],
                              df_corrected_pval_1=df_corrected_pval_m['Intercept'],
                              df_corrected_pval_2=df_corrected_pval_m['Intercept+C(ko_parent)[T.MAT]'],
                              ylim=ylim, color_1='C0', color_2='C1', label_1='PAT', label_2='MAT')
    plt.title('Male', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.ylim(ylim)

    plt.subplot(4,2,3)
    if depot == 'gwat':
        ylim = (-0.5, 9)
    elif depot == 'sqwat':
        ylim = (-0.5, 4.5)
    cytometer.stats.plot_model_coeff(q, df_coeff_f['C(ko_parent)[T.MAT]'] * 1e-3,
                     df_ci_lo_f['C(ko_parent)[T.MAT]'] * 1e-3,
                     df_ci_hi_f['C(ko_parent)[T.MAT]'] * 1e-3,
                     df_pval_f['C(ko_parent)[T.MAT]'],
                     df_corrected_pval=df_corrected_pval_f['C(ko_parent)[T.MAT]'],
                     ylim=ylim, color='k', label='PAT$\mapsto$MAT')
    plt.ylabel(r'$\Delta\beta_{0}\ (10^3\ \mu m^2)$', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.ylim(ylim)

    plt.subplot(4,2,4)
    if depot == 'gwat':
        ylim = (-0.5, 9)
    elif depot == 'sqwat':
        ylim = (-0.5, 4.5)
    cytometer.stats.plot_model_coeff(q, df_coeff_m['C(ko_parent)[T.MAT]'] * 1e-3,
                     df_ci_lo_m['C(ko_parent)[T.MAT]'] * 1e-3,
                     df_ci_hi_m['C(ko_parent)[T.MAT]'] * 1e-3,
                     df_pval_m['C(ko_parent)[T.MAT]'],
                     df_corrected_pval=df_corrected_pval_m['C(ko_parent)[T.MAT]'],
                     ylim=ylim, color='k', label='PAT$\mapsto$MAT')
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.ylim(ylim)

    plt.subplot(4,2,5)
    if depot == 'gwat':
        ylim = (-1.70, 2.75)
    elif depot == 'sqwat':
        ylim = (-1, 2.5)
    h1, h2 = cytometer.stats.cytometer.stats.plot_model_coeff_compare2(q, df_coeff_f['DW_BW'] * 1e-5,
                              df_ci_lo_f['DW_BW'] * 1e-5,
                              df_ci_hi_f['DW_BW'] * 1e-5,
                              df_pval_f['DW_BW'],
                              df_coeff_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                              df_ci_lo_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                              df_ci_hi_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                              df_pval_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'],
                              df_corrected_pval_1=df_corrected_pval_f['DW_BW'],
                              df_corrected_pval_2=df_corrected_pval_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'],
                              ylim=ylim, color_1='C0', color_2='C1', label_1='PAT', label_2='MAT')
    plt.ylabel(r'$\beta_{DW/BW}\ (10^5\ \mu m^2)$', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.ylim(ylim)

    plt.subplot(4,2,6)
    if depot == 'gwat':
        ylim = (-1.70, 2.75)
    elif depot == 'sqwat':
        ylim = (-1, 2.5)
    h1, h2 = cytometer.stats.cytometer.stats.plot_model_coeff_compare2(q, df_coeff_m['DW_BW'] * 1e-5,
                              df_ci_lo_m['DW_BW'] * 1e-5,
                              df_ci_hi_m['DW_BW'] * 1e-5,
                              df_pval_m['DW_BW'],
                              df_coeff_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                              df_ci_lo_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                              df_ci_hi_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                              df_pval_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'],
                              df_corrected_pval_1=df_corrected_pval_m['DW_BW'],
                              df_corrected_pval_2=df_corrected_pval_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'],
                              ylim=ylim, color_1='C0', color_2='C1', label_1='PAT', label_2='MAT')
    plt.tick_params(labelsize=14)
    plt.ylim(ylim)

    plt.subplot(4, 2, 7)
    if depot == 'gwat':
        ylim = (-2.20, 1.00)
    elif depot == 'sqwat':
        ylim = (-2.1, 2)
    cytometer.stats.plot_model_coeff(q, df_coeff_f['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                     df_ci_lo_f['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                     df_ci_hi_f['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                     df_pval_f['DW_BW:C(ko_parent)[T.MAT]'],
                     df_corrected_pval=df_corrected_pval_f['DW_BW:C(ko_parent)[T.MAT]'],
                     ylim=ylim, color='k', label='PAT$\mapsto$MAT')
    plt.xlabel('Quantile (%)', fontsize=14)
    plt.ylabel(r'$\Delta \beta_{DW/BW}\ (10^5\ \mu m^2)$', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.ylim(ylim)

    plt.subplot(4, 2, 8)
    if depot == 'gwat':
        ylim = (-2.20, 1.00)
    elif depot == 'sqwat':
        ylim = (-2.1, 2)
    cytometer.stats.plot_model_coeff(q, df_coeff_m['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                     df_ci_lo_m['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                     df_ci_hi_m['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
                     df_pval_m['DW_BW:C(ko_parent)[T.MAT]'],
                     df_corrected_pval=df_corrected_pval_m['DW_BW:C(ko_parent)[T.MAT]'],
                     ylim=ylim, color='k', label='PAT$\mapsto$MAT')
    plt.xlabel('Quantile (%)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.ylim(ylim)

    plt.tight_layout()

    if depot == 'gwat':
        plt.suptitle('Gonadal', fontsize=14)
    elif depot == 'sqwat':
        plt.suptitle('Subcutaneous', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_decile_dw_bw_quantreg_coeffs_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_decile_dw_bw_quantreg_coeffs_' + depot + '.svg'))
