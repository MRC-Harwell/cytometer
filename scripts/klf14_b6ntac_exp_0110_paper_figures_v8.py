"""
Generate figures for the DeepCytometer paper for v8 of the pipeline.

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
import scipy.stats
import sklearn.neighbors, sklearn.model_selection
import pandas as pd
# from mlxtend.evaluate import permutation_test
# from statsmodels.stats.multitest import multipletests
# import math
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

# 2.00646604, 2.02207953, ..., 5.11355093, 5.12916443
log10_area_bin_edges = np.linspace(np.log10(min_area), np.log10(max_area), 201)
log10_area_bin_centers = (log10_area_bin_edges[0:-1] + log10_area_bin_edges[1:]) / 2.0

for method in ['auto', 'corrected']:

    dataframe_areas_filename = os.path.join(dataframe_dir, 'klf14_b6ntac_exp_0110_dataframe_areas_' + method + '.csv')

    if os.path.isfile(dataframe_areas_filename):

        # load dataframe with cell population quantiles and histograms
        df_all = pd.read_csv(dataframe_areas_filename)

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
                df = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(json_file),
                                                               values=[depot, ], values_tag='depot',
                                                               tags_to_keep=['id', 'ko_parent', 'sex', 'genotype', 'BW'])

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

                # compute areas of the cells
                areas = np.array([shapely.geometry.Polygon(cell).area for cell in cells])

                # smooth out histogram
                kde = sklearn.neighbors.KernelDensity(bandwidth=1000, kernel='gaussian').fit(areas.reshape(-1, 1))
                log_dens = kde.score_samples((10 ** log10_area_bin_centers).reshape(-1, 1))
                pdf = np.exp(log_dens)

                # compute mode
                df['area_smoothed_mode'] = (10 ** log10_area_bin_centers)[np.argmax(pdf)]

                # compute areas at population quantiles
                areas_at_quantiles = stats.mstats.hdquantiles(areas, prob=quantiles, axis=0)
                for j in range(len(quantiles)):
                    df['area_q_' + '{0:02d}'.format(j)] = areas_at_quantiles[j]

                # compute histograms with log10(area) binning
                histo, _ = np.histogram(areas, bins=10**log10_area_bin_edges, density=True)
                for j in range(len(log10_area_bin_centers)):
                    df['histo_bin_' + '{0:03d}'.format(j)] = histo[j]

                # smoothed histogram
                for j in range(len(log10_area_bin_centers)):
                    df['smoothed_histo_bin_' + '{0:03d}'.format(j)] = pdf[j]

                if DEBUG:
                    plt.clf()
                    plt.plot(10 ** log10_area_bin_centers, histo, label='Areas')
                    plt.plot(10 ** log10_area_bin_centers, pdf, label='Kernel')
                    plt.plot([df['area_smoothed_mode'], df['area_smoothed_mode']], [0, pdf.max()], 'k', label='Mode')
                    plt.legend()

                # add results to total dataframe
                df_all = pd.concat([df_all, df], ignore_index=True)

                # save dataframe
                df_all.to_csv(dataframe_areas_filename, index=False)

########################################################################################################################
## Analyse cell populations from automatically segmented images in two depots: SQWAT and GWAT:
##   Cell area histograms
##   HD quantiles of cell areas
## Cell populations were computed in previous section
### USED IN PAPER
########################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import seaborn as sns

# directories
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
annotations_dir = os.path.join(home, 'Data/cytometer_data/aida_data_Klf14_v8/annotations')
ndpi_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
dataframe_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
metainfo_dir = os.path.join(home, 'Data/cytometer_data/klf14')

DEBUG = False

method = 'corrected'

# load dataframe with cell population quantiles and histograms
dataframe_areas_filename = os.path.join(dataframe_dir, 'klf14_b6ntac_exp_0110_dataframe_areas_' + method + '.csv')
df_all = pd.read_csv(dataframe_areas_filename)
df_all = df_all[~np.isnan(df_all['BW'])]
df_all = df_all.reset_index()

df_all['sex'] = df_all['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
df_all['ko_parent'] = df_all['ko_parent'].astype(
    pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
df_all['genotype'] = df_all['genotype'].astype(
    pd.api.types.CategoricalDtype(categories=['KLF14-KO:WT', 'KLF14-KO:Het'], ordered=True))

## auxiliary functions

def models_coeff_ci_pval(models):
    df_coeff = pd.DataFrame()
    df_ci_lo = pd.DataFrame()
    df_ci_hi = pd.DataFrame()
    df_pval = pd.DataFrame()
    for model in models:
        # values of coefficients
        df = pd.DataFrame(data=model.params).transpose()
        df_coeff = pd.concat((df_coeff, df))
        # values of coefficient's confidence interval
        df = pd.DataFrame(data=model.conf_int()[0]).transpose()
        df_ci_lo = pd.concat((df_ci_lo, df))
        df = pd.DataFrame(data=model.conf_int()[1]).transpose()
        df_ci_hi = pd.concat((df_ci_hi, df))
        # p-values
        df = pd.DataFrame(data=model.pvalues).transpose()
        df_pval = pd.concat((df_pval, df))
    df_coeff = df_coeff.reset_index()
    df_ci_lo = df_ci_lo.reset_index()
    df_ci_hi = df_ci_hi.reset_index()
    df_pval = df_pval.reset_index()
    df_coeff.drop(labels='index', axis='columns', inplace=True)
    df_ci_lo.drop(labels='index', axis='columns', inplace=True)
    df_ci_hi.drop(labels='index', axis='columns', inplace=True)
    df_pval.drop(labels='index', axis='columns', inplace=True)
    return df_coeff, df_ci_lo, df_ci_hi, df_pval

def plot_linear_regression(model, df, sex=None, ko_parent=None, genotype=None, style=None, sy=1.0):
    if np.std(df['BW'] / df['BW__']) > 1e-8:
        raise ValueError('BW / BW__ is not the same for all rows')
    BW__factor = (df['BW'] / df['BW__']).mean()
    BW__lim = np.array([df['BW__'].min(), df['BW__'].max()])
    X = pd.DataFrame(data={'BW__': BW__lim, 'sex': [sex, sex], 'ko_parent': [ko_parent, ko_parent],
                           'genotype': [genotype, genotype]})
    y_pred = model.predict(X)
    plt.plot(BW__lim * BW__factor, y_pred * sy, style)

def pval_to_asterisk(pval):
    if pval > 0.05:
        return 'ns'
    elif pval > 0.01:
        return '*'
    elif pval > 0.001:
        return '**'
    elif pval > 0.0001:
        return '***'
    else:
        return '****'

def plot_pvals(pvals, xs, ys):
    ylim = plt.gca().get_ylim()
    offset = (np.max(ylim) - np.min(ylim)) * 0.10
    for pval, x, y in zip(pvals, xs, ys):
        str = pval_to_asterisk(pval)
        if pval > 0.05:
            plt.text(x, y + offset, str, ha='center')
        else:
            plt.text(x, y + offset, str, ha='center', rotation=90)

def plot_model_coeff(q, df_coeff, df_ci_lo, df_ci_hi, df_pval):
    plt.plot(q, df_coeff)
    plt.fill_between(q, df_ci_lo, df_ci_hi, alpha=0.5)
    plot_pvals(df_pval, q, df_coeff)


## histograms
## NOT USED IN PAPER (we use smoothed histograms instead)
########################################################################################################################

# # 2.00646604, 2.02207953, ..., 5.11355093, 5.12916443
# log10_area_bin_edges = np.linspace(np.log10(min_area), np.log10(max_area), 201)
# log10_area_bin_centers = (log10_area_bin_edges[0:-1] + log10_area_bin_edges[1:]) / 2.0
#
# columns = []
# for j in range(len(log10_area_bin_edges) - 1):
#     columns += ['histo_bin_' + '{0:03d}'.format(j),]
#
# # f PAT
# df = df_all[(df_all['depot'] == 'gwat') & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'PAT')]
# df = df.reset_index()
# histo = df[columns]
#
# if DEBUG:
#     plt.clf()
#     plt.plot(10 ** log10_area_bin_centers * 1e-3, np.transpose(histo))
#     plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)
#     plt.title('GWAT f PAT')
#
# # f MAT
# df = df_all[(df_all['depot'] == 'gwat') & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')]
# df = df.reset_index()
# histo = df[columns]
#
# if DEBUG:
#     plt.clf()
#     plt.plot(10 ** log10_area_bin_centers, np.transpose(histo))
#     plt.xlabel('Area ($\mu m^2$)')
#     plt.title('GWAT f MAT')

## smoothed histograms
## USED IN THE PAPER
########################################################################################################################

# 2.00646604, 2.02207953, ..., 5.11355093, 5.12916443
log10_area_bin_edges = np.linspace(np.log10(min_area), np.log10(max_area), 201)
log10_area_bin_centers = (log10_area_bin_edges[0:-1] + log10_area_bin_edges[1:]) / 2.0

columns = []
for j in range(len(log10_area_bin_edges) - 1):
    columns += ['smoothed_histo_bin_' + '{0:03d}'.format(j),]

depot = 'gwat'
# depot = 'sqwat'

if SAVEFIG:
    plt.clf()

    # f PAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'PAT')]
    df = df.reset_index()
    histo = df[columns]

    plt.subplot(221)
    plt.plot(10 ** log10_area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.text(125, 0.8, 'female PAT', horizontalalignment='right', fontsize=14)
    plt.xlim(-6.5, max_area * 1e-3)

    # f MAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')]
    df = df.reset_index()
    histo = df[columns]

    plt.subplot(222)
    plt.plot(10 ** log10_area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.tick_params(axis='y', left=False, labelleft=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.text(125, 0.8, 'female MAT', horizontalalignment='right', fontsize=14)
    plt.xlim(-6.5, max_area * 1e-3)

    # m PAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'PAT')]
    df = df.reset_index()
    histo = df[columns]

    plt.subplot(223)
    plt.plot(10 ** log10_area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.tick_params(axis='y', left=False, labelleft=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)
    plt.text(125, 0.8, 'male PAT', horizontalalignment='right', fontsize=14)
    plt.xlim(-6.5, max_area * 1e-3)

    # m MAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'MAT')]
    df = df.reset_index()
    histo = df[columns]

    plt.subplot(224)
    plt.plot(10 ** log10_area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.tick_params(axis='y', left=False, labelleft=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)
    plt.text(125, 0.8, 'male MAT', horizontalalignment='right', fontsize=14)
    plt.xlim(-6.5, max_area * 1e-3)

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
    histo = df[columns]
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)

    plt.subplot(221)
    plt.fill_between(10 ** log10_area_bin_centers * 1e-3, histo_q1[0,] / histo_q3.max(), histo_q3[0,] / histo_q3.max(),
                     alpha=0.5, color='C0')
    plt.plot(10 ** log10_area_bin_centers * 1e-3, histo_q2[0,] / histo_q3.max(), 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.text(125, 0.8, 'female PAT', horizontalalignment='right', fontsize=14)
    plt.xlim(-6.5, max_area * 1e-3)

    # f MAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')]
    df = df.reset_index()
    histo = df[columns]
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)

    plt.subplot(222)
    plt.fill_between(10 ** log10_area_bin_centers * 1e-3, histo_q1[0,] / histo_q3.max(), histo_q3[0,] / histo_q3.max(),
                     alpha=0.5, color='C0')
    plt.plot(10 ** log10_area_bin_centers * 1e-3, histo_q2[0,] / histo_q3.max(), 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.text(125, 0.8, 'female MAT', horizontalalignment='right', fontsize=14)
    plt.xlim(-6.5, max_area * 1e-3)

    # m PAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'PAT')]
    df = df.reset_index()
    histo = df[columns]
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)

    plt.subplot(223)
    plt.fill_between(10 ** log10_area_bin_centers * 1e-3, histo_q1[0,] / histo_q3.max(), histo_q3[0,] / histo_q3.max(),
                     alpha=0.5, color='C0')
    plt.plot(10 ** log10_area_bin_centers * 1e-3, histo_q2[0,] / histo_q3.max(), 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)
    plt.text(125, 0.8, 'male PAT', horizontalalignment='right', fontsize=14)
    plt.xlim(-6.5, max_area * 1e-3)

    # m MAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'MAT')]
    df = df.reset_index()
    histo = df[columns]
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)

    plt.subplot(224)
    plt.fill_between(10 ** log10_area_bin_centers * 1e-3, histo_q1[0,] / histo_q3.max(), histo_q3[0,] / histo_q3.max(),
                     alpha=0.5, color='C0')
    plt.plot(10 ** log10_area_bin_centers * 1e-3, histo_q2[0,] / histo_q3.max(), 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)
    plt.text(125, 0.8, 'male MAT', horizontalalignment='right', fontsize=14)
    plt.xlim(-6.5, max_area * 1e-3)

    if depot == 'gwat':
        plt.suptitle('Gonadal', fontsize=14)
    elif depot == 'sqwat':
        plt.suptitle('Subcutaneous', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_smoothed_histo_quartiles_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_smoothed_histo_quartiles_' + depot + '.svg'))

## Plots of body weight vs cull age, and swarm plots of body weight
## USED IN PAPER
########################################################################################################################

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


if SAVEFIG:
    # plot body weight vs. age of culling
    plt.clf()
    plt.scatter(metainfo_f[metainfo_f['ko_parent'] == 'PAT']['cull_age'],
                metainfo_f[metainfo_f['ko_parent'] == 'PAT']['BW'], c='C0')
    plt.scatter(metainfo_f[metainfo_f['ko_parent'] == 'MAT']['cull_age'],
                metainfo_f[metainfo_f['ko_parent'] == 'MAT']['BW'], c='C1')
    plt.scatter(metainfo_m[metainfo_f['ko_parent'] == 'PAT']['cull_age'],
                metainfo_m[metainfo_f['ko_parent'] == 'PAT']['BW'], c='C2')
    plt.scatter(metainfo_m[metainfo_f['ko_parent'] == 'MAT']['cull_age'],
                metainfo_m[metainfo_f['ko_parent'] == 'MAT']['BW'], c='C3')
    plt.xlabel('Cull age (days)', fontsize=14)
    plt.ylabel('Body weight (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

    cull_age__lim = np.array([metainfo['cull_age__'].min(), metainfo['cull_age__'].max()])
    cull_age_lim = np.array([metainfo['cull_age'].min(), metainfo['cull_age'].max()])
    X = pd.DataFrame(data={'cull_age__': cull_age__lim, 'ko_parent': ['PAT', 'PAT']})
    y_pred = bw_model_f.predict(X)
    plt.plot(cull_age_lim, y_pred, 'C0', linewidth=2, label='f PAT')
    X = pd.DataFrame(data={'cull_age__': cull_age__lim, 'ko_parent': ['MAT', 'MAT']})
    y_pred = bw_model_f.predict(X)
    plt.plot(cull_age_lim, y_pred, 'C1', linewidth=2, label='f MAT')
    pval_text = '$p_{cull\ age}$=' + '{0:.2f}'.format(bw_model_f.pvalues['cull_age__']) + '\n' + \
                '$p_{MAT}$=' + '{0:.3f}'.format(bw_model_f.pvalues['C(ko_parent)[T.MAT]'])
    plt.text(143.75, 26.3, pval_text, va='top')

    X = pd.DataFrame(data={'cull_age__': cull_age__lim, 'ko_parent': ['PAT', 'PAT']})
    y_pred = bw_model_m.predict(X)
    plt.plot(cull_age_lim, y_pred, 'C2', linewidth=2, label='m PAT')
    X = pd.DataFrame(data={'cull_age__': cull_age__lim, 'ko_parent': ['MAT', 'MAT']})
    y_pred = bw_model_m.predict(X)
    plt.plot(cull_age_lim, y_pred, 'C3', linewidth=2, label='m MAT')
    pval_text = '$p_{cull\ age}$=' + '{0:.2f}'.format(bw_model_m.pvalues['cull_age__']) + '\n' + \
                '$p_{MAT}$=' + '{0:.2f}'.format(bw_model_m.pvalues['C(ko_parent)[T.MAT]'])
    plt.text(143.75, 44, pval_text, va='bottom')

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
    pval_text = '$p_{MAT}$=' + '{0:.3f}'.format(bw_model_f.pvalues['C(ko_parent)[T.MAT]']) + \
                ' (' + pval_to_asterisk(bw_model_f.pvalues['C(ko_parent)[T.MAT]']) + ')'
    plt.text(0, 44.5, pval_text, ha='center', va='bottom', fontsize=14)
    plt.plot([0.8, 0.8, 1.2, 1.2], [52, 54, 54, 52], 'k', lw=1.5)
    pval_text = '$p_{MAT}$=' + '{0:.2f}'.format(bw_model_m.pvalues['C(ko_parent)[T.MAT]']) + \
                ' (' + pval_to_asterisk(bw_model_m.pvalues['C(ko_parent)[T.MAT]']) + ')'
    plt.text(1, 54.5, pval_text, ha='center', va='bottom', fontsize=14)
    plt.ylim(18, 58)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_swarm_bw.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_swarm_bw.svg'))


# scale cull_age to avoid large condition numbers
BW_mean = metainfo['BW'].mean()
metainfo['BW__'] = metainfo['BW'] / BW_mean

# for convenience
metainfo_f = metainfo[metainfo['sex'] == 'f']
metainfo_m = metainfo[metainfo['sex'] == 'm']

# models of depot weight ~ BW * ko_parent
gwat_model_f = sm.RLM.from_formula('gWAT ~ BW__ * C(ko_parent)', data=metainfo_f, M=sm.robust.norms.HuberT()).fit()
gwat_model_m = sm.RLM.from_formula('gWAT ~ BW__ * C(ko_parent)', data=metainfo_m, M=sm.robust.norms.HuberT()).fit()
sqwat_model_f = sm.RLM.from_formula('SC ~ BW__ * C(ko_parent)', data=metainfo_f, M=sm.robust.norms.HuberT()).fit()
sqwat_model_m = sm.RLM.from_formula('SC ~ BW__ * C(ko_parent)', data=metainfo_m, M=sm.robust.norms.HuberT()).fit()

print(gwat_model_f.summary())
print(gwat_model_m.summary())
print(sqwat_model_f.summary())
print(sqwat_model_m.summary())

if SAVEFIG:
    plt.clf()

    plt.subplot(221)
    df = metainfo_f[metainfo_f['ko_parent'] == 'PAT']
    plt.scatter(df['BW'], df['gWAT'], c='C0', label='PAT')
    df = metainfo_f[metainfo_f['ko_parent'] == 'MAT']
    plt.scatter(df['BW'], df['gWAT'], c='C1', label='MAT')
    plot_linear_regression(gwat_model_f, metainfo_f, sex='f', ko_parent='PAT', style='C0')
    plot_linear_regression(gwat_model_f, metainfo_f, sex='f', ko_parent='MAT', style='C1')
    pval_bw = gwat_model_f.pvalues['BW__']
    pval_mat = gwat_model_f.pvalues['C(ko_parent)[T.MAT]']
    pval_text = '$p_{BW}$=' + '{0:.2e}'.format(pval_bw) + ' (' + pval_to_asterisk(pval_bw) + ')' + \
                '\n' + \
                '$p_{MAT}$=' + '{0:.2f}'.format(pval_mat) + ' (' + pval_to_asterisk(pval_mat) + ')'
    plt.text(0.05, 0.95, pval_text, transform=plt.gca().transAxes, va='top')
    plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    plt.tick_params(labelsize=14)
    plt.title('Female', fontsize=14)
    plt.ylabel('Gonadal\ndepot weight (g)', fontsize=14)
    plt.legend(loc='lower right')

    plt.subplot(223)
    df = metainfo_f[metainfo_f['ko_parent'] == 'PAT']
    plt.scatter(df['BW'], df['SC'], c='C0', label='PAT')
    df = metainfo_f[metainfo_f['ko_parent'] == 'MAT']
    plt.scatter(df['BW'], df['SC'], c='C1', label='MAT')
    plot_linear_regression(sqwat_model_f, metainfo_f, sex='f', ko_parent='PAT', style='C0')
    plot_linear_regression(sqwat_model_f, metainfo_f, sex='f', ko_parent='MAT', style='C1')
    pval_bw = sqwat_model_f.pvalues['BW__']
    pval_mat = sqwat_model_f.pvalues['C(ko_parent)[T.MAT]']
    pval_text = '$p_{BW}$=' + '{0:.2e}'.format(pval_bw) + ' (' + pval_to_asterisk(pval_bw) + ')' + \
                '\n' + \
                '$p_{MAT}$=' + '{0:.3f}'.format(pval_mat) + ' (' + pval_to_asterisk(pval_mat) + ')'
    plt.text(0.2, 0.95, pval_text, transform=plt.gca().transAxes, va='top')
    plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    plt.tick_params(labelsize=14)
    plt.xlabel('Body weight (g)', fontsize=14)
    plt.ylabel('Subcutaneous\ndepot weight (g)', fontsize=14)

    plt.subplot(222)
    df = metainfo_m[metainfo_m['ko_parent'] == 'PAT']
    plt.scatter(df['BW'], df['gWAT'], c='C0', label='PAT')
    df = metainfo_m[metainfo_m['ko_parent'] == 'MAT']
    plt.scatter(df['BW'], df['gWAT'], c='C1', label='MAT')
    plot_linear_regression(gwat_model_m, metainfo_m, sex='m', ko_parent='PAT', style='C0')
    plot_linear_regression(gwat_model_m, metainfo_m, sex='m', ko_parent='MAT', style='C1')
    pval_bw = gwat_model_m.pvalues['BW__']
    pval_mat = gwat_model_m.pvalues['C(ko_parent)[T.MAT]']
    pval_text = '$p_{BW}$=' + '{0:.2f}'.format(pval_bw) + ' (' + pval_to_asterisk(pval_bw) + ')' + \
                '\n' + \
                '$p_{MAT}$=' + '{0:.3f}'.format(pval_mat) + ' (' + pval_to_asterisk(pval_mat) + ')'
    plt.text(0.5, 0.25, pval_text, transform=plt.gca().transAxes, va='top', ha='center')
    plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    plt.tick_params(labelsize=14)
    plt.title('Male', fontsize=14)

    plt.subplot(224)
    df = metainfo_m[metainfo_m['ko_parent'] == 'PAT']
    plt.scatter(df['BW'], df['SC'], c='C0', label='PAT')
    df = metainfo_m[metainfo_m['ko_parent'] == 'MAT']
    plt.scatter(df['BW'], df['SC'], c='C1', label='MAT')
    plot_linear_regression(sqwat_model_m, metainfo_m, sex='m', ko_parent='PAT', style='C0')
    plot_linear_regression(sqwat_model_m, metainfo_m, sex='m', ko_parent='MAT', style='C1')
    pval_bw = sqwat_model_m.pvalues['BW__']
    pval_mat = sqwat_model_m.pvalues['C(ko_parent)[T.MAT]']
    pval_text = '$p_{BW}$=' + '{0:.2f}'.format(pval_bw) + ' (' + pval_to_asterisk(pval_bw) + ')' + \
                '\n' + \
                '$p_{MAT}$=' + '{0:.3f}'.format(pval_mat) + ' (' + pval_to_asterisk(pval_mat) + ')'
    plt.text(0.2, 0.95, pval_text, transform=plt.gca().transAxes, va='top')
    plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    plt.tick_params(labelsize=14)
    plt.xlabel('Body weight (g)', fontsize=14)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_depot_linear_model.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_depot_linear_model.svg'))


## linear regression analysis of quantile_area ~ BW * ko_parent
## (only mode, 25%-, 50%- and 75%-quantiles for illustration purposes and debugging)
## USED IN PAPER
########################################################################################################################

# 0.05, 0.1 , 0.15, 0.2, ..., 0.9 , 0.95
quantiles = np.linspace(0, 1, 21)
assert(quantiles[5]  == 0.25)  # check: we are selecting the 25% quantile
assert(quantiles[10] == 0.5)  # check: we are selecting the median
assert(quantiles[15] == 0.75)  # check: we are selecting the 75% quantile

depot = 'gwat'
# depot = 'sqwat'

# fit linear models
df = df_all[(df_all['depot'] == depot)].copy()
BW__factor = df['BW'].mean()
df['BW__'] = df['BW'] / BW__factor
df_f = df[df['sex'] == 'f'].reset_index()
df_m = df[df['sex'] == 'm'].reset_index()

mode_model_f = sm.RLM.from_formula('area_smoothed_mode ~ BW__ * C(ko_parent)', data=df_f, M=sm.robust.norms.HuberT()).fit()
q25_model_f = sm.RLM.from_formula('area_q_05 ~ BW__ * C(ko_parent)', data=df_f, M=sm.robust.norms.HuberT()).fit()
q50_model_f = sm.RLM.from_formula('area_q_10 ~ BW__ * C(ko_parent)', data=df_f, M=sm.robust.norms.HuberT()).fit()
q75_model_f = sm.RLM.from_formula('area_q_15 ~ BW__ * C(ko_parent)', data=df_f, M=sm.robust.norms.HuberT()).fit()
# mode_model_f = sm.formula.ols('area_smoothed_mode ~ BW__ * C(ko_parent)', data=df_f).fit()
# q25_model_f = sm.formula.ols('area_q_05 ~ BW__ * C(ko_parent)', data=df_f).fit()
# q50_model_f = sm.formula.ols('area_q_10 ~ BW__ * C(ko_parent)', data=df_f).fit()
# q75_model_f = sm.formula.ols('area_q_15 ~ BW__ * C(ko_parent)', data=df_f).fit()
print(mode_model_f.summary())
print(q25_model_f.summary())
print(q50_model_f.summary())
print(q75_model_f.summary())

mode_model_m = sm.RLM.from_formula('area_smoothed_mode ~ BW__ * C(ko_parent)', data=df_m, M=sm.robust.norms.HuberT()).fit()
q25_model_m = sm.RLM.from_formula('area_q_05 ~ BW__ * C(ko_parent)', data=df_m, M=sm.robust.norms.HuberT()).fit()
q50_model_m = sm.RLM.from_formula('area_q_10 ~ BW__ * C(ko_parent)', data=df_m, M=sm.robust.norms.HuberT()).fit()
q75_model_m = sm.RLM.from_formula('area_q_15 ~ BW__ * C(ko_parent)', data=df_m, M=sm.robust.norms.HuberT()).fit()
# mode_model_m = sm.formula.ols('area_smoothed_mode ~ BW__ * C(ko_parent)', data=df_m).fit()
# q25_model_m = sm.formula.ols('area_q_05 ~ BW__ * C(ko_parent)', data=df_m).fit()
# q50_model_m = sm.formula.ols('area_q_10 ~ BW__ * C(ko_parent)', data=df_m).fit()
# q75_model_m = sm.formula.ols('area_q_15 ~ BW__ * C(ko_parent)', data=df_m).fit()
print(mode_model_m.summary())
print(q25_model_m.summary())
print(q50_model_m.summary())
print(q75_model_m.summary())

# plot
if SAVEFIG:
    plt.clf()
    plt.gcf().set_size_inches([6.4, 9.99])

    plt.subplot(421)
    df = df_f[df_f['ko_parent'] == 'PAT']
    plt.scatter(df['BW'], df['area_smoothed_mode'] * 1e-3, c='C0', label='PAT')
    df = df_f[df_f['ko_parent'] == 'MAT']
    plt.scatter(df['BW'], df['area_smoothed_mode'] * 1e-3, c='C1', label='MAT')
    plot_linear_regression(mode_model_f, df_f, ko_parent='PAT', style='C0', sy=1e-3)
    plot_linear_regression(mode_model_f, df_f, ko_parent='MAT', style='C1', sy=1e-3)
    plt.tick_params(labelsize=14)
    plt.ylabel('Area$_{\mathrm{mode}}$ ($10^3\ \mu m^2$)', fontsize=14)
    plt.title('Female', fontsize=14)
    plt.legend()

    plt.subplot(423)
    df = df_f[df_f['ko_parent'] == 'PAT']
    plt.scatter(df['BW'], df['area_q_05'] * 1e-3, c='C0', label='PAT')
    df = df_f[df_f['ko_parent'] == 'MAT']
    plt.scatter(df['BW'], df['area_q_05'] * 1e-3, c='C1', label='MAT')
    plot_linear_regression(q25_model_f, df_f, ko_parent='PAT', style='C0', sy=1e-3)
    plot_linear_regression(q25_model_f, df_f, ko_parent='MAT', style='C1', sy=1e-3)
    plt.tick_params(labelsize=14)
    plt.ylabel('Area$_{\mathrm{Q1}}$ ($10^3\ \mu m^2$)', fontsize=14)

    plt.subplot(425)
    df = df_f[df_f['ko_parent'] == 'PAT']
    plt.scatter(df['BW'], df['area_q_10'] * 1e-3, c='C0', label='PAT')
    df = df_f[df_f['ko_parent'] == 'MAT']
    plt.scatter(df['BW'], df['area_q_10'] * 1e-3, c='C1', label='MAT')
    plot_linear_regression(q50_model_f, df_f, ko_parent='PAT', style='C0', sy=1e-3)
    plot_linear_regression(q50_model_f, df_f, ko_parent='MAT', style='C1', sy=1e-3)
    plt.tick_params(labelsize=14)
    plt.ylabel('Area$_{\mathrm{Q2}}$ ($10^3\ \mu m^2$)', fontsize=14)

    plt.subplot(427)
    df = df_f[df_f['ko_parent'] == 'PAT']
    plt.scatter(df['BW'], df['area_q_15'] * 1e-3, c='C0', label='PAT')
    df = df_f[df_f['ko_parent'] == 'MAT']
    plt.scatter(df['BW'], df['area_q_15'] * 1e-3, c='C1', label='MAT')
    plot_linear_regression(q75_model_f, df_f, ko_parent='PAT', style='C0', sy=1e-3)
    plot_linear_regression(q75_model_f, df_f, ko_parent='MAT', style='C1', sy=1e-3)
    plt.tick_params(labelsize=14)
    plt.xlabel('Body weight (g)', fontsize=14)
    plt.ylabel('Area$_{\mathrm{Q3}}$ ($10^3\ \mu m^2$)', fontsize=14)

    plt.subplot(422)
    df = df_m[df_m['ko_parent'] == 'PAT']
    plt.scatter(df['BW'], df['area_smoothed_mode'] * 1e-3, c='C0', label='PAT')
    df = df_m[df_m['ko_parent'] == 'MAT']
    plt.scatter(df['BW'], df['area_smoothed_mode'] * 1e-3, c='C1', label='MAT')
    plot_linear_regression(mode_model_m, df_m, ko_parent='PAT', style='C0', sy=1e-3)
    plot_linear_regression(mode_model_m, df_m, ko_parent='MAT', style='C1', sy=1e-3)
    plt.tick_params(labelsize=14)
    plt.title('Male', fontsize=14)

    plt.subplot(424)
    df = df_m[df_m['ko_parent'] == 'PAT']
    plt.scatter(df['BW'], df['area_q_05'] * 1e-3, c='C0', label='PAT')
    df = df_m[df_m['ko_parent'] == 'MAT']
    plt.scatter(df['BW'], df['area_q_05'] * 1e-3, c='C1', label='MAT')
    plot_linear_regression(q25_model_m, df_m, ko_parent='PAT', style='C0', sy=1e-3)
    plot_linear_regression(q25_model_m, df_m, ko_parent='MAT', style='C1', sy=1e-3)
    plt.tick_params(labelsize=14)

    plt.subplot(426)
    df = df_m[df_m['ko_parent'] == 'PAT']
    plt.scatter(df['BW'], df['area_q_10'] * 1e-3, c='C0', label='PAT')
    df = df_m[df_m['ko_parent'] == 'MAT']
    plt.scatter(df['BW'], df['area_q_10'] * 1e-3, c='C1', label='MAT')
    plot_linear_regression(q50_model_m, df_m, ko_parent='PAT', style='C0', sy=1e-3)
    plot_linear_regression(q50_model_m, df_m, ko_parent='MAT', style='C1', sy=1e-3)
    plt.tick_params(labelsize=14)

    plt.subplot(428)
    df = df_m[df_m['ko_parent'] == 'PAT']
    plt.scatter(df['BW'], df['area_q_15'] * 1e-3, c='C0', label='PAT')
    df = df_m[df_m['ko_parent'] == 'MAT']
    plt.scatter(df['BW'], df['area_q_15'] * 1e-3, c='C1', label='MAT')
    plot_linear_regression(q75_model_m, df_m, ko_parent='PAT', style='C0', sy=1e-3)
    plot_linear_regression(q75_model_m, df_m, ko_parent='MAT', style='C1', sy=1e-3)
    plt.tick_params(labelsize=14)
    plt.xlabel('Body weight (g)', fontsize=14)

    plt.tight_layout()

    if depot == 'gwat':
        plt.suptitle('Gonadal', fontsize=14)
    elif depot == 'sqwat':
        plt.suptitle('Subcutaneous', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_quartile_linear_model_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_quartile_linear_model_' + depot + '.svg'))

# extract coefficients, errors and p-values from quartile models
df_coeff_f, df_ci_lo_f, df_ci_hi_f, df_pval_f = models_coeff_ci_pval([q25_model_f, q50_model_f, q75_model_f])
df_coeff_m, df_ci_lo_m, df_ci_hi_m, df_pval_m = models_coeff_ci_pval([q25_model_m, q50_model_m, q75_model_m])

if SAVEFIG:
    plt.clf()
    plt.gcf().set_size_inches([6.4, 7.5])
    q = [25, 50, 75]

    plt.subplot(321)
    plot_model_coeff(q, df_coeff_f['C(ko_parent)[T.MAT]'] / BW__factor * 1e-3,
                     df_ci_lo_f['C(ko_parent)[T.MAT]'] / BW__factor * 1e-3,
                     df_ci_hi_f['C(ko_parent)[T.MAT]'] / BW__factor * 1e-3,
                     df_pval_f['C(ko_parent)[T.MAT]'])
    plt.xticks(q)
    plt.ylabel(r'$\beta_{KO\ parent}\ (10^3\ \mu m^2)$', fontsize=14)
    plt.title('Female', fontsize=14)
    plt.tick_params(labelsize=14)

    plt.subplot(322)
    plot_model_coeff(q, df_coeff_m['C(ko_parent)[T.MAT]'] / BW__factor * 1e-3,
                     df_ci_lo_m['C(ko_parent)[T.MAT]'] / BW__factor * 1e-3,
                     df_ci_hi_m['C(ko_parent)[T.MAT]'] / BW__factor * 1e-3,
                     df_pval_m['C(ko_parent)[T.MAT]'])
    plt.xticks(q)
    plt.title('Male', fontsize=14)
    plt.tick_params(labelsize=14)

    plt.subplot(323)
    plot_model_coeff(q, df_coeff_f['BW__'] / BW__factor * 1e-3,
                     df_ci_lo_f['BW__'] / BW__factor * 1e-3,
                     df_ci_hi_f['BW__'] / BW__factor * 1e-3,
                     df_pval_f['BW__'])
    plt.xticks(q)
    plt.ylabel(r'$\beta_{BW}\ (10^3\ \mu m^2/g)$', fontsize=14)
    plt.tick_params(labelsize=14)

    plt.subplot(324)
    plot_model_coeff(q, df_coeff_m['BW__'] / BW__factor * 1e-3,
                     df_ci_lo_m['BW__'] / BW__factor * 1e-3,
                     df_ci_hi_m['BW__'] / BW__factor * 1e-3,
                     df_pval_m['BW__'])
    plt.xticks(q)
    plt.tick_params(labelsize=14)

    plt.subplot(325)
    plot_model_coeff(q, df_coeff_f['BW__:C(ko_parent)[T.MAT]'] / BW__factor * 1e-3,
                     df_ci_lo_f['BW__:C(ko_parent)[T.MAT]'] / BW__factor * 1e-3,
                     df_ci_hi_f['BW__:C(ko_parent)[T.MAT]'] / BW__factor * 1e-3,
                     df_pval_f['BW__:C(ko_parent)[T.MAT]'])
    plt.xticks(q)
    plt.ylabel(r'$\beta_{BW\cdot KO\ parent}\ (10^3\ \mu m^2/g)$', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.xlabel('Quantile (%)', fontsize=14)

    plt.subplot(326)
    plot_model_coeff(q, df_coeff_m['BW__:C(ko_parent)[T.MAT]'] / BW__factor * 1e-3,
                     df_ci_lo_m['BW__:C(ko_parent)[T.MAT]'] / BW__factor * 1e-3,
                     df_ci_hi_m['BW__:C(ko_parent)[T.MAT]'] / BW__factor * 1e-3,
                     df_pval_m['BW__:C(ko_parent)[T.MAT]'])
    plt.xticks(q)
    plt.tick_params(labelsize=14)
    plt.xlabel('Quantile (%)', fontsize=14)

    plt.tight_layout()

    if depot == 'gwat':
        plt.suptitle('Gonadal', fontsize=14)
    elif depot == 'sqwat':
        plt.suptitle('Subcutaneous', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_quartile_linear_model_coeffs_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_quartile_linear_model_coeffs_' + depot + '.svg'))

## linear regression analysis of area ~ BW * ko_parent * genotype
## (all deciles)
########################################################################################################################

# 0.1 , 0.2, ..., 0.9
quantiles = np.linspace(0, 1, 21)
deciles_idx = list(range(2, 20, 2))
deciles = quantiles[deciles_idx]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

depot = 'gwat'
# depot = 'sqwat'

# for convenience create two dataframes (female and male) with the data for the current depot
df = df_all[(df_all['depot'] == depot)].copy()
df = df[~np.isnan(df['BW'])]
BW__factor = df['BW'].mean()
df['BW__'] = df['BW'] / BW__factor
df_f = df[df['sex'] == 'f'].reset_index()
df_m = df[df['sex'] == 'm'].reset_index()

# fit linear models for each decile
decile_models_f = [sm.RLM.from_formula('area_q_' + '{0:02d}'.format(j) + ' ~ BW__ * C(ko_parent)', data=df_f, M=sm.robust.norms.HuberT()).fit()
                   for j in deciles_idx]
decile_models_m = [sm.RLM.from_formula('area_q_' + '{0:02d}'.format(j) + ' ~ BW__ * C(ko_parent)', data=df_m, M=sm.robust.norms.HuberT()).fit()
                   for j in deciles_idx]
# decile_models_f = [sm.formula.ols('area_q_' + '{0:02d}'.format(j) + ' ~ BW__ * C(ko_parent)', data=df_f).fit()
#                    for j in deciles_idx]
# decile_models_m = [sm.formula.ols('area_q_' + '{0:02d}'.format(j) + ' ~ BW__ * C(ko_parent)', data=df_m).fit()
#                    for j in deciles_idx]

print(decile_models_f[4].summary())
print(decile_models_m[4].summary())

# extract coefficients, errors and p-values from quartile models
df_coeff_f, df_ci_lo_f, df_ci_hi_f, df_pval_f = models_coeff_ci_pval(decile_models_f)
df_coeff_m, df_ci_lo_m, df_ci_hi_m, df_pval_m = models_coeff_ci_pval(decile_models_m)

# multitest correction using Benjamini-Hochberg
for coeff in df_pval_f.columns:
    _, df_pval_f[coeff], _, _ = multipletests(df_pval_f[coeff], method='fdr_bh', alpha=0.05, returnsorted=False)
for coeff in df_pval_f.columns:
    _, df_pval_m[coeff], _, _ = multipletests(df_pval_m[coeff], method='fdr_bh', alpha=0.05, returnsorted=False)

if SAVEFIG:
    plt.clf()
    plt.gcf().set_size_inches([6.4, 9.99])
    q = np.array(deciles) * 100

    plt.subplot(3,2,1)
    plot_model_coeff(q, df_coeff_f['C(ko_parent)[T.MAT]'] * 1e-3,
                     df_ci_lo_f['C(ko_parent)[T.MAT]'] * 1e-3,
                     df_ci_hi_f['C(ko_parent)[T.MAT]'] * 1e-3,
                     df_pval_f['C(ko_parent)[T.MAT]'])
    plt.title('Female', fontsize=14)
    plt.ylabel(r'$\beta_{KO\ parent}\ (10^3\ \mu m^2)$', fontsize=14)
    plt.tick_params(labelsize=14)

    plt.subplot(3,2,2)
    plot_model_coeff(q, df_coeff_m['C(ko_parent)[T.MAT]'] * 1e-3,
                     df_ci_lo_m['C(ko_parent)[T.MAT]'] * 1e-3,
                     df_ci_hi_m['C(ko_parent)[T.MAT]'] * 1e-3,
                     df_pval_m['C(ko_parent)[T.MAT]'])
    plt.title('Male', fontsize=14)
    plt.tick_params(labelsize=14)

    plt.subplot(3,2,3)
    plot_model_coeff(q, df_coeff_f['BW__'] / BW__factor * 1e-3,
                     df_ci_lo_f['BW__'] / BW__factor * 1e-3,
                     df_ci_hi_f['BW__'] / BW__factor * 1e-3,
                     df_pval_f['BW__'])
    plt.ylabel(r'$\beta_{BW}\ (10^3\ \mu m^2/g)$', fontsize=14)
    plt.tick_params(labelsize=14)

    plt.subplot(3,2,4)
    plot_model_coeff(q, df_coeff_m['BW__'] / BW__factor * 1e-3,
                     df_ci_lo_m['BW__'] / BW__factor * 1e-3,
                     df_ci_hi_m['BW__'] / BW__factor * 1e-3,
                     df_pval_m['BW__'])
    plt.tick_params(labelsize=14)

    plt.subplot(3,2,5)
    plot_model_coeff(q, df_coeff_f['BW__:C(ko_parent)[T.MAT]'] / BW__factor * 1e-3,
                     df_ci_lo_f['BW__:C(ko_parent)[T.MAT]'] / BW__factor * 1e-3,
                     df_ci_hi_f['BW__:C(ko_parent)[T.MAT]'] / BW__factor * 1e-3,
                     df_pval_f['BW__:C(ko_parent)[T.MAT]'])
    plt.ylabel(r'$\beta_{BW \cdot KO\ parent}\ (10^3\ \mu m^2/g)$', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.xlabel('Quantile (%)', fontsize=14)

    plt.subplot(3,2,6)
    plot_model_coeff(q, df_coeff_m['BW__:C(ko_parent)[T.MAT]'] / BW__factor * 1e-3,
                     df_ci_lo_m['BW__:C(ko_parent)[T.MAT]'] / BW__factor * 1e-3,
                     df_ci_hi_m['BW__:C(ko_parent)[T.MAT]'] / BW__factor * 1e-3,
                     df_pval_m['BW__:C(ko_parent)[T.MAT]'])
    plt.tick_params(labelsize=14)
    plt.xlabel('Quantile (%)', fontsize=14)

    plt.tight_layout()

    if depot == 'gwat':
        plt.suptitle('Gonadal', fontsize=14)
    elif depot == 'sqwat':
        plt.suptitle('Subcutaneous', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_decile_linear_model_coeffs_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_decile_linear_model_coeffs_' + depot + '.svg'))

## linear regression analysis of quantile_area ~ BW * ko_parent * genotype
## (only mode, 25%-, 50%- and 75%-quantiles for illustration purposes and debugging)
##
## NOT USED: commented out
########################################################################################################################
#
# # 0.05, 0.1 , 0.15, 0.2, ..., 0.9 , 0.95
# quantiles = np.linspace(0, 1, 21)
# assert(quantiles[5]  == 0.25)  # check: we are selecting the 25% quantile
# assert(quantiles[10] == 0.5)  # check: we are selecting the median
# assert(quantiles[15] == 0.75)  # check: we are selecting the 75% quantile
#
# depot = 'gwat'
# # depot = 'sqwat'
#
# # fit linear models
# df = df_all[(df_all['depot'] == depot)]
# BW__factor = df['BW'].mean()
# df['BW__'] = df['BW'] / BW__factor
# df_f = df[df['sex'] == 'f'].reset_index()
# df_m = df[df['sex'] == 'm'].reset_index()
#
# mode_model_f = sm.formula.ols('area_smoothed_mode ~ BW__ * C(ko_parent)', data=df_f).fit()
# q25_model_f = sm.formula.ols('area_q_05 ~ BW__ * C(ko_parent)', data=df_f).fit()
# q50_model_f = sm.formula.ols('area_q_10 ~ BW__ * C(ko_parent)', data=df_f).fit()
# q75_model_f = sm.formula.ols('area_q_15 ~ BW__ * C(ko_parent)', data=df_f).fit()
# print(mode_model_f.summary())
# print(q25_model_f.summary())
# print(q50_model_f.summary())
# print(q75_model_f.summary())
#
# mode_model_m = sm.formula.ols('area_smoothed_mode ~ BW__ * C(ko_parent)', data=df_m).fit()
# q25_model_m = sm.formula.ols('area_q_05 ~ BW__ * C(ko_parent)', data=df_m).fit()
# q50_model_m = sm.formula.ols('area_q_10 ~ BW__ * C(ko_parent)', data=df_m).fit()
# q75_model_m = sm.formula.ols('area_q_15 ~ BW__ * C(ko_parent)', data=df_m).fit()
# print(mode_model_m.summary())
# print(q25_model_m.summary())
# print(q50_model_m.summary())
# print(q75_model_m.summary())
#
# # plot
# if DEBUG:
#     plt.clf()
#     plt.gcf().set_size_inches([6.4, 9.99])
#
#     plt.subplot(421)
#     df = df_f[df_f['ko_parent'] == 'PAT']
#     plt.scatter(df['BW'], df['area_smoothed_mode'] * 1e-3, c='C0', label='f PAT')
#     df = df_f[df_f['ko_parent'] == 'MAT']
#     plt.scatter(df['BW'], df['area_smoothed_mode'] * 1e-3, c='C1', label='f MAT')
#     plot_linear_regression(mode_model_f, df_f, ko_parent='PAT', style='C0', sy=1e-3)
#     plot_linear_regression(mode_model_f, df_f, ko_parent='MAT', style='C1', sy=1e-3)
#     plt.tick_params(labelsize=14)
#     plt.ylabel('Mode ($10^3\ \mu m^2$)', fontsize=14)
#     if depot == 'sqwat':
#         plt.ylim(2, 15)
#     elif depot == 'qwat':
#         pass
#     plt.legend()
#
#     plt.subplot(423)
#     df = df_f[df_f['ko_parent'] == 'PAT']
#     plt.scatter(df['BW'], df['area_q_05'] * 1e-3, c='C0', label='f PAT')
#     df = df_f[df_f['ko_parent'] == 'MAT']
#     plt.scatter(df['BW'], df['area_q_05'] * 1e-3, c='C1', label='f MAT')
#     plot_linear_regression(q25_model_f, df_f, ko_parent='PAT', style='C0', sy=1e-3)
#     plot_linear_regression(q25_model_f, df_f, ko_parent='MAT', style='C1', sy=1e-3)
#     plt.tick_params(labelsize=14)
#     if depot == 'sqwat':
#         plt.ylim(2, 15)
#     elif depot == 'qwat':
#         plt.ylim(20, 43)
#     plt.ylabel('25%-quant. ($10^3\ \mu m^2$)', fontsize=14)
#     plt.legend()
#
#     plt.subplot(425)
#     df = df_f[df_f['ko_parent'] == 'PAT']
#     plt.scatter(df['BW'], df['area_q_10'] * 1e-3, c='C0', label='f PAT')
#     df = df_f[df_f['ko_parent'] == 'MAT']
#     plt.scatter(df['BW'], df['area_q_10'] * 1e-3, c='C1', label='f MAT')
#     plot_linear_regression(q50_model_f, df_f, ko_parent='PAT', style='C0', sy=1e-3)
#     plot_linear_regression(q50_model_f, df_f, ko_parent='MAT', style='C1', sy=1e-3)
#     plt.tick_params(labelsize=14)
#     plt.ylabel('Median ($10^3\ \mu m^2$)', fontsize=14)
#     plt.legend()
#
#     plt.subplot(427)
#     df = df_f[df_f['ko_parent'] == 'PAT']
#     plt.scatter(df['BW'], df['area_q_15'] * 1e-3, c='C0', label='f PAT')
#     df = df_f[df_f['ko_parent'] == 'MAT']
#     plt.scatter(df['BW'], df['area_q_15'] * 1e-3, c='C1', label='f MAT')
#     plot_linear_regression(q75_model_f, df_f, ko_parent='PAT', style='C0', sy=1e-3)
#     plot_linear_regression(q75_model_f, df_f, ko_parent='MAT', style='C1', sy=1e-3)
#     plt.tick_params(labelsize=14)
#     plt.xlabel('Body weight (g)', fontsize=14)
#     plt.ylabel('75%-quant. ($10^3\ \mu m^2$)', fontsize=14)
#     plt.legend()
#
#     plt.subplot(422)
#     df = df_m[df_m['ko_parent'] == 'PAT']
#     plt.scatter(df['BW'], df['area_smoothed_mode'] * 1e-3, c='C0', label='m PAT')
#     df = df_m[df_m['ko_parent'] == 'MAT']
#     plt.scatter(df['BW'], df['area_smoothed_mode'] * 1e-3, c='C1', label='m MAT')
#     plot_linear_regression(mode_model_m, df_m, ko_parent='PAT', style='C0', sy=1e-3)
#     plot_linear_regression(mode_model_m, df_m, ko_parent='MAT', style='C1', sy=1e-3)
#     plt.tick_params(labelsize=14)
#     plt.ylabel('Mode ($10^3\ \mu m^2$)', fontsize=14)
#     plt.legend()
#
#     plt.subplot(424)
#     df = df_m[df_m['ko_parent'] == 'PAT']
#     plt.scatter(df['BW'], df['area_q_05'] * 1e-3, c='C0', label='m PAT')
#     df = df_m[df_m['ko_parent'] == 'MAT']
#     plt.scatter(df['BW'], df['area_q_05'] * 1e-3, c='C1', label='m MAT')
#     plot_linear_regression(q25_model_m, df_m, ko_parent='PAT', style='C0', sy=1e-3)
#     plot_linear_regression(q25_model_m, df_m, ko_parent='MAT', style='C1', sy=1e-3)
#     plt.tick_params(labelsize=14)
#     if depot == 'sqwat':
#         plt.ylim(4, 15)
#     elif depot == 'qwat':
#         plt.ylim(5, 22)
#     plt.legend()
#
#     plt.subplot(426)
#     df = df_m[df_m['ko_parent'] == 'PAT']
#     plt.scatter(df['BW'], df['area_q_10'] * 1e-3, c='C0', label='m PAT')
#     df = df_m[df_m['ko_parent'] == 'MAT']
#     plt.scatter(df['BW'], df['area_q_10'] * 1e-3, c='C1', label='m MAT')
#     plot_linear_regression(q50_model_m, df_m, ko_parent='PAT', style='C0', sy=1e-3)
#     plot_linear_regression(q50_model_m, df_m, ko_parent='MAT', style='C1', sy=1e-3)
#     plt.tick_params(labelsize=14)
#     plt.ylabel('Median ($10^3\ \mu m^2$)', fontsize=14)
#     if depot == 'sqwat':
#         plt.ylim(6, 29)
#     elif depot == 'qwat':
#         plt.ylim(20, 43)
#     plt.legend()
#
#     plt.subplot(428)
#     df = df_m[df_m['ko_parent'] == 'PAT']
#     plt.scatter(df['BW'], df['area_q_15'] * 1e-3, c='C0', label='m PAT')
#     df = df_m[df_m['ko_parent'] == 'MAT']
#     plt.scatter(df['BW'], df['area_q_15'] * 1e-3, c='C1', label='m MAT')
#     plot_linear_regression(q75_model_m, df_m, ko_parent='PAT', style='C0', sy=1e-3)
#     plot_linear_regression(q75_model_m, df_m, ko_parent='MAT', style='C1', sy=1e-3)
#     plt.tick_params(labelsize=14)
#     plt.xlabel('Body weight (g)', fontsize=14)
#     plt.ylabel('75%-quant. ($10^3\ \mu m^2$)', fontsize=14)
#     plt.legend()
#
#     plt.tight_layout()

## linear regression analysis of area ~ BW * ko_parent * genotype
## (all deciles)
##
## NOT USED: commented out because it leads nowhere
########################################################################################################################
#
# # 0.1 , 0.2, ..., 0.9
# quantiles = np.linspace(0, 1, 21)
# deciles_idx = list(range(2, 20, 2))
# deciles = quantiles[deciles_idx]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#
# depot = 'gwat'
# # depot = 'sqwat'
#
# # for convenience create two dataframes (female and male) with the data for the current depot
# df = df_all[(df_all['depot'] == depot)]
# df = df[~np.isnan(df['BW'])]
# BW__factor = df['BW'].mean()
# df['BW__'] = df['BW'] / BW__factor
# df_f = df[df['sex'] == 'f'].reset_index()
# df_m = df[df['sex'] == 'm'].reset_index()
#
# # fit linear models for each decile
# decile_models_f = [sm.formula.ols('area_q_' + '{0:02d}'.format(j) + ' ~ BW__ * C(ko_parent) * C(genotype)', data=df_f).fit()
#                    for j in deciles_idx]
# decile_models_m = [sm.formula.ols('area_q_' + '{0:02d}'.format(j) + ' ~ BW__ * C(ko_parent) * C(genotype)', data=df_m).fit()
#                    for j in deciles_idx]
#
# print(decile_models_f[4].summary())
# print(decile_models_m[4].summary())
#
# # extract coefficients, errors and p-values from quartile models
# df_coeff_f, df_ci_lo_f, df_ci_hi_f, df_pval_f = models_coeff_ci_pval(decile_models_f)
# df_coeff_m, df_ci_lo_m, df_ci_hi_m, df_pval_m = models_coeff_ci_pval(decile_models_m)
#
# # multitest correction using Benjamini-Hochberg
# for coeff in df_pval_f.columns:
#     _, df_pval_f[coeff], _, _ = multipletests(df_pval_f[coeff], method='fdr_bh', alpha=0.05, returnsorted=False)
# for coeff in df_pval_f.columns:
#     _, df_pval_m[coeff], _, _ = multipletests(df_pval_m[coeff], method='fdr_bh', alpha=0.05, returnsorted=False)
#
# if DEBUG:
#     plt.clf()
#     plt.gcf().set_size_inches([6.4, 9.99])
#     q = np.array(deciles) * 100
#
#     plt.subplot(7,2,1)
#     plot_model_coeff(q, df_coeff_f['C(ko_parent)[T.MAT]'],
#                      df_stderr_f['C(ko_parent)[T.MAT]'],
#                      df_pval_f['C(ko_parent)[T.MAT]'])
#     plt.title('Female', fontsize=14)
#     plt.ylabel(r'$\beta_{KO\ parent}$', fontsize=14)
#     plt.tick_params(labelsize=14)
#
#     plt.subplot(722)
#     plot_model_coeff(q, df_coeff_m['C(ko_parent)[T.MAT]'],
#                      df_stderr_m['C(ko_parent)[T.MAT]'],
#                      df_pval_m['C(ko_parent)[T.MAT]'])
#     plt.title('Male', fontsize=14)
#     plt.tick_params(labelsize=14)
#
#     plt.subplot(723)
#     plot_model_coeff(q, df_coeff_f['C(genotype)[T.KLF14-KO:Het]'],
#                      df_stderr_f['C(genotype)[T.KLF14-KO:Het]'],
#                      df_pval_f['C(genotype)[T.KLF14-KO:Het]'])
#     plt.ylabel(r'$\beta_{genotype}$', fontsize=14)
#     plt.tick_params(labelsize=14)
#
#     plt.subplot(724)
#     plot_model_coeff(q, df_coeff_m['C(genotype)[T.KLF14-KO:Het]'],
#                      df_stderr_m['C(genotype)[T.KLF14-KO:Het]'],
#                      df_pval_m['C(genotype)[T.KLF14-KO:Het]'])
#     plt.tick_params(labelsize=14)
#
#     plt.subplot(725)
#     plot_model_coeff(q, df_coeff_f['C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]'],
#                      df_stderr_f['C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]'],
#                      df_pval_f['C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]'])
#     plt.ylabel(r'$\beta_{KO\ parent\cdot genotype}$', fontsize=14)
#     plt.tick_params(labelsize=14)
#
#     plt.subplot(726)
#     plot_model_coeff(q, df_coeff_m['C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]'],
#                      df_stderr_m['C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]'],
#                      df_pval_m['C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]'])
#     plt.tick_params(labelsize=14)
#
#     plt.subplot(727)
#     plot_model_coeff(q, df_coeff_f['BW__'] / BW__factor, df_stderr_f['BW__'] / BW__factor,
#                      df_pval_f['BW__'])
#     plt.ylabel(r'$\beta_{BW}$', fontsize=14)
#     plt.tick_params(labelsize=14)
#
#     plt.subplot(728)
#     plot_model_coeff(q, df_coeff_m['BW__'] / BW__factor, df_stderr_m['BW__'] / BW__factor,
#                      df_pval_m['BW__'])
#     plt.tick_params(labelsize=14)
#
#     plt.subplot(729)
#     plot_model_coeff(q, df_coeff_f['BW__:C(ko_parent)[T.MAT]'] / BW__factor,
#                      df_stderr_f['BW__:C(ko_parent)[T.MAT]'] / BW__factor,
#                      df_pval_f['BW__:C(ko_parent)[T.MAT]'])
#     plt.ylabel(r'$\beta_{BW \cdot KO\ parent}$', fontsize=14)
#     plt.tick_params(labelsize=14)
#
#     plt.subplot(7,2,10)
#     plot_model_coeff(q, df_coeff_m['BW__:C(ko_parent)[T.MAT]'] / BW__factor,
#                      df_stderr_m['BW__:C(ko_parent)[T.MAT]'] / BW__factor,
#                      df_pval_m['BW__:C(ko_parent)[T.MAT]'])
#     plt.tick_params(labelsize=14)
#
#     plt.subplot(7,2,11)
#     plot_model_coeff(q, df_coeff_f['BW__:C(genotype)[T.KLF14-KO:Het]'] / BW__factor,
#                      df_stderr_f['BW__:C(genotype)[T.KLF14-KO:Het]'] / BW__factor,
#                      df_pval_f['BW__:C(genotype)[T.KLF14-KO:Het]'])
#     plt.ylabel(r'$\beta_{BW \cdot genotype}$', fontsize=14)
#     plt.tick_params(labelsize=14)
#
#     plt.subplot(7,2,12)
#     plot_model_coeff(q, df_coeff_m['BW__:C(genotype)[T.KLF14-KO:Het]'] / BW__factor,
#                      df_stderr_m['BW__:C(genotype)[T.KLF14-KO:Het]'] / BW__factor,
#                      df_pval_m['BW__:C(genotype)[T.KLF14-KO:Het]'])
#     plt.tick_params(labelsize=14)
#
#     plt.subplot(7,2,13)
#     plot_model_coeff(q, df_coeff_f['BW__:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]'] / BW__factor,
#                      df_stderr_f['BW__:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]'] / BW__factor,
#                      df_pval_f['BW__:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]'])
#     plt.ylabel(r'$\beta_{BW \cdot genotype \cdot KO\ parent}$', fontsize=14)
#     plt.tick_params(labelsize=14)
#     plt.xlabel('Quantile (%)', fontsize=14)
#
#     plt.subplot(7,2,14)
#     plot_model_coeff(q, df_coeff_m['BW__:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]'] / BW__factor,
#                      df_stderr_m['BW__:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]'] / BW__factor,
#                      df_pval_m['BW__:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]'])
#     plt.tick_params(labelsize=14)
#     plt.xlabel('Quantile (%)', fontsize=14)
#
#     plt.tight_layout()
