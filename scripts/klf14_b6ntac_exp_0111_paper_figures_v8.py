"""
Generate figures for the DeepCytometer paper for v8 of the pipeline.

We repeat the phenotyping from klf14_b6ntac_exp_0110_paper_figures_v8.py, but change the stratification of the data so
that we have Control (PATs + WT MATs) vs. Het MATs.

This script partly deprecates klf14_b6ntac_exp_0099_paper_figures_v7.py:
* Figures have been updated to have v8 of the pipeline in the paper.

This script partly deprecates klf14_b6ntac_exp_0110_paper_figures_v8.py:
* We repeat the phenotyping, but change the stratification of the data so that we have Control (PATs + WT MATs) vs.
  Het MATs.

"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0111_paper_figures'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])

DEBUG = False
SAVE_FIGS = False

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
## Common code to the rest of this script:
## Import packages and auxiliary functions
## USED IN PAPER
########################################################################################################################

import pickle
from toolz import interleave
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import skimage
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import openslide
from PIL import Image, ImageDraw
import cytometer.data
import cytometer.stats
import shapely

# directories
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
hand_traced_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training_v2')
annotations_dir = os.path.join(home, 'Data/cytometer_data/aida_data_Klf14_v8/annotations')
histo_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
dataframe_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
metainfo_dir = os.path.join(home, 'Data/cytometer_data/klf14')
area2quantile_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v8')
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
metainfo['functional_ko'] = 'Control'
metainfo.loc[(metainfo['ko_parent'] == 'MAT') & (metainfo['genotype'] == 'KLF14-KO:Het'), 'functional_ko'] = 'FKO'
metainfo.loc[(metainfo['ko_parent'] == 'MAT') & (metainfo['genotype'] == 'KLF14-KO:WT'), 'functional_ko'] = 'MAT_WT'
metainfo['functional_ko'] = metainfo['functional_ko'].astype(
    pd.api.types.CategoricalDtype(categories=['Control', 'MAT_WT', 'FKO'], ordered=True))

# remove BW=NaNs
metainfo = metainfo[~np.isnan(metainfo['BW'])]
metainfo = metainfo.reset_index()

# load dataframe with cell population quantiles and histograms
dataframe_areas_filename = os.path.join(dataframe_dir, 'klf14_b6ntac_exp_0110_dataframe_areas_' + method + '.pkl')
df_all = pd.read_pickle(dataframe_areas_filename)
df_all = df_all.reset_index()

df_all['sex'] = df_all['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
df_all['ko_parent'] = df_all['ko_parent'].astype(
    pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
df_all['genotype'] = df_all['genotype'].astype(
    pd.api.types.CategoricalDtype(categories=['KLF14-KO:WT', 'KLF14-KO:Het'], ordered=True))
df_all['functional_ko'] = 'Control'
df_all.loc[(df_all['ko_parent'] == 'MAT') & (df_all['genotype'] == 'KLF14-KO:Het'), 'functional_ko'] = 'FKO'
df_all.loc[(df_all['ko_parent'] == 'MAT') & (df_all['genotype'] == 'KLF14-KO:WT'), 'functional_ko'] = 'MAT_WT'
df_all['functional_ko'] = df_all['functional_ko'].astype(
    pd.api.types.CategoricalDtype(categories=['Control', 'MAT_WT', 'FKO'], ordered=True))

# load extra info needed for the histograms
dataframe_areas_extra_filename = os.path.join(dataframe_dir, 'klf14_b6ntac_exp_0110_dataframe_areas_extra.npz')
with np.load(dataframe_areas_extra_filename) as aux:
    quantiles = aux['quantiles']
    area_bin_edges = aux['area_bin_edges']
    area_bin_centers = aux['area_bin_centers']

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



print('PAT WT: ' + str(np.count_nonzero((metainfo['genotype'] == 'KLF14-KO:WT') & (metainfo['ko_parent'] == 'PAT'))))
print('PAT Het: ' + str(np.count_nonzero((metainfo['genotype'] == 'KLF14-KO:Het') & (metainfo['ko_parent'] == 'PAT'))))
print('MAT WT: ' + str(np.count_nonzero((metainfo['genotype'] == 'KLF14-KO:WT') & (metainfo['ko_parent'] == 'MAT'))))
print('MAT Het: ' + str(np.count_nonzero((metainfo['genotype'] == 'KLF14-KO:Het') & (metainfo['ko_parent'] == 'MAT'))))


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

df_all = df_all[~np.isnan(df_all['BW'])]

bw_model = sm.RLM.from_formula('BW ~ C(sex)', data=metainfo, subset=metainfo['ko_parent']=='PAT', M=sm.robust.norms.HuberT()).fit()
print(bw_model.summary())
print(bw_model.pvalues)
print('Males are ' + str(bw_model.params['C(sex)[T.m]'] / bw_model.params['Intercept'] * 100)
      + ' % larger than females')


# ## effect of cull age on body weight
# ########################################################################################################################
#
# # check whether mice killed later are bigger
# bw_model_f = sm.OLS.from_formula('BW ~ cull_age', data=metainfo_f).fit()
# print(bw_model_f.summary())
# bw_model_m = sm.OLS.from_formula('BW ~ cull_age', data=metainfo_m).fit()
# print(bw_model_m.summary())
#
# # check whether mice in groups that are going to be compared (PAT vs MAT and WT vs Het) don't have a cull age effect
# cull_model_f = sm.OLS.from_formula('cull_age ~ genotype', data=metainfo_f).fit()
# print(cull_model_f.summary())
# cull_model_m = sm.OLS.from_formula('cull_age ~ genotype', data=metainfo_m).fit()
# print(cull_model_m.summary())
#
# cull_model_f = sm.OLS.from_formula('cull_age ~ ko_parent', data=metainfo_f).fit()
# print(cull_model_f.summary())
# cull_model_m = sm.OLS.from_formula('cull_age ~ ko_parent', data=metainfo_m).fit()
# print(cull_model_m.summary())
#
# # does cull_age make a difference in the BW ~ genotype model?
# bw_null_model_f = sm.OLS.from_formula('BW ~ C(genotype)', data=metainfo_f).fit()
# bw_null_model_m = sm.OLS.from_formula('BW ~ C(genotype)', data=metainfo_m).fit()
# bw_model_f = sm.OLS.from_formula('BW ~ C(genotype) * cull_age__', data=metainfo_f).fit()
# bw_model_m = sm.OLS.from_formula('BW ~ C(genotype) * cull_age__', data=metainfo_m).fit()
#
# print('Cull age effect in BW ~ Genotype')
# print('Female')
# null_model = bw_null_model_f
# alt_model = bw_model_f
# lr, pval = cytometer.stats.lrtest(null_model.llf, alt_model.llf)
# pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
# print('p-val: ' + pval_text)
# print('AIC_null=' + '{0:.2f}'.format(null_model.aic) + ', AIC_alt=' + '{0:.2f}'.format(alt_model.aic)
#       + ', ΔAIC=' + '{0:.2f}'.format(alt_model.aic - null_model.aic))
#
# print('Male')
# null_model = bw_null_model_m
# alt_model = bw_model_m
# lr, pval = cytometer.stats.lrtest(null_model.llf, alt_model.llf)
# pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.3g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
# print('p-val: ' + pval_text)
# print('AIC_null=' + '{0:.2f}'.format(null_model.aic) + ', AIC_alt=' + '{0:.2f}'.format(alt_model.aic)
#       + ', ΔAIC=' + '{0:.2f}'.format(alt_model.aic - null_model.aic))
#
# ## does cull_age make a difference in the BW ~ parent model?
# bw_null_model_f = sm.OLS.from_formula('BW ~ C(ko_parent)', data=metainfo_f).fit()
# bw_null_model_m = sm.OLS.from_formula('BW ~ C(ko_parent)', data=metainfo_m).fit()
# bw_model_f = sm.OLS.from_formula('BW ~ C(ko_parent) * cull_age__', data=metainfo_f).fit()
# bw_model_m = sm.OLS.from_formula('BW ~ C(ko_parent) * cull_age__', data=metainfo_m).fit()
#
# if DEBUG:
#     print(bw_null_model_f.summary())
#     print(bw_null_model_m.summary())
#     print(bw_model_f.summary())
#     print(bw_model_m.summary())
#
# print('Cull age effect in BW ~ Parent')
# print('Female')
# null_model = bw_null_model_f
# alt_model = bw_model_f
# lr, pval = cytometer.stats.lrtest(null_model.llf, alt_model.llf)
# pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
# print('p-val: ' + pval_text)
# print('AIC_null=' + '{0:.2f}'.format(null_model.aic) + ', AIC_alt=' + '{0:.2f}'.format(alt_model.aic)
#       + ', ΔAIC=' + '{0:.2f}'.format(alt_model.aic - null_model.aic))
#
# print('Male')
# null_model = bw_null_model_m
# alt_model = bw_model_m
# lr, pval = cytometer.stats.lrtest(null_model.llf, alt_model.llf)
# pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.3g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
# print('p-val: ' + pval_text)
# print('AIC_null=' + '{0:.2f}'.format(null_model.aic) + ', AIC_alt=' + '{0:.2f}'.format(alt_model.aic)
#       + ', ΔAIC=' + '{0:.2f}'.format(alt_model.aic - null_model.aic))
#
# if DEBUG:
#     print(bw_null_model_f.summary())
#     print(bw_model_f.summary())
#
# if SAVE_FIGS:
#     # plot body weight vs. age of culling
#     plt.clf()
#     # 'BW ~ C(ko_parent) * cull_age__'
#     cytometer.stats.plot_linear_regression(bw_model_f, metainfo_f,
#                                            ind_var='cull_age__', other_vars={'ko_parent': 'PAT'}, dep_var='BW',
#                                            sx=np.std(metainfo['cull_age']), tx=np.mean(metainfo['cull_age']),
#                                            c='C2', marker='o', line_label='f PAT')
#     cytometer.stats.plot_linear_regression(bw_model_f, metainfo_f,
#                                            ind_var='cull_age__', other_vars={'ko_parent': 'MAT'}, dep_var='BW',
#                                            sx=np.std(metainfo['cull_age']), tx=np.mean(metainfo['cull_age']),
#                                            c='C3', marker='o', line_label='f MAT')
#     cytometer.stats.plot_linear_regression(bw_model_m, metainfo_m,
#                                            ind_var='cull_age__', other_vars={'ko_parent': 'PAT'}, dep_var='BW',
#                                            sx=np.std(metainfo['cull_age']), tx=np.mean(metainfo['cull_age']),
#                                            c='C4', marker='o', line_label='m PAT')
#     cytometer.stats.plot_linear_regression(bw_model_m, metainfo_m,
#                                            ind_var='cull_age__', other_vars={'ko_parent': 'MAT'}, dep_var='BW',
#                                            sx=np.std(metainfo['cull_age']), tx=np.mean(metainfo['cull_age']),
#                                            c='C5', marker='o', line_label='m MAT')
#
#     plt.xlabel('Cull age (days)', fontsize=14)
#     plt.ylabel('Body weight (g)', fontsize=14)
#     plt.tick_params(labelsize=14)
#     plt.legend(fontsize=12)
#     plt.tight_layout()
#
#     plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_bw_vs_cull_age.png'))
#     plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_bw_vs_cull_age.svg'))

## effect of FKO on body weight
########################################################################################################################

# BW ~ functional_ko for female/male
bw_model_f = sm.OLS.from_formula('BW ~ C(functional_ko)', data=metainfo_f).fit()
bw_model_m = sm.OLS.from_formula('BW ~ C(functional_ko)', data=metainfo_m).fit()

print(bw_model_f.summary())
print(bw_model_m.summary())

if SAVE_FIGS:
    plt.clf()
    plt.gcf().set_size_inches([5.48, 4.8 ])

    ax = sns.swarmplot(x='sex', y='BW', hue='functional_ko', data=metainfo, dodge=True, palette=['C2', 'C3', 'C4'])
    plt.xlabel('')
    plt.ylabel('Body weight (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.xticks([0, 1], labels=['Female', 'Male'])
    ax.get_legend().set_title('')
    ax.legend(['Control (PAT)', 'MAT WT', 'MAT Het (FKO)'], loc='lower right', fontsize=12)

    # plt.plot([-0.2, -0.2, 0.2, 0.2], [42, 44, 44, 42], 'k', lw=1.5)
    # pval_text = '$p$=' + '{0:.2f}'.format(bw_model_f.pvalues['C(functional_ko)[T.True]']) + \
    #             ' ' + cytometer.stats.pval_to_asterisk(bw_model_f.pvalues['C(functional_ko)[T.True]'])
    # plt.text(0, 44.5, pval_text, ha='center', va='bottom', fontsize=14)
    # plt.plot([0.8, 0.8, 1.2, 1.2], [52, 54, 54, 52], 'k', lw=1.5)
    # pval_text = '$p$=' + '{0:.2f}'.format(bw_model_m.pvalues['C(functional_ko)[T.True]']) + \
    #             ' ' + cytometer.stats.pval_to_asterisk(bw_model_m.pvalues['C(functional_ko)[T.True]'])
    # plt.text(1, 54.5, pval_text, ha='center', va='bottom', fontsize=14)
    # plt.ylim(18, 58)

    plt.plot([-0.3, -0.3, 0.0, 0.0], [42, 44, 44, 42], 'k', lw=1.5)
    pval_text = '$p$=' + '{0:.4f}'.format(bw_model_f.pvalues['C(functional_ko)[T.MAT_WT]']) + \
                ' ' + cytometer.stats.pval_to_asterisk(bw_model_f.pvalues['C(functional_ko)[T.MAT_WT]'])
    plt.text(-0.15, 44.5, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([-0.3, -0.3, 0.3, 0.3], [48, 50, 50, 48], 'k', lw=1.5)
    pval_text = '$p$=' + '{0:.3f}'.format(bw_model_f.pvalues['C(functional_ko)[T.FKO]']) + \
                ' ' + cytometer.stats.pval_to_asterisk(bw_model_f.pvalues['C(functional_ko)[T.FKO]'])
    plt.text(0.0, 50.5, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([0.7, 0.7, 1.0, 1.0], [52, 54, 54, 52], 'k', lw=1.5)
    pval_text = '$p$=' + '{0:.2f}'.format(bw_model_m.pvalues['C(functional_ko)[T.MAT_WT]']) + \
                ' ' + cytometer.stats.pval_to_asterisk(bw_model_m.pvalues['C(functional_ko)[T.MAT_WT]'])
    plt.text(1.7/2, 54.5, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([0.7, 0.7, 1.3, 1.3], [58, 60, 60, 58], 'k', lw=1.5)
    pval_text = '$p$=' + '{0:.2f}'.format(bw_model_m.pvalues['C(functional_ko)[T.FKO]']) + \
                ' ' + cytometer.stats.pval_to_asterisk(bw_model_m.pvalues['C(functional_ko)[T.FKO]'])
    plt.text(1.0, 60.5, pval_text, ha='center', va='bottom', fontsize=14)

    plt.ylim(18, 65)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_swarm_bw_fko.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_swarm_bw_fko.svg'))

#
# # BW ~ fko + cull_age
# bw_model_f = sm.OLS.from_formula('BW ~ C(functional_ko) + cull_age', data=metainfo_f).fit()
# bw_model_m = sm.OLS.from_formula('BW ~ C(functional_ko) + cull_age', data=metainfo_m).fit()
#
# print(bw_model_f.summary())
# print(bw_model_f.pvalues)
# print('FKO females are ' + str(bw_model_f.params['C(functional_ko)[T.True]'] / bw_model_f.params['Intercept'] * 100)
#       + ' % larger than Control')
#
# print(bw_model_m.summary())
#
#
# ## effect of genotype, parent and body weight on depot weight
# ########################################################################################################################
#
# # scale BW to avoid large condition numbers
# BW_mean = metainfo['BW'].mean()
# metainfo['BW__'] = metainfo['BW'] / BW_mean
#
# # update the sub-dataframes we created for convenience
# metainfo_f = metainfo[metainfo['sex'] == 'f']
# metainfo_m = metainfo[metainfo['sex'] == 'm']
#
#
# ## depot ~ BW * kfo models
#
# # models for Likelihood Ratio Test, to check whether genotype variable has an effect
# gwat_null_model_f = sm.OLS.from_formula('gWAT ~ BW__', data=metainfo_f).fit()
# gwat_null_model_m = sm.OLS.from_formula('gWAT ~ BW__', data=metainfo_m).fit()
# sqwat_null_model_f = sm.OLS.from_formula('SC ~ BW__', data=metainfo_f).fit()
# sqwat_null_model_m = sm.OLS.from_formula('SC ~ BW__', data=metainfo_m).fit()
#
# gwat_model_f = sm.OLS.from_formula('gWAT ~ BW__ * C(functional_ko)', data=metainfo_f).fit()
# gwat_model_m = sm.OLS.from_formula('gWAT ~ BW__ * C(functional_ko)', data=metainfo_m).fit()
# sqwat_model_f = sm.OLS.from_formula('SC ~ BW__ * C(functional_ko)', data=metainfo_f).fit()
# sqwat_model_m = sm.OLS.from_formula('SC ~ BW__ * C(functional_ko)', data=metainfo_m).fit()
#
# # Likelihood ratio tests of the FKO variable
# print('Likelihood Ratio Tests: FKO')
#
# print('Female')
# lr, pval = cytometer.stats.lrtest(gwat_null_model_f.llf, gwat_model_f.llf)
# pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
# print('Gonadal: ' + pval_text)
# print('Gonadal: AIC_null=' + '{0:.2f}'.format(gwat_null_model_f.aic) + ', AIC_alt=' + '{0:.2f}'.format(gwat_model_f.aic))
#
# lr, pval = cytometer.stats.lrtest(sqwat_null_model_f.llf, sqwat_model_f.llf)
# pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
# print('Subcutaneous: ' + pval_text)
# print('Subcutaneous: AIC_null=' + '{0:.2f}'.format(sqwat_null_model_f.aic) + ', AIC_alt=' + '{0:.2f}'.format(sqwat_model_f.aic))
#
# print('Male')
# lr, pval = cytometer.stats.lrtest(gwat_null_model_m.llf, gwat_model_m.llf)
# pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
# print('Gonadal: ' + pval_text)
# print('Gonadal: AIC_null=' + '{0:.2f}'.format(gwat_null_model_m.aic) + ', AIC_alt=' + '{0:.2f}'.format(gwat_model_m.aic))
#
# lr, pval = cytometer.stats.lrtest(sqwat_null_model_m.llf, sqwat_model_m.llf)
# pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
# print('Subcutaneous: ' + pval_text)
# print('Subcutaneous: AIC_null=' + '{0:.2f}'.format(sqwat_null_model_m.aic) + ', AIC_alt=' + '{0:.2f}'.format(sqwat_model_m.aic))
#
# ## fit linear models DW ~ BW__, stratified by sex and genotype
# # female Control vs. FKO
# gwat_model_f_control = sm.OLS.from_formula('gWAT ~ BW__', data=metainfo_f, subset=metainfo_f['functional_ko'] == False).fit()
# gwat_model_f_fko = sm.OLS.from_formula('gWAT ~ BW__', data=metainfo_f, subset=metainfo_f['functional_ko'] == True).fit()
# sqwat_model_f_control = sm.OLS.from_formula('SC ~ BW__', data=metainfo_f, subset=metainfo_f['functional_ko'] == False).fit()
# sqwat_model_f_fko = sm.OLS.from_formula('SC ~ BW__', data=metainfo_f, subset=metainfo_f['functional_ko'] == True).fit()
#
# # male Control vs. FKO
# gwat_model_m_control = sm.OLS.from_formula('gWAT ~ BW__', data=metainfo_m, subset=metainfo_m['functional_ko'] == False).fit()
# gwat_model_m_fko = sm.OLS.from_formula('gWAT ~ BW__', data=metainfo_m, subset=metainfo_m['functional_ko'] == True).fit()
# sqwat_model_m_control = sm.OLS.from_formula('SC ~ BW__', data=metainfo_m, subset=metainfo_m['functional_ko'] == False).fit()
# sqwat_model_m_fko = sm.OLS.from_formula('SC ~ BW__', data=metainfo_m, subset=metainfo_m['functional_ko'] == True).fit()
#
# # extract coefficients, errors and p-values from models
# model_names = ['gwat_model_f_control', 'gwat_model_f_fko',
#          'sqwat_model_f_control', 'sqwat_model_f_fko',
#          'gwat_model_m_control', 'gwat_model_m_fko',
#          'sqwat_model_m_control', 'sqwat_model_m_fko']
# df_coeff, df_ci_lo, df_ci_hi, df_pval = \
#     cytometer.stats.models_coeff_ci_pval(
#         [gwat_model_f_control, gwat_model_f_fko,
#          sqwat_model_f_control, sqwat_model_f_fko,
#          gwat_model_m_control, gwat_model_m_fko,
#          sqwat_model_m_control, sqwat_model_m_fko],
#     model_names=model_names)
#
# # multitest correction using Benjamini-Krieger-Yekutieli
# col = df_pval.columns[1]
# df_corrected_pval = df_pval.copy()
# _, df_corrected_pval[col], _, _ = multipletests(df_pval[col], method='fdr_tsbky', alpha=0.05, returnsorted=False)
# df_corrected_pval['Intercept'] = -1.0
#
# # convert p-values to asterisks
# df_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval, brackets=False), columns=df_coeff.columns,
#                            index=model_names)
# df_corrected_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_corrected_pval, brackets=False),
#                                      columns=df_coeff.columns, index=model_names)
#
# if SAVE_FIGS:
#     df_concat = pd.concat([df_coeff, df_pval, df_asterisk, df_corrected_pval, df_corrected_asterisk],
#                           axis=1)
#     idx = list(interleave(np.array_split(range(df_concat.shape[1]), 5)))
#     df_concat = df_concat.iloc[:, idx]
#     df_concat.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_depot_weight_models_coeffs_pvals_fko.csv'))
#
# if SAVE_FIGS:
#     plt.clf()
#     plt.subplot(221)
#     sex = 'f'
#     cytometer.stats.plot_linear_regression(gwat_null_model_f, metainfo_f, 'BW__',
#                                            other_vars={'sex':sex}, sx=BW_mean, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(gwat_model_f_control, metainfo_f, 'BW__',
#                                            other_vars={'sex':sex, 'functional_ko':False},
#                                            dep_var='gWAT', sx=BW_mean, c='C2', marker='x',
#                                            line_label='Control')
#     cytometer.stats.plot_linear_regression(gwat_model_f_fko, metainfo_f, 'BW__',
#                                            other_vars={'sex':sex, 'functional_ko':True},
#                                            dep_var='gWAT', sx=BW_mean, c='C3', marker='+',
#                                            line_label='FKO')
#     plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
#     plt.ylim(0, 2.1)
#     plt.tick_params(labelsize=14)
#     plt.title('Female', fontsize=14)
#     plt.ylabel('Gonadal\ndepot weight (g)', fontsize=14)
#
#     plt.subplot(222)
#     sex = 'm'
#     cytometer.stats.plot_linear_regression(gwat_null_model_m, metainfo_m, 'BW__',
#                                            other_vars={'sex':sex}, sx=BW_mean, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(gwat_model_m_control, metainfo_m, 'BW__',
#                                            other_vars={'sex':sex, 'functional_ko':False},
#                                            dep_var='gWAT', sx=BW_mean, c='C2', marker='x',
#                                            line_label='Control')
#     cytometer.stats.plot_linear_regression(gwat_model_m_fko, metainfo_m, 'BW__',
#                                            other_vars={'sex':sex, 'functional_ko':True},
#                                            dep_var='gWAT', sx=BW_mean, c='C3', marker='+',
#                                            line_label='FKO')
#     plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
#     plt.ylim(0, 2.1)
#     plt.tick_params(labelsize=14)
#     plt.title('Male', fontsize=14)
#
#     plt.subplot(223)
#     sex = 'f'
#     cytometer.stats.plot_linear_regression(sqwat_null_model_f, metainfo_f, 'BW__',
#                                            other_vars={'sex':sex}, sx=BW_mean, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(sqwat_model_f_control, metainfo_f, 'BW__',
#                                            other_vars={'sex':sex, 'functional_ko':False},
#                                            dep_var='SC', sx=BW_mean, c='C2', marker='x',
#                                            line_label='Control')
#     cytometer.stats.plot_linear_regression(sqwat_model_f_fko, metainfo_f, 'BW__',
#                                            other_vars={'sex':sex, 'functional_ko':True},
#                                            dep_var='SC', sx=BW_mean, c='C3', marker='+',
#                                            line_label='FKO')
#     plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
#     plt.tick_params(labelsize=14)
#     plt.ylim(0, 2.1)
#     plt.xlabel('Body weight (g)', fontsize=14)
#     plt.ylabel('Subcutaneous\ndepot weight (g)', fontsize=14)
#     plt.legend(loc='upper right')
#
#     plt.subplot(224)
#     sex = 'm'
#     cytometer.stats.plot_linear_regression(sqwat_null_model_m, metainfo_m, 'BW__',
#                                            other_vars={'sex':sex}, sx=BW_mean, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(sqwat_model_m_control, metainfo_m, 'BW__',
#                                            other_vars={'sex':sex, 'functional_ko':False},
#                                            dep_var='SC', sx=BW_mean, c='C2', marker='x',
#                                            line_label='Control')
#     cytometer.stats.plot_linear_regression(sqwat_model_m_fko, metainfo_m, 'BW__',
#                                            other_vars={'sex':sex, 'functional_ko':True},
#                                            dep_var='SC', sx=BW_mean, c='C3', marker='+',
#                                            line_label='KFO')
#     plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
#     plt.ylim(0, 2.1)
#     plt.tick_params(labelsize=14)
#     plt.xlabel('Body weight (g)', fontsize=14)
#
#     plt.tight_layout()
#
#     plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_depot_linear_model_fko.png'))
#     plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_depot_linear_model_fko.svg'))
#
#
# ########################################################################################################################
# ## Analyse cell populations from automatically segmented images in two depots: SQWAT and GWAT:
# ########################################################################################################################
#
# ## one data point per animal
# ## linear regression analysis of quantile_area ~ DW * functional_ko
# ## USED IN PAPER
# ########################################################################################################################
#
# ## (only mode, 25%-, 50%- and 75%-quantiles for illustration purposes and debugging)
#
# # 0.05, 0.1 , 0.15, 0.2, ..., 0.9 , 0.95
# quantiles = np.linspace(0, 1, 21)  #
#
# # indices of the quantiles we are going to model
# i_quantiles = [5, 10, 15]  # Q1, Q2, Q3
#
# # for convenience
# df_all_f = df_all[df_all['sex'] == 'f']
# df_all_m = df_all[df_all['sex'] == 'm']
#
# depot = 'gwat'
# # depot = 'sqwat'
#
# # fit linear models to area quantiles
# q_models_f_control = []
# q_models_f_fko = []
# q_models_m_control = []
# q_models_m_fko = []
# q_models_f_null = []
# q_models_m_null = []
# q_models_f = []
# q_models_m = []
# for i_q in i_quantiles:
#
#     # choose one area_at_quantile value as the output of the linear model
#     df_all['area_at_quantile'] = np.array(df_all['area_at_quantiles'].to_list())[:, i_q]
#
#     # fit WT/Het linear models
#     idx = (df_all['sex'] == 'f') & (df_all['depot'] == depot) & (df_all['functional_ko'] == False)
#     q_model_f_wt = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
#     idx = (df_all['sex'] == 'f') & (df_all['depot'] == depot) & (df_all['functional_ko'] == True)
#     q_model_f_het = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
#     idx = (df_all['sex'] == 'm') & (df_all['depot'] == depot) & (df_all['functional_ko'] == False)
#     q_model_m_wt = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
#     idx = (df_all['sex'] == 'm') & (df_all['depot'] == depot) & (df_all['functional_ko'] == True)
#     q_model_m_het = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
#
#     # fit null models
#     idx = (df_all['sex'] == 'f') & (df_all['depot'] == depot)
#     q_model_f_null = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
#     idx = (df_all['sex'] == 'm') & (df_all['depot'] == depot)
#     q_model_m_null = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
#
#     # fit models with effect variable
#     idx = (df_all['sex'] == 'f') & (df_all['depot'] == depot)
#     q_model_f = sm.OLS.from_formula('area_at_quantile ~ DW * C(functional_ko)', data=df_all, subset=idx).fit()
#     idx = (df_all['sex'] == 'm') & (df_all['depot'] == depot)
#     q_model_m = sm.OLS.from_formula('area_at_quantile ~ DW * C(functional_ko)', data=df_all, subset=idx).fit()
#
#     q_models_f_control.append(q_model_f_wt)
#     q_models_f_fko.append(q_model_f_het)
#     q_models_m_control.append(q_model_m_wt)
#     q_models_m_fko.append(q_model_m_het)
#     q_models_f_null.append(q_model_f_null)
#     q_models_m_null.append(q_model_m_null)
#     q_models_f.append(q_model_f)
#     q_models_m.append(q_model_m)
#
#     if DEBUG:
#         print(q_model_f_wt.summary())
#         print(q_model_f_het.summary())
#         print(q_model_m_wt.summary())
#         print(q_model_m_het.summary())
#         print(q_model_f_null.summary())
#         print(q_model_m_null.summary())
#         print(q_model_f.summary())
#         print(q_model_m.summary())
#
# # extract coefficients, errors and p-values from PAT and MAT models
# model_names = []
# for model_name in ['model_f_control', 'model_f_fko', 'model_m_control', 'model_m_fko']:
#     for i_q in i_quantiles:
#         model_names.append('q_' + '{0:.0f}'.format(quantiles[i_q] * 100) + '_' + model_name)
# df_coeff, df_ci_lo, df_ci_hi, df_pval = \
#     cytometer.stats.models_coeff_ci_pval(
#         q_models_f_control + q_models_f_fko + q_models_m_control + q_models_m_fko,
#     model_names=model_names)
#
# # convert p-values to asterisks
# df_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval, brackets=False), columns=df_coeff.columns,
#                            index=model_names)
#
# if SAVE_FIGS:
#     df_concat = pd.concat([df_coeff, df_pval, df_asterisk],
#                           axis=1)
#     idx = list(interleave(np.array_split(range(df_concat.shape[1]), 3)))
#     df_concat = df_concat.iloc[:, idx]
#     df_concat.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_area_at_quartiles_fko_models_coeffs_pvals_' + depot + '.csv'))
#
# # plot
# if SAVE_FIGS:
#
#     plt.clf()
#     plt.gcf().set_size_inches([6.4, 7.6])
#
#     plt.subplot(321)
#     # Q1 Female
#     i = 0  # quantile index for "i_quantiles"
#     i_q = i_quantiles[i]  # quantile index for "quantiles"
#     sex = 'f'
#     df = df_all_f[df_all_f['depot'] == depot].copy()
#     df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
#     cytometer.stats.plot_linear_regression(q_models_f_null[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(q_models_f_control[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'functional_ko': False},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
#                                            line_label='Control')
#     cytometer.stats.plot_linear_regression(q_models_f_fko[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'functional_ko': True},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
#                                            line_label='FKO')
#     plt.tick_params(labelsize=14)
#     plt.ylabel('Area$_{\mathrm{Q1}}$ ($10^3\ \mu m^2$)', fontsize=14)
#     plt.title('Female', fontsize=14)
#     if depot == 'gwat':
#         plt.legend(loc='best', fontsize=12)
#     if depot == 'gwat':
#         plt.ylim(0.9, 4.3)
#     elif depot == 'sqwat':
#         plt.ylim(0.5, 3)
#
#     plt.subplot(322)
#     # Q1 Male
#     i = 0  # quantile index for "i_quantiles"
#     i_q = i_quantiles[i]  # quantile index for "quantiles"
#     sex = 'm'
#     df = df_all_m[df_all_m['depot'] == depot].copy()
#     df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
#     cytometer.stats.plot_linear_regression(q_models_m_null[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(q_models_m_control[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'functional_ko': False},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
#                                            line_label='Control')
#     cytometer.stats.plot_linear_regression(q_models_m_fko[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'functional_ko': True},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
#                                            line_label='FKO')
#     plt.tick_params(labelsize=14)
#     plt.title('Male', fontsize=14)
#     if depot == 'gwat':
#         plt.ylim(0.9, 4.3)
#     elif depot == 'sqwat':
#         plt.ylim(0.5, 3)
#
#     plt.subplot(323)
#     # Q2 Female
#     i = 1  # quantile index for "i_quantiles"
#     i_q = i_quantiles[i]  # quantile index for "quantiles"
#     sex = 'f'
#     df = df_all_f[df_all_f['depot'] == depot].copy()
#     df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
#     cytometer.stats.plot_linear_regression(q_models_f_null[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(q_models_f_control[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'functional_ko': False},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
#                                            line_label='Control')
#     cytometer.stats.plot_linear_regression(q_models_f_fko[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'functional_ko': True},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
#                                            line_label='FKO')
#     plt.tick_params(labelsize=14)
#     plt.ylabel('Area$_{\mathrm{Q2}}$ ($10^3\ \mu m^2$)', fontsize=14)
#     if depot == 'gwat':
#         plt.ylim(1.4, 8.5)
#     elif depot == 'sqwat':
#         plt.ylim(0.8, 6)
#
#     plt.subplot(324)
#     # Q2 Male
#     i = 1  # quantile index for "i_quantiles"
#     i_q = i_quantiles[i]  # quantile index for "quantiles"
#     sex = 'm'
#     df = df_all_m[df_all_m['depot'] == depot].copy()
#     df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
#     cytometer.stats.plot_linear_regression(q_models_m_null[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(q_models_m_control[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'functional_ko': False},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
#                                            line_label='WT')
#     cytometer.stats.plot_linear_regression(q_models_m_fko[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'functional_ko': True},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
#                                            line_label='Het')
#     plt.tick_params(labelsize=14)
#     if depot == 'gwat':
#         plt.ylim(1.4, 8.5)
#     elif depot == 'sqwat':
#         plt.ylim(0.8, 6)
#
#     plt.subplot(325)
#     # Q3 Female
#     i = 2  # quantile index for "i_quantiles"
#     i_q = i_quantiles[i]  # quantile index for "quantiles"
#     sex = 'f'
#     df = df_all_f[df_all_f['depot'] == depot].copy()
#     df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
#     cytometer.stats.plot_linear_regression(q_models_f_null[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(q_models_f_control[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'functional_ko': False},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
#                                            line_label='Control')
#     cytometer.stats.plot_linear_regression(q_models_f_fko[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'functional_ko': True},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
#                                            line_label='FKO')
#     plt.tick_params(labelsize=14)
#     plt.ylabel('Area$_{\mathrm{Q3}}$ ($10^3\ \mu m^2$)', fontsize=14)
#     plt.xlabel('Depot weight (g)', fontsize=14)
#     if depot == 'gwat':
#         plt.ylim(1, 14)
#         # plt.text(0.02, 0.98, pval_text, transform=plt.gca().transAxes, va='top', fontsize=12)
#     elif depot == 'sqwat':
#         plt.ylim(1, 10.5)
#
#     plt.subplot(326)
#     # Q3 Male
#     i = 2  # quantile index for "i_quantiles"
#     i_q = i_quantiles[i]  # quantile index for "quantiles"
#     sex = 'm'
#     df = df_all_m[df_all_m['depot'] == depot].copy()
#     df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
#     cytometer.stats.plot_linear_regression(q_models_m_null[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(q_models_m_control[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'functional_ko': False},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
#                                            line_label='WT')
#     cytometer.stats.plot_linear_regression(q_models_m_fko[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'functional_ko': True},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
#                                            line_label='Het')
#     plt.tick_params(labelsize=14)
#     plt.xlabel('Depot weight (g)', fontsize=14)
#     if depot == 'gwat':
#         plt.ylim(1, 14)
#     elif depot == 'sqwat':
#         plt.ylim(1, 10.5)
#
#     depot_title = depot.replace('gwat', 'Gonadal').replace('sqwat', 'Subcutaneous')
#     plt.suptitle(depot_title, fontsize=14)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#
#     plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_area_at_quartile_genotype_models_' + depot + '.png'))
#     plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_area_at_quartile_genotype_models_' + depot + '.svg'))
#
# # Likelihood ratio tests of the genotype variable
# print('Likelihood Ratio Test')
#
# print('Female')
# for i, i_q in enumerate(i_quantiles):
#     lr, pval = cytometer.stats.lrtest(q_models_f_null[i].llf, q_models_f[i].llf)
#     pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
#     print('q=' + str(quantiles[i_q]) + ', ' + depot + ': ' + pval_text)
#     print('AIC_null=' + '{0:.2f}'.format(q_models_f_null[i].aic) + ', AIC_alt=' + '{0:.2f}'.format(q_models_f[i].aic))
#
# print('Male')
# for i, i_q in enumerate(i_quantiles):
#     lr, pval = cytometer.stats.lrtest(q_models_m_null[i].llf, q_models_m[i].llf)
#     pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
#     print('q=' + str(quantiles[i_q]) + ', ' + depot + ': ' + pval_text)
#     print('AIC_null=' + '{0:.2f}'.format(q_models_m_null[i].aic) + ', AIC_alt=' + '{0:.2f}'.format(q_models_m[i].aic))
#
# ## multitest correction of all betas from both depots
#
# gwat_df = pd.read_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_area_at_quartiles_genotype_models_coeffs_pvals_gwat.csv'))
# sqwat_df = pd.read_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_area_at_quartiles_genotype_models_coeffs_pvals_sqwat.csv'))
#
# # add a depot tag to the model name
# gwat_df['model'] = ['gwat_' + x for x in gwat_df['model']]
# sqwat_df['model'] = ['sqwat_' + x for x in gwat_df['model']]
#
# # concatenate p-values from gwat and sqwat, so that we have: female gwat, female sqwat, male gwat, male sqwat
# col = 'DW.1'  # column name with the uncorrected p-values for the slope coefficients
# pval = np.concatenate((gwat_df[col], sqwat_df[col]))
#
# # multitest correction using Benjamini-Krieger-Yekutieli
# _, corrected_pval, _, _ = multipletests(pval, method='fdr_tsbky', alpha=0.05, returnsorted=False)
#
# # convert p-values to asterisks
# corrected_asterisk = cytometer.stats.pval_to_asterisk(corrected_pval, brackets=False)
#
# # add to dataframe
# gwat_df['DW.3'] = corrected_pval[0:12]
# gwat_df['DW.4'] = corrected_asterisk[0:12]
# sqwat_df['DW.3'] = corrected_pval[12:]
# sqwat_df['DW.4'] = corrected_asterisk[12:]
#
# if SAVE_FIGS:
#     gwat_df.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_area_at_quartiles_fko_models_coeffs_pvals_gwat.csv'))
#     sqwat_df.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_area_at_quartiles_fko_models_coeffs_pvals_sqwat.csv'))
#
# ## one data point per animal
# ## linear regression analysis of quantile_area ~ DW * ko_parent
# ## USED IN PAPER
# ########################################################################################################################
#
# ## (only mode, 25%-, 50%- and 75%-quantiles for illustration purposes and debugging)
#
# # 0.05, 0.1 , 0.15, 0.2, ..., 0.9 , 0.95
# quantiles = np.linspace(0, 1, 21)  #
#
# # indices of the quantiles we are going to model
# i_quantiles = [5, 10, 15]  # Q1, Q2, Q3
#
# # for convenience
# df_all_f = df_all[df_all['sex'] == 'f']
# df_all_m = df_all[df_all['sex'] == 'm']
#
# depot = 'gwat'
# # depot = 'sqwat'
#
# # fit linear models to area quantiles
# q_models_f_pat = []
# q_models_f_mat = []
# q_models_m_pat = []
# q_models_m_mat = []
# q_models_f_null = []
# q_models_m_null = []
# q_models_f = []
# q_models_m = []
# for i_q in i_quantiles:
#
#     # choose one area_at_quantile value as the output of the linear model
#     df_all['area_at_quantile'] = np.array(df_all['area_at_quantiles'].to_list())[:, i_q]
#
#     # fit WT/Het linear models
#     idx = (df_all['sex'] == 'f') & (df_all['depot'] == depot) & (df_all['ko_parent'] == 'PAT')
#     q_model_f_pat = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
#     idx = (df_all['sex'] == 'f') & (df_all['depot'] == depot) & (df_all['ko_parent'] == 'MAT')
#     q_model_f_mat = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
#     idx = (df_all['sex'] == 'm') & (df_all['depot'] == depot) & (df_all['ko_parent'] == 'PAT')
#     q_model_m_pat = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
#     idx = (df_all['sex'] == 'm') & (df_all['depot'] == depot) & (df_all['ko_parent'] == 'MAT')
#     q_model_m_mat = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
#
#     # fit null models
#     idx = (df_all['sex'] == 'f') & (df_all['depot'] == depot)
#     q_model_f_null = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
#     idx = (df_all['sex'] == 'm') & (df_all['depot'] == depot)
#     q_model_m_null = sm.OLS.from_formula('area_at_quantile ~ DW', data=df_all, subset=idx).fit()
#
#     # fit models with parent variable
#     idx = (df_all['sex'] == 'f') & (df_all['depot'] == depot)
#     q_model_f = sm.OLS.from_formula('area_at_quantile ~ DW * C(ko_parent)', data=df_all, subset=idx).fit()
#     idx = (df_all['sex'] == 'm') & (df_all['depot'] == depot)
#     q_model_m = sm.OLS.from_formula('area_at_quantile ~ DW * C(ko_parent)', data=df_all, subset=idx).fit()
#
#     q_models_f_pat.append(q_model_f_pat)
#     q_models_f_mat.append(q_model_f_mat)
#     q_models_m_pat.append(q_model_m_pat)
#     q_models_m_mat.append(q_model_m_mat)
#     q_models_f_null.append(q_model_f_null)
#     q_models_m_null.append(q_model_m_null)
#     q_models_f.append(q_model_f)
#     q_models_m.append(q_model_m)
#
#     if DEBUG:
#         print(q_model_f_pat.summary())
#         print(q_model_f_mat.summary())
#         print(q_model_m_pat.summary())
#         print(q_model_m_mat.summary())
#         print(q_model_f_null.summary())
#         print(q_model_m_null.summary())
#         print(q_model_f.summary())
#         print(q_model_m.summary())
#
# # extract coefficients, errors and p-values from PAT and MAT models
# model_names = []
# for model_name in ['model_f_pat', 'model_f_mat', 'model_m_pat', 'model_m_mat']:
#     for i_q in i_quantiles:
#         model_names.append('q_' + '{0:.0f}'.format(quantiles[i_q] * 100) + '_' + model_name)
# df_coeff, df_ci_lo, df_ci_hi, df_pval = \
#     cytometer.stats.models_coeff_ci_pval(
#         q_models_f_pat + q_models_f_mat + q_models_m_pat + q_models_m_mat,
#     model_names=model_names)
#
# # convert p-values to asterisks
# df_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval, brackets=False), columns=df_coeff.columns,
#                            index=model_names)
#
# if SAVE_FIGS:
#     df_concat = pd.concat([df_coeff, df_pval, df_asterisk],
#                           axis=1)
#     idx = list(interleave(np.array_split(range(df_concat.shape[1]), 3)))
#     df_concat = df_concat.iloc[:, idx]
#     df_concat.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_area_at_quartiles_parent_models_coeffs_pvals_' + depot + '.csv'))
#
# # plot
# if SAVE_FIGS:
#
#     plt.clf()
#     plt.gcf().set_size_inches([6.4, 7.6])
#
#     plt.subplot(321)
#     # Q1 Female
#     i = 0  # quantile index for "i_quantiles"
#     i_q = i_quantiles[i]  # quantile index for "quantiles"
#     sex = 'f'
#     df = df_all_f[df_all_f['depot'] == depot].copy()
#     df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
#     cytometer.stats.plot_linear_regression(q_models_f_null[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(q_models_f_pat[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C0', marker='x',
#                                            line_label='PAT')
#     cytometer.stats.plot_linear_regression(q_models_f_mat[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C1', marker='+',
#                                            line_label='MAT')
#     plt.tick_params(labelsize=14)
#     plt.ylabel('Area$_{\mathrm{Q1}}$ ($10^3\ \mu m^2$)', fontsize=14)
#     plt.title('Female', fontsize=14)
#     if depot == 'gwat':
#         plt.legend(loc='best', fontsize=12)
#     if depot == 'gwat':
#         plt.ylim(0.9, 4.3)
#     elif depot == 'sqwat':
#         plt.ylim(0.5, 3)
#
#     plt.subplot(322)
#     # Q1 Male
#     i = 0  # quantile index for "i_quantiles"
#     i_q = i_quantiles[i]  # quantile index for "quantiles"
#     sex = 'm'
#     df = df_all_m[df_all_m['depot'] == depot].copy()
#     df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
#     cytometer.stats.plot_linear_regression(q_models_m_null[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(q_models_m_pat[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C0', marker='x',
#                                            line_label='PAT')
#     cytometer.stats.plot_linear_regression(q_models_m_mat[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C1', marker='+',
#                                            line_label='MAT')
#     plt.tick_params(labelsize=14)
#     plt.title('Male', fontsize=14)
#     if depot == 'gwat':
#         plt.ylim(0.9, 4.3)
#     elif depot == 'sqwat':
#         plt.ylim(0.5, 3)
#
#     plt.subplot(323)
#     # Q2 Female
#     i = 1  # quantile index for "i_quantiles"
#     i_q = i_quantiles[i]  # quantile index for "quantiles"
#     sex = 'f'
#     df = df_all_f[df_all_f['depot'] == depot].copy()
#     df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
#     cytometer.stats.plot_linear_regression(q_models_f_null[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(q_models_f_pat[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C0', marker='x',
#                                            line_label='PAT')
#     cytometer.stats.plot_linear_regression(q_models_f_mat[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C1', marker='+',
#                                            line_label='MAT')
#     plt.tick_params(labelsize=14)
#     plt.ylabel('Area$_{\mathrm{Q2}}$ ($10^3\ \mu m^2$)', fontsize=14)
#     if depot == 'gwat':
#         plt.ylim(1.4, 8.5)
#     elif depot == 'sqwat':
#         plt.ylim(0.8, 6)
#
#     plt.subplot(324)
#     # Q2 Male
#     i = 1  # quantile index for "i_quantiles"
#     i_q = i_quantiles[i]  # quantile index for "quantiles"
#     sex = 'm'
#     df = df_all_m[df_all_m['depot'] == depot].copy()
#     df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
#     cytometer.stats.plot_linear_regression(q_models_m_null[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(q_models_m_pat[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C0', marker='x',
#                                            line_label='PAT')
#     cytometer.stats.plot_linear_regression(q_models_m_mat[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C1', marker='+',
#                                            line_label='MAT')
#     plt.tick_params(labelsize=14)
#     if depot == 'gwat':
#         plt.ylim(1.4, 8.5)
#     elif depot == 'sqwat':
#         plt.ylim(0.8, 6)
#
#     plt.subplot(325)
#     # Q3 Female
#     i = 2  # quantile index for "i_quantiles"
#     i_q = i_quantiles[i]  # quantile index for "quantiles"
#     sex = 'f'
#     df = df_all_f[df_all_f['depot'] == depot].copy()
#     df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
#     cytometer.stats.plot_linear_regression(q_models_f_null[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(q_models_f_pat[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C0', marker='x',
#                                            line_label='PAT')
#     cytometer.stats.plot_linear_regression(q_models_f_mat[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C1', marker='+',
#                                            line_label='MAT')
#     plt.tick_params(labelsize=14)
#     plt.ylabel('Area$_{\mathrm{Q3}}$ ($10^3\ \mu m^2$)', fontsize=14)
#     plt.xlabel('Depot weight (g)', fontsize=14)
#     if depot == 'gwat':
#         plt.ylim(1, 14)
#         # plt.text(0.02, 0.98, pval_text, transform=plt.gca().transAxes, va='top', fontsize=12)
#     elif depot == 'sqwat':
#         plt.ylim(1, 10.5)
#
#     plt.subplot(326)
#     # Q3 Male
#     i = 2  # quantile index for "i_quantiles"
#     i_q = i_quantiles[i]  # quantile index for "quantiles"
#     sex = 'm'
#     df = df_all_m[df_all_m['depot'] == depot].copy()
#     df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
#     cytometer.stats.plot_linear_regression(q_models_m_null[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-3, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(q_models_m_pat[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C0', marker='x',
#                                            line_label='PAT')
#     cytometer.stats.plot_linear_regression(q_models_m_mat[i], df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
#                                            dep_var='area_at_quantile', sy=1e-3, c='C1', marker='+',
#                                            line_label='MAT')
#     plt.tick_params(labelsize=14)
#     plt.xlabel('Depot weight (g)', fontsize=14)
#     if depot == 'gwat':
#         plt.ylim(1, 14)
#     elif depot == 'sqwat':
#         plt.ylim(1, 10.5)
#
#     depot_title = depot.replace('gwat', 'Gonadal').replace('sqwat', 'Subcutaneous')
#     plt.suptitle(depot_title, fontsize=14)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#
#     plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_area_at_quartile_parent_models_' + depot + '.png'))
#     plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_area_at_quartile_parent_models_' + depot + '.svg'))
#
# # Likelihood ratio tests of the parent variable
# print('Likelihood Ratio Test')
#
# print('Female')
# for i, i_q in enumerate(i_quantiles):
#     lr, pval = cytometer.stats.lrtest(q_models_f_null[i].llf, q_models_f[i].llf)
#     pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
#     print('q=' + str(quantiles[i_q]) + ', ' + depot + ': ' + pval_text)
#     print('AIC_null=' + '{0:.2f}'.format(q_models_f_null[i].aic) + ', AIC_alt=' + '{0:.2f}'.format(q_models_f[i].aic))
#
# print('Male')
# for i, i_q in enumerate(i_quantiles):
#     lr, pval = cytometer.stats.lrtest(q_models_m_null[i].llf, q_models_m[i].llf)
#     pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
#     print('q=' + str(quantiles[i_q]) + ', ' + depot + ': ' + pval_text)
#     print('AIC_null=' + '{0:.2f}'.format(q_models_m_null[i].aic) + ', AIC_alt=' + '{0:.2f}'.format(q_models_m[i].aic))
#
# ## multitest correction of all betas from both depots
#
# gwat_df = pd.read_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_area_at_quartiles_parent_models_coeffs_pvals_gwat.csv'))
# sqwat_df = pd.read_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_area_at_quartiles_parent_models_coeffs_pvals_sqwat.csv'))
#
# # add a depot tag to the model name
# gwat_df['model'] = ['gwat_' + x for x in gwat_df['model']]
# sqwat_df['model'] = ['sqwat_' + x for x in gwat_df['model']]
#
# # concatenate p-values from gwat and sqwat, so that we have: female gwat, female sqwat, male gwat, male sqwat
# col = 'DW.1'  # column name with the uncorrected p-values for the slope coefficients
# pval = np.concatenate((gwat_df[col], sqwat_df[col]))
#
# # multitest correction using Benjamini-Krieger-Yekutieli
# _, corrected_pval, _, _ = multipletests(pval, method='fdr_tsbky', alpha=0.05, returnsorted=False)
#
# # convert p-values to asterisks
# corrected_asterisk = cytometer.stats.pval_to_asterisk(corrected_pval, brackets=False)
#
# # add to dataframe
# gwat_df['DW.3'] = corrected_pval[0:12]
# gwat_df['DW.4'] = corrected_asterisk[0:12]
# sqwat_df['DW.3'] = corrected_pval[12:]
# sqwat_df['DW.4'] = corrected_asterisk[12:]
#
# if SAVE_FIGS:
#     gwat_df.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_area_at_quartiles_parent_models_coeffs_pvals_gwat.csv'))
#     sqwat_df.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_area_at_quartiles_parent_models_coeffs_pvals_sqwat.csv'))
#
#
# ## one data point per animal
# ## linear regression analysis of kN ~ DW * genotype
# ## USED IN PAPER
# ########################################################################################################################
#
# # fit models kN ~ DW
# idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'gwat') & (df_all['genotype'] == 'KLF14-KO:WT')
# gwat_model_f_control = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'gwat') & (df_all['genotype'] == 'KLF14-KO:Het')
# gwat_model_f_fko = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'gwat') & (df_all['genotype'] == 'KLF14-KO:WT')
# gwat_model_m_control = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'gwat') & (df_all['genotype'] == 'KLF14-KO:Het')
# gwat_model_m_fko = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
#
# idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'sqwat') & (df_all['genotype'] == 'KLF14-KO:WT')
# sqwat_model_f_control = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'sqwat') & (df_all['genotype'] == 'KLF14-KO:Het')
# sqwat_model_f_fko = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'sqwat') & (df_all['genotype'] == 'KLF14-KO:WT')
# sqwat_model_m_control = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'sqwat') & (df_all['genotype'] == 'KLF14-KO:Het')
# sqwat_model_m_fko = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
#
# # fit null models and models with effect variable
# idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'gwat')
# gwat_model_f_null = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# gwat_model_f = sm.OLS.from_formula('kN ~ DW * C(genotype)', data=df_all, subset=idx).fit()
# idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'gwat')
# gwat_model_m_null = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# gwat_model_m = sm.OLS.from_formula('kN ~ DW * C(genotype)', data=df_all, subset=idx).fit()
#
# idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'sqwat')
# sqwat_model_f_null = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# sqwat_model_f = sm.OLS.from_formula('kN ~ DW * C(genotype)', data=df_all, subset=idx).fit()
# idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'sqwat')
# sqwat_model_m_null = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# sqwat_model_m = sm.OLS.from_formula('kN ~ DW * C(genotype)', data=df_all, subset=idx).fit()
#
# # Likelihood Ratio Tests
# print('Genotype effect')
# print('Female')
# lr, pval = cytometer.stats.lrtest(gwat_model_f_null.llf, gwat_model_f.llf)
# pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(
#     pval)
# print('\t' + 'gwat: ' + pval_text)
# lr, pval = cytometer.stats.lrtest(sqwat_model_f_null.llf, sqwat_model_f.llf)
# pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(
#     pval)
# print('\t' + 'sqwat: ' + pval_text)
#
# print('Male')
# lr, pval = cytometer.stats.lrtest(gwat_model_m_null.llf, gwat_model_m.llf)
# pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(
#     pval)
# print('\t' + 'gwat: ' + pval_text)
# lr, pval = cytometer.stats.lrtest(sqwat_model_m_null.llf, sqwat_model_m.llf)
# pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(
#     pval)
# print('\t' + 'sqwat: ' + pval_text)
#
# # extract coefficients, errors and p-values from PAT and MAT models
# model_names = ['gwat_model_f_control', 'gwat_model_f_fko', 'sqwat_model_f_control', 'sqwat_model_f_fko',
#                'gwat_model_m_control', 'gwat_model_m_fko', 'sqwat_model_m_control', 'sqwat_model_m_fko']
# df_coeff, df_ci_lo, df_ci_hi, df_pval = \
#     cytometer.stats.models_coeff_ci_pval(
#         [gwat_model_f_control, gwat_model_f_fko, sqwat_model_f_control, sqwat_model_f_fko,
#          gwat_model_m_control, gwat_model_m_fko, sqwat_model_m_control, sqwat_model_m_fko],
#         model_names=model_names)
#
# # multitest correction using Benjamini-Krieger-Yekutieli
#
# # _, df_corrected_pval, _, _ = multipletests(df_pval.values.flatten(), method='fdr_tsbky', alpha=0.05, returnsorted=False)
# # df_corrected_pval = pd.DataFrame(df_corrected_pval.reshape(df_pval.shape), columns=df_pval.columns, index=model_names)
#
# col = df_pval.columns[1]
# df_corrected_pval = df_pval.copy()
# _, df_corrected_pval[col], _, _ = multipletests(df_pval[col], method='fdr_tsbky', alpha=0.05, returnsorted=False)
# df_corrected_pval['Intercept'] = -1.0
#
# # convert p-values to asterisks
# df_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval, brackets=False), columns=df_coeff.columns,
#                            index=model_names)
# df_corrected_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_corrected_pval, brackets=False),
#                                      columns=df_coeff.columns, index=model_names)
#
# if SAVE_FIGS:
#     df_concat = pd.concat([df_coeff, df_pval, df_asterisk, df_corrected_pval, df_corrected_asterisk],
#                           axis=1)
#     idx = list(interleave(np.array_split(range(df_concat.shape[1]), 5)))
#     df_concat = df_concat.iloc[:, idx]
#     df_concat.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_kN_genotype_models_coeffs_pvals.csv'))
#
# # plot
# if SAVE_FIGS:
#
#     plt.clf()
#
#     plt.subplot(221)
#     # Female
#     depot = 'gwat'; sex = 'f'
#     idx = (df_all['depot'] == depot) & (df_all['sex'] == sex)
#     df = df_all[idx]
#     cytometer.stats.plot_linear_regression(gwat_model_f_null, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-6, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(gwat_model_f_control, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:WT'},
#                                            dep_var='kN', sy=1e-6, c='C2', marker='x',
#                                            line_label='WT')
#     cytometer.stats.plot_linear_regression(gwat_model_f_fko, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:Het'},
#                                            dep_var='kN', sy=1e-6, c='C3', marker='x',
#                                            line_label='Het')
#     plt.tick_params(labelsize=14)
#     plt.ylabel('Gonadal\nk N ($10^6$ cells)', fontsize=14)
#     plt.title('Female', fontsize=14)
#     plt.legend(loc='best', fontsize=12)
#     plt.ylim(-1, 16)
#
#     plt.subplot(222)
#     # Male
#     depot = 'gwat'; sex = 'm'
#     idx = (df_all['depot'] == depot) & (df_all['sex'] == sex)
#     df = df_all[idx]
#     cytometer.stats.plot_linear_regression(gwat_model_m_null, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-6, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(gwat_model_m_control, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:WT'},
#                                            dep_var='kN', sy=1e-6, c='C2', marker='x',
#                                            line_label='WT')
#     cytometer.stats.plot_linear_regression(gwat_model_m_fko, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:Het'},
#                                            dep_var='kN', sy=1e-6, c='C3', marker='x',
#                                            line_label='Het')
#     plt.tick_params(labelsize=14)
#     plt.title('Male', fontsize=14)
#     plt.ylim(-1, 10)
#
#     plt.subplot(223)
#     depot = 'sqwat'; sex = 'f'
#     idx = (df_all['depot'] == depot) & (df_all['sex'] == sex)
#     df = df_all[idx]
#     cytometer.stats.plot_linear_regression(sqwat_model_f_null, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-6, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(sqwat_model_f_control, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:WT'},
#                                            dep_var='kN', sy=1e-6, c='C2', marker='x',
#                                            line_label='WT')
#     cytometer.stats.plot_linear_regression(sqwat_model_f_fko, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:Het'},
#                                            dep_var='kN', sy=1e-6, c='C3', marker='x',
#                                            line_label='Het')
#     plt.tick_params(labelsize=14)
#     plt.ylabel('Subcutaneous\nk N ($10^6$ cells)', fontsize=14)
#     plt.xlabel('Depot weight (g)', fontsize=14)
#     plt.ylim(-1, 16)
#
#     plt.subplot(224)
#     # Male
#     depot = 'sqwat'; sex = 'm'
#     idx = (df_all['depot'] == depot) & (df_all['sex'] == sex)
#     df = df_all[idx]
#     cytometer.stats.plot_linear_regression(sqwat_model_m_null, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-6, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(sqwat_model_m_control, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:WT'},
#                                            dep_var='kN', sy=1e-6, c='C2', marker='x',
#                                            line_label='WT')
#     cytometer.stats.plot_linear_regression(sqwat_model_m_fko, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'genotype': 'KLF14-KO:Het'},
#                                            dep_var='kN', sy=1e-6, c='C3', marker='x',
#                                            line_label='Het')
#     plt.tick_params(labelsize=14)
#     plt.xlabel('Depot weight (g)', fontsize=14)
#     plt.ylim(-1, 10)
#
#     plt.tight_layout()
#
#     plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_kN_linear_genotype_model.png'))
#     plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_kN_linear_genotype_model.svg'))
#
# ## one data point per animal
# ## linear regression analysis of kN ~ DW * parent
# ## USED IN PAPER
# ########################################################################################################################
#
# # fit models kN ~ DW
# idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'gwat') & (df_all['ko_parent'] == 'PAT')
# gwat_model_f_pat = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'gwat') & (df_all['ko_parent'] == 'MAT')
# gwat_model_f_mat = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'gwat') & (df_all['ko_parent'] == 'PAT')
# gwat_model_m_pat = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'gwat') & (df_all['ko_parent'] == 'MAT')
# gwat_model_m_mat = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
#
# idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'sqwat') & (df_all['ko_parent'] == 'PAT')
# sqwat_model_f_pat = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'sqwat') & (df_all['ko_parent'] == 'MAT')
# sqwat_model_f_mat = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'sqwat') & (df_all['ko_parent'] == 'PAT')
# sqwat_model_m_pat = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'sqwat') & (df_all['ko_parent'] == 'MAT')
# sqwat_model_m_mat = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
#
# # fit null models and models with effect variable
# idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'gwat')
# gwat_model_f_null = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# gwat_model_f = sm.OLS.from_formula('kN ~ DW * C(ko_parent)', data=df_all, subset=idx).fit()
# idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'gwat')
# gwat_model_m_null = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# gwat_model_m = sm.OLS.from_formula('kN ~ DW * C(ko_parent)', data=df_all, subset=idx).fit()
#
# idx = (df_all['sex'] == 'f') & (df_all['depot'] == 'sqwat')
# sqwat_model_f_null = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# sqwat_model_f = sm.OLS.from_formula('kN ~ DW * C(ko_parent)', data=df_all, subset=idx).fit()
# idx = (df_all['sex'] == 'm') & (df_all['depot'] == 'sqwat')
# sqwat_model_m_null = sm.OLS.from_formula('kN ~ DW', data=df_all, subset=idx).fit()
# sqwat_model_m = sm.OLS.from_formula('kN ~ DW * C(ko_parent)', data=df_all, subset=idx).fit()
#
# # Likelihood Ratio Tests
# print('Parent effect')
# print('Female')
# lr, pval = cytometer.stats.lrtest(gwat_model_f_null.llf, gwat_model_f.llf)
# pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(
#     pval)
# print('\t' + 'gwat: ' + pval_text)
# lr, pval = cytometer.stats.lrtest(sqwat_model_f_null.llf, sqwat_model_f.llf)
# pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(
#     pval)
# print('\t' + 'sqwat: ' + pval_text)
#
# print('Male')
# lr, pval = cytometer.stats.lrtest(gwat_model_m_null.llf, gwat_model_m.llf)
# pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(
#     pval)
# print('\t' + 'gwat: ' + pval_text)
# lr, pval = cytometer.stats.lrtest(sqwat_model_m_null.llf, sqwat_model_m.llf)
# pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(
#     pval)
# print('\t' + 'sqwat: ' + pval_text)
#
# # extract coefficients, errors and p-values from PAT and MAT models
# model_names = ['gwat_model_f_pat', 'gwat_model_f_mat', 'sqwat_model_f_pat', 'sqwat_model_f_mat',
#                'gwat_model_m_pat', 'gwat_model_m_mat', 'sqwat_model_m_pat', 'sqwat_model_m_mat']
# df_coeff, df_ci_lo, df_ci_hi, df_pval = \
#     cytometer.stats.models_coeff_ci_pval(
#         [gwat_model_f_pat, gwat_model_f_mat, sqwat_model_f_pat, sqwat_model_f_mat,
#          gwat_model_m_pat, gwat_model_m_mat, sqwat_model_m_pat, sqwat_model_m_mat],
#         model_names=model_names)
#
# # multitest correction using Benjamini-Krieger-Yekutieli
#
# # _, df_corrected_pval, _, _ = multipletests(df_pval.values.flatten(), method='fdr_tsbky', alpha=0.05, returnsorted=False)
# # df_corrected_pval = pd.DataFrame(df_corrected_pval.reshape(df_pval.shape), columns=df_pval.columns, index=model_names)
#
# col = df_pval.columns[1]
# df_corrected_pval = df_pval.copy()
# _, df_corrected_pval[col], _, _ = multipletests(df_pval[col], method='fdr_tsbky', alpha=0.05, returnsorted=False)
# df_corrected_pval['Intercept'] = -1.0
#
# # convert p-values to asterisks
# df_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval, brackets=False), columns=df_coeff.columns,
#                            index=model_names)
# df_corrected_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_corrected_pval, brackets=False),
#                                      columns=df_coeff.columns, index=model_names)
#
# if SAVE_FIGS:
#     df_concat = pd.concat([df_coeff, df_pval, df_asterisk, df_corrected_pval, df_corrected_asterisk],
#                           axis=1)
#     idx = list(interleave(np.array_split(range(df_concat.shape[1]), 5)))
#     df_concat = df_concat.iloc[:, idx]
#     df_concat.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_kN_parent_models_coeffs_pvals.csv'))
#
# # plot
# if SAVE_FIGS:
#
#     plt.clf()
#
#     plt.subplot(221)
#     # Female
#     depot = 'gwat'; sex = 'f'
#     idx = (df_all['depot'] == depot) & (df_all['sex'] == sex)
#     df = df_all[idx]
#     cytometer.stats.plot_linear_regression(gwat_model_f_null, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-6, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(gwat_model_f_pat, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
#                                            dep_var='kN', sy=1e-6, c='C0', marker='x',
#                                            line_label='PAT')
#     cytometer.stats.plot_linear_regression(gwat_model_f_mat, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
#                                            dep_var='kN', sy=1e-6, c='C1', marker='x',
#                                            line_label='MAT')
#     plt.tick_params(labelsize=14)
#     plt.ylabel('Gonadal\nk N ($10^6$ cells)', fontsize=14)
#     plt.title('Female', fontsize=14)
#     plt.legend(loc='best', fontsize=12)
#     plt.ylim(-1, 16)
#
#     plt.subplot(222)
#     # Male
#     depot = 'gwat'; sex = 'm'
#     idx = (df_all['depot'] == depot) & (df_all['sex'] == sex)
#     df = df_all[idx]
#     cytometer.stats.plot_linear_regression(gwat_model_m_null, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-6, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(gwat_model_m_pat, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
#                                            dep_var='kN', sy=1e-6, c='C0', marker='x',
#                                            line_label='PAT')
#     cytometer.stats.plot_linear_regression(gwat_model_m_mat, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
#                                            dep_var='kN', sy=1e-6, c='C1', marker='x',
#                                            line_label='MAT')
#     plt.tick_params(labelsize=14)
#     plt.title('Male', fontsize=14)
#     plt.ylim(-1, 10)
#
#     plt.subplot(223)
#     depot = 'sqwat'; sex = 'f'
#     idx = (df_all['depot'] == depot) & (df_all['sex'] == sex)
#     df = df_all[idx]
#     cytometer.stats.plot_linear_regression(sqwat_model_f_null, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-6, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(sqwat_model_f_pat, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
#                                            dep_var='kN', sy=1e-6, c='C0', marker='x',
#                                            line_label='PAT')
#     cytometer.stats.plot_linear_regression(sqwat_model_f_mat, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
#                                            dep_var='kN', sy=1e-6, c='C1', marker='x',
#                                            line_label='MAT')
#     plt.tick_params(labelsize=14)
#     plt.ylabel('Subcutaneous\nk N ($10^6$ cells)', fontsize=14)
#     plt.xlabel('Depot weight (g)', fontsize=14)
#     plt.ylim(-1, 16)
#
#     plt.subplot(224)
#     # Male
#     depot = 'sqwat'; sex = 'm'
#     idx = (df_all['depot'] == depot) & (df_all['sex'] == sex)
#     df = df_all[idx]
#     cytometer.stats.plot_linear_regression(sqwat_model_m_null, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex}, sy=1e-6, c='k',
#                                            line_label='Null')
#     cytometer.stats.plot_linear_regression(sqwat_model_m_pat, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'PAT'},
#                                            dep_var='kN', sy=1e-6, c='C0', marker='x',
#                                            line_label='PAT')
#     cytometer.stats.plot_linear_regression(sqwat_model_m_mat, df, 'DW',
#                                            other_vars={'depot': depot, 'sex': sex, 'ko_parent': 'MAT'},
#                                            dep_var='kN', sy=1e-6, c='C1', marker='x',
#                                            line_label='MAT')
#     plt.tick_params(labelsize=14)
#     plt.xlabel('Depot weight (g)', fontsize=14)
#     plt.ylim(-1, 10)
#
#     plt.tight_layout()
#
#     plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_kN_linear_parent_model.png'))
#     plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_kN_linear_parent_model.svg'))
#
# ########################################################################################################################
# ## Heatmaps of whole slides
# ########################################################################################################################
#
# # parameters to constrain cell areas
# min_cell_area = 0  # pixel; we want all small objects
# max_cell_area = 200e3  # pixel
# xres_ref = 0.4538234626730202
# yres_ref = 0.4537822752643282
# min_cell_area_um2 = min_cell_area * xres_ref * yres_ref
# max_cell_area_um2 = max_cell_area * xres_ref * yres_ref
#
# # CSV file with metainformation of all mice
# metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
# metainfo = pd.read_csv(metainfo_csv_file)
#
# # make sure that in the boxplots PAT comes before MAT
# metainfo['sex'] = metainfo['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
# metainfo['ko_parent'] = metainfo['ko_parent'].astype(
#     pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
# metainfo['genotype'] = metainfo['genotype'].astype(
#     pd.api.types.CategoricalDtype(categories=['KLF14-KO:WT', 'KLF14-KO:Het'], ordered=True))
#
# # load area to quantile maps computed in exp_0106
# filename_area2quantile = os.path.join(area2quantile_dir, 'klf14_b6ntac_exp_0106_filename_area2quantile_v8.npz')
# with np.load(filename_area2quantile, allow_pickle=True) as aux:
#     f_area2quantile_f = aux['f_area2quantile_f'].item()
#     f_area2quantile_m = aux['f_area2quantile_m'].item()
#
# # load AIDA's colourmap
# cm = cytometer.data.aida_colourmap()
#
# # downsample factor that we used in klf14_b6ntac_exp_0106_full_slide_pipeline_v8
# downsample_factor = 16
#
# # auxiliary function to keep a list of cropping coordinates for each image
# # this coordinates correspond to the reduced size when saving figures to kernel_file + '_exp_0110_cell_size_heatmap.png'
# def get_crop_box(i_file):
#     # [x0, xend, y0, yend]
#     crop_box = {
#         0: [22, 353, 99, 333],
#         1: [30, 507, 37, 446],
#         2: [62, 254, 13, 235],
#         3: [92, 570, 31, 406],
#         4: [11, 259, 30, 263],
#         5: [19, 279, 14, 342],
#         6: [43, 213, 8, 285],
#         7: [39, 229, 31, 314],
#         8: [53, 539, 34, 450],
#         9: [24, 167, 38, 210],
#         10: [361, 574, 21, 146],
#         11: [447, 609, 24, 253],
#         12: [45, 278, 32, 307],
#         13: [33, 207, 26, 216],
#         14: [80, 567, 52, 439],
#         15: [349, 594, 9, 292],
#         16: [22, 237, 22, 233],
#         17: [353, 599, 92, 308],
#         18: [33, 219, 37, 252],
#         19: [8, 270, 24, 311],
#         20: [27, 234, 39, 268],
#         21: [19, 232, 38, 219],
#         22: [80, 187, 47, 240],
#         23: [51, 517, 47, 379],
#         24: [30, 289, 18, 316],
#         25: [31, 226, 54, 272],
#         26: [47, 498, 27, 394],
#         27: [335, 621, 61, 340],
#         28: [395, 617, 39, 268],
#         29: [30, 176, 47, 188],
#         30: [30, 273, 13, 306],
#         31: [60, 240, 63, 236],
#         32: [8, 223, 30, 365],
#         33: [41, 215, 28, 224],
#         34: [424, 623, 31, 306],
#         35: [27, 223, 21, 203],
#         36: [58, 218, 13, 260],
#         37: [29, 186, 24, 181],
#         38: [55, 204, 88, 233],
#         39: [67, 241, 40, 261],
#         40: [36, 212, 55, 312],
#         41: [449, 567, 89, 279],
#         42: [39, 229, 42, 323],
#         43: [95, 262, 28, 204],
#         44: [400, 584, 21, 276],
#         45: [34, 218, 33, 223],
#         46: [359, 612, 27, 292],
#         47: [444, 603, 22, 216],
#         48: [38, 217, 12, 306],
#         49: [393, 596, 40, 228],
#         50: [73, 268, 42, 276],
#         51: [442, 585, 41, 211],
#         52: [43, 285, 38, 270],
#         53: [133, 255, 24, 193],
#         54: [421, 621, 11, 246],
#         55: [31, 227, 18, 332],
#         56: [24, 165, 21, 247],
#         57: [71, 246, 83, 249],
#         58: [295, 511, 27, 286],
#         59: [39, 246, 20, 234],
#         60: [24, 201, 50, 224],
#         61: [421, 614, 27, 266],
#         62: [383, 622, 27, 273],
#         63: [62, 579, 116, 429],
#         64: [389, 618, 66, 301],
#         65: [55, 292, 11, 258],
#         66: [426, 622, 67, 299],
#         67: [66, 214, 23, 263],
#         68: [86, 270, 30, 345],
#         69: [62, 234, 186, 318],
#         70: [57, 250, 58, 283],
#         71: [45, 240, 31, 212],
#         72: [350, 555, 8, 182],
#         73: [8, 196, 47, 207],
#         74: [12, 193, 22, 175],
#         75: [105, 345, 28, 275],
#         76: [387, 558, 27, 332],
#         77: [62, 288, 29, 341],
#         78: [54, 406, 28, 310],
#         79: [58, 328, 25, 347],
#         80: [8, 275, 26, 339],
#         81: [11, 260, 7, 329],
#         82: [348, 610, 17, 336],
#         83: [7, 327, 22, 358],
#         84: [8, 170, 57, 208],
#         85: [19, 241, 40, 257],
#         86: [382, 600, 50, 246],
#         87: [436, 605, 33, 253],
#         88: [389, 563, 20, 227],
#         89: [31, 289, 32, 276],
#         90: [308, 602, 7, 343],
#         91: [6, 276, 25, 303],
#         92: [15, 268, 6, 305],
#         93: [82, 520, 47, 314],
#         94: [46, 520, 27, 431],
#         95: [15, 519, 11, 396],
#         96: [30, 594, 4, 406],
#         97: [27, 612, 18, 404],
#         98: [22, 246, 17, 293],
#         99: [102, 550, 36, 462],
#         100: [43, 221, 30, 237],
#         101: [52, 504, 85, 361],
#         102: [29, 440, 24, 435],
#         103: [283, 598, 151, 392],
#         104: [46, 232, 26, 274],
#         105: [45, 243, 31, 264],
#         106: [48, 556, 42, 305],
#         107: [52, 572, 3, 347],
#         108: [98, 547, 25, 425],
#         109: [19, 570, 13, 350],
#         110: [395, 618, 8, 304],
#         111: [51, 508, 46, 426],
#         112: [52, 608, 9, 330],
#         113: [34, 538, 55, 427],
#         114: [127, 530, 31, 405],
#         115: [406, 605, 35, 264],
#         116: [42, 577, 38, 440],
#         117: [50, 590, 63, 413],
#         118: [33, 259, 37, 216],
#         119: [12, 486, 32, 346],
#         120: [56, 612, 37, 421],
#         121: [429, 599, 66, 243],
#         122: [78, 458, 24, 349],
#         123: [25, 278, 24, 278],
#         124: [8, 585, 60, 450],
#         125: [372, 591, 41, 224],
#         126: [6, 242, 11, 269],
#         127: [41, 543, 18, 301],
#         128: [52, 415, 25, 369],
#         129: [231, 616, 34, 302],
#         130: [17, 278, 17, 324],
#         131: [358, 604, 67, 303],
#         132: [17, 583, 18, 415],
#         133: [24, 214, 23, 256],
#         134: [71, 228, 19, 228],
#         135: [43, 199, 28, 254],
#         136: [26, 481, 63, 332],
#         137: [157, 594, 9, 295],
#         138: [140, 529, 11, 399],
#         139: [52, 541, 9, 404],
#         140: [44, 572, 38, 361],
#         141: [344, 599, 11, 305],
#         142: [7, 549, 36, 440],
#         143: [64, 593, 32, 377],
#         144: [20, 270, 14, 254],
#         145: [39, 220, 32, 303],
#         146: [9, 552, 30, 344]
#     }
#     return crop_box.get(i_file, 'Invalid file index')
#
# ## compute whole slide heatmaps
#
# # put all the annotation files in a single list
# json_annotation_files = json_annotation_files_dict['sqwat'] + json_annotation_files_dict['gwat']
#
# # loop annotations files to compute the heatmaps
# for i_file, json_file in enumerate(json_annotation_files):
#
#     print('File ' + str(i_file) + '/' + str(len(json_annotation_files) - 1) + ': ' + json_file)
#
#     # name of corresponding .ndpi file
#     ndpi_file = json_file.replace('.json', '.ndpi')
#     kernel_file = os.path.splitext(ndpi_file)[0]
#
#     # add path to file
#     json_file = os.path.join(annotations_dir, json_file)
#     ndpi_file = os.path.join(histo_dir, ndpi_file)
#
#     # open full resolution histology slide
#     im = openslide.OpenSlide(ndpi_file)
#
#     # pixel size
#     assert (im.properties['tiff.ResolutionUnit'] == 'centimeter')
#     xres = 1e-2 / float(im.properties['tiff.XResolution']) * 1e6  # um^2
#     yres = 1e-2 / float(im.properties['tiff.YResolution']) * 1e6  # um^2
#
#     # name of file where rough mask was saved to
#     coarse_mask_file = os.path.basename(ndpi_file)
#     coarse_mask_file = coarse_mask_file.replace('.ndpi', '_coarse_mask.npz')
#     coarse_mask_file = os.path.join(annotations_dir, coarse_mask_file)
#
#     # load coarse tissue mask
#     with np.load(coarse_mask_file) as aux:
#         lores_istissue0 = aux['lores_istissue0']
#         im_downsampled = aux['im_downsampled']
#
#     if DEBUG:
#         plt.clf()
#         plt.subplot(211)
#         plt.imshow(im_downsampled)
#         plt.subplot(212)
#         plt.imshow(lores_istissue0)
#
#     # load list of contours in Corrected segmentations
#     json_file_corrected = os.path.join(annotations_dir, json_file.replace('.json', '_exp_0106_corrected.json'))
#
#     # list of items (there's a contour in each item)
#     contours_corrected = cytometer.data.aida_get_contours(json_file_corrected, layer_name='White adipocyte.*')
#
#     # init array for interpolated quantiles
#     quantiles_grid = np.zeros(shape=lores_istissue0.shape, dtype=np.float32)
#
#     # init array for mask where there are segmentations
#     areas_mask = Image.new("1", lores_istissue0.shape[::-1], "black")
#     draw = ImageDraw.Draw(areas_mask)
#
#     # init lists for contour centroids and areas
#     areas_all = []
#     centroids_all = []
#     centroids_down_all = []
#
#     # loop items (one contour per item)
#     for c in contours_corrected:
#
#         # convert to downsampled coordinates
#         c = np.array(c)
#         c_down = c / downsample_factor
#
#         if DEBUG:
#             plt.fill(c_down[:, 0], c_down[:, 1], fill=False, color='r')
#
#         # compute cell area
#         area = shapely.geometry.Polygon(c).area * xres * yres  # (um^2)
#         areas_all.append(area)
#
#         # compute centroid of contour
#         centroid = np.mean(c, axis=0)
#         centroids_all.append(centroid)
#         centroids_down_all.append(centroid / downsample_factor)
#
#         # add object described by contour to areas_mask
#         draw.polygon(list(c_down.flatten()), outline="white", fill="white")
#
#     # convert mask from RGBA to binary
#     areas_mask = np.array(areas_mask, dtype=np.bool)
#
#     areas_all = np.array(areas_all)
#
#     # interpolate scattered area data to regular grid
#     grid_row, grid_col = np.mgrid[0:areas_mask.shape[0], 0:areas_mask.shape[1]]
#     quantiles_grid = scipy.interpolate.griddata(centroids_down_all, areas_all, (grid_col, grid_row), method='linear', fill_value=0)
#     quantiles_grid[~areas_mask] = 0
#
#     if DEBUG:
#         plt.clf()
#         plt.subplot(211)
#         plt.imshow(im_downsampled)
#         plt.axis('off')
#         plt.subplot(212)
#         plt.imshow(quantiles_grid)
#
#     # get metainfo for this slide
#     df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(ndpi_file),
#                                                           values=[i_file,], values_tag='i',
#                                                           tags_to_keep=['id', 'ko_parent', 'sex'])
#
#     # mouse ID as a string
#     id = df_common['id'].values[0]
#     sex = df_common['sex'].values[0]
#     ko = df_common['ko_parent'].values[0]
#
#     # convert area values to quantiles
#     if sex == 'f':
#         quantiles_grid = f_area2quantile_f(quantiles_grid)
#     elif sex == 'm':
#         quantiles_grid = f_area2quantile_m(quantiles_grid)
#     else:
#         raise ValueError('Wrong sex value')
#
#     # make background white in the plot
#     quantiles_grid[~areas_mask] = np.nan
#
#     if DEBUG:
#         plt.clf()
#         plt.subplot(211)
#         plt.imshow(im_downsampled)
#         plt.axis('off')
#         plt.subplot(212)
#         # plt.imshow(areas_grid, vmin=0.0, vmax=1.0, cmap='gnuplot2')
#         plt.imshow(quantiles_grid, vmin=0.0, vmax=1.0, cmap=cm)
#         cbar = plt.colorbar(shrink=1.0)
#         cbar.ax.tick_params(labelsize=14)
#         cbar.ax.set_ylabel('Cell area quantile', rotation=90, fontsize=14)
#         plt.axis('off')
#         plt.tight_layout()
#
#     if DEBUG:
#         plt.clf()
#         plt.hist(areas_all, bins=50, density=True, histtype='step')
#
#     # plot cell areas for paper
#     plt.clf()
#     plt.imshow(quantiles_grid, vmin=0.0, vmax=1.0, cmap=cm)
#     plt.axis('off')
#     plt.tight_layout()
#
#     plt.savefig(os.path.join(figures_dir, kernel_file + '_exp_0110_cell_size_heatmap.png'),
#                 bbox_inches='tight')
#
#     # save heatmap at full size
#     quantiles_grid_im = Image.fromarray((cm(quantiles_grid) * 255).astype(np.uint8))
#     quantiles_grid_im.save(os.path.join(figures_dir, kernel_file + '_exp_0110_cell_size_heatmap_large.png'))
#
#
# ## crop heatmaps for paper
#
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
#     plt.clf()
#     plt.gcf().set_size_inches([12.8, 9.6])
#     plt.imshow(cropped_heatmap_large, interpolation='none')
#     plt.axis('off')
#     plt.tight_layout()
#
#     plt.savefig(os.path.join(figures_dir, kernel_file + '_exp_0110_cell_size_heatmap_large_cropped.png'),
#                 bbox_inches='tight')
#
# ## plot colourmaps and cell area density functions
#
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
#
# ## plot area distributions
#
# # load auxiliary file with all Corrected areas
# filename_corrected_areas = os.path.join(figures_dir, 'klf14_b6ntac_exp_0106_corrected_areas.npz')
# with np.load(filename_corrected_areas, allow_pickle=True) as aux:
#     areas_corrected_f = aux['areas_corrected_f']
#     areas_corrected_m = aux['areas_corrected_m']
#
# plt.close()
# plt.clf()
# areas = np.concatenate(areas_corrected_f)
# areas = areas[(areas >= min_cell_area_um2) & (areas <= max_cell_area_um2)]
# aq = stats.mstats.hdquantiles(areas, prob=np.linspace(0, 1, 11), axis=0)
# for a in aq:
#     plt.plot([a * 1e-3, a * 1e-3], [0, 0.335], 'k', linewidth=2, zorder=0)
# plt.hist(areas * 1e-3, histtype='step', bins=100, density=True, linewidth=4, zorder=1, color='r')
# plt.tick_params(labelsize=14)
# plt.title('Cell area density with decile divisions', fontsize=14)
# plt.xlabel('Cell area ($\cdot 10^3\ \mu m^2$)', fontsize=14)
# plt.ylabel('Density', fontsize=14)
# plt.yticks([])
# plt.tight_layout()
#
# plt.savefig(os.path.join(figures_dir, 'exp_0106_dist_quantiles_corrected_f.png'), bbox_inches='tight')
#
# plt.clf()
# areas = np.concatenate(areas_corrected_m)
# areas = areas[(areas >= min_cell_area_um2) & (areas <= max_cell_area_um2)]
# aq = stats.mstats.hdquantiles(areas, prob=np.linspace(0, 1, 11), axis=0)
# for a in aq:
#     plt.plot([a * 1e-3, a * 1e-3], [0, 0.242], 'k', linewidth=2, zorder=0)
# plt.hist(areas * 1e-3, histtype='step', bins=100, density=True, linewidth=4, zorder=1, color='r')
# plt.tick_params(labelsize=14)
# plt.title('Cell area density with decile divisions', fontsize=14)
# plt.xlabel('Cell area ($\cdot 10^3\ \mu m^2$)', fontsize=14)
# plt.ylabel('Density', fontsize=14)
# plt.yticks([])
# plt.tight_layout()
#
# plt.savefig(os.path.join(figures_dir, 'exp_0106_dist_quantiles_corrected_m.png'), bbox_inches='tight')