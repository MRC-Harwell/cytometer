"""
Generate figures for the DeepCytometer paper for v8 of the pipeline.

Environment: cytometer_tensorflow_v2.

We repeat the phenotyping from klf14_b6ntac_exp_0110_paper_figures_v8.py, but change the stratification of the data so
that we have Control (PATs + WT MATs) vs. Het MATs.

The comparisons we do are:
  * Control vs. MAT WT
  * MAT WT vs. MAT Het

This script partly deprecates klf14_b6ntac_exp_0099_paper_figures_v7.py:
* Figures have been updated to have v8 of the pipeline in the paper.

This script partly deprecates klf14_b6ntac_exp_0110_paper_figures_v8.py:
* We repeat the phenotyping, but change the stratification of the data so that we have Control (PATs + WT MATs) vs.
  Het MATs.

"""

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
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

# import pickle
from toolz import interleave
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import scipy
import scipy.stats as stats
# import skimage
import sklearn.neighbors, sklearn.model_selection
import statsmodels.api as sm
# import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import seaborn as sns
# import openslide
import PIL
# from PIL import Image, ImageDraw
import cytometer.data
import cytometer.stats
import shapely

# directories
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
hand_traced_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training_v2')
annotations_dir = os.path.join(home, 'Data/cytometer_data/aida_data_Klf14_v8/annotations')
histo_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
dataframe_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')
paper_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')
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


## BW ~ functional_ko
########################################################################################################################

# BW ~ functional_ko for female/male
bw_model_f = sm.OLS.from_formula('BW ~ C(functional_ko)', data=metainfo_f).fit()
bw_model_m = sm.OLS.from_formula('BW ~ C(functional_ko)', data=metainfo_m).fit()

print(bw_model_f.summary())
print(bw_model_m.summary())

extra_tests_f = bw_model_f.t_test('Intercept + C(functional_ko)[T.MAT_WT], Intercept + C(functional_ko)[T.FKO]')
extra_tests_m = bw_model_m.t_test('Intercept + C(functional_ko)[T.MAT_WT], Intercept + C(functional_ko)[T.FKO]')

# mean BW
bwmean_control_f = np.mean(metainfo_f[metainfo_f['ko_parent'] == 'PAT']['BW'])
bwmean_matwt_f = np.mean(metainfo_f[(metainfo_f['ko_parent'] == 'MAT') & (metainfo_f['genotype'] == 'KLF14-KO:WT')]['BW'])
bwmean_fko_f = np.mean(metainfo_f[(metainfo_f['ko_parent'] == 'MAT') & (metainfo_f['genotype'] == 'KLF14-KO:Het')]['BW'])

bwmean_control_m = np.mean(metainfo_m[metainfo_m['ko_parent'] == 'PAT']['BW'])
bwmean_matwt_m = np.mean(metainfo_m[(metainfo_m['ko_parent'] == 'MAT') & (metainfo_m['genotype'] == 'KLF14-KO:WT')]['BW'])
bwmean_fko_m = np.mean(metainfo_m[(metainfo_m['ko_parent'] == 'MAT') & (metainfo_m['genotype'] == 'KLF14-KO:Het')]['BW'])

# Tukey HSD
multicomp_f = sm.stats.multicomp.MultiComparison(metainfo_f['BW'], metainfo_f['functional_ko'])
tukeyhsd_f = multicomp_f.tukeyhsd()
tukeyhsd_f = pd.DataFrame(data=tukeyhsd_f._results_table.data[1:], columns=tukeyhsd_f._results_table.data[0])
print(tukeyhsd_f)

multicomp_m = sm.stats.multicomp.MultiComparison(metainfo_m['BW'], metainfo_m['functional_ko'])
tukeyhsd_m = multicomp_m.tukeyhsd()
tukeyhsd_m = pd.DataFrame(data=tukeyhsd_m._results_table.data[1:], columns=tukeyhsd_m._results_table.data[0])
print(tukeyhsd_m)

if SAVE_FIGS:
    plt.clf()
    plt.gcf().set_size_inches([5.48, 4.8 ])

    ax = sns.swarmplot(x='sex', y='BW', hue='functional_ko', data=metainfo, dodge=True, palette=['C2', 'C3', 'C4'])
    plt.xlabel('')
    plt.ylabel('Body weight (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.xticks([0, 1], labels=['Female', 'Male'])
    ax.get_legend().set_title('')
    ax.legend(['Control (PAT)', 'MAT WT', 'FKO (MAT Het)'], loc='lower right', fontsize=12)

    # mean values
    plt.plot([-0.35, -0.15], [bwmean_control_f,]*2, 'k', linewidth=2)
    plt.plot([-0.10,  0.10], [bwmean_matwt_f,]*2, 'k', linewidth=2)
    plt.plot([ 0.17,  0.35], [bwmean_fko_f,]*2, 'k', linewidth=2)

    plt.plot([ 0.65,  0.85], [bwmean_control_m,]*2, 'k', linewidth=2)
    plt.plot([ 0.90,  1.10], [bwmean_matwt_m,]*2, 'k', linewidth=2)
    plt.plot([ 1.17,  1.35], [bwmean_fko_m,]*2, 'k', linewidth=2)

    # female
    plt.plot([-0.3, -0.3, 0.0, 0.0], [42, 44, 44, 42], 'k', lw=1.5)
    idx = (tukeyhsd_f['group1'] == 'Control') & (tukeyhsd_f['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_f.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.3f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(-0.15, 44.5, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([0.0, 0.0, 0.3, 0.3], [47, 49, 49, 47], 'k', lw=1.5)
    idx = (tukeyhsd_f['group1'] == 'FKO') & (tukeyhsd_f['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_f.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(0.15, 49.5, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([-0.3, -0.3, 0.3, 0.3], [52, 54, 54, 52], 'k', lw=1.5)
    idx = (tukeyhsd_f['group1'] == 'Control') & (tukeyhsd_f['group2'] == 'FKO')
    pval = list(tukeyhsd_f.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(0.0, 54.5, pval_text, ha='center', va='bottom', fontsize=14)

    # male
    plt.plot([1.0, 1.0, 1.3, 1.3], [47.5, 49.5, 49.5, 47.5], 'k', lw=1.5)
    idx = (tukeyhsd_m['group1'] == 'FKO') & (tukeyhsd_m['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_m.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(1.15, 50, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([0.7, 0.7, 1.0, 1.0], [52.5, 54.5, 54.5, 52.5], 'k', lw=1.5)
    idx = (tukeyhsd_m['group1'] == 'Control') & (tukeyhsd_m['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_m.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(0.85, 55, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([0.7, 0.7, 1.3, 1.3], [57.5, 59.5, 59.5, 57.5], 'k', lw=1.5)
    idx = (tukeyhsd_m['group1'] == 'Control') & (tukeyhsd_m['group2'] == 'FKO')
    pval = list(tukeyhsd_m.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(1.00, 60, pval_text, ha='center', va='bottom', fontsize=14)

    plt.ylim(18, 65)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_swarm_bw_fko.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_swarm_bw_fko.svg'))

## DW ~ functional_ko
########################################################################################################################

# mean DW Gonadal
dwmean_control_f_gwat = np.mean(metainfo_f[metainfo_f['ko_parent'] == 'PAT']['gWAT'])
dwmean_matwt_f_gwat = np.mean(metainfo_f[(metainfo_f['ko_parent'] == 'MAT') & (metainfo_f['genotype'] == 'KLF14-KO:WT')]['gWAT'])
dwmean_fko_f_gwat = np.mean(metainfo_f[(metainfo_f['ko_parent'] == 'MAT') & (metainfo_f['genotype'] == 'KLF14-KO:Het')]['gWAT'])

dwmean_control_m_gwat = np.mean(metainfo_m[metainfo_m['ko_parent'] == 'PAT']['gWAT'])
dwmean_matwt_m_gwat = np.mean(metainfo_m[(metainfo_m['ko_parent'] == 'MAT') & (metainfo_m['genotype'] == 'KLF14-KO:WT')]['gWAT'])
dwmean_fko_m_gwat = np.mean(metainfo_m[(metainfo_m['ko_parent'] == 'MAT') & (metainfo_m['genotype'] == 'KLF14-KO:Het')]['gWAT'])

# Tukey HSD for gWAT ~ functional_ko
multicomp_f = sm.stats.multicomp.MultiComparison(metainfo_f['gWAT'], metainfo_f['functional_ko'])
tukeyhsd_f = multicomp_f.tukeyhsd()
tukeyhsd_f = pd.DataFrame(data=tukeyhsd_f._results_table.data[1:], columns=tukeyhsd_f._results_table.data[0])
print(tukeyhsd_f)

multicomp_m = sm.stats.multicomp.MultiComparison(metainfo_m['gWAT'], metainfo_m['functional_ko'])
tukeyhsd_m = multicomp_m.tukeyhsd()
tukeyhsd_m = pd.DataFrame(data=tukeyhsd_m._results_table.data[1:], columns=tukeyhsd_m._results_table.data[0])
print(tukeyhsd_m)

if SAVE_FIGS:
    plt.clf()
    plt.gcf().set_size_inches([5.48, 4.8 ])

    ax = sns.swarmplot(x='sex', y='gWAT', hue='functional_ko', data=metainfo, dodge=True, palette=['C2', 'C3', 'C4'])
    plt.xlabel('')
    plt.ylabel('Gonadal depot weight (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.xticks([0, 1], labels=['Female', 'Male'])
    ax.get_legend().set_title('')
    ax.legend([])
    # ax.legend(['Control (PAT)', 'MAT WT', 'FKO (MAT Het)'], loc='lower right', fontsize=12)

    # mean values
    plt.plot([-0.35, -0.15], [dwmean_control_f_gwat,]*2, 'k', linewidth=2)
    plt.plot([-0.10,  0.10], [dwmean_matwt_f_gwat,]*2, 'k', linewidth=2)
    plt.plot([ 0.17,  0.35], [dwmean_fko_f_gwat,]*2, 'k', linewidth=2)

    plt.plot([ 0.65,  0.85], [dwmean_control_m_gwat,]*2, 'k', linewidth=2)
    plt.plot([ 0.90,  1.10], [dwmean_matwt_m_gwat,]*2, 'k', linewidth=2)
    plt.plot([ 1.17,  1.35], [dwmean_fko_m_gwat,]*2, 'k', linewidth=2)

    # female
    plt.plot([-0.3, -0.3, 0.0, 0.0], [1.45, 1.55, 1.55, 1.45], 'k', lw=1.5)
    idx = (tukeyhsd_f['group1'] == 'Control') & (tukeyhsd_f['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_f.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(-0.15, 1.56, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([0.0, 0.0, 0.3, 0.3], [1.75, 1.85, 1.85, 1.75], 'k', lw=1.5)
    idx = (tukeyhsd_f['group1'] == 'FKO') & (tukeyhsd_f['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_f.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(0.15, 1.86, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([-0.3, -0.3, 0.3, 0.3], [2.05, 2.15, 2.15, 2.05], 'k', lw=1.5)
    idx = (tukeyhsd_f['group1'] == 'Control') & (tukeyhsd_f['group2'] == 'FKO')
    pval = list(tukeyhsd_f.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(0.0, 2.16, pval_text, ha='center', va='bottom', fontsize=14)

    # male
    plt.plot([1.0, 1.0, 1.3, 1.3], [1.75, 1.85, 1.85, 1.75], 'k', lw=1.5)
    idx = (tukeyhsd_m['group1'] == 'FKO') & (tukeyhsd_m['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_m.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(1.15, 1.86, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([0.7, 0.7, 1.0, 1.0], [2.05, 2.15, 2.15, 2.05], 'k', lw=1.5)
    idx = (tukeyhsd_m['group1'] == 'Control') & (tukeyhsd_m['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_m.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(0.85, 2.16, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([0.7, 0.7, 1.3, 1.3], [2.35, 2.45, 2.45, 2.35], 'k', lw=1.5)
    idx = (tukeyhsd_m['group1'] == 'Control') & (tukeyhsd_m['group2'] == 'FKO')
    pval = list(tukeyhsd_m.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(1.00, 2.46, pval_text, ha='center', va='bottom', fontsize=14)

    plt.ylim(0, 2.7)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_swarm_gwat_fko.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_swarm_gwat_fko.svg'))

# mean DW Subcut.
dwmean_control_f_sqwat = np.mean(metainfo_f[metainfo_f['ko_parent'] == 'PAT']['SC'])
dwmean_matwt_f_sqwat = np.mean(metainfo_f[(metainfo_f['ko_parent'] == 'MAT') & (metainfo_f['genotype'] == 'KLF14-KO:WT')]['SC'])
dwmean_fko_f_sqwat = np.mean(metainfo_f[(metainfo_f['ko_parent'] == 'MAT') & (metainfo_f['genotype'] == 'KLF14-KO:Het')]['SC'])

dwmean_control_m_sqwat = np.mean(metainfo_m[metainfo_m['ko_parent'] == 'PAT']['SC'])
dwmean_matwt_m_sqwat = np.mean(metainfo_m[(metainfo_m['ko_parent'] == 'MAT') & (metainfo_m['genotype'] == 'KLF14-KO:WT')]['SC'])
dwmean_fko_m_sqwat = np.mean(metainfo_m[(metainfo_m['ko_parent'] == 'MAT') & (metainfo_m['genotype'] == 'KLF14-KO:Het')]['SC'])

# Tukey HSD for SC ~ functional_ko
multicomp_f = sm.stats.multicomp.MultiComparison(metainfo_f['SC'], metainfo_f['functional_ko'])
tukeyhsd_f = multicomp_f.tukeyhsd()
tukeyhsd_f = pd.DataFrame(data=tukeyhsd_f._results_table.data[1:], columns=tukeyhsd_f._results_table.data[0])
print(tukeyhsd_f)

multicomp_m = sm.stats.multicomp.MultiComparison(metainfo_m['SC'], metainfo_m['functional_ko'])
tukeyhsd_m = multicomp_m.tukeyhsd()
tukeyhsd_m = pd.DataFrame(data=tukeyhsd_m._results_table.data[1:], columns=tukeyhsd_m._results_table.data[0])
print(tukeyhsd_m)

if SAVE_FIGS:
    plt.clf()
    plt.gcf().set_size_inches([5.48, 4.8 ])

    ax = sns.swarmplot(x='sex', y='SC', hue='functional_ko', data=metainfo, dodge=True, palette=['C2', 'C3', 'C4'])
    plt.xlabel('')
    plt.ylabel('Subcutaneous depot weight (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.xticks([0, 1], labels=['Female', 'Male'])
    ax.get_legend().set_title('')
    ax.legend([])
    # ax.legend(['Control (PAT)', 'MAT WT', 'FKO (MAT Het)'], loc='lower right', fontsize=12)

    # mean values
    plt.plot([-0.35, -0.15], [dwmean_control_f_sqwat,]*2, 'k', linewidth=2)
    plt.plot([-0.10,  0.10], [dwmean_matwt_f_sqwat,]*2, 'k', linewidth=2)
    plt.plot([ 0.17,  0.35], [dwmean_fko_f_sqwat,]*2, 'k', linewidth=2)

    plt.plot([ 0.65,  0.85], [dwmean_control_m_sqwat,]*2, 'k', linewidth=2)
    plt.plot([ 0.90,  1.10], [dwmean_matwt_m_sqwat,]*2, 'k', linewidth=2)
    plt.plot([ 1.17,  1.35], [dwmean_fko_m_sqwat,]*2, 'k', linewidth=2)

    # female
    plt.plot([0.0, 0.0, 0.3, 0.3], [1.05, 1.15, 1.15, 1.05], 'k', lw=1.5)
    idx = (tukeyhsd_f['group1'] == 'FKO') & (tukeyhsd_f['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_f.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(0.15, 1.16, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([-0.3, -0.3, 0.0, 0.0], [1.65, 1.75, 1.75, 1.65], 'k', lw=1.5)
    idx = (tukeyhsd_f['group1'] == 'Control') & (tukeyhsd_f['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_f.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(-0.15, 1.76, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([-0.3, -0.3, 0.3, 0.3], [1.95, 2.05, 2.05, 1.95], 'k', lw=1.5)
    idx = (tukeyhsd_f['group1'] == 'Control') & (tukeyhsd_f['group2'] == 'FKO')
    pval = list(tukeyhsd_f.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(0.0, 2.06, pval_text, ha='center', va='bottom', fontsize=14)

    # male
    plt.plot([1.0, 1.0, 1.3, 1.3], [1.3, 1.4, 1.4, 1.3], 'k', lw=1.5)
    idx = (tukeyhsd_m['group1'] == 'FKO') & (tukeyhsd_m['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_m.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(1.15, 1.41, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([0.7, 0.7, 1.0, 1.0], [1.6, 1.7, 1.7, 1.6], 'k', lw=1.5)
    idx = (tukeyhsd_m['group1'] == 'Control') & (tukeyhsd_m['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_m.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(0.85, 1.71, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([0.7, 0.7, 1.3, 1.3], [1.9, 2.0, 2.0, 1.9], 'k', lw=1.5)
    idx = (tukeyhsd_m['group1'] == 'Control') & (tukeyhsd_m['group2'] == 'FKO')
    pval = list(tukeyhsd_m.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(1.00, 2.01, pval_text, ha='center', va='bottom', fontsize=14)

    plt.ylim(0, 2.3)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_swarm_sqwat_fko.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_swarm_sqwat_fko.svg'))

print('mean DW')
print('\tfemale gonadal Control: ' + str(dwmean_control_f_gwat))
print('\tfemale gonadal MAT WT: ' + str(dwmean_matwt_f_gwat))
print('\tfemale gonadal FKO: ' + str(dwmean_fko_f_gwat))

print('\tfemale subcut. Control: ' + str(dwmean_control_f_sqwat))
print('\tfemale subcut. MAT WT: ' + str(dwmean_matwt_f_sqwat))
print('\tfemale subcut. FKO: ' + str(dwmean_fko_f_sqwat))

print('\tmale gonadal Control: ' + str(dwmean_control_m_gwat))
print('\tmale gonadal MAT WT: ' + str(dwmean_matwt_m_gwat))
print('\tmale gonadal FKO: ' + str(dwmean_fko_m_gwat))

print('\tmale subcut. Control: ' + str(dwmean_control_m_sqwat))
print('\tmale subcut. MAT WT: ' + str(dwmean_matwt_m_sqwat))
print('\tmale subcut. FKO: ' + str(dwmean_fko_m_sqwat))

## DW ~ BW * functional_ko
########################################################################################################################

# scale BW to avoid large condition numbers
BW_mean = metainfo['BW'].mean()
metainfo['BW__'] = metainfo['BW'] / BW_mean

# auxiliary variables to create the null models for the (Control vs. MAT WT) and (MAT WT vs. FKO) comparisons
metainfo['functional_ko_a'] = metainfo['functional_ko'].astype(
    pd.api.types.CategoricalDtype(categories=['Control_MAT_WT', 'FKO'], ordered=True))
metainfo.loc[metainfo['functional_ko'] != 'FKO', 'functional_ko_a'] = 'Control_MAT_WT'

metainfo['functional_ko_b'] = metainfo['functional_ko'].astype(
    pd.api.types.CategoricalDtype(categories=['Control', 'MAT_WT_FKO'], ordered=True))
metainfo.loc[metainfo['functional_ko'] != 'Control', 'functional_ko_b'] = 'MAT_WT_FKO'

metainfo['functional_ko_c'] = metainfo['functional_ko'].astype(
    pd.api.types.CategoricalDtype(categories=['Control_FKO', 'MAT_WT'], ordered=True))
metainfo.loc[metainfo['functional_ko'] != 'MAT_WT', 'functional_ko_c'] = 'Control_FKO'

# for convenience create two dataframes (female and male) with the data for the current depot
metainfo_f = metainfo[metainfo['sex'] == 'f']
metainfo_m = metainfo[metainfo['sex'] == 'm']

## depot ~ BW * kfo models

# global models fitted to 3 strata (Control, MAT WT and FKO):
# These are the models that we are going to use to test for correlation, apart from the LRTs
model_gwat_f_global = sm.OLS.from_formula('gWAT ~ BW__ * C(functional_ko)', data=metainfo_f).fit()
model_sqwat_f_global = sm.OLS.from_formula('SC ~ BW__ * C(functional_ko)', data=metainfo_f).fit()

model_gwat_m_global = sm.OLS.from_formula('gWAT ~ BW__ * C(functional_ko)', data=metainfo_m).fit()
model_sqwat_m_global = sm.OLS.from_formula('SC ~ BW__ * C(functional_ko)', data=metainfo_m).fit()

# models fitted to 2 strata (combining Control and MAT WT) to be used as null models
model_gwat_f_control_matwt = sm.OLS.from_formula('gWAT ~ BW__ * C(functional_ko_a)', data=metainfo_f).fit()
model_sqwat_f_control_matwt = sm.OLS.from_formula('SC ~ BW__ * C(functional_ko_a)', data=metainfo_f).fit()

model_gwat_m_control_matwt = sm.OLS.from_formula('gWAT ~ BW__ * C(functional_ko_a)', data=metainfo_m).fit()
model_sqwat_m_control_matwt = sm.OLS.from_formula('SC ~ BW__ * C(functional_ko_a)', data=metainfo_m).fit()

# models fitted to 2 strata (combining MAT WT and FKO) to be used as null models
model_gwat_f_matwt_fko = sm.OLS.from_formula('gWAT ~ BW__ * C(functional_ko_b)', data=metainfo_f).fit()
model_sqwat_f_matwt_fko = sm.OLS.from_formula('SC ~ BW__ * C(functional_ko_b)', data=metainfo_f).fit()

model_gwat_m_matwt_fko = sm.OLS.from_formula('gWAT ~ BW__ * C(functional_ko_b)', data=metainfo_m).fit()
model_sqwat_m_matwt_fko = sm.OLS.from_formula('SC ~ BW__ * C(functional_ko_b)', data=metainfo_m).fit()

# models fitted to 2 strata (combining Control and FKO) to be used as null models
model_gwat_f_control_fko = sm.OLS.from_formula('gWAT ~ BW__ * C(functional_ko_c)', data=metainfo_f).fit()
model_sqwat_f_control_fko = sm.OLS.from_formula('SC ~ BW__ * C(functional_ko_c)', data=metainfo_f).fit()

model_gwat_m_control_fko = sm.OLS.from_formula('gWAT ~ BW__ * C(functional_ko_c)', data=metainfo_m).fit()
model_sqwat_m_control_fko = sm.OLS.from_formula('SC ~ BW__ * C(functional_ko_c)', data=metainfo_m).fit()

# compute LRTs and extract p-values and LRs
lrt = pd.DataFrame(columns=['lr', 'pval', 'pval_ast'])

lr, pval = cytometer.stats.lrtest(model_gwat_f_control_matwt.llf, model_gwat_f_global.llf)
lrt.loc['model_gwat_f_control_matwt', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(model_sqwat_f_control_matwt.llf, model_sqwat_f_global.llf)
lrt.loc['model_sqwat_f_control_matwt', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(model_gwat_m_control_matwt.llf, model_gwat_m_global.llf)
lrt.loc['model_gwat_m_control_matwt', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(model_sqwat_m_control_matwt.llf, model_sqwat_m_global.llf)
lrt.loc['model_sqwat_m_control_matwt', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(model_gwat_f_matwt_fko.llf, model_gwat_f_global.llf)
lrt.loc['model_gwat_f_matwt_fko', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(model_sqwat_f_matwt_fko.llf, model_sqwat_f_global.llf)
lrt.loc['model_sqwat_f_matwt_fko', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(model_gwat_m_matwt_fko.llf, model_gwat_m_global.llf)
lrt.loc['model_gwat_m_matwt_fko', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(model_sqwat_m_matwt_fko.llf, model_sqwat_m_global.llf)
lrt.loc['model_sqwat_m_matwt_fko', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(model_gwat_f_control_fko.llf, model_gwat_f_global.llf)
lrt.loc['model_gwat_f_control_fko', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(model_sqwat_f_control_fko.llf, model_sqwat_f_global.llf)
lrt.loc['model_sqwat_f_control_fko', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(model_gwat_m_control_fko.llf, model_gwat_m_global.llf)
lrt.loc['model_gwat_m_control_fko', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(model_sqwat_m_control_fko.llf, model_sqwat_m_global.llf)
lrt.loc['model_sqwat_m_control_fko', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))

# multitest correction using Benjamini-Krieger-Yekutieli
_, lrt['pval_adj'], _, _ = multipletests(lrt['pval'], method='fdr_tsbky', alpha=0.05, returnsorted=False)
lrt['pval_adj_ast'] = cytometer.stats.pval_to_asterisk(lrt['pval_adj'])

if SAVE_FIGS:
    lrt.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_depot_weight_models_lrt_fko.csv'), na_rep='nan')

# Likelihood ratio tests: Control vs. MAT WT
print('Likelihood Ratio Tests: Control vs. MAT WT')

print('Female')
lr, pval, pval_ast, pval_adj, pval_adj_ast = lrt.loc['model_gwat_f_control_matwt', :]
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + pval_ast \
            + ', p-adj=' + '{0:.2g}'.format(pval_adj) + ' ' + pval_adj_ast
print('Gonadal: ' + pval_text)

lr, pval, pval_ast, pval_adj, pval_adj_ast = lrt.loc['model_sqwat_f_control_matwt', :]
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + pval_ast \
            + ', p-adj=' + '{0:.2g}'.format(pval_adj) + ' ' + pval_adj_ast
print('Subcutaneous: ' + pval_text)

print('Male')
lr, pval, pval_ast, pval_adj, pval_adj_ast = lrt.loc['model_gwat_m_control_matwt', :]
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + pval_ast \
            + ', p-adj=' + '{0:.2g}'.format(pval_adj) + ' ' + pval_adj_ast
print('Gonadal: ' + pval_text)

lr, pval, pval_ast, pval_adj, pval_adj_ast = lrt.loc['model_sqwat_m_control_matwt', :]
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + pval_ast \
            + ', p-adj=' + '{0:.2g}'.format(pval_adj) + ' ' + pval_adj_ast
print('Subcutaneous: ' + pval_text)

# Likelihood ratio tests: MAT WT vs. FKO (MAT Het)
print('')
print('Likelihood Ratio Tests: MAT WT vs. FKO (MAT Het)')

print('Female')
lr, pval, pval_ast, pval_adj, pval_adj_ast = lrt.loc['model_gwat_f_matwt_fko', :]
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + pval_ast \
            + ', p-adj=' + '{0:.2g}'.format(pval_adj) + ' ' + pval_adj_ast
print('Gonadal: ' + pval_text)

lr, pval, pval_ast, pval_adj, pval_adj_ast = lrt.loc['model_sqwat_f_matwt_fko', :]
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + pval_ast \
            + ', p-adj=' + '{0:.2g}'.format(pval_adj) + ' ' + pval_adj_ast
print('Subcutaneous: ' + pval_text)

print('Male')
lr, pval, pval_ast, pval_adj, pval_adj_ast = lrt.loc['model_gwat_m_matwt_fko', :]
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + pval_ast \
            + ', p-adj=' + '{0:.2g}'.format(pval_adj) + ' ' + pval_adj_ast
print('Gonadal: ' + pval_text)

lr, pval, pval_ast, pval_adj, pval_adj_ast = lrt.loc['model_sqwat_m_matwt_fko', :]
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + pval_ast \
            + ', p-adj=' + '{0:.2g}'.format(pval_adj) + ' ' + pval_adj_ast
print('Subcutaneous: ' + pval_text)

# Likelihood ratio tests: Control vs. FKO (MAT Het)
print('')
print('Likelihood Ratio Tests: Control vs. FKO (MAT Het)')

print('Female')
lr, pval, pval_ast, pval_adj, pval_adj_ast = lrt.loc['model_gwat_f_control_fko', :]
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + pval_ast \
            + ', p-adj=' + '{0:.2g}'.format(pval_adj) + ' ' + pval_adj_ast
print('Gonadal: ' + pval_text)

lr, pval, pval_ast, pval_adj, pval_adj_ast = lrt.loc['model_sqwat_f_control_fko', :]
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + pval_ast \
            + ', p-adj=' + '{0:.2g}'.format(pval_adj) + ' ' + pval_adj_ast
print('Subcutaneous: ' + pval_text)

print('Male')
lr, pval, pval_ast, pval_adj, pval_adj_ast = lrt.loc['model_gwat_m_control_fko', :]
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + pval_ast \
            + ', p-adj=' + '{0:.2g}'.format(pval_adj) + ' ' + pval_adj_ast
print('Gonadal: ' + pval_text)

lr, pval, pval_ast, pval_adj, pval_adj_ast = lrt.loc['model_sqwat_m_control_fko', :]
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + pval_ast \
            + ', p-adj=' + '{0:.2g}'.format(pval_adj) + ' ' + pval_adj_ast
print('Subcutaneous: ' + pval_text)

# extract coefficients, errors and p-values from models
model_names = ['model_gwat_f_global', 'model_sqwat_f_global', 'model_gwat_m_global', 'model_sqwat_m_global']
extra_hypotheses='Intercept+C(functional_ko)[T.MAT_WT],Intercept+C(functional_ko)[T.FKO]'\
                 + ',BW__+BW__:C(functional_ko)[T.MAT_WT],BW__+BW__:C(functional_ko)[T.FKO]'

df_coeff, df_ci_lo, df_ci_hi, df_pval = \
    cytometer.stats.models_coeff_ci_pval(
        [model_gwat_f_global, model_sqwat_f_global, model_gwat_m_global, model_sqwat_m_global],
        extra_hypotheses=extra_hypotheses,
        model_names=model_names)

# multitest correction using Benjamini-Krieger-Yekutieli
# we only need to correct the slopes' p-values, because we are not testing the values of the intercepts
col = ['BW__', 'BW__+BW__:C(functional_ko)[T.MAT_WT]', 'BW__+BW__:C(functional_ko)[T.FKO]']
df_corrected_pval = df_pval.copy()
_, aux, _, _ = multipletests(np.array(df_pval[col]).flatten(), method='fdr_tsbky', alpha=0.05, returnsorted=False)
df_corrected_pval[:] = np.nan
df_corrected_pval[col] = aux.reshape(df_corrected_pval[col].shape)

# convert p-values to asterisks
df_pval_ast = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval, brackets=False), columns=df_coeff.columns,
                           index=model_names)
df_corrected_pval_ast = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_corrected_pval, brackets=False),
                                     columns=df_coeff.columns, index=model_names)

if SAVE_FIGS:
    df_concat = pd.concat([df_coeff, df_ci_lo, df_ci_hi, df_pval, df_pval_ast, df_corrected_pval, df_corrected_pval_ast], axis=1)
    idx = list(interleave(np.array_split(range(df_concat.shape[1]), 7)))
    df_concat = df_concat.iloc[:, idx]
    df_concat.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_depot_weight_models_coeffs_pvals_fko.csv'), na_rep='nan')

if SAVE_FIGS:
    plt.clf()
    plt.subplot(221)
    sex = 'f'
    cytometer.stats.plot_linear_regression(model_gwat_f_global, metainfo_f, 'BW__',
                                           other_vars={'sex':sex, 'functional_ko':'Control'},
                                           dep_var='gWAT', sx=BW_mean, c='C2', marker='x',
                                           line_label='Control')
    cytometer.stats.plot_linear_regression(model_gwat_f_global, metainfo_f, 'BW__',
                                           other_vars={'sex':sex, 'functional_ko':'MAT_WT'},
                                           dep_var='gWAT', sx=BW_mean, c='C3', marker='+',
                                           line_label='MAT WT')
    cytometer.stats.plot_linear_regression(model_gwat_f_global, metainfo_f, 'BW__',
                                           other_vars={'sex':sex, 'functional_ko':'FKO'},
                                           dep_var='gWAT', sx=BW_mean, c='C4', marker='o',
                                           line_label='FKO')
    plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    plt.ylim(0, 2.1)
    plt.tick_params(labelsize=14)
    plt.title('Female', fontsize=14)
    plt.ylabel('Gonadal\ndepot weight (g)', fontsize=14)

    plt.subplot(222)
    sex = 'm'
    cytometer.stats.plot_linear_regression(model_gwat_m_global, metainfo_m, 'BW__',
                                           other_vars={'sex'print('mean DW')
print('\tfemale gonadal Control: ' + str(dwmean_control_f_gwat))
print('\tfemale gonadal MAT WT: ' + str(dwmean_matwt_f_gwat))
print('\tfemale gonadal FKO: ' + str(dwmean_fko_f_gwat))

print('\tfemale subcut. Control: ' + str(dwmean_control_f_sqwat))
print('\tfemale subcut. MAT WT: ' + str(dwmean_matwt_f_sqwat))
print('\tfemale subcut. FKO: ' + str(dwmean_fko_f_sqwat))

print('\tmale gonadal Control: ' + str(dwmean_control_m_gwat))
print('\tmale gonadal MAT WT: ' + str(dwmean_matwt_m_gwat))
print('\tmale gonadal FKO: ' + str(dwmean_fko_m_gwat))

print('\tmale subcut. Control: ' + str(dwmean_control_m_sqwat))
print('\tmale subcut. MAT WT: ' + str(dwmean_matwt_m_sqwat))
print('\tmale subcut. FKO: ' + str(dwmean_fko_m_sqwat))

:sex, 'functional_ko':'Control'},
                                           dep_var='gWAT', sx=BW_mean, c='C2', marker='x',
                                           line_label='Control')
    cytometer.stats.plot_linear_regression(model_gwat_m_global, metainfo_m, 'BW__',
                                           other_vars={'sex':sex, 'functional_ko':'MAT_WT'},
                                           dep_var='gWAT', sx=BW_mean, c='C3', marker='+',
                                           line_label='MAT WT')
    cytometer.stats.plot_linear_regression(model_gwat_m_global, metainfo_m, 'BW__',
                                           other_vars={'sex':sex, 'functional_ko':'FKO'},
                                           dep_var='gWAT', sx=BW_mean, c='C4', marker='o',
                                           line_label='FKO')
    plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    plt.ylim(0, 2.1)
    plt.tick_params(labelsize=14)
    plt.title('Male', fontsize=14)

    plt.subplot(223)
    sex = 'f'
    cytometer.stats.plot_linear_regression(model_sqwat_f_global, metainfo_f, 'BW__',
                                           other_vars={'sex':sex, 'functional_ko':'Control'},
                                           dep_var='SC', sx=BW_mean, c='C2', marker='x',
                                           line_label='Control')
    cytometer.stats.plot_linear_regression(model_sqwat_f_global, metainfo_f, 'BW__',
                                           other_vars={'sex':sex, 'functional_ko':'MAT_WT'},
                                           dep_var='SC', sx=BW_mean, c='C3', marker='+',
                                           line_label='MAT WT')
    cytometer.stats.plot_linear_regression(model_sqwat_f_global, metainfo_f, 'BW__',
                                           other_vars={'sex':sex, 'functional_ko':'FKO'},
                                           dep_var='SC', sx=BW_mean, c='C4', marker='o',
                                           line_label='FKO')
    plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    plt.tick_params(labelsize=14)
    plt.ylim(0, 2.1)
    plt.xlabel('Body weight (g)', fontsize=14)
    plt.ylabel('Subcutaneous\ndepot weight (g)', fontsize=14)
    plt.legend(loc='upper right')

    plt.subplot(224)
    sex = 'm'
    cytometer.stats.plot_linear_regression(model_sqwat_m_global, metainfo_m, 'BW__',
                                           other_vars={'sex':sex, 'functional_ko':'Control'},
                                           dep_var='SC', sx=BW_mean, c='C2', marker='x',
                                           line_label='Control')
    cytometer.stats.plot_linear_regression(model_sqwat_m_global, metainfo_m, 'BW__',
                                           other_vars={'sex':sex, 'functional_ko':'MAT_WT'},
                                           dep_var='SC', sx=BW_mean, c='C3', marker='+',
                                           line_label='MAT WT')
    cytometer.stats.plot_linear_regression(model_sqwat_m_global, metainfo_m, 'BW__',
                                           other_vars={'sex':sex, 'functional_ko':'FKO'},
                                           dep_var='SC', sx=BW_mean, c='C4', marker='o',
                                           line_label='KFO')
    plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    plt.ylim(0, 2.1)
    plt.tick_params(labelsize=14)
    plt.xlabel('Body weight (g)', fontsize=14)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_depot_linear_model_fko.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_depot_linear_model_fko.jpg'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_depot_linear_model_fko.svg'))


########################################################################################################################
## Analyse cell populations from automatically segmented images in two depots: SQWAT and GWAT:
########################################################################################################################

## area_at_quantile ~ functional_ko
########################################################################################################################

# (only mode, 25%-, 50%- and 75%-quantiles for illustration purposes and debugging)
# 0.05, 0.1 , 0.15, 0.2, ..., 0.9 , 0.95
quantiles = np.linspace(0, 1, 21)  #

# indices of the quantiles we are going to model
i_quantiles = [5, 10, 15]  # Q1, Q2, Q3

# we are going to compare median values, like in Small et al.
i_q = i_quantiles[1]

# choose one area_at_quantile value as the output of the linear model
df_all['area_at_quantile'] = np.array(df_all['area_at_quantiles'].to_list())[:, i_q]
df_all['area_at_quantile_10e3'] = df_all['area_at_quantile'] * 1e-3

# for convenience create auxiliary dataframes
df_gwat = df_all[df_all['depot'] == 'gwat']
df_sqwat = df_all[df_all['depot'] == 'sqwat']
df_f_gwat = df_all[(df_all['sex'] == 'f') & (df_all['depot'] == 'gwat')]
df_m_gwat = df_all[(df_all['sex'] == 'm') & (df_all['depot'] == 'gwat')]
df_f_sqwat = df_all[(df_all['sex'] == 'f') & (df_all['depot'] == 'sqwat')]
df_m_sqwat = df_all[(df_all['sex'] == 'm') & (df_all['depot'] == 'sqwat')]

# mean areaq Gonadal
areaqmean_control_f_gwat = np.mean(df_f_gwat[df_f_gwat['ko_parent'] == 'PAT']['area_at_quantile_10e3'])
areaqmean_matwt_f_gwat = np.mean(df_f_gwat[(df_f_gwat['ko_parent'] == 'MAT') & (df_f_gwat['genotype'] == 'KLF14-KO:WT')]['area_at_quantile_10e3'])
areaqmean_fko_f_gwat = np.mean(df_f_gwat[(df_f_gwat['ko_parent'] == 'MAT') & (df_f_gwat['genotype'] == 'KLF14-KO:Het')]['area_at_quantile_10e3'])

areaqmean_control_m_gwat = np.mean(df_m_gwat[df_m_gwat['ko_parent'] == 'PAT']['area_at_quantile_10e3'])
areaqmean_matwt_m_gwat = np.mean(df_m_gwat[(df_m_gwat['ko_parent'] == 'MAT') & (df_m_gwat['genotype'] == 'KLF14-KO:WT')]['area_at_quantile_10e3'])
areaqmean_fko_m_gwat = np.mean(df_m_gwat[(df_m_gwat['ko_parent'] == 'MAT') & (df_m_gwat['genotype'] == 'KLF14-KO:Het')]['area_at_quantile_10e3'])

# mean areaq Subcut.
areaqmean_control_f_sqwat = np.mean(df_f_sqwat[df_f_sqwat['ko_parent'] == 'PAT']['area_at_quantile_10e3'])
areaqmean_matwt_f_sqwat = np.mean(df_f_sqwat[(df_f_sqwat['ko_parent'] == 'MAT') & (df_f_sqwat['genotype'] == 'KLF14-KO:WT')]['area_at_quantile_10e3'])
areaqmean_fko_f_sqwat = np.mean(df_f_sqwat[(df_f_sqwat['ko_parent'] == 'MAT') & (df_f_sqwat['genotype'] == 'KLF14-KO:Het')]['area_at_quantile_10e3'])

areaqmean_control_m_sqwat = np.mean(df_m_sqwat[df_m_sqwat['ko_parent'] == 'PAT']['area_at_quantile_10e3'])
areaqmean_matwt_m_sqwat = np.mean(df_m_sqwat[(df_m_sqwat['ko_parent'] == 'MAT') & (df_m_sqwat['genotype'] == 'KLF14-KO:WT')]['area_at_quantile_10e3'])
areaqmean_fko_m_sqwat = np.mean(df_m_sqwat[(df_m_sqwat['ko_parent'] == 'MAT') & (df_m_sqwat['genotype'] == 'KLF14-KO:Het')]['area_at_quantile_10e3'])

# Tukey HSD for area_at_quantile ~ functional_ko
multicomp_gwat_f = sm.stats.multicomp.MultiComparison(df_f_gwat['area_at_quantile_10e3'], df_f_gwat['functional_ko'])
tukeyhsd_gwat_f = multicomp_gwat_f.tukeyhsd()
tukeyhsd_gwat_f = pd.DataFrame(data=tukeyhsd_gwat_f._results_table.data[1:], columns=tukeyhsd_gwat_f._results_table.data[0])
print(tukeyhsd_gwat_f)

multicomp_gwat_m = sm.stats.multicomp.MultiComparison(df_m_gwat['area_at_quantile_10e3'], df_m_gwat['functional_ko'])
tukeyhsd_gwat_m = multicomp_gwat_m.tukeyhsd()
tukeyhsd_gwat_m = pd.DataFrame(data=tukeyhsd_gwat_m._results_table.data[1:], columns=tukeyhsd_gwat_m._results_table.data[0])
print(tukeyhsd_gwat_m)

multicomp_sqwat_f = sm.stats.multicomp.MultiComparison(df_f_sqwat['area_at_quantile_10e3'], df_f_sqwat['functional_ko'])
tukeyhsd_sqwat_f = multicomp_sqwat_f.tukeyhsd()
tukeyhsd_sqwat_f = pd.DataFrame(data=tukeyhsd_sqwat_f._results_table.data[1:], columns=tukeyhsd_sqwat_f._results_table.data[0])
print(tukeyhsd_sqwat_f)

multicomp_sqwat_m = sm.stats.multicomp.MultiComparison(df_m_sqwat['area_at_quantile_10e3'], df_m_sqwat['functional_ko'])
tukeyhsd_sqwat_m = multicomp_sqwat_m.tukeyhsd()
tukeyhsd_sqwat_m = pd.DataFrame(data=tukeyhsd_sqwat_m._results_table.data[1:], columns=tukeyhsd_sqwat_m._results_table.data[0])
print(tukeyhsd_sqwat_m)

if SAVE_FIGS:
    plt.clf()
    plt.gcf().set_size_inches([5.48, 4.8 ])

    ax = sns.swarmplot(x='sex', y='area_at_quantile_10e3', hue='functional_ko', data=df_gwat, dodge=True, palette=['C2', 'C3', 'C4'])
    plt.xlabel('')
    plt.ylabel('Area$_{\mathrm{Q2}}$ ($10^3 \ \mu m^2$)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.xticks([0, 1], labels=['Female', 'Male'])
    ax.get_legend().set_title('')
    ax.legend(['Control (PAT)', 'MAT WT', 'FKO (MAT Het)'], loc='lower right', fontsize=12)

    # mean values
    plt.plot([-0.35, -0.15], [areaqmean_control_f_gwat,]*2, 'k', linewidth=2)
    plt.plot([-0.10,  0.10], [areaqmean_matwt_f_gwat,]*2, 'k', linewidth=2)
    plt.plot([ 0.17,  0.35], [areaqmean_fko_f_gwat,]*2, 'k', linewidth=2)

    plt.plot([ 0.65,  0.85], [areaqmean_control_m_gwat,]*2, 'k', linewidth=2)
    plt.plot([ 0.90,  1.10], [areaqmean_matwt_m_gwat,]*2, 'k', linewidth=2)
    plt.plot([ 1.17,  1.35], [areaqmean_fko_m_gwat,]*2, 'k', linewidth=2)

    # female
    plt.plot([-0.3, -0.3, 0.0, 0.0], [7.350, 7.550, 7.550, 7.350], 'k', lw=1.5)
    idx = (tukeyhsd_gwat_f['group1'] == 'Control') & (tukeyhsd_gwat_f['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_gwat_f.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.3f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(-0.15, 7.600, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([0.0, 0.0, 0.3, 0.3], [8.050, 8.250, 8.250, 8.050], 'k', lw=1.5)
    idx = (tukeyhsd_gwat_f['group1'] == 'FKO') & (tukeyhsd_gwat_f['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_gwat_f.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(0.15, 8.300, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([-0.3, -0.3, 0.3, 0.3], [8.750, 8.950, 8.950, 8.750], 'k', lw=1.5)
    idx = (tukeyhsd_gwat_f['group1'] == 'Control') & (tukeyhsd_gwat_f['group2'] == 'FKO')
    pval = list(tukeyhsd_gwat_f.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(0.0, 9.000, pval_text, ha='center', va='bottom', fontsize=14)

    # male
    plt.plot([0.7, 0.7, 1.0, 1.0], [7.700, 7.900, 7.900, 7.700], 'k', lw=1.5)
    idx = (tukeyhsd_gwat_m['group1'] == 'Control') & (tukeyhsd_gwat_m['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_gwat_m.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(0.85, 7.950, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([1.0, 1.0, 1.3, 1.3], [8.400, 8.600, 8.600, 8.400], 'k', lw=1.5)
    idx = (tukeyhsd_gwat_m['group1'] == 'FKO') & (tukeyhsd_gwat_m['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_gwat_m.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(1.15, 8.650, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([0.7, 0.7, 1.3, 1.3], [9.100, 9.300, 9.300, 9.100], 'k', lw=1.5)
    idx = (tukeyhsd_gwat_m['group1'] == 'Control') & (tukeyhsd_gwat_m['group2'] == 'FKO')
    pval = list(tukeyhsd_gwat_m.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(1.00, 9.350, pval_text, ha='center', va='bottom', fontsize=14)

    plt.ylim(1.000, 10.500)
    plt.title('Gonadal', fontsize=14)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_swarm_areaq_fko_gwat.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_swarm_areaq_fko_gwat.svg'))

if SAVE_FIGS:
    plt.clf()
    plt.gcf().set_size_inches([5.48, 4.8 ])

    ax = sns.swarmplot(x='sex', y='area_at_quantile_10e3', hue='functional_ko', data=df_sqwat, dodge=True, palette=['C2', 'C3', 'C4'])
    plt.xlabel('')
    plt.ylabel('Area$_{\mathrm{Q2}}$ ($10^3 \ \mu m^2$)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.xticks([0, 1], labels=['Female', 'Male'])
    ax.get_legend().set_title('')
    ax.get_legend().remove()

    # mean values
    plt.plot([-0.35, -0.15], [areaqmean_control_f_sqwat,]*2, 'k', linewidth=2)
    plt.plot([-0.10,  0.10], [areaqmean_matwt_f_sqwat,]*2, 'k', linewidth=2)
    plt.plot([ 0.17,  0.35], [areaqmean_fko_f_sqwat,]*2, 'k', linewidth=2)

    plt.plot([ 0.65,  0.85], [areaqmean_control_m_sqwat,]*2, 'k', linewidth=2)
    plt.plot([ 0.90,  1.10], [areaqmean_matwt_m_sqwat,]*2, 'k', linewidth=2)
    plt.plot([ 1.17,  1.35], [areaqmean_fko_m_sqwat,]*2, 'k', linewidth=2)

    # female
    plt.plot([-0.3, -0.3, 0.0, 0.0], [5.4, 5.6, 5.6, 5.4], 'k', lw=1.5)
    idx = (tukeyhsd_sqwat_f['group1'] == 'Control') & (tukeyhsd_sqwat_f['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_sqwat_f.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(-0.15, 5.65, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([0.0, 0.0, 0.3, 0.3], [6.1, 6.3, 6.3, 6.1], 'k', lw=1.5)
    idx = (tukeyhsd_sqwat_f['group1'] == 'FKO') & (tukeyhsd_sqwat_f['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_sqwat_f.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(0.15, 6.35, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([-0.3, -0.3, 0.3, 0.3], [6.8, 7.0, 7.0, 6.8], 'k', lw=1.5)
    idx = (tukeyhsd_sqwat_f['group1'] == 'Control') & (tukeyhsd_sqwat_f['group2'] == 'FKO')
    pval = list(tukeyhsd_sqwat_f.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(0.0, 7.05, pval_text, ha='center', va='bottom', fontsize=14)

    # male
    plt.plot([0.7, 0.7, 1.0, 1.0], [5.15, 5.35, 5.35, 5.15], 'k', lw=1.5)
    idx = (tukeyhsd_sqwat_m['group1'] == 'Control') & (tukeyhsd_sqwat_m['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_sqwat_m.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(0.85, 5.4, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([1.0, 1.0, 1.3, 1.3], [5.85, 6.05, 6.05, 5.85], 'k', lw=1.5)
    idx = (tukeyhsd_sqwat_m['group1'] == 'FKO') & (tukeyhsd_sqwat_m['group2'] == 'MAT_WT')
    pval = list(tukeyhsd_sqwat_m.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(1.15, 6.1, pval_text, ha='center', va='bottom', fontsize=14)

    plt.plot([0.7, 0.7, 1.3, 1.3], [6.55, 6.75, 6.75, 6.55], 'k', lw=1.5)
    idx = (tukeyhsd_sqwat_m['group1'] == 'Control') & (tukeyhsd_sqwat_m['group2'] == 'FKO')
    pval = list(tukeyhsd_sqwat_m.loc[idx, 'p-adj'])[0]
    pval_text = '{0:.2f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(1.00, 6.8, pval_text, ha='center', va='bottom', fontsize=14)

    plt.ylim(1.000, 10.500)
    plt.title('Subcutaneous', fontsize=14)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_swarm_areaq_fko_sqwat.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_swarm_areaq_fko_sqwat.svg'))

print('mean areaq')
print('\tfemale gonadal Control: ' + str(areaqmean_control_f_gwat))
print('\tfemale gonadal MAT WT: ' + str(areaqmean_matwt_f_gwat))
print('\tfemale gonadal FKO: ' + str(areaqmean_fko_f_gwat))

print('\tfemale subcut. Control: ' + str(areaqmean_control_f_sqwat))
print('\tfemale subcut. MAT WT: ' + str(areaqmean_matwt_f_sqwat))
print('\tfemale subcut. FKO: ' + str(areaqmean_fko_f_sqwat))

print('\tmale gonadal Control: ' + str(areaqmean_control_m_gwat))
print('\tmale gonadal MAT WT: ' + str(areaqmean_matwt_m_gwat))
print('\tmale gonadal FKO: ' + str(areaqmean_fko_m_gwat))

print('\tmale subcut. Control: ' + str(areaqmean_control_m_sqwat))
print('\tmale subcut. MAT WT: ' + str(areaqmean_matwt_m_sqwat))
print('\tmale subcut. FKO: ' + str(areaqmean_fko_m_sqwat))

## one data point per animal
## linear regression analysis of quantile_area ~ DW * functional_ko
## USED IN PAPER
########################################################################################################################

## (only mode, 25%-, 50%- and 75%-quantiles for illustration purposes and debugging)

# 0.05, 0.1 , 0.15, 0.2, ..., 0.9 , 0.95
quantiles = np.linspace(0, 1, 21)  #

# indices of the quantiles we are going to model
i_quantiles = [5, 10, 15]  # Q1, Q2, Q3

# auxiliary variables for LRT null-models
df_all['functional_ko_a'] = df_all['functional_ko'].astype(
    pd.api.types.CategoricalDtype(categories=['Control_MAT_WT', 'FKO'], ordered=True))
df_all.loc[df_all['functional_ko'] != 'FKO', 'functional_ko_a'] = 'Control_MAT_WT'

df_all['functional_ko_b'] = df_all['functional_ko'].astype(
    pd.api.types.CategoricalDtype(categories=['Control', 'MAT_WT_FKO'], ordered=True))
df_all.loc[df_all['functional_ko'] != 'Control', 'functional_ko_b'] = 'MAT_WT_FKO'

df_all['functional_ko_c'] = df_all['functional_ko'].astype(
    pd.api.types.CategoricalDtype(categories=['Control_FKO', 'MAT_WT'], ordered=True))
df_all.loc[df_all['functional_ko'] != 'MAT_WT', 'functional_ko_c'] = 'Control_FKO'

# fit linear models to area quantiles
models_gwat_f_global = []
models_gwat_m_global = []
models_sqwat_f_global = []
models_sqwat_m_global = []
models_gwat_f_control_matwt = []
models_gwat_m_control_matwt = []
models_sqwat_f_control_matwt = []
models_sqwat_m_control_matwt = []
models_gwat_f_matwt_fko = []
models_gwat_m_matwt_fko = []
models_sqwat_f_matwt_fko = []
models_sqwat_m_matwt_fko = []
models_gwat_f_control_fko = []
models_gwat_m_control_fko = []
models_sqwat_f_control_fko = []
models_sqwat_m_control_fko = []
for i_q in i_quantiles:

    # choose one area_at_quantile value as the output of the linear model
    df_all['area_at_quantile'] = np.array(df_all['area_at_quantiles'].to_list())[:, i_q]

    # for convenience create two dataframes (female and male) with the data for the current depot
    df_f_gwat = df_all[(df_all['sex'] == 'f') & (df_all['depot'] == 'gwat')]
    df_m_gwat = df_all[(df_all['sex'] == 'm') & (df_all['depot'] == 'gwat')]
    df_f_sqwat = df_all[(df_all['sex'] == 'f') & (df_all['depot'] == 'sqwat')]
    df_m_sqwat = df_all[(df_all['sex'] == 'm') & (df_all['depot'] == 'sqwat')]

    # global models fitted to 3 strata (Control, MAT WT and FKO):
    # These are the models that we are going to use to test for correlation, apart from the LRTs
    model_gwat_f_global = sm.OLS.from_formula('area_at_quantile ~ DW * C(functional_ko)', data=df_f_gwat).fit()
    model_gwat_m_global = sm.OLS.from_formula('area_at_quantile ~ DW * C(functional_ko)', data=df_m_gwat).fit()
    model_sqwat_f_global = sm.OLS.from_formula('area_at_quantile ~ DW * C(functional_ko)', data=df_f_sqwat).fit()
    model_sqwat_m_global = sm.OLS.from_formula('area_at_quantile ~ DW * C(functional_ko)', data=df_m_sqwat).fit()

    # models fitted to 2 strata (combining Control and MAT WT) to be used as null models
    model_gwat_f_control_matwt = sm.OLS.from_formula('area_at_quantile ~ DW * C(functional_ko_a)', data=df_f_gwat).fit()
    model_gwat_m_control_matwt = sm.OLS.from_formula('area_at_quantile ~ DW * C(functional_ko_a)', data=df_m_gwat).fit()
    model_sqwat_f_control_matwt = sm.OLS.from_formula('area_at_quantile ~ DW * C(functional_ko_a)', data=df_f_sqwat).fit()
    model_sqwat_m_control_matwt = sm.OLS.from_formula('area_at_quantile ~ DW * C(functional_ko_a)', data=df_m_sqwat).fit()

    # models fitted to 2 strata (combining MAT WT and FKO) to be used as null models
    model_gwat_f_matwt_fko = sm.OLS.from_formula('area_at_quantile ~ DW * C(functional_ko_b)', data=df_f_gwat).fit()
    model_gwat_m_matwt_fko = sm.OLS.from_formula('area_at_quantile ~ DW * C(functional_ko_b)', data=df_m_gwat).fit()
    model_sqwat_f_matwt_fko = sm.OLS.from_formula('area_at_quantile ~ DW * C(functional_ko_b)', data=df_f_sqwat).fit()
    model_sqwat_m_matwt_fko = sm.OLS.from_formula('area_at_quantile ~ DW * C(functional_ko_b)', data=df_m_sqwat).fit()

    # models fitted to 2 strata (combining Control and FKO) to be used as null models
    model_gwat_f_control_fko = sm.OLS.from_formula('area_at_quantile ~ DW * C(functional_ko_c)', data=df_f_gwat).fit()
    model_gwat_m_control_fko = sm.OLS.from_formula('area_at_quantile ~ DW * C(functional_ko_c)', data=df_m_gwat).fit()
    model_sqwat_f_control_fko = sm.OLS.from_formula('area_at_quantile ~ DW * C(functional_ko_c)', data=df_f_sqwat).fit()
    model_sqwat_m_control_fko = sm.OLS.from_formula('area_at_quantile ~ DW * C(functional_ko_c)', data=df_m_sqwat).fit()

    models_gwat_f_global.append(model_gwat_f_global)
    models_gwat_m_global.append(model_gwat_m_global)
    models_sqwat_f_global.append(model_sqwat_f_global)
    models_sqwat_m_global.append(model_sqwat_m_global)
    models_gwat_f_control_matwt.append(model_gwat_f_control_matwt)
    models_gwat_m_control_matwt.append(model_gwat_m_control_matwt)
    models_sqwat_f_control_matwt.append(model_sqwat_f_control_matwt)
    models_sqwat_m_control_matwt.append(model_sqwat_m_control_matwt)
    models_gwat_f_matwt_fko.append(model_gwat_f_matwt_fko)
    models_gwat_m_matwt_fko.append(model_gwat_m_matwt_fko)
    models_sqwat_f_matwt_fko.append(model_sqwat_f_matwt_fko)
    models_sqwat_m_matwt_fko.append(model_sqwat_m_matwt_fko)
    models_gwat_f_control_fko.append(model_gwat_f_control_fko)
    models_gwat_m_control_fko.append(model_gwat_m_control_fko)
    models_sqwat_f_control_fko.append(model_sqwat_f_control_fko)
    models_sqwat_m_control_fko.append(model_sqwat_m_control_fko)

    if DEBUG:
        print(model_gwat_f_global.summary())
        print(model_gwat_m_global.summary())
        print(model_sqwat_f_global.summary())
        print(model_sqwat_m_global.summary())
        print(model_gwat_f_control_matwt.summary())
        print(model_gwat_m_control_matwt.summary())
        print(model_sqwat_f_control_matwt.summary())
        print(model_sqwat_m_control_matwt.summary())
        print(model_gwat_f_matwt_fko.summary())
        print(model_gwat_m_matwt_fko.summary())
        print(model_sqwat_f_matwt_fko.summary())
        print(model_sqwat_m_matwt_fko.summary())
        print(model_gwat_f_control_fko.summary())
        print(model_gwat_m_control_fko.summary())
        print(model_sqwat_f_control_fko.summary())
        print(model_sqwat_m_control_fko.summary())

# extract coefficients, errors and p-values from PAT and MAT models
model_names = ['model_gwat_f_global_q1', 'model_gwat_f_global_q2', 'model_gwat_f_global_q3',
               'model_sqwat_f_global_q1', 'model_sqwat_f_global_q2', 'model_sqwat_f_global_q3',
               'model_gwat_m_global_q1', 'model_gwat_m_global_q2', 'model_gwat_m_global_q3',
               'model_sqwat_m_global_q1', 'model_sqwat_m_global_q2', 'model_sqwat_m_global_q3'
               ]
extra_hypotheses='Intercept+C(functional_ko)[T.MAT_WT],Intercept+C(functional_ko)[T.FKO]'\
                 + ',DW+DW:C(functional_ko)[T.MAT_WT],DW+DW:C(functional_ko)[T.FKO]'

df_coeff, df_ci_lo, df_ci_hi, df_pval = \
    cytometer.stats.models_coeff_ci_pval(
        [models_gwat_f_global[0], models_gwat_f_global[1], models_gwat_f_global[2],
        models_sqwat_f_global[0], models_sqwat_f_global[1], models_sqwat_f_global[2],
        models_gwat_m_global[0], models_gwat_m_global[1], models_gwat_m_global[2],
        models_sqwat_m_global[0], models_sqwat_m_global[1], models_sqwat_m_global[2]],
    extra_hypotheses=extra_hypotheses,
    model_names=model_names)

# multitest correction using Benjamini-Krieger-Yekutieli
# we only need to correct the slopes' p-values, because we are not testing the values of the intercepts
col = ['DW', 'DW+DW:C(functional_ko)[T.MAT_WT]', 'DW+DW:C(functional_ko)[T.FKO]']
df_corrected_pval = df_pval.copy()
_, aux, _, _ = multipletests(np.array(df_pval[col]).flatten(), method='fdr_tsbky', alpha=0.05, returnsorted=False)
df_corrected_pval[:] = np.nan
df_corrected_pval[col] = aux.reshape(df_corrected_pval[col].shape)

# convert p-values to asterisks
df_pval_ast = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval, brackets=False), columns=df_coeff.columns,
                           index=model_names)
df_corrected_pval_ast = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_corrected_pval, brackets=False),
                                     columns=df_coeff.columns, index=model_names)

if SAVE_FIGS:
    df_concat = pd.concat(
        [df_coeff, df_ci_lo, df_ci_hi, df_pval, df_pval_ast, df_corrected_pval, df_corrected_pval_ast], axis=1)
    idx = list(interleave(np.array_split(range(df_concat.shape[1]), 7)))
    df_concat = df_concat.iloc[:, idx]
    df_concat.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_area_at_quartiles_fko_models_coeffs_pvals.csv'),
                     na_rep='nan')

# plot
if SAVE_FIGS:

    plt.clf()
    plt.gcf().set_size_inches([6.4, 7.6])
    depot = 'gwat'

    plt.subplot(321)
    # Q1 Female
    i = 0  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'f'
    idx = (df_all['sex'] == sex) & (df_all['depot'] == depot)
    df = df_all[idx].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(models_gwat_f_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'Control'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
                                           line_label='Control')
    cytometer.stats.plot_linear_regression(models_gwat_f_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'MAT_WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
                                           line_label='MAT WT')
    cytometer.stats.plot_linear_regression(models_gwat_f_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'FKO'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C4', marker='o',
                                           line_label='FKO')
    plt.tick_params(labelsize=14)
    plt.ylabel('Area$_{\mathrm{Q1}}$ ($10^3\ \mu m^2$)', fontsize=14)
    plt.title('Female', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.ylim(0.9, 5)

    plt.subplot(322)
    # Q1 Male
    i = 0  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'm'
    idx = (df_all['sex'] == sex) & (df_all['depot'] == depot)
    df = df_all[idx].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(models_gwat_m_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'Control'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
                                           line_label='Control')
    cytometer.stats.plot_linear_regression(models_gwat_m_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'MAT_WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
                                           line_label='MAT WT')
    cytometer.stats.plot_linear_regression(models_gwat_m_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'FKO'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C4', marker='o',
                                           line_label='FKO')
    plt.tick_params(labelsize=14)
    plt.title('Male', fontsize=14)
    plt.ylim(0.9, 5)

    plt.subplot(323)
    # Q2 Female
    i = 1  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'f'
    idx = (df_all['sex'] == sex) & (df_all['depot'] == depot)
    df = df_all[idx].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(models_gwat_f_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'Control'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
                                           line_label='Control')
    cytometer.stats.plot_linear_regression(models_gwat_f_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'MAT_WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
                                           line_label='MAT WT')
    cytometer.stats.plot_linear_regression(models_gwat_f_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'FKO'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C4', marker='o',
                                           line_label='FKO')
    plt.tick_params(labelsize=14)
    plt.ylabel('Area$_{\mathrm{Q2}}$ ($10^3\ \mu m^2$)', fontsize=14)
    plt.ylim(1.4, 8.5)

    plt.subplot(324)
    # Q2 Male
    i = 1  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'm'
    idx = (df_all['sex'] == sex) & (df_all['depot'] == depot)
    df = df_all[idx].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(models_gwat_m_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'Control'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
                                           line_label='Control')
    cytometer.stats.plot_linear_regression(models_gwat_m_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'MAT_WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
                                           line_label='MAT WT')
    cytometer.stats.plot_linear_regression(models_gwat_m_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'FKO'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C4', marker='o',
                                           line_label='FKO')
    plt.tick_params(labelsize=14)
    plt.ylim(1.4, 8.5)

    plt.subplot(325)
    # Q3 Female
    i = 2  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'f'
    idx = (df_all['sex'] == sex) & (df_all['depot'] == depot)
    df = df_all[idx].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(models_gwat_f_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'Control'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
                                           line_label='Control')
    cytometer.stats.plot_linear_regression(models_gwat_f_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'MAT_WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
                                           line_label='MAT WT')
    cytometer.stats.plot_linear_regression(models_gwat_f_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'FKO'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C4', marker='o',
                                           line_label='FKO')
    plt.tick_params(labelsize=14)
    plt.ylabel('Area$_{\mathrm{Q3}}$ ($10^3\ \mu m^2$)', fontsize=14)
    plt.xlabel('Depot weight (g)', fontsize=14)
    plt.ylim(1, 14)

    plt.subplot(326)
    # Q3 Male
    i = 2  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'm'
    idx = (df_all['sex'] == sex) & (df_all['depot'] == depot)
    df = df_all[idx].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(models_gwat_m_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'Control'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
                                           line_label='Control')
    cytometer.stats.plot_linear_regression(models_gwat_m_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'MAT_WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
                                           line_label='MAT WT')
    cytometer.stats.plot_linear_regression(models_gwat_m_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'FKO'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C4', marker='o',
                                           line_label='FKO')
    plt.tick_params(labelsize=14)
    plt.xlabel('Depot weight (g)', fontsize=14)
    plt.ylim(1, 14)

    depot_title = depot.replace('gwat', 'Gonadal').replace('sqwat', 'Subcutaneous')
    plt.suptitle(depot_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_area_at_quartile_genotype_models_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_area_at_quartile_genotype_models_' + depot + '.svg'))

if SAVE_FIGS:

    plt.clf()
    plt.gcf().set_size_inches([6.4, 7.6])
    depot = 'sqwat'

    plt.subplot(321)
    # Q1 Female
    i = 0  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'f'
    idx = (df_all['sex'] == sex) & (df_all['depot'] == depot)
    df = df_all[idx].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(models_sqwat_f_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'Control'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
                                           line_label='Control')
    cytometer.stats.plot_linear_regression(models_sqwat_f_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'MAT_WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
                                           line_label='MAT WT')
    cytometer.stats.plot_linear_regression(models_sqwat_f_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'FKO'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C4', marker='o',
                                           line_label='FKO')
    plt.tick_params(labelsize=14)
    plt.ylabel('Area$_{\mathrm{Q1}}$ ($10^3\ \mu m^2$)', fontsize=14)
    plt.title('Female', fontsize=14)
    plt.ylim(0.5, 3)

    plt.subplot(322)
    # Q1 Male
    i = 0  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'm'
    idx = (df_all['sex'] == sex) & (df_all['depot'] == depot)
    df = df_all[idx].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(models_sqwat_m_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'Control'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
                                           line_label='Control')
    cytometer.stats.plot_linear_regression(models_sqwat_m_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'MAT_WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
                                           line_label='MAT WT')
    cytometer.stats.plot_linear_regression(models_sqwat_m_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'FKO'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C4', marker='o',
                                           line_label='FKO')
    plt.tick_params(labelsize=14)
    plt.title('Male', fontsize=14)
    plt.ylim(0.5, 3)

    plt.subplot(323)
    # Q2 Female
    i = 1  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'f'
    idx = (df_all['sex'] == sex) & (df_all['depot'] == depot)
    df = df_all[idx].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(models_sqwat_f_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'Control'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
                                           line_label='Control')
    cytometer.stats.plot_linear_regression(models_sqwat_f_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'MAT_WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
                                           line_label='MAT WT')
    cytometer.stats.plot_linear_regression(models_sqwat_f_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'FKO'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C4', marker='o',
                                           line_label='FKO')
    plt.tick_params(labelsize=14)
    plt.ylabel('Area$_{\mathrm{Q2}}$ ($10^3\ \mu m^2$)', fontsize=14)
    plt.ylim(0.8, 6)

    plt.subplot(324)
    # Q2 Male
    i = 1  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'm'
    idx = (df_all['sex'] == sex) & (df_all['depot'] == depot)
    df = df_all[idx].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(models_sqwat_m_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'Control'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
                                           line_label='Control')
    cytometer.stats.plot_linear_regression(models_sqwat_m_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'MAT_WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
                                           line_label='MAT WT')
    cytometer.stats.plot_linear_regression(models_sqwat_m_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'FKO'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C4', marker='o',
                                           line_label='FKO')
    plt.tick_params(labelsize=14)
    plt.ylim(0.8, 6)

    plt.subplot(325)
    # Q3 Female
    i = 2  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'f'
    idx = (df_all['sex'] == sex) & (df_all['depot'] == depot)
    df = df_all[idx].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(models_sqwat_f_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'Control'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
                                           line_label='Control')
    cytometer.stats.plot_linear_regression(models_sqwat_f_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'MAT_WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
                                           line_label='MAT WT')
    cytometer.stats.plot_linear_regression(models_sqwat_f_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'FKO'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C4', marker='o',
                                           line_label='FKO')
    plt.tick_params(labelsize=14)
    plt.ylabel('Area$_{\mathrm{Q3}}$ ($10^3\ \mu m^2$)', fontsize=14)
    plt.xlabel('Depot weight (g)', fontsize=14)
    plt.ylim(1, 10.5)

    plt.subplot(326)
    # Q3 Male
    i = 2  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'm'
    idx = (df_all['sex'] == sex) & (df_all['depot'] == depot)
    df = df_all[idx].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(models_sqwat_m_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'Control'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C2', marker='x',
                                           line_label='Control')
    cytometer.stats.plot_linear_regression(models_sqwat_m_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'MAT_WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C3', marker='+',
                                           line_label='MAT WT')
    cytometer.stats.plot_linear_regression(models_sqwat_m_global[i], df, 'DW',
                                           other_vars={'depot': depot, 'sex': sex, 'functional_ko': 'FKO'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C4', marker='o',
                                           line_label='FKO')
    plt.tick_params(labelsize=14)
    plt.xlabel('Depot weight (g)', fontsize=14)
    plt.ylim(1, 10.5)

    depot_title = depot.replace('gwat', 'Gonadal').replace('sqwat', 'Subcutaneous')
    plt.suptitle(depot_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_area_at_quartile_genotype_models_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_area_at_quartile_genotype_models_' + depot + '.svg'))


# compute LRTs and extract p-values and LRs
lrt = pd.DataFrame(columns=['lr', 'pval', 'pval_ast'])

# Control vs. MAT WT
lr, pval = cytometer.stats.lrtest(models_gwat_f_control_matwt[0].llf, models_gwat_f_global[0].llf)
lrt.loc['model_gwat_f_control_matwt_Q1', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_gwat_f_control_matwt[1].llf, models_gwat_f_global[1].llf)
lrt.loc['model_gwat_f_control_matwt_Q2', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_gwat_f_control_matwt[2].llf, models_gwat_f_global[2].llf)
lrt.loc['model_gwat_f_control_matwt_Q3', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(models_sqwat_f_control_matwt[0].llf, models_sqwat_f_global[0].llf)
lrt.loc['model_sqwat_f_control_matwt_Q1', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_sqwat_f_control_matwt[1].llf, models_sqwat_f_global[1].llf)
lrt.loc['model_sqwat_f_control_matwt_Q2', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_sqwat_f_control_matwt[2].llf, models_sqwat_f_global[2].llf)
lrt.loc['model_sqwat_f_control_matwt_Q3', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(models_gwat_m_control_matwt[0].llf, models_gwat_m_global[0].llf)
lrt.loc['model_gwat_m_control_matwt_Q1', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_gwat_m_control_matwt[1].llf, models_gwat_m_global[1].llf)
lrt.loc['model_gwat_m_control_matwt_Q2', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_gwat_m_control_matwt[2].llf, models_gwat_m_global[2].llf)
lrt.loc['model_gwat_m_control_matwt_Q3', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(models_sqwat_m_control_matwt[0].llf, models_sqwat_m_global[0].llf)
lrt.loc['model_sqwat_m_control_matwt_Q1', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_sqwat_m_control_matwt[1].llf, models_sqwat_m_global[1].llf)
lrt.loc['model_sqwat_m_control_matwt_Q2', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_sqwat_m_control_matwt[2].llf, models_sqwat_m_global[2].llf)
lrt.loc['model_sqwat_m_control_matwt_Q3', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))

# MAT WT vs FKO (MAT Het)
lr, pval = cytometer.stats.lrtest(models_gwat_f_matwt_fko[0].llf, models_gwat_f_global[0].llf)
lrt.loc['model_gwat_f_matwt_fko_Q1', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_gwat_f_matwt_fko[1].llf, models_gwat_f_global[1].llf)
lrt.loc['model_gwat_f_matwt_fko_Q2', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_gwat_f_matwt_fko[2].llf, models_gwat_f_global[2].llf)
lrt.loc['model_gwat_f_matwt_fko_Q3', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(models_sqwat_f_matwt_fko[0].llf, models_sqwat_f_global[0].llf)
lrt.loc['model_sqwat_f_matwt_fko_Q1', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_sqwat_f_matwt_fko[1].llf, models_sqwat_f_global[1].llf)
lrt.loc['model_sqwat_f_matwt_fko_Q2', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_sqwat_f_matwt_fko[2].llf, models_sqwat_f_global[2].llf)
lrt.loc['model_sqwat_f_matwt_fko_Q3', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(models_gwat_m_matwt_fko[0].llf, models_gwat_m_global[0].llf)
lrt.loc['model_gwat_m_matwt_fko_Q1', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_gwat_m_matwt_fko[1].llf, models_gwat_m_global[1].llf)
lrt.loc['model_gwat_m_matwt_fko_Q2', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_gwat_m_matwt_fko[2].llf, models_gwat_m_global[2].llf)
lrt.loc['model_gwat_m_matwt_fko_Q3', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(models_sqwat_m_matwt_fko[0].llf, models_sqwat_m_global[0].llf)
lrt.loc['model_sqwat_m_matwt_fko_Q1', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_sqwat_m_matwt_fko[1].llf, models_sqwat_m_global[1].llf)
lrt.loc['model_sqwat_m_cmatwt_fko_Q2', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_sqwat_m_matwt_fko[2].llf, models_sqwat_m_global[2].llf)
lrt.loc['model_sqwat_m_matwt_fko_Q3', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))

# Control vs FKO (MAT Het)
lr, pval = cytometer.stats.lrtest(models_gwat_f_control_fko[0].llf, models_gwat_f_global[0].llf)
lrt.loc['model_gwat_f_control_fko_Q1', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_gwat_f_control_fko[1].llf, models_gwat_f_global[1].llf)
lrt.loc['model_gwat_f_control_fko_Q2', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_gwat_f_control_fko[2].llf, models_gwat_f_global[2].llf)
lrt.loc['model_gwat_f_control_fko_Q3', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(models_sqwat_f_control_fko[0].llf, models_sqwat_f_global[0].llf)
lrt.loc['model_sqwat_f_control_fko_Q1', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_sqwat_f_control_fko[1].llf, models_sqwat_f_global[1].llf)
lrt.loc['model_sqwat_f_control_fko_Q2', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_sqwat_f_control_fko[2].llf, models_sqwat_f_global[2].llf)
lrt.loc['model_sqwat_f_control_fko_Q3', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(models_gwat_m_control_fko[0].llf, models_gwat_m_global[0].llf)
lrt.loc['model_gwat_m_control_fko_Q1', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_gwat_m_control_fko[1].llf, models_gwat_m_global[1].llf)
lrt.loc['model_gwat_m_control_fko_Q2', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_gwat_m_control_fko[2].llf, models_gwat_m_global[2].llf)
lrt.loc['model_gwat_m_control_fko_Q3', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(models_sqwat_m_control_fko[0].llf, models_sqwat_m_global[0].llf)
lrt.loc['model_sqwat_m_control_fko_Q1', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_sqwat_m_control_fko[1].llf, models_sqwat_m_global[1].llf)
lrt.loc['model_sqwat_m_control_fko_Q2', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))
lr, pval = cytometer.stats.lrtest(models_sqwat_m_control_fko[2].llf, models_sqwat_m_global[2].llf)
lrt.loc['model_sqwat_m_control_fko_Q3', :] =  (lr, pval, cytometer.stats.pval_to_asterisk(pval))

# multitest correction using Benjamini-Krieger-Yekutieli
_, lrt['pval_adj'], _, _ = multipletests(lrt['pval'], method='fdr_tsbky', alpha=0.05, returnsorted=False)
lrt['pval_adj_ast'] = cytometer.stats.pval_to_asterisk(lrt['pval_adj'])

if SAVE_FIGS:
    lrt.to_csv(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_area_at_quartiles_models_lrt_fko.csv'), na_rep='nan')

########################################################################################################################
## smoothed histograms
##
## We can use all animals for this, even the ones where BW=NaN, because we don't need BW or DW
## USED IN THE PAPER
########################################################################################################################

## only training windows used for hand tracing (there are only Control and MAT Het mice in the dataset)

# a previous version of this section was in klf14_b6ntac_exp_0109_pipeline_v8_validation.py, but now we have updated it
# so that plots are labelled with Control, MAT WT, FKO instead of PAT, MAT

# list of hand traced contours
# The list contains 126 XCF (Gimp format) files with the contours that were used for training DeepCytometer,
# plus 5 files (131 in total) with extra contours for 2 mice where the cell population was not well
# represented.
import pandas as pd

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

# filename of the dataframe with the hand traced cell data
df_hand_all_filename = os.path.join(paper_dir, 'klf14_b6ntac_exp_0111_pipeline_v8_validation_smoothed_histo_hand_' + depot + '.csv')

if os.path.isfile(df_hand_all_filename):

    # load dataframe with the hand traced data
    df_hand_all = pd.read_csv(df_hand_all_filename)

else:  # compute dataframe with the hand traced data

    # loop hand traced files and make a dataframe with the cell sizes
    df_hand_all = pd.DataFrame()
    for i, file_svg in enumerate(hand_file_svg_list):

        print('File ' + str(i) + '/' + str(len(hand_file_svg_list) - 1) + ': ' + os.path.basename(file_svg))

        # load hand traced contours
        cells = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False,
                                                        minimum_npoints=3)

        print('Cells: ' + str(len(cells)))

        if (len(cells) == 0):
            continue

        # load training image
        file_im = file_svg.replace('.svg', '.tif')
        im = PIL.Image.open(file_im)

        # read pixel size information
        xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
        yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

        im = np.array(im)

        if DEBUG:
            plt.clf()
            plt.imshow(im)
            for j in range(len(cells)):
                cell = np.array(cells[j])
                plt.fill(cell[:, 0], cell[:, 1], edgecolor='C0', fill=False)
                plt.text(np.mean(cell[:, 0]), np.mean(cell[:, 1]), str(j))

        # compute cell areas
        cell_areas = np.array([shapely.geometry.Polygon(x).area for x in cells]) * xres * yres

        df = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_svg),
                                                       values=cell_areas, values_tag='area',
                                                       tags_to_keep=['id', 'ko_parent', 'sex', 'genotype'])

        # figure out what depot these cells belong to
        # NOTE: this code is here only for completion, because there are no gonadal slides in the training dataset, only
        # subcutaneous
        aux = os.path.basename(file_svg).replace('KLF14-B6NTAC', '')
        if 'B' in aux and 'C' in aux:
            raise ValueError('Slice appears to be both gonadal and subcutaneous')
        elif 'B' in aux:
            depot = 'gwat'
        elif 'C' in aux:
            depot = 'sqwat'
        else:
            raise ValueError('Slice is neither gonadal nor subcutaneous')
        df['depot'] = depot
        df_hand_all = df_hand_all.append(df, ignore_index=True)

    # save dataframe for later use
    df_hand_all.to_csv(df_hand_all_filename, index=False)


print('Min cell size = ' + '{0:.1f}'.format(np.min(df_hand_all['area'])) + ' um^2 = '
      + '{0:.1f}'.format(np.min(df_hand_all['area']) / xres_ref / yres_ref) + ' pixels')
print('Max cell size = ' + '{0:.1f}'.format(np.max(df_hand_all['area'])) + ' um^2 = '
      + '{0:.1f}'.format(np.max(df_hand_all['area']) / xres_ref / yres_ref) + ' pixels')

# these are the same quantiles as the ones for automatic segmentations in exp 0110
quantiles = np.linspace(0, 1, 21)
area_bin_edges = np.linspace(min_area_um2, max_area_um2, 201)
area_bin_centers = (area_bin_edges[0:-1] + area_bin_edges[1:]) / 2.0

# 1-alpha is the % of confidence interval, e.g. alpha=0.05 => 95% CI
alpha = 0.05
k = stats.norm.ppf(1 - alpha / 2, loc=0, scale=1)  # multiplier for CI length (~1.96 for 95% CI)

if SAVE_FIGS:

    plt.clf()

    plt.subplot(221)
    idx = (df_hand_all['depot'] == 'sqwat') & (df_hand_all['sex'] == 'f') & (df_hand_all['ko_parent'] == 'PAT')
    kde = sklearn.neighbors.KernelDensity(bandwidth=100, kernel='gaussian').fit(
        np.array(df_hand_all[idx]['area']).reshape(-1, 1))
    log_dens = kde.score_samples((area_bin_centers).reshape(-1, 1))
    pdf = np.exp(log_dens)
    plt.plot((area_bin_centers) * 1e-3, pdf / pdf.max())
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.text(0.9, 0.9, 'Female Control', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xticks([0, 10, 20])
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    area_q = stats.mstats.hdquantiles(df_hand_all[idx]['area'] * 1e-3, prob=[0.25, 0.50, 0.75], axis=0)
    area_stderr = stats.mstats.hdquantiles_sd(df_hand_all[idx]['area'] * 1e-3, prob=[0.25, 0.50, 0.75], axis=0)
    ci_lo = area_q - k * area_stderr
    ci_hi = area_q + k * area_stderr

    print('female Control')
    print('\tQ1: ' + '{0:.2f}'.format(area_q[0]) + ' (' + '{0:.2f}'.format(ci_lo[0]) + ', ' + '{0:.2f}'.format(ci_hi[0]) + ')')
    print('\tQ2: ' + '{0:.2f}'.format(area_q[1]) + ' (' + '{0:.2f}'.format(ci_lo[1]) + ', ' + '{0:.2f}'.format(ci_hi[1]) + ')')
    print('\tQ3: ' + '{0:.2f}'.format(area_q[2]) + ' (' + '{0:.2f}'.format(ci_lo[2]) + ', ' + '{0:.2f}'.format(ci_hi[2]) + ')')

    plt.plot([area_q[0], area_q[0]], [0, 1], 'k', linewidth=1)
    plt.plot([area_q[1], area_q[1]], [0, 1], 'k', linewidth=1)
    plt.plot([area_q[2], area_q[2]], [0, 1], 'k', linewidth=1)

    plt.subplot(222)
    idx = (df_hand_all['depot'] == 'sqwat') & (df_hand_all['sex'] == 'm') & (df_hand_all['ko_parent'] == 'PAT')
    kde = sklearn.neighbors.KernelDensity(bandwidth=100, kernel='gaussian').fit(
        np.array(df_hand_all[idx]['area']).reshape(-1, 1))
    log_dens = kde.score_samples(area_bin_centers.reshape(-1, 1))
    pdf = np.exp(log_dens)
    plt.plot(area_bin_centers * 1e-3, pdf / pdf.max())
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.text(0.9, 0.9, 'Male Control', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xticks([0, 10, 20])
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    area_q = stats.mstats.hdquantiles(df_hand_all[idx]['area'] * 1e-3, prob=[0.25, 0.50, 0.75], axis=0)
    area_stderr = stats.mstats.hdquantiles_sd(df_hand_all[idx]['area'] * 1e-3, prob=[0.25, 0.50, 0.75], axis=0)
    ci_lo = area_q - k * area_stderr
    ci_hi = area_q + k * area_stderr

    print('male PAT')
    print('\tQ1: ' + '{0:.2f}'.format(area_q[0]) + ' (' + '{0:.2f}'.format(ci_lo[0]) + ', ' + '{0:.2f}'.format(ci_hi[0]) + ')')
    print('\tQ2: ' + '{0:.2f}'.format(area_q[1]) + ' (' + '{0:.2f}'.format(ci_lo[1]) + ', ' + '{0:.2f}'.format(ci_hi[1]) + ')')
    print('\tQ3: ' + '{0:.2f}'.format(area_q[2]) + ' (' + '{0:.2f}'.format(ci_lo[2]) + ', ' + '{0:.2f}'.format(ci_hi[2]) + ')')

    plt.plot([area_q[0], area_q[0]], [0, 1], 'k', linewidth=1)
    plt.plot([area_q[1], area_q[1]], [0, 1], 'k', linewidth=1)
    plt.plot([area_q[2], area_q[2]], [0, 1], 'k', linewidth=1)

    plt.subplot(223)
    idx = (df_hand_all['depot'] == 'sqwat') & (df_hand_all['sex'] == 'f') & (df_hand_all['ko_parent'] == 'MAT') \
          & (df_hand_all['genotype'] == 'KLF14-KO:Het')
    kde = sklearn.neighbors.KernelDensity(bandwidth=100, kernel='gaussian').fit(
        np.array(df_hand_all[idx]['area']).reshape(-1, 1))
    log_dens = kde.score_samples(area_bin_centers.reshape(-1, 1))
    pdf = np.exp(log_dens)
    plt.plot(area_bin_centers * 1e-3, pdf / pdf.max())
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.text(0.9, 0.9, 'Female FKO\n(MAT Het)', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xticks([0, 10, 20])
    plt.xlim(-1.2, max_area_um2 * 1e-3)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)

    area_q = stats.mstats.hdquantiles(df_hand_all[idx]['area'] * 1e-3, prob=[0.25, 0.50, 0.75], axis=0)
    area_stderr = stats.mstats.hdquantiles_sd(df_hand_all[idx]['area'] * 1e-3, prob=[0.25, 0.50, 0.75], axis=0)
    ci_lo = area_q - k * area_stderr
    ci_hi = area_q + k * area_stderr

    print('female MAT WT')
    print('\tQ1: ' + '{0:.2f}'.format(area_q[0]) + ' (' + '{0:.2f}'.format(ci_lo[0]) + ', ' + '{0:.2f}'.format(ci_hi[0]) + ')')
    print('\tQ2: ' + '{0:.2f}'.format(area_q[1]) + ' (' + '{0:.2f}'.format(ci_lo[1]) + ', ' + '{0:.2f}'.format(ci_hi[1]) + ')')
    print('\tQ3: ' + '{0:.2f}'.format(area_q[2]) + ' (' + '{0:.2f}'.format(ci_lo[2]) + ', ' + '{0:.2f}'.format(ci_hi[2]) + ')')

    plt.plot([area_q[0], area_q[0]], [0, 1], 'k', linewidth=1)
    plt.plot([area_q[1], area_q[1]], [0, 1], 'k', linewidth=1)
    plt.plot([area_q[2], area_q[2]], [0, 1], 'k', linewidth=1)

    plt.subplot(224)
    idx = (df_hand_all['depot'] == 'sqwat') & (df_hand_all['sex'] == 'm') & (df_hand_all['ko_parent'] == 'MAT')
    kde = sklearn.neighbors.KernelDensity(bandwidth=100, kernel='gaussian').fit(
        np.array(df_hand_all[idx]['area']).reshape(-1, 1))
    log_dens = kde.score_samples(area_bin_centers.reshape(-1, 1))
    pdf = np.exp(log_dens)
    plt.plot(area_bin_centers * 1e-3, pdf / pdf.max())
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.text(0.9, 0.9, 'Male FKO\n(MAT Het)', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xticks([0, 10, 20])
    plt.xlim(-1.2, max_area_um2 * 1e-3)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)

    area_q = stats.mstats.hdquantiles(df_hand_all[idx]['area'] * 1e-3, prob=[0.25, 0.50, 0.75], axis=0)
    area_stderr = stats.mstats.hdquantiles_sd(df_hand_all[idx]['area'] * 1e-3, prob=[0.25, 0.50, 0.75], axis=0)
    ci_lo = area_q - k * area_stderr
    ci_hi = area_q + k * area_stderr

    print('male MAT WT')
    print('\tQ1: ' + '{0:.2f}'.format(area_q[0]) + ' (' + '{0:.2f}'.format(ci_lo[0]) + ', ' + '{0:.2f}'.format(ci_hi[0]) + ')')
    print('\tQ2: ' + '{0:.2f}'.format(area_q[1]) + ' (' + '{0:.2f}'.format(ci_lo[1]) + ', ' + '{0:.2f}'.format(ci_hi[1]) + ')')
    print('\tQ3: ' + '{0:.2f}'.format(area_q[2]) + ' (' + '{0:.2f}'.format(ci_lo[2]) + ', ' + '{0:.2f}'.format(ci_hi[2]) + ')')

    plt.plot([area_q[0], area_q[0]], [0, 1], 'k', linewidth=1)
    plt.plot([area_q[1], area_q[1]], [0, 1], 'k', linewidth=1)
    plt.plot([area_q[2], area_q[2]], [0, 1], 'k', linewidth=1)

    if depot == 'gwat':
        plt.suptitle('Gonadal', fontsize=14)
    elif depot == 'sqwat':
        plt.suptitle('Subcutaneous', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_pipeline_v8_validation_smoothed_histo_hand_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_pipeline_v8_validation_smoothed_histo_hand_' + depot + '.svg'))


## whole slides used for hand tracing (there are only Control and MAT Het mice in the dataset)

# all hand traced slides are subcutaneous, so we only need to compare against subcutaneous
depot = 'sqwat'

# identify whole slides used for the hand traced dataset
idx_used_in_hand_traced = np.full((df_all.shape[0],), False)
for hand_file_svg in hand_file_svg_list:
    df = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(hand_file_svg),
                                                   values=[depot, ], values_tag='depot',
                                                   tags_to_keep=['id', 'ko_parent', 'sex', 'genotype'])

    idx_used_in_hand_traced[(df_all[df.columns] == df.values).all(1)] = True


if SAVE_FIGS:
    plt.clf()

    # f PAT
    df = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'PAT')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())

    plt.subplot(221)
    plt.plot(area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.text(0.9, 0.9, 'Female Control', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xticks([0, 10, 20])
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # m PAT
    df = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'PAT')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())

    plt.subplot(222)
    plt.plot(area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.text(0.9, 0.9, 'Male Control', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xticks([0, 10, 20])
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # f MAT WT
    df = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')
                & (df_all['genotype'] == 'KLF14-KO:Het')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())

    plt.subplot(223)
    plt.plot(area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.text(0.9, 0.9, 'Female FKO\n(MAT Het)', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xticks([0, 10, 20])
    plt.xlim(-1.2, max_area_um2 * 1e-3)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)

    # m MAT
    df = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'MAT')
                & (df_all['genotype'] == 'KLF14-KO:Het')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())

    plt.subplot(224)
    plt.plot(area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.xticks([0, 10, 20])
    plt.text(0.9, 0.9, 'Male FKO\n(MAT Het)', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)

    plt.suptitle('Subcutaneous', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_smoothed_histo_' + depot + '_hand_subset.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_smoothed_histo_' + depot + '_hand_subset.svg'))

if SAVE_FIGS:
    plt.clf()

    # f PAT
    df = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'PAT')]
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
    plt.text(0.9, 0.9, 'Female Control', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # m PAT
    df = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'PAT')]
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
    plt.text(0.9, 0.9, 'Male Control', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # f MAT Het
    df = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')
                & (df_all['genotype'] == 'KLF14-KO:Het')]
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
    plt.text(0.9, 0.9, 'Female FKO\n(MAT Het)', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)

    # m MAT Het
    df = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'MAT')
                & (df_all['genotype'] == 'KLF14-KO:Het')]
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
    plt.text(0.9, 0.9, 'Male FKO\n(MAT Het)', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    plt.suptitle('Subcutaneous', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# numerical quartiles and CIs associated to the histograms

idx_q1 = np.where(quantiles == 0.25)[0][0]
idx_q2 = np.where(quantiles == 0.50)[0][0]
idx_q3 = np.where(quantiles == 0.75)[0][0]

# f PAT
df = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'PAT')]
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

if SAVE_FIGS:
    plt.subplot(221)
    plt.plot([q1_hat, q1_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q2_hat, q2_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q3_hat, q3_hat], [0, 1], 'k', linewidth=1)

# m PAT
df = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'PAT')]
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

if SAVE_FIGS:
    plt.subplot(222)
    plt.plot([q1_hat, q1_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q2_hat, q2_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q3_hat, q3_hat], [0, 1], 'k', linewidth=1)

# f MAT Het
df = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')
            & (df_all['genotype'] == 'KLF14-KO:Het')]
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

print('f MAT Het')
print('\t' + '{0:.2f}'.format(q1_hat) + ' (' + '{0:.2f}'.format(q1_ci_lo) + ', ' + '{0:.2f}'.format(q1_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q2_hat) + ' (' + '{0:.2f}'.format(q2_ci_lo) + ', ' + '{0:.2f}'.format(q2_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q3_hat) + ' (' + '{0:.2f}'.format(q3_ci_lo) + ', ' + '{0:.2f}'.format(q3_ci_hi) + ')')

if SAVE_FIGS:
    plt.subplot(223)
    plt.plot([q1_hat, q1_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q2_hat, q2_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q3_hat, q3_hat], [0, 1], 'k', linewidth=1)

# m MAT Het
df = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'MAT')
            & (df_all['genotype'] == 'KLF14-KO:Het')]
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

print('m MAT Het')
print('\t' + '{0:.2f}'.format(q1_hat) + ' (' + '{0:.2f}'.format(q1_ci_lo) + ', ' + '{0:.2f}'.format(q1_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q2_hat) + ' (' + '{0:.2f}'.format(q2_ci_lo) + ', ' + '{0:.2f}'.format(q2_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q3_hat) + ' (' + '{0:.2f}'.format(q3_ci_lo) + ', ' + '{0:.2f}'.format(q3_ci_hi) + ')')

if SAVE_FIGS:
    plt.subplot(224)
    plt.plot([q1_hat, q1_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q2_hat, q2_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q3_hat, q3_hat], [0, 1], 'k', linewidth=1)

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_smoothed_histo_quartiles_' + depot + '_hand_subset.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_smoothed_histo_quartiles_' + depot + '_hand_subset.svg'))


## all slides (75 subcutaneous and 72 gonadal whole slides)

# reload load dataframe with cell population quantiles and histograms, because previously we removed 3 slides that we
# didn't have the BW for. But for this section, we don't need BW, so we can use all the data
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

depot = 'gwat'
# depot = 'sqwat'

if SAVE_FIGS:
    plt.clf()
    plt.gcf().set_size_inches([6.4 , 6.6])

    # f PAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'PAT')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())

    plt.subplot(321)
    plt.plot(area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.text(0.9, 0.9, 'Female Control', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xticks([0, 10, 20])
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # m PAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'PAT')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())

    plt.subplot(322)
    plt.plot(area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.text(0.9, 0.9, 'Male Control', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xticks([0, 10, 20])
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # f MAT WT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')
                & (df_all['genotype'] == 'KLF14-KO:WT')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())

    plt.subplot(323)
    plt.plot(area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.text(0.9, 0.9, 'Female MAT WT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xticks([0, 10, 20])
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # m MAT WT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'MAT')
                & (df_all['genotype'] == 'KLF14-KO:WT')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())

    plt.subplot(324)
    plt.plot(area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.xticks([0, 10, 20])
    plt.text(0.9, 0.9, 'Male MAT WT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # f MAT Het
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')
                & (df_all['genotype'] == 'KLF14-KO:Het')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())

    plt.subplot(325)
    plt.plot(area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.text(0.9, 0.9, 'Female FKO\n(MAT Het)', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xticks([0, 10, 20])
    plt.xlim(-1.2, max_area_um2 * 1e-3)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)

    # m MAT Het
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'MAT')
                & (df_all['genotype'] == 'KLF14-KO:Het')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())

    plt.subplot(326)
    plt.plot(area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.xticks([0, 10, 20])
    plt.text(0.9, 0.9, 'Male FKO\n(MAT Het)', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)

    if depot == 'gwat':
        plt.suptitle('Gonadal', fontsize=14)
    elif depot == 'sqwat':
        plt.suptitle('Subcutaneous', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.975])

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_smoothed_histo_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_smoothed_histo_' + depot + '.svg'))

if SAVE_FIGS:
    plt.clf()
    plt.gcf().set_size_inches([6.4 , 6.6])

    # f PAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'PAT')]
    histo = np.array(df['smoothed_histo'].tolist())
    histo_beg = stats.mstats.hdquantiles(histo, prob=0.025, axis=0)
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)
    histo_end = stats.mstats.hdquantiles(histo, prob=0.975, axis=0)

    plt.subplot(321)
    hist_max = histo_end.max()
    plt.fill_between(area_bin_centers * 1e-3, histo_beg[0,] / hist_max, histo_end[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.fill_between(area_bin_centers * 1e-3, histo_q1[0,] / hist_max, histo_q3[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.plot(area_bin_centers * 1e-3, histo_q2[0,] / hist_max, 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.text(0.9, 0.9, 'Female Control', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # m PAT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'PAT')]
    histo = np.array(df['smoothed_histo'].tolist())
    histo_beg = stats.mstats.hdquantiles(histo, prob=0.025, axis=0)
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)
    histo_end = stats.mstats.hdquantiles(histo, prob=0.975, axis=0)

    plt.subplot(322)
    hist_max = histo_end.max()
    plt.fill_between(area_bin_centers * 1e-3, histo_beg[0,] / hist_max, histo_end[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.fill_between(area_bin_centers * 1e-3, histo_q1[0,] / hist_max, histo_q3[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.plot(area_bin_centers * 1e-3, histo_q2[0,] / hist_max, 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.text(0.9, 0.9, 'Male Control', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # f MAT WT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')
                & (df_all['genotype'] == 'KLF14-KO:WT')]
    histo = np.array(df['smoothed_histo'].tolist())
    histo_beg = stats.mstats.hdquantiles(histo, prob=0.025, axis=0)
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)
    histo_end = stats.mstats.hdquantiles(histo, prob=0.975, axis=0)

    plt.subplot(323)
    hist_max = histo_end.max()
    plt.fill_between(area_bin_centers * 1e-3, histo_beg[0,] / hist_max, histo_end[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.fill_between(area_bin_centers * 1e-3, histo_q1[0,] / hist_max, histo_q3[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.plot(area_bin_centers * 1e-3, histo_q2[0,] / hist_max, 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.text(0.9, 0.9, 'Female MAT WT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # m MAT WT
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'MAT')
                & (df_all['genotype'] == 'KLF14-KO:WT')]
    histo = np.array(df['smoothed_histo'].tolist())
    histo_beg = stats.mstats.hdquantiles(histo, prob=0.025, axis=0)
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)
    histo_end = stats.mstats.hdquantiles(histo, prob=0.975, axis=0)

    plt.subplot(324)
    hist_max = histo_end.max()
    plt.fill_between(area_bin_centers * 1e-3, histo_beg[0,] / hist_max, histo_end[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.fill_between(area_bin_centers * 1e-3, histo_q1[0,] / hist_max, histo_q3[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.plot(area_bin_centers * 1e-3, histo_q2[0,] / hist_max, 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.text(0.9, 0.9, 'Male MAT WT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # f MAT Het
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')
                & (df_all['genotype'] == 'KLF14-KO:Het')]
    histo = np.array(df['smoothed_histo'].tolist())
    histo_beg = stats.mstats.hdquantiles(histo, prob=0.025, axis=0)
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)
    histo_end = stats.mstats.hdquantiles(histo, prob=0.975, axis=0)

    plt.subplot(325)
    hist_max = histo_end.max()
    plt.fill_between(area_bin_centers * 1e-3, histo_beg[0,] / hist_max, histo_end[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.fill_between(area_bin_centers * 1e-3, histo_q1[0,] / hist_max, histo_q3[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.plot(area_bin_centers * 1e-3, histo_q2[0,] / hist_max, 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.text(0.9, 0.9, 'Female FKO\n(MAT Het)', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)

    # m MAT Het
    df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'MAT')
                & (df_all['genotype'] == 'KLF14-KO:Het')]
    histo = np.array(df['smoothed_histo'].tolist())
    histo_beg = stats.mstats.hdquantiles(histo, prob=0.025, axis=0)
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)
    histo_end = stats.mstats.hdquantiles(histo, prob=0.975, axis=0)

    plt.subplot(326)
    hist_max = histo_end.max()
    plt.fill_between(area_bin_centers * 1e-3, histo_beg[0,] / hist_max, histo_end[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.fill_between(area_bin_centers * 1e-3, histo_q1[0,] / hist_max, histo_q3[0,] / hist_max,
                     alpha=0.5, color='C0')
    plt.plot(area_bin_centers * 1e-3, histo_q2[0,] / hist_max, 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)
    plt.text(0.9, 0.9, 'Male FKO\n(MAT Het)', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
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

if SAVE_FIGS:
    plt.subplot(321)
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

if SAVE_FIGS:
    plt.subplot(322)
    plt.plot([q1_hat, q1_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q2_hat, q2_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q3_hat, q3_hat], [0, 1], 'k', linewidth=1)

# f MAT WT
df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')
            & (df_all['genotype'] == 'KLF14-KO:WT')]
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

print('f MAT WT')
print('\t' + '{0:.2f}'.format(q1_hat) + ' (' + '{0:.2f}'.format(q1_ci_lo) + ', ' + '{0:.2f}'.format(q1_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q2_hat) + ' (' + '{0:.2f}'.format(q2_ci_lo) + ', ' + '{0:.2f}'.format(q2_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q3_hat) + ' (' + '{0:.2f}'.format(q3_ci_lo) + ', ' + '{0:.2f}'.format(q3_ci_hi) + ')')

if SAVE_FIGS:
    plt.subplot(323)
    plt.plot([q1_hat, q1_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q2_hat, q2_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q3_hat, q3_hat], [0, 1], 'k', linewidth=1)

# m MAT WT
df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'MAT')
            & (df_all['genotype'] == 'KLF14-KO:WT')]
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

print('m MAT WT')
print('\t' + '{0:.2f}'.format(q1_hat) + ' (' + '{0:.2f}'.format(q1_ci_lo) + ', ' + '{0:.2f}'.format(q1_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q2_hat) + ' (' + '{0:.2f}'.format(q2_ci_lo) + ', ' + '{0:.2f}'.format(q2_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q3_hat) + ' (' + '{0:.2f}'.format(q3_ci_lo) + ', ' + '{0:.2f}'.format(q3_ci_hi) + ')')

if SAVE_FIGS:
    plt.subplot(324)
    plt.plot([q1_hat, q1_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q2_hat, q2_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q3_hat, q3_hat], [0, 1], 'k', linewidth=1)

# f MAT Het
df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')
            & (df_all['genotype'] == 'KLF14-KO:Het')]
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())
stderr_at_quantiles = np.array(df['stderr_at_quantiles'].to_list())
stderr_at_quantiles[:, [0, -1]] = np.nan  ## first and last values are artifacts of saving to the CSV file

# inverse-variance method to combine quantiles and sdterr values from multiple mice
areas_at_quantiles_hat, stderr_at_quantiles_hat = \
    cytometer.stats.inverse_variance_method(areas_at_quantiles, stderr_at_quantiles)

# compute combined value and CIs in 10^3 um^2 units
alpha = 0.05
q1_hat, q2_hat, q3_hat = areas_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]] * 1e-3
k = stats.norm.ppf(1 - alpha / 2, loc=0, scale=1)  # multiplier for CI length (~1.96 for 95% CI)
q1_ci_lo, q2_ci_lo, q3_ci_lo = [q1_hat, q2_hat, q3_hat] - k * stderr_at_quantiles_hat[
    [idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_hi, q2_ci_hi, q3_ci_hi = [q1_hat, q2_hat, q3_hat] + k * stderr_at_quantiles_hat[
    [idx_q1, idx_q2, idx_q3]] * 1e-3

print('f MAT Het')
print('\t' + '{0:.2f}'.format(q1_hat) + ' (' + '{0:.2f}'.format(q1_ci_lo) + ', ' + '{0:.2f}'.format(q1_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q2_hat) + ' (' + '{0:.2f}'.format(q2_ci_lo) + ', ' + '{0:.2f}'.format(q2_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q3_hat) + ' (' + '{0:.2f}'.format(q3_ci_lo) + ', ' + '{0:.2f}'.format(q3_ci_hi) + ')')

if SAVE_FIGS:
    plt.subplot(325)
    plt.plot([q1_hat, q1_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q2_hat, q2_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q3_hat, q3_hat], [0, 1], 'k', linewidth=1)

# m MAT Het
df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'MAT')
            & (df_all['genotype'] == 'KLF14-KO:Het')]
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())
stderr_at_quantiles = np.array(df['stderr_at_quantiles'].to_list())
stderr_at_quantiles[:, [0, -1]] = np.nan  ## first and last values are artifacts of saving to the CSV file

# inverse-variance method to combine quantiles and sdterr values from multiple mice
areas_at_quantiles_hat, stderr_at_quantiles_hat = \
    cytometer.stats.inverse_variance_method(areas_at_quantiles, stderr_at_quantiles)

# compute combined value and CIs in 10^3 um^2 units
alpha = 0.05
q1_hat, q2_hat, q3_hat = areas_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]] * 1e-3
k = stats.norm.ppf(1 - alpha / 2, loc=0, scale=1)  # multiplier for CI length (~1.96 for 95% CI)
q1_ci_lo, q2_ci_lo, q3_ci_lo = [q1_hat, q2_hat, q3_hat] - k * stderr_at_quantiles_hat[
    [idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_hi, q2_ci_hi, q3_ci_hi = [q1_hat, q2_hat, q3_hat] + k * stderr_at_quantiles_hat[
    [idx_q1, idx_q2, idx_q3]] * 1e-3

print('m MAT Het')
print('\t' + '{0:.2f}'.format(q1_hat) + ' (' + '{0:.2f}'.format(q1_ci_lo) + ', ' + '{0:.2f}'.format(q1_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q2_hat) + ' (' + '{0:.2f}'.format(q2_ci_lo) + ', ' + '{0:.2f}'.format(q2_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q3_hat) + ' (' + '{0:.2f}'.format(q3_ci_lo) + ', ' + '{0:.2f}'.format(q3_ci_hi) + ')')

if SAVE_FIGS:
    plt.subplot(326)
    plt.plot([q1_hat, q1_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q2_hat, q2_hat], [0, 1], 'k', linewidth=1)
    plt.plot([q3_hat, q3_hat], [0, 1], 'k', linewidth=1)

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_smoothed_histo_quartiles_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0111_paper_figures_smoothed_histo_quartiles_' + depot + '.svg'))

########################################################################################################################
## comparison of populations quartiles from smoothed histograms of DeepCytometer whole slides and hand tracing
##
## We can use all animals for this, even the ones where BW=NaN, because we don't need BW or DW
## USED IN THE PAPER
########################################################################################################################

# indices for the quartiles
idx_q1 = np.where(quantiles == 0.25)[0][0]
idx_q2 = np.where(quantiles == 0.50)[0][0]
idx_q3 = np.where(quantiles == 0.75)[0][0]

depot = 'sqwat'

# load hand traced areas
df_hand_all = pd.read_csv(os.path.join(dataframe_dir, 'klf14_b6ntac_exp_0111_pipeline_v8_validation_smoothed_histo_hand_' + depot + '.csv'))

# identify whole slides used for the hand traced dataset
idx_used_in_hand_traced = np.full((df_all.shape[0],), False)
for hand_file_svg in hand_file_svg_list:
    df = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(hand_file_svg),
                                                   values=[depot, ], values_tag='depot',
                                                   tags_to_keep=['id', 'ko_parent', 'sex', 'genotype'])

    idx_used_in_hand_traced[(df_all[df.columns] == df.values).all(1)] = True

## whole slides used for hand tracing segmented by Deep Cytometer compared to the smaller hand traced dataset

print('DeepCytometer whole slide quartiles compared to hand tracing, same slides both')

# f PAT
df_hand = df_hand_all[(df_hand_all['depot'] == depot) & (df_hand_all['sex'] == 'f') & (df_hand_all['ko_parent'] == 'PAT')]
df = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'PAT')]

areas_at_quantiles_hand = stats.mstats.hdquantiles(df_hand['area'], prob=quantiles, axis=0)
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())
stderr_at_quantiles = np.array(df['stderr_at_quantiles'].to_list())

areas_at_quantiles_hat, stderr_at_quantiles_hat = \
    cytometer.stats.inverse_variance_method(areas_at_quantiles, stderr_at_quantiles)

areas_at_quantiles_hand = areas_at_quantiles_hand[[idx_q1, idx_q2, idx_q3]]
areas_at_quantiles_hat = areas_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]]

area_ratio_1, area_ratio_2, area_ratio_3 = (areas_at_quantiles_hat / areas_at_quantiles_hand - 1) * 100

print('f PAT')
print('\t' + '{0:.2f}'.format(area_ratio_1) + ' %')
print('\t' + '{0:.2f}'.format(area_ratio_2) + ' %')
print('\t' + '{0:.2f}'.format(area_ratio_3) + ' %')

# f MAT Het
df_hand = df_hand_all[(df_hand_all['depot'] == depot) & (df_hand_all['sex'] == 'f') & (df_hand_all['ko_parent'] == 'MAT')
                      & (df_hand_all['genotype'] == 'KLF14-KO:Het')]
df = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')
            & (df_all['genotype'] == 'KLF14-KO:Het')]

areas_at_quantiles_hand = stats.mstats.hdquantiles(df_hand['area'], prob=quantiles, axis=0)
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())
stderr_at_quantiles = np.array(df['stderr_at_quantiles'].to_list())

areas_at_quantiles_hat, stderr_at_quantiles_hat = \
    cytometer.stats.inverse_variance_method(areas_at_quantiles, stderr_at_quantiles)

areas_at_quantiles_hand = areas_at_quantiles_hand[[idx_q1, idx_q2, idx_q3]]
areas_at_quantiles_hat = areas_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]]

area_ratio_1, area_ratio_2, area_ratio_3 = (areas_at_quantiles_hat / areas_at_quantiles_hand - 1) * 100

print('f MAT Het')
print('\t' + '{0:.2f}'.format(area_ratio_1) + ' %')
print('\t' + '{0:.2f}'.format(area_ratio_2) + ' %')
print('\t' + '{0:.2f}'.format(area_ratio_3) + ' %')

# m PAT
df_hand = df_hand_all[(df_hand_all['depot'] == depot) & (df_hand_all['sex'] == 'm') & (df_hand_all['ko_parent'] == 'PAT')]
df = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'PAT')]

areas_at_quantiles_hand = stats.mstats.hdquantiles(df_hand['area'], prob=quantiles, axis=0)
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())
stderr_at_quantiles = np.array(df['stderr_at_quantiles'].to_list())

areas_at_quantiles_hat, stderr_at_quantiles_hat = \
    cytometer.stats.inverse_variance_method(areas_at_quantiles, stderr_at_quantiles)

areas_at_quantiles_hand = areas_at_quantiles_hand[[idx_q1, idx_q2, idx_q3]]
areas_at_quantiles_hat = areas_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]]

area_ratio_1, area_ratio_2, area_ratio_3 = (areas_at_quantiles_hat / areas_at_quantiles_hand - 1) * 100

print('m PAT')
print('\t' + '{0:.2f}'.format(area_ratio_1) + ' %')
print('\t' + '{0:.2f}'.format(area_ratio_2) + ' %')
print('\t' + '{0:.2f}'.format(area_ratio_3) + ' %')

# m MAT Het
df_hand = df_hand_all[(df_hand_all['depot'] == depot) & (df_hand_all['sex'] == 'm') & (df_hand_all['ko_parent'] == 'MAT')
                      & (df_hand_all['genotype'] == 'KLF14-KO:Het')]
df = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'MAT')
            & (df_all['genotype'] == 'KLF14-KO:Het')]

areas_at_quantiles_hand = stats.mstats.hdquantiles(df_hand['area'], prob=quantiles, axis=0)
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())
stderr_at_quantiles = np.array(df['stderr_at_quantiles'].to_list())

areas_at_quantiles_hat, stderr_at_quantiles_hat = \
    cytometer.stats.inverse_variance_method(areas_at_quantiles, stderr_at_quantiles)

areas_at_quantiles_hand = areas_at_quantiles_hand[[idx_q1, idx_q2, idx_q3]]
areas_at_quantiles_hat = areas_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]]

area_ratio_1, area_ratio_2, area_ratio_3 = (areas_at_quantiles_hat / areas_at_quantiles_hand - 1) * 100

print('m MAT Het')
print('\t' + '{0:.2f}'.format(area_ratio_1) + ' %')
print('\t' + '{0:.2f}'.format(area_ratio_2) + ' %')
print('\t' + '{0:.2f}'.format(area_ratio_3) + ' %')


## whole slides used for hand tracing compared to hand tracing

print('DeepCytometer whole slide quartiles compared to DeepCytometer segmentations from hand traced whole slides')

# f PAT
df_hand = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'PAT')]
df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'PAT')]

areas_at_quantiles_hand = np.array(df_hand['area_at_quantiles'].to_list())
stderr_at_quantiles_hand = np.array(df_hand['stderr_at_quantiles'].to_list())
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())
stderr_at_quantiles = np.array(df['stderr_at_quantiles'].to_list())

areas_at_quantiles_hand_hat, stderr_at_quantiles_hand_hat = \
    cytometer.stats.inverse_variance_method(areas_at_quantiles_hand, stderr_at_quantiles_hand)
areas_at_quantiles_hat, stderr_at_quantiles_hat = \
    cytometer.stats.inverse_variance_method(areas_at_quantiles, stderr_at_quantiles)

areas_at_quantiles_hand_hat = areas_at_quantiles_hand_hat[[idx_q1, idx_q2, idx_q3]]
areas_at_quantiles_hat = areas_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]]

area_ratio_1, area_ratio_2, area_ratio_3 = (areas_at_quantiles_hat / areas_at_quantiles_hand_hat - 1) * 100

print('f PAT')
print('\t' + '{0:.2f}'.format(area_ratio_1) + ' %')
print('\t' + '{0:.2f}'.format(area_ratio_2) + ' %')
print('\t' + '{0:.2f}'.format(area_ratio_3) + ' %')

# f MAT Het
df_hand = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')
                 & (df_all['genotype'] == 'KLF14-KO:Het')]
df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'f') & (df_all['ko_parent'] == 'MAT')
            & (df_all['genotype'] == 'KLF14-KO:Het')]

areas_at_quantiles_hand = np.array(df_hand['area_at_quantiles'].to_list())
stderr_at_quantiles_hand = np.array(df_hand['stderr_at_quantiles'].to_list())
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())
stderr_at_quantiles = np.array(df['stderr_at_quantiles'].to_list())

areas_at_quantiles_hand_hat, stderr_at_quantiles_hand_hat = \
    cytometer.stats.inverse_variance_method(areas_at_quantiles_hand, stderr_at_quantiles_hand)
areas_at_quantiles_hat, stderr_at_quantiles_hat = \
    cytometer.stats.inverse_variance_method(areas_at_quantiles, stderr_at_quantiles)

areas_at_quantiles_hand_hat = areas_at_quantiles_hand_hat[[idx_q1, idx_q2, idx_q3]]
areas_at_quantiles_hat = areas_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]]

area_ratio_1, area_ratio_2, area_ratio_3 = (areas_at_quantiles_hat / areas_at_quantiles_hand_hat - 1) * 100

print('f MAT Het')
print('\t' + '{0:.2f}'.format(area_ratio_1) + ' %')
print('\t' + '{0:.2f}'.format(area_ratio_2) + ' %')
print('\t' + '{0:.2f}'.format(area_ratio_3) + ' %')

# m PAT
df_hand = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'PAT')]
df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'PAT')]

areas_at_quantiles_hand = np.array(df_hand['area_at_quantiles'].to_list())
stderr_at_quantiles_hand = np.array(df_hand['stderr_at_quantiles'].to_list())
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())
stderr_at_quantiles = np.array(df['stderr_at_quantiles'].to_list())

areas_at_quantiles_hand_hat, stderr_at_quantiles_hand_hat = \
    cytometer.stats.inverse_variance_method(areas_at_quantiles_hand, stderr_at_quantiles_hand)
areas_at_quantiles_hat, stderr_at_quantiles_hat = \
    cytometer.stats.inverse_variance_method(areas_at_quantiles, stderr_at_quantiles)

areas_at_quantiles_hand_hat = areas_at_quantiles_hand_hat[[idx_q1, idx_q2, idx_q3]]
areas_at_quantiles_hat = areas_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]]

area_ratio_1, area_ratio_2, area_ratio_3 = (areas_at_quantiles_hat / areas_at_quantiles_hand_hat - 1) * 100

print('m PAT')
print('\t' + '{0:.2f}'.format(area_ratio_1) + ' %')
print('\t' + '{0:.2f}'.format(area_ratio_2) + ' %')
print('\t' + '{0:.2f}'.format(area_ratio_3) + ' %')

# m MAT Het
df_hand = df_all[idx_used_in_hand_traced & (df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'MAT')
                 & (df_all['genotype'] == 'KLF14-KO:Het')]
df = df_all[(df_all['depot'] == depot) & (df_all['sex'] == 'm') & (df_all['ko_parent'] == 'MAT')
            & (df_all['genotype'] == 'KLF14-KO:Het')]

areas_at_quantiles_hand = np.array(df_hand['area_at_quantiles'].to_list())
stderr_at_quantiles_hand = np.array(df_hand['stderr_at_quantiles'].to_list())
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())
stderr_at_quantiles = np.array(df['stderr_at_quantiles'].to_list())

areas_at_quantiles_hand_hat, stderr_at_quantiles_hand_hat = \
    cytometer.stats.inverse_variance_method(areas_at_quantiles_hand, stderr_at_quantiles_hand)
areas_at_quantiles_hat, stderr_at_quantiles_hat = \
    cytometer.stats.inverse_variance_method(areas_at_quantiles, stderr_at_quantiles)

areas_at_quantiles_hand_hat = areas_at_quantiles_hand_hat[[idx_q1, idx_q2, idx_q3]]
areas_at_quantiles_hat = areas_at_quantiles_hat[[idx_q1, idx_q2, idx_q3]]

area_ratio_1, area_ratio_2, area_ratio_3 = (areas_at_quantiles_hat / areas_at_quantiles_hand_hat - 1) * 100

print('m MAT Het')
print('\t' + '{0:.2f}'.format(area_ratio_1) + ' %')
print('\t' + '{0:.2f}'.format(area_ratio_2) + ' %')
print('\t' + '{0:.2f}'.format(area_ratio_3) + ' %')

