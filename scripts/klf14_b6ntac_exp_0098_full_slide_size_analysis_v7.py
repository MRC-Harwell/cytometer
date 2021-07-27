"""
Read annotation files from full slides processed by pipeline v7.
"""

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0098_full_slide_size_analysis_v7'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import sys
import pickle
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import tensorflow as tf
from PIL import Image, ImageDraw
import pandas as pd
import scipy.stats as stats

# Filter out INFO & WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

os.environ['KERAS_BACKEND'] = 'tensorflow'

import openslide
import numpy as np
import matplotlib.pyplot as plt
import cytometer.data
import itertools
from shapely.geometry import Polygon
import scipy
import seaborn as sns
import statannot

LIMIT_GPU_MEM = False

# limit GPU memory used
if LIMIT_GPU_MEM:
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    set_session(tf.Session(config=config))

DEBUG = False
SAVE_FIGS = False

root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
data_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
training_dir = os.path.join(home, root_data_dir, 'klf14_b6ntac_training')
# figures_dir = os.path.join(root_data_dir, 'figures')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
area2quantile_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v7')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')
results_dir = os.path.join(root_data_dir, 'klf14_b6ntac_results')
annotations_dir = os.path.join(home, 'Data/cytometer_data/aida_data_Klf14_v7/annotations')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
metainfo_dir = os.path.join(home, 'Data/cytometer_data/klf14')

# k-folds file
saved_kfolds_filename = 'klf14_b6ntac_exp_0094_generate_extra_training_images.pickle'

# rough_foreground_mask() parameters
downsample_factor = 8.0
# dilation_size = 25
# component_size_threshold = 1e6
# hole_size_treshold = 8000

# json_annotation_files_dict here needs to have the same files as in
# klf14_b6ntac_exp_0099_paper_figures_v7.py

# NOTE: This list has the base filenames that map to the .ndpi files. However, the script later modifies these filenames
# to add the '_exp_0097_corrected.json' suffix, so that corrected segmentations are loaded.

# list of annotation files
json_annotation_files = [
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
    # 'KLF14-B6NTAC-PAT 37.2d 411-16 C4 - 2020-02-14 10.34.10.json',

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
## Hand traced cell areas
########################################################################################################################

# load svg files from manual dataset
saved_kfolds_filename = os.path.join(saved_models_dir, saved_kfolds_filename)
with open(saved_kfolds_filename, 'rb') as f:
    aux = pickle.load(f)
file_svg_list = aux['file_list']# load list of images, and indices for training vs. testing indices

## extra files that were not used in the CNN training and segmentation, but that they were added so that we could have
## representative ECDFs for animals that were undersampled
file_svg_list_extra = [
    os.path.join(training_dir, 'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_020824_col_018688.svg'),
    os.path.join(training_dir, 'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_013256_col_007952.svg'),
    os.path.join(training_dir, 'KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46_row_006040_col_005272.svg'),
    os.path.join(training_dir, 'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_012680_col_023936.svg'),
    os.path.join(training_dir, 'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_017360_col_024712.svg')
]

# add extra files
file_svg_list += file_svg_list_extra

# correct home directory in file paths
file_svg_list = cytometer.data.change_home_directory(list(file_svg_list), '/users/rittscher/rcasero', home,
                                                     check_isfile=True)

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# loop files with hand traced contours
manual_areas_f = []
manual_areas_m = []
for i, file_svg in enumerate(file_svg_list):

    print('file ' + str(i) + '/' + str(len(file_svg_list) - 1) + ': ' + os.path.basename(file_svg))

    # change file extension from .svg to .tif
    file_tif = file_svg.replace('.svg', '.tif')

    # open histology training image
    im = Image.open(file_tif)

    # read pixel size information
    xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
    yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

    # read the ground truth cell contours in the SVG file. This produces a list [contour_0, ..., contour_N-1]
    # where each contour_i = [(X_0, Y_0), ..., (X_P-1, X_P-1)]
    contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False,
                                                       minimum_npoints=3)

    # create dataframe for this image
    df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_svg),
                                                          values=[i,], values_tag='i',
                                                          tags_to_keep=['id', 'ko_parent', 'sex'])

    # mouse ID as a string
    id = df_common['id'].values[0]
    sex = df_common['sex'].values[0]
    ko = df_common['ko_parent'].values[0]

    # compute cell area
    if sex == 'f':
        manual_areas_f.append([Polygon(c).area * xres * yres for c in contours])  # (um^2)
    elif sex == 'm':
        manual_areas_m.append([Polygon(c).area * xres * yres for c in contours])  # (um^2)
    else:
        raise ValueError('Wrong sex value')


manual_areas_f = list(itertools.chain.from_iterable(manual_areas_f))
manual_areas_m = list(itertools.chain.from_iterable(manual_areas_m))

########################################################################################################################
## Automatic segmentation areas, but only from full slides where the hand tracing came from
########################################################################################################################

# from
# ['/path/to/files/klf14_b6ntac_training/KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11_row_019220_col_061724.svg', ...]
# to
# ['KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11_row_019220_col_061724.svg', ...]
file_training_full_slide_svg_list = [os.path.basename(file) for file in file_svg_list]

# to
# ['KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11', ...]
file_training_full_slide_svg_list = [file.split('_')[0] for file in file_training_full_slide_svg_list]

# remove duplicates
file_training_full_slide_svg_list = list(dict.fromkeys(file_training_full_slide_svg_list))

# loop files
areas_auto_training_slides_f = []
areas_auto_training_slides_m = []
areas_corrected_training_slides_f = []
areas_corrected_training_slides_m = []
for i_file, file in enumerate(file_training_full_slide_svg_list):

    print('file ' + str(i_file) + '/' + str(len(file_training_full_slide_svg_list) - 1) + ': ' + file)

    # get corresponding .ndpi file
    file_ndpi = os.path.join(data_dir, file + '.ndpi')

    # open histology training image
    im = openslide.OpenSlide(file_ndpi)

    # pixel size
    assert (im.properties['tiff.ResolutionUnit'] == 'centimeter')
    xres = 1e-2 / float(im.properties['tiff.XResolution']) * 1e6  # um^2
    yres = 1e-2 / float(im.properties['tiff.YResolution']) * 1e6  # um^2

    # create dataframe for this image
    df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file),
                                                          values=[i_file, ], values_tag='i',
                                                          tags_to_keep=['id', 'ko_parent', 'sex'])

    # mouse ID as a string
    id = df_common['id'].values[0]
    sex = df_common['sex'].values[0]
    ko = df_common['ko_parent'].values[0]

    # load list of contours in Auto and Corrected segmentations
    json_file_auto = os.path.join(annotations_dir, file + '_exp_0097_auto.json')
    contours_auto = cytometer.data.aida_get_contours(json_file_auto, layer_name='White adipocyte.*')
    json_file_corrected = os.path.join(annotations_dir, file + '_exp_0097_corrected.json')
    contours_corrected = cytometer.data.aida_get_contours(json_file_corrected, layer_name='White adipocyte.*')

    # loop items (one contour per item)
    areas_all = []
    for c in contours_auto:
        # compute cell area
        area = Polygon(c).area * xres * yres  # (um^2)
        areas_all.append(area)
    if sex == 'f':
        areas_auto_training_slides_f.append(np.array(areas_all))
    elif sex == 'm':
        areas_auto_training_slides_m.append(np.array(areas_all))

    areas_all = []
    for c in contours_corrected:
        # compute cell area
        area = Polygon(c).area * xres * yres  # (um^2)
        areas_all.append(area)
    if sex == 'f':
        areas_corrected_training_slides_f.append(np.array(areas_all))
    elif sex == 'm':
        areas_corrected_training_slides_m.append(np.array(areas_all))

# create dataframe for seaborn.boxplot()
df_all = pd.DataFrame()
df_all['Area'] = manual_areas_f
df_all['Sex'] = 'Female'
df_all['Method'] = 'Hand traced'

df = pd.DataFrame()
df['Area'] = manual_areas_m
df['Sex'] = 'Male'
df['Method'] = 'Hand traced'
df_all = pd.concat((df_all, df))

df = pd.DataFrame()
df['Area'] = np.concatenate(areas_auto_training_slides_f)
df['Sex'] = 'Female'
df['Method'] = 'Auto training'
df_all = pd.concat((df_all, df))

df = pd.DataFrame()
df['Area'] = np.concatenate(areas_auto_training_slides_m)
df['Sex'] = 'Male'
df['Method'] = 'Auto training'
df_all = pd.concat((df_all, df))

df = pd.DataFrame()
df['Area'] = np.concatenate(areas_corrected_training_slides_f)
df['Sex'] = 'Female'
df['Method'] = 'Corrected training'
df_all = pd.concat((df_all, df))

df = pd.DataFrame()
df['Area'] = np.concatenate(areas_corrected_training_slides_m)
df['Sex'] = 'Male'
df['Method'] = 'Corrected training'
df_all = pd.concat((df_all, df))

# clean outliers from the Corrected data, in dataframe and in vectors of vectors
max_cell_size = 22500
df_all = df_all.loc[df_all['Area'] <= max_cell_size]

areas_corrected_training_slides_f = [x[x <= max_cell_size] for x in areas_corrected_training_slides_f]
areas_corrected_training_slides_m = [x[x <= max_cell_size] for x in areas_corrected_training_slides_m]

# boxplots of Auto vs. Corrected
plt.clf()
ax = sns.boxplot(x='Sex', y='Area', data=df_all, hue='Method')
plt.tick_params(labelsize=14)
plt.xlabel('Sex', fontsize=14)
plt.ylabel('White adipocyte area ($\mu m^2$)', fontsize=14)
plt.tight_layout()

statannot.add_stat_annotation(ax,
                              x='Sex', y='Area', data=df_all, hue='Method',
                              box_pairs=[(('Female', 'Hand traced'), ('Female', 'Auto training')),
                                         (('Female', 'Hand traced'), ('Female', 'Corrected training')),
                                         (('Male', 'Hand traced'), ('Male', 'Auto training')),
                                         (('Male', 'Hand traced'), ('Male', 'Corrected training'))
                                         ],
                              test='Mann-Whitney', comparisons_correction='bonferroni',
                              text_format='star', loc='inside', verbose=2)
plt.legend(loc='center left')
plt.tight_layout()

plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0098_manual_auto_corrected_training_slides_area_boxplots.png'))
plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0098_manual_auto_corrected_training_slides_area_boxplots.svg'))

########################################################################################################################
## Colourmap for AIDA (using all automatically segmented data)
########################################################################################################################

filename_corrected_areas = os.path.join(figures_dir, 'klf14_b6ntac_exp_0098_corrected_areas.npz')

if os.path.isfile(filename_corrected_areas):

    with np.load(filename_corrected_areas, allow_pickle=True) as aux:
        json_files_f = aux['json_files_f']
        json_files_m = aux['json_files_m']
        areas_auto_f = aux['areas_auto_f']
        areas_auto_m = aux['areas_auto_m']
        areas_corrected_f = aux['areas_corrected_f']
        areas_corrected_m = aux['areas_corrected_m']

else:

    json_files_f = []
    json_files_m = []
    areas_auto_f = []
    areas_auto_m = []
    areas_corrected_f = []
    areas_corrected_m = []
    for i, json_file in enumerate(json_annotation_files):

        print('file ' + str(i) + '/' + str(len(json_annotation_files) - 1) + ': ' + os.path.basename(json_file))

        # change file extension from .svg to .tif
        file_ndpi = json_file.replace('.json', '.ndpi')
        file_ndpi = os.path.join(data_dir, file_ndpi)

        # open histology training image
        im = openslide.OpenSlide(file_ndpi)

        # pixel size
        assert (im.properties['tiff.ResolutionUnit'] == 'centimeter')
        xres = 1e-2 / float(im.properties['tiff.XResolution']) * 1e6  # um^2
        yres = 1e-2 / float(im.properties['tiff.YResolution']) * 1e6  # um^2

        # create dataframe for this image
        df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(json_file),
                                                              values=[i,], values_tag='i',
                                                              tags_to_keep=['id', 'ko_parent', 'sex'])

        # mouse ID as a string
        id = df_common['id'].values[0]
        sex = df_common['sex'].values[0]
        ko = df_common['ko_parent'].values[0]

        # load list of contours in Auto and Corrected segmentations
        json_file_auto = os.path.join(annotations_dir, json_file.replace('.json', '_exp_0097_auto.json'))
        contours_auto = cytometer.data.aida_get_contours(json_file_auto, layer_name='White adipocyte.*')
        json_file_corrected = os.path.join(annotations_dir, json_file.replace('.json', '_exp_0097_corrected.json'))
        contours_corrected = cytometer.data.aida_get_contours(json_file_corrected, layer_name='White adipocyte.*')

        # loop items (one contour per item)
        areas_all = []
        for c in contours_auto:
            # compute cell area
            area = Polygon(c).area * xres * yres  # (um^2)
            areas_all.append(area)
        if sex == 'f':
            json_files_f.append(json_file)
            areas_auto_f.append(np.array(areas_all))
        elif sex == 'm':
            json_files_m.append(json_file)
            areas_auto_m.append(np.array(areas_all))

        areas_all = []
        for c in contours_corrected:
            # compute cell area
            area = Polygon(c).area * xres * yres  # (um^2)
            areas_all.append(area)
        if sex == 'f':
            areas_corrected_f.append(np.array(areas_all))
        elif sex == 'm':
            areas_corrected_m.append(np.array(areas_all))

    # save cell areas
    # corrected_areas_f = np.array(corrected_areas_f)
    # corrected_areas_m = np.array(corrected_areas_m)
    np.savez(filename_corrected_areas, json_files_f=json_files_f, json_files_m=json_files_m,
             areas_auto_f=areas_auto_f, areas_auto_m=areas_auto_m,
             areas_corrected_f=areas_corrected_f, areas_corrected_m=areas_corrected_m)

# add DeepCytometer data from all frames to the dataframe
df = pd.DataFrame()
df['Area'] = np.concatenate(areas_auto_f)
df['Sex'] = 'Female'
df['Method'] = 'Auto all'
df_all = df

df = pd.DataFrame()
df['Area'] = np.concatenate(areas_auto_m)
df['Sex'] = 'Male'
df['Method'] = 'Auto all'
df_all = pd.concat((df_all, df))

df = pd.DataFrame()
df['Area'] = np.concatenate(areas_corrected_f)
df['Sex'] = 'Female'
df['Method'] = 'Corrected all'
df_all = pd.concat((df_all, df))

df = pd.DataFrame()
df['Area'] = np.concatenate(areas_corrected_m)
df['Sex'] = 'Male'
df['Method'] = 'Corrected all'
df_all = pd.concat((df_all, df))

if DEBUG:
    plt.clf()
    plt.hist(np.concatenate(areas_auto_f), histtype='step', bins=200, density=True, label='Auto female')
    plt.hist(np.concatenate(areas_auto_m), histtype='step', bins=200, density=True, label='Auto male')
    plt.hist(np.concatenate(areas_corrected_f), histtype='step', bins=200, density=True, label='Corrected female')
    plt.hist(np.concatenate(areas_corrected_m), histtype='step', bins=200, density=True, label='Corrected male')
    plt.tick_params(labelsize=14)
    plt.legend()

    # boxplots of Auto vs. Corrected
    plt.clf()
    sns.boxplot(x='Sex', y='Area', data=df_all, hue='Method')
    plt.tick_params(labelsize=14)
    plt.xlabel('Sex', fontsize=14)
    plt.ylabel('White adipocyte area ($\mu m^2$)', fontsize=14)
    plt.tight_layout()

if DEBUG:
    # inspect outliers in Corrected (very slow)
    q = np.linspace(0, 1, 1001)
    quant_corrected_f = stats.mstats.hdquantiles(np.concatenate(areas_corrected_f), prob=q, axis=0)
    quant_corrected_m = stats.mstats.hdquantiles(np.concatenate(areas_corrected_m), prob=q, axis=0)

if DEBUG:
    plt.clf()
    plt.plot(q, quant_corrected_f)
    plt.tick_params(labelsize=14)
    plt.xlabel('Quantile', fontsize=14)
    plt.ylabel('White adipocyte area $\mu m^2$', fontsize=14)
    plt.tight_layout()

# clean outliers from the Corrected data, in dataframe and in vectors of vectors
max_cell_size = 22500
df_all = df_all.loc[df_all['Area'] <= max_cell_size]

areas_corrected_f = [x[x <= max_cell_size] for x in areas_corrected_f]
areas_corrected_m = [x[x <= max_cell_size] for x in areas_corrected_m]

if DEBUG:
    plt.clf()
    plt.hist(np.concatenate(areas_auto_f), histtype='step', bins=200, density=True, label='Auto female')
    plt.hist(np.concatenate(areas_auto_m), histtype='step', bins=200, density=True, label='Auto male')
    plt.hist(np.concatenate(areas_corrected_f), histtype='step', bins=200, density=True, label='Corrected female')
    plt.hist(np.concatenate(areas_corrected_m), histtype='step', bins=200, density=True, label='Corrected male')
    plt.legend()

# boxplots of Auto vs. Corrected
plt.clf()
ax = sns.boxplot(x='Sex', y='Area', data=df_all, hue='Method',
                 hue_order=('Hand traced', 'Auto training', 'Auto all', 'Corrected training', 'Corrected all'),
                 palette='muted', fliersize=1)
plt.tick_params(labelsize=14)
plt.xlabel('Sex', fontsize=14)
plt.ylabel('White adipocyte area ($\mu m^2$)', fontsize=14)
plt.tight_layout()

statannot.add_stat_annotation(ax,
                              x='Sex', y='Area', data=df_all, hue='Method',
                              hue_order=('Hand traced', 'Auto training', 'Auto all', 'Corrected training', 'Corrected all'),
                              box_pairs=[(('Female', 'Hand traced'), ('Female', 'Auto training')),
                                         (('Female', 'Hand traced'), ('Female', 'Corrected training')),
                                         (('Male', 'Hand traced'), ('Male', 'Auto training')),
                                         (('Male', 'Hand traced'), ('Male', 'Corrected training')),
                                         (('Female', 'Auto training'), ('Female', 'Auto all')),
                                         (('Female', 'Corrected training'), ('Female', 'Corrected all')),
                                         (('Male', 'Auto training'), ('Male', 'Auto all')),
                                         (('Male', 'Corrected training'), ('Male', 'Corrected all')),
                                         ],
                              test='Mann-Whitney', comparisons_correction='bonferroni',
                              text_format='star', loc='inside', verbose=2)
plt.legend(loc='lower left', bbox_to_anchor=(1.0, 0.0))
plt.tight_layout()

plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0098_boxplots_comparing_manual_auto_corrected_populations_in_training_set_or_all_slides.png'))
plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0098_boxplots_comparing_manual_auto_corrected_populations_in_training_set_or_all_slides.svg'))

# file that contains quantile-to-area functions
filename_area2quantile = os.path.join(area2quantile_dir, 'klf14_b6ntac_exp_0098_filename_area2quantile.npz')

if os.path.isfile(filename_area2quantile):
    with np.load(filename_area2quantile, allow_pickle=True) as aux:
        f_area2quantile_f = aux['f_area2quantile_f'].item()
        f_area2quantile_m = aux['f_area2quantile_m'].item()
else:
    # compute function to map between cell areas and [0.0, 1.0], that we can use to sample the colourmap uniformly according
    # to area quantiles
    f_area2quantile_f = cytometer.data.area2quantile(np.concatenate(areas_corrected_f), quantiles=np.linspace(0.0, 1.0, 1001))
    f_area2quantile_m = cytometer.data.area2quantile(np.concatenate(areas_corrected_m), quantiles=np.linspace(0.0, 1.0, 1001))
    np.savez(filename_area2quantile, f_area2quantile_f=f_area2quantile_f, f_area2quantile_m=f_area2quantile_m)

# load AIDA's colourmap
cm = cytometer.data.aida_colourmap()

########################################################################################################################
## Heatmaps of whole slides
########################################################################################################################

# loop annotations files
for i_file, json_file in enumerate(json_annotation_files):

    print('File: ' + str(i_file) + ': JSON annotations file: ' + os.path.basename(json_file))

    # name of corresponding .ndpi file
    ndpi_file = json_file.replace('.json', '.ndpi')
    kernel_file = os.path.splitext(ndpi_file)[0]

    # add path to file
    json_file = os.path.join(annotations_dir, json_file)
    ndpi_file = os.path.join(data_dir, ndpi_file)

    # open full resolution histology slide
    im = openslide.OpenSlide(ndpi_file)

    # pixel size
    assert (im.properties['tiff.ResolutionUnit'] == 'centimeter')
    xres = 1e-2 / float(im.properties['tiff.XResolution']) * 1e6  # um^2
    yres = 1e-2 / float(im.properties['tiff.YResolution']) * 1e6  # um^2

    # change pixel size to downsampled size
    xres *= downsample_factor
    yres *= downsample_factor

    # name of file to save rough mask, current mask, and time steps
    rough_mask_file = os.path.basename(ndpi_file)
    rough_mask_file = rough_mask_file.replace('.ndpi', '_rough_mask.npz')
    rough_mask_file = os.path.join(annotations_dir, rough_mask_file)

    # load coarse tissue mask
    with np.load(rough_mask_file) as aux:
        lores_istissue0 = aux['lores_istissue0']
        im_downsampled = aux['im_downsampled']

    if DEBUG:
        plt.clf()
        plt.subplot(211)
        plt.imshow(im_downsampled)
        plt.subplot(212)
        plt.imshow(lores_istissue0)

    # load list of contours in Auto and Corrected segmentations
    json_file_corrected = os.path.join(annotations_dir, json_file.replace('.json', '_exp_0097_corrected.json'))

    # list of items (there's a contour in each item)
    contours_corrected = cytometer.data.aida_get_contours(json_file_corrected, layer_name='White adipocyte.*')

    # init array for interpolated quantiles
    quantiles_grid = np.zeros(shape=lores_istissue0.shape, dtype=np.float32)

    # init array for mask where there are segmentations
    areas_mask = Image.new("1", lores_istissue0.shape[::-1], "black")
    draw = ImageDraw.Draw(areas_mask)

    # init lists for contour centroids and areas
    areas_all = []
    centroids_all = []

    # loop items (one contour per item)
    for c in contours_corrected:

        # convert to downsampled coordinates
        c = np.array(c) / downsample_factor

        if DEBUG:
            plt.fill(c[:, 0], c[:, 1], fill=False, color='r')

        # compute cell area
        area = Polygon(c).area * xres * yres  # (um^2)
        areas_all.append(area)

        # compute centroid of contour
        centroid = np.mean(c, axis=0)
        centroids_all.append(centroid)

        # add object described by contour to mask
        draw.polygon(list(c.flatten()), outline="white", fill="white")

    # convert mask
    areas_mask = np.array(areas_mask, dtype=np.bool)

    areas_all = np.array(areas_all)

    # interpolate scattered area data to regular grid
    idx = areas_mask
    xi = np.transpose(np.array(np.where(idx)))[:, [1, 0]]
    quantiles_grid[idx] = scipy.interpolate.griddata(centroids_all, areas_all, xi, method='linear', fill_value=0)

    if DEBUG:
        plt.clf()
        plt.subplot(211)
        plt.imshow(im_downsampled)
        plt.axis('off')
        plt.subplot(212)
        plt.imshow(quantiles_grid)

    # create dataframe for this image
    df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(ndpi_file),
                                                          values=[i,], values_tag='i',
                                                          tags_to_keep=['id', 'ko_parent', 'sex'])

    # mouse ID as a string
    id = df_common['id'].values[0]
    sex = df_common['sex'].values[0]
    ko = df_common['ko_parent'].values[0]

    # convert area values to quantiles
    if sex == 'f':
        quantiles_grid = f_area2quantile_f(quantiles_grid)
    elif sex == 'm':
        quantiles_grid = f_area2quantile_m(quantiles_grid)
    else:
        raise ValueError('Wrong sex value')

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

    # plot cell areas for paper
    plt.clf()
    plt.imshow(quantiles_grid, vmin=0.0, vmax=1.0, cmap=cm)
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, kernel_file + '_exp_0098_cell_segmentation.png'),
                bbox_inches='tight')

## Colourmaps for hand traced data

# colourmap plot
a = np.array([[0,1]])
plt.figure(figsize=(9, 1.5))
img = plt.imshow(a, cmap=cm)
plt.gca().set_visible(False)
cax = plt.axes([0.1, 0.2, 0.8, 0.6])
cbar = plt.colorbar(orientation='horizontal', cax=cax)
cbar.ax.tick_params(labelsize=14)
plt.title('Cell area quantile (w.r.t. manual dataset)', rotation=0, fontsize=14)
plt.tight_layout()

plt.savefig(os.path.join(figures_dir, 'exp_0098_aida_colourmap.png'), bbox_inches='tight')

# plot area distributions
plt.clf()
aq_f = stats.mstats.hdquantiles(np.array(manual_areas_f) * 1e-3, prob=np.linspace(0, 1, 11), axis=0)
for a in aq_f:
    plt.plot([a, a], [0, 0.55], 'r', linewidth=3)
plt.hist(np.array(manual_areas_f) * 1e-3, histtype='stepfilled', bins=50, density=True, linewidth=4, zorder=0)
plt.tick_params(labelsize=14)
plt.xlabel('Cell area ($\cdot 10^3\ \mu m^2$)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.yticks([])
plt.tight_layout()

plt.savefig(os.path.join(figures_dir, 'exp_0098_dist_quantiles_manual_f.png'), bbox_inches='tight')

plt.clf()
aq_m = stats.mstats.hdquantiles(np.array(manual_areas_m) * 1e-3, prob=np.linspace(0, 1, 11), axis=0)
for a in aq_m:
    plt.plot([a, a], [0, 0.28], 'r', linewidth=3)
plt.hist(np.array(manual_areas_m) * 1e-3, histtype='stepfilled', bins=50, density=True, linewidth=4, zorder=0)
plt.tick_params(labelsize=14)
plt.xlabel('Cell area ($\cdot 10^3\ \mu m^2$)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.yticks([])
plt.tight_layout()

plt.savefig(os.path.join(figures_dir, 'exp_0098_dist_quantiles_manual_m.png'), bbox_inches='tight')

## Colourmaps for all slides Corrected data

# plot area distributions
plt.clf()
aq_f = stats.mstats.hdquantiles(np.concatenate(areas_corrected_f) * 1e-3, prob=np.linspace(0, 1, 11), axis=0)
for a in aq_f:
    plt.plot([a, a], [0, 0.25], 'k', linewidth=3)
plt.hist(np.concatenate(areas_corrected_f) * 1e-3, histtype='stepfilled', bins=50, density=True, linewidth=4, zorder=0)
plt.tick_params(labelsize=14)
plt.xlabel('White adipocyte area ($\cdot 10^3\ \mu m^2$)', fontsize=14)
plt.yticks([])
plt.ylabel('Density', fontsize=14)
plt.tight_layout()

plt.savefig(os.path.join(figures_dir, 'exp_0098_dist_quantiles_corrected_all_f.png'), bbox_inches='tight')

# plot area distributions
plt.clf()
aq_m = stats.mstats.hdquantiles(np.concatenate(areas_corrected_m) * 1e-3, prob=np.linspace(0, 1, 11), axis=0)
for a in aq_m:
    plt.plot([a, a], [0, 0.17], 'k', linewidth=3)
plt.hist(np.concatenate(areas_corrected_m) * 1e-3, histtype='stepfilled', bins=50, density=True, linewidth=4, zorder=0)
plt.tick_params(labelsize=14)
plt.xlabel('White adipocyte area ($\cdot 10^3\ \mu m^2$)', fontsize=14)
plt.yticks([])
plt.ylabel('Density', fontsize=14)
plt.tight_layout()

plt.savefig(os.path.join(figures_dir, 'exp_0098_dist_quantiles_corrected_all_m.png'), bbox_inches='tight')

# colourmap plot
a = np.array([[0,1]])
plt.figure(figsize=(9, 1.75))
img = plt.imshow(a, cmap=cm)
plt.gca().set_visible(False)
cax = plt.axes([0.1, 0.4, 0.8, 0.4])
cbar = plt.colorbar(orientation='horizontal', cax=cax)
# for q in np.linspace(0, 1, 11):
#     plt.plot([q, q], [0, 1.0], 'k', linewidth=3)
cbar.ax.tick_params(labelsize=14)
cbar.set_ticks(np.linspace(0, 1, 11))
plt.title('Quantile colour map', rotation=0, fontsize=14)
cbar.ax.set_xlabel('Quantile', fontsize=14)
plt.tight_layout()

plt.savefig(os.path.join(figures_dir, 'exp_0098_aida_colourmap.png'), bbox_inches='tight')
