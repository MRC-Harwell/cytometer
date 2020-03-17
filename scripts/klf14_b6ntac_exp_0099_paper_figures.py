"""
Generate figures for the DeepCytometer paper.

Code cannibalised from:
* klf14_b6ntac_exp_0097_full_slide_pipeline_v7.py

"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0099_paper_figures'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])

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
## Explore training/test data of different folds
########################################################################################################################

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K

import cytometer
import cytometer.data
import tensorflow as tf

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

LIMIT_GPU_MEMORY = False

# limit GPU memory used
if LIMIT_GPU_MEMORY:
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    set_session(tf.Session(config=config))

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False

'''Directories and filenames'''

# data paths
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
klf14_training_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training')
klf14_training_non_overlap_data_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training_non_overlap')
klf14_training_augmented_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training_augmented')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
saved_models_dir = os.path.join(klf14_root_data_dir, 'saved_models')
ndpi_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
metainfo_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')

saved_kfolds_filename = 'klf14_b6ntac_exp_0079_generate_kfolds.pickle'
saved_extra_kfolds_filename = 'klf14_b6ntac_exp_0094_generate_extra_training_images.pickle'


# original dataset used in pipelines up to v6 + extra "other" tissue images
kfold_filename = os.path.join(saved_models_dir, saved_extra_kfolds_filename)
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

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# loop the folds to get the ndpi files that correspond to testing of each fold
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
    background_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Background', add_offset_from_filename=False,
                                                                  minimum_npoints=3)
    contours = cell_contours + other_contours + brown_contours + background_contours

    # make a list with the type of cell each contour is classified as
    contour_type = [np.zeros(shape=(len(cell_contours),), dtype=np.uint8),  # 0: white-adipocyte
                    np.ones(shape=(len(other_contours),), dtype=np.uint8),  # 1: other types of tissue
                    np.ones(shape=(len(brown_contours),), dtype=np.uint8),  # 1: brown cells (treated as "other" tissue)
                    np.zeros(shape=(len(background_contours),), dtype=np.uint8)] # 0: background
    contour_type = np.concatenate(contour_type)

    print('Cells: ' + str(len(cell_contours)) + '. Other: ' + str(len(other_contours))
          + '. Brown: ' + str(len(brown_contours)) + '. Background: ' + str(len(background_contours)))

    # create dataframe for this image
    df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_svg),
                                                          values=[i,], values_tag='i',
                                                          tags_to_keep=['id', 'ko_parent', 'sex'])

    # mouse ID as a string
    id = df_common['id'].values[0]
    sex = df_common['sex'].values[0]
    ko = df_common['ko_parent'].values[0]

    # row to add to the table
    df = pd.DataFrame(
        [(sex, ko,
          len(cell_contours), len(other_contours) + len(brown_contours), len(background_contours), 1, int(len(cell_contours)>0))],
        columns=['Sex', 'Genotype', 'Cells', 'Other', 'Background', 'Windows', 'Windows with cells'], index=[id])

    if id in table.index:

        num_cols = ['Cells', 'Other', 'Background', 'Windows', 'Windows with cells']
        table.loc[id, num_cols] = (table.loc[id, num_cols] + df.loc[id, num_cols])

    else:

        table = table.append(df, sort=False, ignore_index=False, verify_integrity=True)

# alphabetical order by mouse IDs
table = table.sort_index()

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

print('f PAT: ' + str(np.sum(table.loc[idx_f * idx_pat, 'Cells'])))
print('f MAT: ' + str(np.sum(table.loc[idx_f * idx_mat, 'Cells'])))
print('m PAT: ' + str(np.sum(table.loc[idx_m * idx_pat, 'Cells'])))
print('m MAT: ' + str(np.sum(table.loc[idx_m * idx_mat, 'Cells'])))

# find folds that test images belong to
for i_file, ndpi_file_kernel in enumerate(ndpi_files_test_list):

    # fold  where the current .ndpi image was not used for training
    i_fold = ndpi_files_test_list[ndpi_file_kernel]

    print('File ' + str(i_file) + '/' + str(len(ndpi_files_test_list) - 1) + ': ' + ndpi_file_kernel
          + '. Fold = ' + str(i_fold))

# mean and std of mouse weight
weight_f_mat = [22.07, 26.39, 30.65, 24.28, 27.72]
weight_f_pat = [31.42, 29.25, 27.18, 23.69, 21.20]
weight_m_mat = [46.19, 40.87, 40.02, 41.98, 34.52, 36.08]
weight_m_pat = [36.55, 40.77, 36.98, 36.11]

print('f MAT: mean = ' + str(np.mean(weight_f_mat)) + ', std = ' + str(np.std(weight_f_mat)))
print('f PAT: mean = ' + str(np.mean(weight_f_pat)) + ', std = ' + str(np.std(weight_f_pat)))
print('m MAT: mean = ' + str(np.mean(weight_m_mat)) + ', std = ' + str(np.std(weight_m_mat)))
print('m PAT: mean = ' + str(np.mean(weight_m_pat)) + ', std = ' + str(np.std(weight_m_pat)))

########################################################################################################################
## Plots of get_next_roi_to_process()
########################################################################################################################

import pickle
import cytometer.utils

# Filter out INFO & WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # limit number of GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

os.environ['KERAS_BACKEND'] = 'tensorflow'

import time
import openslide
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import fftconvolve
from cytometer.utils import rough_foreground_mask
import PIL
from keras import backend as K

LIMIT_GPU_MEMORY = False

# limit GPU memory used
if LIMIT_GPU_MEMORY:
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    set_session(tf.Session(config=config))

DEBUG = False
SAVE_FIGS = False

root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
data_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
training_dir = os.path.join(home, root_data_dir, 'klf14_b6ntac_training')
seg_dir = os.path.join(home, root_data_dir, 'klf14_b6ntac_seg')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')
results_dir = os.path.join(root_data_dir, 'klf14_b6ntac_results')
annotations_dir = os.path.join(home, 'Software/AIDA/dist/data/annotations')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')

# k-folds file
saved_extra_kfolds_filename = 'klf14_b6ntac_exp_0094_generate_extra_training_images.pickle'

# model names
dmap_model_basename = 'klf14_b6ntac_exp_0086_cnn_dmap'
contour_model_basename = 'klf14_b6ntac_exp_0091_cnn_contour_after_dmap'
classifier_model_basename = 'klf14_b6ntac_exp_0095_cnn_tissue_classifier_fcn'
correction_model_basename = 'klf14_b6ntac_exp_0089_cnn_segmentation_correction_overlapping_scaled_contours'

# full resolution image window and network expected receptive field parameters
fullres_box_size = np.array([2751, 2751])
receptive_field = np.array([131, 131])

# rough_foreground_mask() parameters
downsample_factor = 8.0
dilation_size = 25
component_size_threshold = 1e6
hole_size_treshold = 8000

# contour parameters
contour_downsample_factor = 0.1
bspline_k = 1

# block_split() parameters in downsampled image
block_len = np.ceil((fullres_box_size - receptive_field) / downsample_factor)
block_overlap = np.ceil((receptive_field - 1) / 2 / downsample_factor).astype(np.int)

# segmentation parameters
min_cell_area = 1500
max_cell_area = 100e3
min_mask_overlap = 0.8
phagocytosis = True
min_class_prop = 0.5
correction_window_len = 401
correction_smoothing = 11
batch_size = 16

# segmentation correction parameters

# load list of images, and indices for training vs. testing indices
saved_kfolds_filename = os.path.join(saved_models_dir, saved_extra_kfolds_filename)
with open(saved_kfolds_filename, 'rb') as f:
    aux = pickle.load(f)
file_svg_list = aux['file_list']
idx_test_all = aux['idx_test']
idx_train_all = aux['idx_train']

# correct home directory
file_svg_list = [x.replace('/users/rittscher/rcasero', home) for x in file_svg_list]
file_svg_list = [x.replace('/home/rcasero', home) for x in file_svg_list]

# loop the folds to get the ndpi files that correspond to testing of each fold
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

# File 4/19: KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04. Fold = 2
i_file = 4
# File 10/19: KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38. Fold = 5
i_file = 10

ndpi_file_kernel = list(ndpi_files_test_list.keys())[i_file]

# for i_file, ndpi_file_kernel in enumerate(ndpi_files_test_list):

# fold  where the current .ndpi image was not used for training
i_fold = ndpi_files_test_list[ndpi_file_kernel]

print('File ' + str(i_file) + '/' + str(len(ndpi_files_test_list) - 1) + ': ' + ndpi_file_kernel
      + '. Fold = ' + str(i_fold))

# make full path to ndpi file
ndpi_file = os.path.join(data_dir, ndpi_file_kernel + '.ndpi')

contour_model_file = os.path.join(saved_models_dir, contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
dmap_model_file = os.path.join(saved_models_dir, dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
classifier_model_file = os.path.join(saved_models_dir,
                                     classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')
correction_model_file = os.path.join(saved_models_dir,
                                     correction_model_basename + '_model_fold_' + str(i_fold) + '.h5')

# name of file to save annotations
annotations_file = os.path.basename(ndpi_file)
annotations_file = os.path.splitext(annotations_file)[0]
annotations_file = os.path.join(annotations_dir, annotations_file + '_exp_0097.json')

# name of file to save areas and contours
results_file = os.path.basename(ndpi_file)
results_file = os.path.splitext(results_file)[0]
results_file = os.path.join(results_dir, results_file + '_exp_0097.npz')

# rough segmentation of the tissue in the image
lores_istissue0, im_downsampled = rough_foreground_mask(ndpi_file, downsample_factor=downsample_factor,
                                                        dilation_size=dilation_size,
                                                        component_size_threshold=component_size_threshold,
                                                        hole_size_treshold=hole_size_treshold,
                                                        return_im=True)

if DEBUG:
    plt.clf()
    plt.imshow(im_downsampled)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_histology_i_file_' + str(i_file) + '.png'),
                bbox_inches='tight')

    plt.clf()
    plt.imshow(im_downsampled)
    plt.contour(lores_istissue0, colors='k')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_rough_mask_i_file_' + str(i_file) + '.png'),
                bbox_inches='tight')

# segmentation copy, to keep track of what's left to do
lores_istissue = lores_istissue0.copy()

# open full resolution histology slide
im = openslide.OpenSlide(ndpi_file)

# pixel size
assert(im.properties['tiff.ResolutionUnit'] == 'centimeter')
xres = 1e-2 / float(im.properties['tiff.XResolution'])
yres = 1e-2 / float(im.properties['tiff.YResolution'])

# # init empty list to store area values and contour coordinates
# areas_all = []
# contours_all = []

# keep extracting histology windows until we have finished
step = -1
time_0 = time_curr = time.time()
while np.count_nonzero(lores_istissue) > 0:

    # next step (it starts from 0)
    step += 1

    time_prev = time_curr
    time_curr = time.time()

    print('File ' + str(i_file) + '/' + str(len(ndpi_files_test_list) - 1) + ': step ' +
          str(step) + ': ' +
          str(np.count_nonzero(lores_istissue)) + '/' + str(np.count_nonzero(lores_istissue0)) + ': ' +
          "{0:.1f}".format(100.0 - np.count_nonzero(lores_istissue) / np.count_nonzero(lores_istissue0) * 100) +
          '% completed: ' +
          'step time ' + "{0:.2f}".format(time_curr - time_prev) + ' s' +
          ', total time ' + "{0:.2f}".format(time_curr - time_0) + ' s')

    ## Code extracted from:
    ## get_next_roi_to_process()

    # variables for get_next_roi_to_process()
    seg = lores_istissue.copy()
    downsample_factor = downsample_factor
    max_window_size = fullres_box_size
    border = np.round((receptive_field - 1) / 2)

    # convert to np.array so that we can use algebraic operators
    max_window_size = np.array(max_window_size)
    border = np.array(border)

    # convert segmentation mask to [0, 1]
    seg = (seg != 0).astype('int')

    # approximate measures in the downsampled image (we don't round them)
    lores_max_window_size = max_window_size / downsample_factor
    lores_border = border / downsample_factor

    # kernels that flipped correspond to top line and left line. They need to be pre-flipped
    # because the convolution operation internally flips them (two flips cancel each other)
    kernel_top = np.zeros(shape=np.round(lores_max_window_size - 2 * lores_border).astype('int'))
    kernel_top[int((kernel_top.shape[0] - 1) / 2), :] = 1
    kernel_left = np.zeros(shape=np.round(lores_max_window_size - 2 * lores_border).astype('int'))
    kernel_left[:, int((kernel_top.shape[1] - 1) / 2)] = 1

    if DEBUG:
        plt.clf()
        plt.imshow(kernel_top)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_kernel_top_i_file_' + str(i_file) + '.png'),
                    bbox_inches='tight')

        plt.imshow(kernel_left)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_kernel_left_i_file_' + str(i_file) + '.png'),
                    bbox_inches='tight')

    seg_top = np.round(fftconvolve(seg, kernel_top, mode='same'))
    seg_left = np.round(fftconvolve(seg, kernel_left, mode='same'))

    # window detections
    detection_idx = np.nonzero(seg_left * seg_top)

    # set top-left corner of the box = top-left corner of first box detected
    lores_first_row = detection_idx[0][0]
    lores_first_col = detection_idx[1][0]

    # first, we look within a window with the maximum size
    lores_last_row = detection_idx[0][0] + lores_max_window_size[0] - 2 * lores_border[0]
    lores_last_col = detection_idx[1][0] + lores_max_window_size[1] - 2 * lores_border[1]

    # second, if the segmentation is smaller than the window, we reduce the window size
    window = seg[lores_first_row:int(np.round(lores_last_row)), lores_first_col:int(np.round(lores_last_col))]

    idx = np.any(window, axis=1)  # reduce rows size
    last_segmented_pixel_len = np.max(np.where(idx))
    lores_last_row = detection_idx[0][0] + np.min((lores_max_window_size[0] - 2 * lores_border[0],
                                                   last_segmented_pixel_len))

    idx = np.any(window, axis=0)  # reduce cols size
    last_segmented_pixel_len = np.max(np.where(idx))
    lores_last_col = detection_idx[1][0] + np.min((lores_max_window_size[1] - 2 * lores_border[1],
                                                   last_segmented_pixel_len))

    # save coordinates for plot (this is only for a figure in the paper and doesn't need to be done in the real
    # implementation)
    lores_first_col_bak = lores_first_col
    lores_first_row_bak = lores_first_row
    lores_last_col_bak = lores_last_col
    lores_last_row_bak = lores_last_row

    # add a border around the window
    lores_first_row = np.max([0, lores_first_row - lores_border[0]])
    lores_first_col = np.max([0, lores_first_col - lores_border[1]])

    lores_last_row = np.min([seg.shape[0], lores_last_row + lores_border[0]])
    lores_last_col = np.min([seg.shape[1], lores_last_col + lores_border[1]])

    # convert low resolution indices to high resolution
    first_row = np.int(np.round(lores_first_row * downsample_factor))
    last_row = np.int(np.round(lores_last_row * downsample_factor))
    first_col = np.int(np.round(lores_first_col * downsample_factor))
    last_col = np.int(np.round(lores_last_col * downsample_factor))

    # round down indices in downsampled segmentation
    lores_first_row = int(lores_first_row)
    lores_last_row = int(lores_last_row)
    lores_first_col = int(lores_first_col)
    lores_last_col = int(lores_last_col)

    # load window from full resolution slide
    tile = im.read_region(location=(first_col, first_row), level=0,
                          size=(last_col - first_col, last_row - first_row))
    tile = np.array(tile)
    tile = tile[:, :, 0:3]

    # interpolate coarse tissue segmentation to full resolution
    istissue_tile = lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col]
    istissue_tile = cytometer.utils.resize(istissue_tile, size=(last_col - first_col, last_row - first_row),
                                           resample=PIL.Image.NEAREST)

    if DEBUG:
        plt.clf()
        plt.imshow(tile)
        plt.imshow(istissue_tile, alpha=0.5)
        plt.contour(istissue_tile, colors='k')
        plt.title('Yellow: Tissue. Purple: Background')
        plt.axis('off')

    # clear keras session to prevent each segmentation iteration from getting slower. Note that this forces us to
    # reload the models every time
    K.clear_session()

    # segment histology, split into individual objects, and apply segmentation correction
    labels, labels_class, todo_edge, \
    window_im, window_labels, window_labels_corrected, window_labels_class, index_list, scaling_factor_list \
        = cytometer.utils.segmentation_pipeline6(tile,
                                                 dmap_model=dmap_model_file,
                                                 contour_model=contour_model_file,
                                                 correction_model=correction_model_file,
                                                 classifier_model=classifier_model_file,
                                                 min_cell_area=min_cell_area,
                                                 mask=istissue_tile,
                                                 min_mask_overlap=min_mask_overlap,
                                                 phagocytosis=phagocytosis,
                                                 min_class_prop=min_class_prop,
                                                 correction_window_len=correction_window_len,
                                                 correction_smoothing=correction_smoothing,
                                                 return_bbox=True, return_bbox_coordinates='xy',
                                                 batch_size=batch_size)

    # downsample "to do" mask so that the rough tissue segmentation can be updated
    lores_todo_edge = PIL.Image.fromarray(todo_edge.astype(np.uint8))
    lores_todo_edge = lores_todo_edge.resize((lores_last_col - lores_first_col,
                                              lores_last_row - lores_first_row),
                                             resample=PIL.Image.NEAREST)
    lores_todo_edge = np.array(lores_todo_edge)

    # update coarse tissue mask (this is only necessary here to plot figures for the paper. In the actual code,
    # the coarse mask gets directly updated, without this intermediate step)
    seg_updated = seg.copy()
    seg_updated[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = lores_todo_edge

    if DEBUG:
        plt.clf()
        fig = plt.imshow(seg, cmap='Greys')
        plt.contour(seg_left * seg_top > 0, colors='r')
        rect = Rectangle((lores_first_col, lores_first_row),
                         lores_last_col - lores_first_col, lores_last_row - lores_first_row,
                         alpha=0.5, facecolor='g', edgecolor='g', zorder=2)
        fig.axes.add_patch(rect)
        rect2 = Rectangle((lores_first_col_bak, lores_first_row_bak),
                          lores_last_col_bak - lores_first_col_bak, lores_last_row_bak - lores_first_row_bak,
                          alpha=1.0, facecolor=None, fill=False, edgecolor='g', lw=1, zorder=3)
        fig.axes.add_patch(rect2)
        plt.scatter(detection_idx[1][0], detection_idx[0][0], color='k', s=5, zorder=3)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_fftconvolve_i_file_' + str(i_file) +
                                 '_step_' + str(step) + '.png'),
                    bbox_inches='tight')

    if DEBUG:
        plt.clf()
        fig = plt.imshow(seg, cmap='Greys')
        plt.contour(seg_left * seg_top > 0, colors='r')
        plt.contour(seg_updated, colors='w', zorder=4)
        rect = Rectangle((lores_first_col, lores_first_row),
                         lores_last_col - lores_first_col, lores_last_row - lores_first_row,
                         alpha=0.5, facecolor='g', edgecolor='g', zorder=2)
        fig.axes.add_patch(rect)
        rect2 = Rectangle((lores_first_col_bak, lores_first_row_bak),
                          lores_last_col_bak - lores_first_col_bak, lores_last_row_bak - lores_first_row_bak,
                          alpha=1.0, facecolor=None, fill=False, edgecolor='g', lw=3, zorder=3)
        fig.axes.add_patch(rect2)
        plt.scatter(detection_idx[1][0], detection_idx[0][0], color='k', s=5, zorder=3)
        plt.axis('off')
        plt.tight_layout()
        plt.xlim(int(lores_first_col - 50), int(lores_last_col + 50))
        plt.ylim(int(lores_last_row + 50), int(lores_first_row - 50))
        plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_fftconvolve_detail_i_file_' + str(i_file) +
                                 '_step_' + str(step) + '.png'),
                    bbox_inches='tight')

        # update coarse tissue mask for next iteration
        lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = lores_todo_edge

########################################################################################################################
## Show examples of what each deep CNN do (code cannibilised from the "inspect" scripts of the networks)
########################################################################################################################

import pickle
import warnings

# other imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

import cytometer.data
import cytometer.utils
import cytometer.model_checkpoint_parallel
import tensorflow as tf

from PIL import Image, ImageDraw
import math

LIMIT_GPU_MEMORY = False

# limit GPU memory used
if LIMIT_GPU_MEMORY:
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    set_session(tf.Session(config=config))

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False

'''Directories and filenames'''

# data paths
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
klf14_training_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training')
klf14_training_non_overlap_data_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training_non_overlap')
klf14_training_augmented_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training_augmented')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
saved_models_dir = os.path.join(klf14_root_data_dir, 'saved_models')

saved_kfolds_filename = 'klf14_b6ntac_exp_0079_generate_kfolds.pickle'

# model names
dmap_model_basename = 'klf14_b6ntac_exp_0086_cnn_dmap'
contour_model_basename = 'klf14_b6ntac_exp_0091_cnn_contour_after_dmap'
classifier_model_basename = 'klf14_b6ntac_exp_0095_cnn_tissue_classifier_fcn'
correction_model_basename = 'klf14_b6ntac_exp_0089_cnn_segmentation_correction_overlapping_scaled_contours'


'''Load folds'''

# load list of images, and indices for training vs. testing indices
contour_model_kfold_filename = os.path.join(saved_models_dir, saved_kfolds_filename)
with open(contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
svg_file_list = aux['file_list']
idx_test_all = aux['idx_test']
idx_train_all = aux['idx_train']

if DEBUG:
    for i, file in enumerate(svg_file_list):
        print(str(i) + ': ' + file)

# correct home directory
svg_file_list = [x.replace('/home/rcasero', home) for x in svg_file_list]

# KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06_row_017204_col_019444.tif (fold 5 for testing. No .svg)
# KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06_row_009644_col_061660.tif (fold 5 for testing. No .svg)
# KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00_row_019228_col_015060.svg (fold 7 for testing. With .svg)

# find which fold the testing image belongs to
np.where(['36.1c' in x for x in svg_file_list])
idx_test_all[7]

# TIFF files that correspond to the SVG files (without augmentation)
im_orig_file_list = []
for i, file in enumerate(svg_file_list):
    im_orig_file_list.append(file.replace('.svg', '.tif'))
    im_orig_file_list[i] = os.path.join(os.path.dirname(im_orig_file_list[i]) + '_augmented',
                                        'im_seed_nan_' + os.path.basename(im_orig_file_list[i]))

    # check that files exist
    if not os.path.isfile(file):
        # warnings.warn('i = ' + str(i) + ': File does not exist: ' + os.path.basename(file))
        warnings.warn('i = ' + str(i) + ': File does not exist: ' + file)
    if not os.path.isfile(im_orig_file_list[i]):
        # warnings.warn('i = ' + str(i) + ': File does not exist: ' + os.path.basename(im_orig_file_list[i]))
        warnings.warn('i = ' + str(i) + ': File does not exist: ' + im_orig_file_list[i])

'''Inspect model results'''

# for i_fold, idx_test in enumerate(idx_test_all):
i_fold = 7; idx_test = idx_test_all[i_fold]

print('Fold ' + str(i_fold) + '/' + str(len(idx_test_all)-1))

'''Load data'''

# split the data list into training and testing lists
im_test_file_list, im_train_file_list = cytometer.data.split_list(im_orig_file_list, idx_test)

# load the test data (im, dmap, mask)
test_dataset, test_file_list, test_shuffle_idx = \
    cytometer.data.load_datasets(im_test_file_list, prefix_from='im', prefix_to=['im', 'dmap', 'mask', 'contour'],
                                 nblocks=1, shuffle_seed=None)

# fill in the little gaps in the mask
kernel = np.ones((3, 3), np.uint8)
for i in range(test_dataset['mask'].shape[0]):
    test_dataset['mask'][i, :, :, 0] = cv2.dilate(test_dataset['mask'][i, :, :, 0].astype(np.uint8),
                                                  kernel=kernel, iterations=1)

# load dmap model, and adjust input size
saved_model_filename = os.path.join(saved_models_dir, dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
dmap_model = keras.models.load_model(saved_model_filename)
if dmap_model.input_shape[1:3] != test_dataset['im'].shape[1:3]:
    dmap_model = cytometer.utils.change_input_size(dmap_model, batch_shape=test_dataset['im'].shape)

# estimate dmaps
pred_dmap = dmap_model.predict(test_dataset['im'], batch_size=4)

if DEBUG:
    for i in range(test_dataset['im'].shape[0]):
        plt.clf()
        plt.subplot(221)
        plt.imshow(test_dataset['im'][i, :, :, :])
        plt.axis('off')
        plt.subplot(222)
        plt.imshow(test_dataset['dmap'][i, :, :, 0])
        plt.axis('off')
        plt.subplot(223)
        plt.imshow(test_dataset['mask'][i, :, :, 0])
        plt.axis('off')
        plt.subplot(224)
        plt.imshow(pred_dmap[i, :, :, 0])
        plt.axis('off')

# KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00_row_019228_col_015060.svg
i = 2

if DEBUG:

    plt.clf()
    plt.imshow(test_dataset['im'][i, :, :, :])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

    plt.clf()
    plt.imshow(test_dataset['dmap'][i, :, :, 0])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'dmap_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

    plt.clf()
    plt.imshow(pred_dmap[i, :, :, 0])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pred_dmap_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

# load dmap to contour model, and adjust input size
saved_model_filename = os.path.join(saved_models_dir, contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
contour_model = keras.models.load_model(saved_model_filename)
if contour_model.input_shape[1:3] != pred_dmap.shape[1:3]:
    contour_model = cytometer.utils.change_input_size(contour_model, batch_shape=pred_dmap.shape)

# estimate contours
pred_contour = contour_model.predict(pred_dmap, batch_size=4)

if DEBUG:
    plt.clf()
    plt.imshow(test_dataset['contour'][i, :, :, 0])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'contour_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

    plt.clf()
    plt.imshow(pred_contour[i, :, :, 0])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pred_contour_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

# load classifier model, and adjust input size
saved_model_filename = os.path.join(saved_models_dir, classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')
classifier_model = keras.models.load_model(saved_model_filename)
if classifier_model.input_shape[1:3] != test_dataset['im'].shape[1:3]:
    classifier_model = cytometer.utils.change_input_size(classifier_model, batch_shape=test_dataset['im'].shape)

# estimate pixel-classification
pred_class = classifier_model.predict(test_dataset['im'], batch_size=4)

if DEBUG:
    plt.clf()
    plt.imshow(pred_class[i, :, :, 0])
    plt.contour(pred_class[i, :, :, 0] > 0.5, colors='r', linewidhts=3)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pred_class_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

    plt.clf()
    plt.imshow(pred_class[i, :, :, 0] > 0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pred_class_thresh_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

## create classifier ground truth

# print('file ' + str(i) + '/' + str(len(file_svg_list) - 1))

# init output
im_array_all = []
out_class_all = []
out_mask_all = []
contour_type_all = []

file_tif = os.path.join(klf14_training_dir, os.path.basename(im_test_file_list[i]))
file_tif = file_tif.replace('im_seed_nan_', '')

# change file extension from .svg to .tif
file_svg = file_tif.replace('.tif', '.svg')

# open histology training image
im = Image.open(file_tif)

# make array copy
im_array = np.array(im)

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
                np.ones(shape=(len(brown_contours),), dtype=np.uint8),  # 1: brown cells (treated as "other" tissue)
                np.zeros(shape=(len(background_contours),), dtype=np.uint8)]  # 0: background
contour_type = np.concatenate(contour_type)
contour_type_all.append(contour_type)

print('Cells: ' + str(len(cell_contours)))
print('Other: ' + str(len(other_contours)))
print('Brown: ' + str(len(brown_contours)))
print('Background: ' + str(len(background_contours)))

# initialise arrays for training
out_class = np.zeros(shape=im_array.shape[0:2], dtype=np.uint8)
out_mask = np.zeros(shape=im_array.shape[0:2], dtype=np.uint8)

# loop ground truth cell contours
for j, contour in enumerate(contours):

    plt.plot([p[0] for p in contour], [p[1] for p in contour])
    plt.text(contour[0][0], contour[0][1], str(j))

    if DEBUG:
        plt.clf()

        plt.subplot(121)
        plt.imshow(im_array)
        plt.plot([p[0] for p in contour], [p[1] for p in contour])
        xy_c = (np.mean([p[0] for p in contour]), np.mean([p[1] for p in contour]))
        plt.scatter(xy_c[0], xy_c[1])

    # rasterise current ground truth segmentation
    cell_seg_gtruth = Image.new("1", im_array.shape[0:2][::-1], "black")  # I = 32-bit signed integer pixels
    draw = ImageDraw.Draw(cell_seg_gtruth)
    draw.polygon(contour, outline="white", fill="white")
    cell_seg_gtruth = np.array(cell_seg_gtruth, dtype=np.bool)

    # we are going to save the ground truth segmentation of the cell that we are going to later use in the figures
    if j == 106:
        cell_seg_gtruth_106 = cell_seg_gtruth.copy()

    if DEBUG:
        plt.subplot(122)
        plt.cla()
        plt.imshow(im_array)
        plt.contour(cell_seg_gtruth.astype(np.uint8))

    # add current object to training output and mask
    out_mask[cell_seg_gtruth] = 1
    out_class[cell_seg_gtruth] = contour_type[j]

if DEBUG:
    plt.clf()
    aux = (1- out_class).astype(np.float32)
    aux = np.ma.masked_where(out_mask < 0.5, aux)
    plt.imshow(aux)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'class_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

## Segmentation correction CNN

# segmentation parameters
min_cell_area = 1500
max_cell_area = 100e3
min_mask_overlap = 0.8
phagocytosis = True
min_class_prop = 0.5
correction_window_len = 401
correction_smoothing = 11
batch_size = 2

# segment histology
labels, labels_class, _ \
    = cytometer.utils.segment_dmap_contour_v6(im_array,
                                              contour_model=contour_model, dmap_model=dmap_model,
                                              classifier_model=classifier_model,
                                              border_dilation=0)
labels = labels[0, :, :]
labels_class = labels_class[0, :, :, 0]

if DEBUG:
    plt.clf()
    plt.imshow(labels)

# remove labels that touch the edges, that are too small or too large, don't overlap enough with the tissue mask,
# are fully surrounded by another label or are not white adipose tissue
labels, todo_edge = cytometer.utils.clean_segmentation(
    labels, min_cell_area=min_cell_area, max_cell_area=max_cell_area,
    remove_edge_labels=True, mask=None, min_mask_overlap=min_mask_overlap,
    phagocytosis=phagocytosis,
    labels_class=labels_class, min_class_prop=min_class_prop)

if DEBUG:
    plt.clf()
    plt.imshow(im_array)
    plt.contour(labels, levels=np.unique(labels), colors='k')
    plt.contourf(labels == 0)

# split image into individual labels
im_array = np.expand_dims(im_array, axis=0)
labels = np.expand_dims(labels, axis=0)
labels_class = np.expand_dims(labels_class, axis=0)
cell_seg_gtruth_106 = np.expand_dims(cell_seg_gtruth_106, axis=0)
window_mask = None
(window_labels, window_im, window_labels_class, window_cell_seg_gtruth_106), index_list, scaling_factor_list \
    = cytometer.utils.one_image_per_label_v2((labels, im_array, labels_class, cell_seg_gtruth_106.astype(np.uint8)),
                                             resize_to=(correction_window_len, correction_window_len),
                                             resample=(Image.NEAREST, Image.LINEAR, Image.NEAREST, Image.NEAREST),
                                             only_central_label=True, return_bbox=False)

# load correction model
saved_model_filename = os.path.join(saved_models_dir, correction_model_basename + '_model_fold_' + str(i_fold) + '.h5')
correction_model = keras.models.load_model(saved_model_filename)
if correction_model.input_shape[1:3] != window_im.shape[1:3]:
    correction_model = cytometer.utils.change_input_size(correction_model, batch_shape=window_im.shape)

# multiply image by mask
window_im_masked = cytometer.utils.quality_model_mask(
    np.expand_dims(window_labels, axis=-1), im=window_im, quality_model_type='-1_1')

# process (histology * mask) to estimate which pixels are underestimated and which overestimated in the segmentation
window_im_masked = correction_model.predict(window_im_masked, batch_size=batch_size)

# compute the correction to be applied to the segmentation
correction = (window_im[:, :, :, 0].copy() * 0).astype(np.float32)
correction[window_im_masked[:, :, :, 0] >= 0.5] = 1  # the segmentation went too far
correction[window_im_masked[:, :, :, 0] <= -0.5] = -1  # the segmentation fell short

if DEBUG:
    j = 0

    plt.clf()
    plt.imshow(correction[j, :, :])
    # plt.contour(window_labels[j, ...], colors='r', linewidths=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pred_correction_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

    plt.clf()
    plt.imshow(correction[j, :, :])
    plt.contour(window_labels[j, ...], colors='r', linewidths=1)
    plt.contour(window_cell_seg_gtruth_106[j, ...], colors='w', linewidths=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(
        os.path.join(figures_dir, 'pred_correction_gtruth_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
        bbox_inches='tight')

# correct segmentation (full operation)
window_im = window_im.astype(np.float32)
window_im /= 255.0
window_labels_corrected = cytometer.utils.correct_segmentation(
    im=window_im, seg=window_labels,
    correction_model=correction_model, model_type='-1_1',
    smoothing=correction_smoothing,
    batch_size=batch_size)

if DEBUG:
    j = 0

    plt.clf()
    plt.imshow(window_im[j, ...])
    plt.contour(window_labels[j, ...], colors='r', linewidths=3)
    plt.text(185, 210, '+1', fontsize=30)
    plt.text(116, 320, '-1', fontsize=30)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'im_for_correction_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

    plt.clf()
    plt.imshow(window_im[j, ...])
    plt.contour(window_labels_corrected[j, ...], colors='g', linewidths=3)
    plt.text(185, 210, '+1', fontsize=30)
    plt.text(116, 320, '-1', fontsize=30)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'corrected_seg_' + os.path.basename(im_test_file_list[i]).replace('.tif', '.png')),
                bbox_inches='tight')

    aux = np.array(contours[j])
    plt.plot(aux[:, 0], aux[:, 1])

########################################################################################################################
## Check whether manual correction of pipeline results makes a difference
########################################################################################################################

import matplotlib.pyplot as plt
import cytometer.data
from shapely.geometry import Polygon
import openslide
import numpy as np
import scipy.stats
import pandas as pd
from mlxtend.evaluate import permutation_test
from statsmodels.stats.multitest import multipletests
import math

# directories
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
annotations_dir = os.path.join(home, 'Software/AIDA/dist/data/annotations')
ndpi_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
metainfo_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')

DEBUG = False

depot = 'sqwat'
# depot = 'gwat'

permutation_sample_size = 9  # the factorial of this number is the number of repetitions in the permutation tests

# list of annotation files for this depot
json_annotation_files = json_annotation_files_dict[depot]

# modify filenames to select the particular segmentation we want (e.g. the automatic ones, or the manually refined ones)
json_pipeline_annotation_files = [x.replace('.json', '_exp_0097_corrected_monolayer_left.json') for x in json_annotation_files]
json_pipeline_annotation_files = [os.path.join(annotations_dir, x) for x in json_pipeline_annotation_files]

# modify filenames to select the particular segmentation we want (e.g. the automatic ones, or the manually refined ones)
json_refined_annotation_files = [x.replace('.json', '_exp_0097_refined_left.json') for x in json_annotation_files]
json_refined_annotation_files = [os.path.join(annotations_dir, x) for x in json_refined_annotation_files]

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# make sure that in the boxplots PAT comes before MAT
metainfo['sex'] = metainfo['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
metainfo['ko_parent'] = metainfo['ko_parent'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
metainfo['genotype'] = metainfo['genotype'].astype(pd.api.types.CategoricalDtype(categories=['KLF14-KO:WT', 'KLF14-KO:Het'], ordered=True))

quantiles = np.linspace(0, 1, 11)
quantiles = quantiles[1:-1]

pipeline_area_q_all = []
refined_area_q_all = []
pipeline_area_mean_all = []
refined_area_mean_all = []
id_all = []
for i_file, (json_pipeline_annotation_file, json_refined_annotation_file) in enumerate(zip(json_pipeline_annotation_files, json_refined_annotation_files)):

    # create dataframe for this image
    df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(json_pipeline_annotation_file),
                                                          values=[i_file, ], values_tag='i_file',
                                                          tags_to_keep=['id', 'ko_parent', 'genotype', 'sex',
                                                                        'BW', 'gWAT', 'SC'])

    # mouse ID as a string
    id = df_common['id'].values[0]

    # we have only refined some of the segmentations for testing
    if not id in ['16.2a', '16.2b', '16.2c', '16.2d', '16.2e', '16.2f', '17.1a', '17.1b', '17.1c', '17.1d', '17.1e',
                  '17.1f', '17.2a', '17.2b', '17.2c', '17.2d', '17.2f', '17.2g', '18.1a', '18.1b', '18.1c', '18.1d']:
       continue

    print('File ' + str(i_file) + '/' + str(len(json_annotation_files)-1) + ': ')
    if os.path.isfile(json_pipeline_annotation_file):
        print('\t' + os.path.basename(json_pipeline_annotation_file))
    else:
        print('\t' + os.path.basename(json_pipeline_annotation_file) + ' ... missing')
    if os.path.isfile(json_refined_annotation_file):
        print('\t' + os.path.basename(json_refined_annotation_file))
    else:
        print('\t' + os.path.basename(json_refined_annotation_file) + ' ... missing')

    # ndpi file that corresponds to this .json file
    ndpi_file = json_pipeline_annotation_file.replace('_exp_0097_corrected_monolayer_left.json', '.ndpi')
    ndpi_file = ndpi_file.replace(annotations_dir, ndpi_dir)

    # open full resolution histology slide
    im = openslide.OpenSlide(ndpi_file)

    # pixel size
    assert (im.properties['tiff.ResolutionUnit'] == 'centimeter')
    xres = 1e-2 / float(im.properties['tiff.XResolution'])  # m
    yres = 1e-2 / float(im.properties['tiff.YResolution'])  # m

    # ko = df_common['ko_parent'].values[0]
    # genotype = df_common['genotype'].values[0]
    # sex = df_common['sex'].values[0]
    # bw = df_common['BW'].values[0]
    # gwat = df_common['gWAT'].values[0]
    # sc = df_common['SC'].values[0]

    # read contours from AIDA annotations
    pipeline_contours = cytometer.data.aida_get_contours(json_pipeline_annotation_file, layer_name='White adipocyte.*')
    refined_contours = cytometer.data.aida_get_contours(json_refined_annotation_file, layer_name='White adipocyte.*')

    # compute area of each contour
    pipeline_areas = [Polygon(c).area * xres * yres for c in pipeline_contours]  # (um^2)
    refined_areas = [Polygon(c).area * xres * yres for c in refined_contours]  # (um^2)

    # compute HD quantiles
    pipeline_area_q = scipy.stats.mstats.hdquantiles(pipeline_areas, prob=quantiles, axis=0)
    refined_area_q = scipy.stats.mstats.hdquantiles(refined_areas, prob=quantiles, axis=0)

    # compute average cell size
    pipeline_area_mean = np.mean(pipeline_areas)
    refined_area_mean = np.mean(refined_areas)

    pipeline_area_q_all.append(pipeline_area_q)
    refined_area_q_all.append(refined_area_q)
    pipeline_area_mean_all.append(pipeline_area_mean)
    refined_area_mean_all.append(refined_area_mean)
    id_all.append(id)

    print('Removed cells: %.2f' % (1 - len(refined_areas) / len(pipeline_areas)))
    print((np.array(pipeline_area_q) - np.array(refined_area_q)) * 1e12)

    if DEBUG:
        plt.clf()
        plt.plot(quantiles, pipeline_area_q * 1e12, label='Pipeline', linewidth=3)
        plt.plot(quantiles, refined_area_q * 1e12, label='Refined', linewidth=3)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel('Quantiles', fontsize=14)
        plt.ylabel('Area ($\mu m^2$)', fontsize=14)
        plt.legend(fontsize=12)

# convert the list of vectors into a matrix
pipeline_area_q_all = np.vstack(pipeline_area_q_all)
refined_area_q_all = np.vstack(refined_area_q_all)
refined_area_mean_all = np.array(refined_area_mean_all)
pipeline_area_mean_all = np.array(pipeline_area_mean_all)

if DEBUG:
    plt.clf()
    pipeline_area_q_mean = np.mean(pipeline_area_q_all, axis=0)
    for i in range(pipeline_area_q_all.shape[0]):
        plt.plot(quantiles, 100 * (refined_area_q_all[i, :] - pipeline_area_q_all[i, :]) / pipeline_area_q_mean, label=id_all[i], linewidth=3)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Quantiles', fontsize=14)
    plt.ylabel('Area change with refinement (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()

if DEBUG:
    plt.clf()
    plt.boxplot(100 * (refined_area_mean_all - pipeline_area_mean_all) / pipeline_area_mean_all, labels=['Mean size'])
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylabel('Area change with refinement (%)', fontsize=14)
    plt.tight_layout()

########################################################################################################################
## Plots of segmented full slides with quantile colourmaps
########################################################################################################################

# This is done in klf14_b6ntac_exp_0098_full_slide_size_analysis_v7

########################################################################################################################
## Segmentation validation
########################################################################################################################

# This is done in klf14_b6ntac_exp_0096_pipeline_v7_validation.py

########################################################################################################################
## Cell populations from automatically segmented images in two depots: SQWAT and GWAT.
## This section needs to be run for each of the depots. But the results are saved, so in later sections, it's possible
## to get all the data together
### USED IN PAPER
########################################################################################################################

import matplotlib.pyplot as plt
import cytometer.data
from shapely.geometry import Polygon
import openslide
import numpy as np
import scipy.stats
import pandas as pd
from mlxtend.evaluate import permutation_test
from statsmodels.stats.multitest import multipletests
import math

# directories
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
annotations_dir = os.path.join(home, 'Software/AIDA/dist/data/annotations')
ndpi_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
metainfo_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')

DEBUG = False

depot = 'sqwat'
# depot = 'gwat'

permutation_sample_size = 9  # the factorial of this number is the number of repetitions in the permutation tests

# list of annotation files for this depot
json_annotation_files = json_annotation_files_dict[depot]

# modify filenames to select the particular segmentation we want (e.g. the automatic ones, or the manually refined ones)
json_annotation_files = [x.replace('.json', '_exp_0097_corrected.json') for x in json_annotation_files]
json_annotation_files = [os.path.join(annotations_dir, x) for x in json_annotation_files]

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# make sure that in the boxplots PAT comes before MAT
metainfo['sex'] = metainfo['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
metainfo['ko_parent'] = metainfo['ko_parent'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
metainfo['genotype'] = metainfo['genotype'].astype(pd.api.types.CategoricalDtype(categories=['KLF14-KO:WT', 'KLF14-KO:Het'], ordered=True))

quantiles = np.linspace(0, 1, 11)
quantiles = quantiles[1:-1]

# compute areas of the rough masks
filename_rough_mask_area = os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_rough_mask_area_' + depot + '.npz')
id_all = []
rough_mask_area_all = []
for i_file, json_file in enumerate(json_annotation_files):

    print('File ' + str(i_file) + '/' + str(len(json_annotation_files)-1) + ': ' + os.path.basename(json_file))

    if not os.path.isfile(json_file):
        print('Missing file')
        # continue

    # open full resolution histology slide
    ndpi_file = json_file.replace('_exp_0097_corrected.json', '.ndpi')
    ndpi_file = os.path.join(ndpi_dir, os.path.basename(ndpi_file))
    im = openslide.OpenSlide(ndpi_file)

    # pixel size
    assert (im.properties['tiff.ResolutionUnit'] == 'centimeter')
    xres = 1e-2 / float(im.properties['tiff.XResolution'])
    yres = 1e-2 / float(im.properties['tiff.YResolution'])

    # load mask
    rough_mask_file = json_file.replace('_exp_0097_corrected.json', '_rough_mask.npz')
    rough_mask_file = os.path.join(annotations_dir, rough_mask_file)

    if not os.path.isfile(rough_mask_file):
        print('No mask: ' + rough_mask_file)

    aux = np.load(rough_mask_file)
    lores_istissue0 = aux['lores_istissue0']

    if DEBUG:
        foo = aux['im_downsampled']
        foo = PIL.Image.fromarray(foo)
        foo = foo.resize(tuple((np.round(np.array(foo.size[0:2]) / 4)).astype(np.int)))
        plt.imshow(foo)
        plt.title(os.path.basename(ndpi_file))

    # compute scaling factor between downsampled mask and original image
    size_orig = np.array(im.dimensions)  # width, height
    size_downsampled = np.array(lores_istissue0.shape)[::-1]  # width, height
    downsample_factor = size_orig / size_downsampled  # width, height

    # create dataframe for this image
    rough_mask_area = np.count_nonzero(lores_istissue0) * (xres * downsample_factor[0]) * (yres * downsample_factor[1])  # m^2
    df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(json_file),
                                                          values=[rough_mask_area,], values_tag='SC_rough_mask_area',
                                                          tags_to_keep=['id', 'ko_parent', 'sex'])

    # mouse ID as a string
    id = df_common['id'].values[0]

    # correct area because most slides contain two slices, but some don't
    if depot == 'sqwat' and not (id in ['16.2d', '17.1e', '17.2g', '16.2e', '18.1f', '37.4a', '37.2e']):
        # two slices in the slide, so slice area is approx. one half
        rough_mask_area /= 2
    elif depot == 'gwat' and not (id in ['36.1d', '16.2a', '16.2b', '16.2c', '16.2d', '16.2e', '17.1b', '17.1d',
                                         '17.1e', '17.1f', '17.2c', '17.2d', '17.2f', '17.2g', '18.1b', '18.1c',
                                         '18.1d', '18.2a', '18.2c', '18.2d', '18.2f', '18.2g', '18.3c', '19.1a',
                                         '19.2e', '19.2f', '19.2g', '36.3d', '37.2e', '37.2f', '37.2g', '37.2h',
                                         '37.3a', '37.4a', '37.4b', '39.2d']):
        # two slices in the slide, so slice area is approx. one half
        rough_mask_area /= 2

    # add to output
    id_all.append(id)
    rough_mask_area_all.append(rough_mask_area)

# save results
np.savez_compressed(filename_rough_mask_area, id_all=id_all, rough_mask_area_all=rough_mask_area_all)


# load or compute area quantiles
filename_quantiles = os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_area_quantiles_' + depot + '.npz')
if os.path.isfile(filename_quantiles):

    aux = np.load(filename_quantiles)
    area_mean_all = aux['area_mean_all']
    area_q_all = aux['area_q_all']
    id_all = aux['id_all']
    ko_all = aux['ko_all']
    genotype_all = aux['genotype_all']
    sex_all = aux['sex_all']

else:

    area_mean_all = []
    area_q_all = []
    id_all = []
    ko_all = []
    genotype_all = []
    sex_all = []
    bw_all = []
    gwat_all = []
    sc_all = []
    for i_file, json_file in enumerate(json_annotation_files):

        print('File ' + str(i_file) + '/' + str(len(json_annotation_files)-1) + ': ' + os.path.basename(json_file))

        if not os.path.isfile(json_file):
            print('Missing file')
            continue

        # ndpi file that corresponds to this .json file
        ndpi_file = json_file.replace('_exp_0097_corrected.json', '.ndpi')
        ndpi_file = ndpi_file.replace(annotations_dir, ndpi_dir)

        # open full resolution histology slide
        im = openslide.OpenSlide(ndpi_file)

        # pixel size
        assert (im.properties['tiff.ResolutionUnit'] == 'centimeter')
        xres = 1e-2 / float(im.properties['tiff.XResolution'])  # m
        yres = 1e-2 / float(im.properties['tiff.YResolution'])  # m

        # create dataframe for this image
        df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(json_file),
                                                              values=[i_file,], values_tag='i_file',
                                                              tags_to_keep=['id', 'ko_parent', 'genotype', 'sex',
                                                                            'BW', 'gWAT', 'SC'])

        # mouse ID as a string
        id = df_common['id'].values[0]
        ko = df_common['ko_parent'].values[0]
        genotype = df_common['genotype'].values[0]
        sex = df_common['sex'].values[0]
        bw = df_common['BW'].values[0]
        gwat = df_common['gWAT'].values[0]
        sc = df_common['SC'].values[0]

        # read contours from AIDA annotations
        contours = cytometer.data.aida_get_contours(os.path.join(annotations_dir, json_file), layer_name='White adipocyte.*')

        # compute area of each contour
        areas = [Polygon(c).area * xres * yres for c in contours]  # (um^2)

        # compute average area of all contours
        area_mean = np.mean(areas)

        # compute HD quantiles
        area_q = scipy.stats.mstats.hdquantiles(areas, prob=quantiles, axis=0)

        # append to totals
        area_mean_all.append(area_mean)
        area_q_all.append(area_q)
        id_all.append(id)
        ko_all.append(ko)
        genotype_all.append(genotype)
        sex_all.append(sex)
        bw_all.append(bw)
        gwat_all.append(gwat)
        sc_all.append(sc)

    # reorder from largest to smallest final area value
    area_mean_all = np.array(area_mean_all)
    area_q_all = np.array(area_q_all)
    id_all = np.array(id_all)
    ko_all = np.array(ko_all)
    genotype_all = np.array(genotype_all)
    sex_all = np.array(sex_all)
    bw_all = np.array(bw_all)
    gwat_all = np.array(gwat_all)
    sc_all = np.array(sc_all)

    idx = np.argsort(area_q_all[:, -1])
    idx = idx[::-1]  # sort from larger to smaller
    area_mean_all = area_mean_all[idx]
    area_q_all = area_q_all[idx, :]
    id_all = id_all[idx]
    ko_all = ko_all[idx]
    genotype_all = genotype_all[idx]
    sex_all = sex_all[idx]
    bw_all = bw_all[idx]
    gwat_all = gwat_all[idx]
    sc_all = sc_all[idx]

    np.savez_compressed(filename_quantiles, area_mean_all=area_mean_all, area_q_all=area_q_all, id_all=id_all,
                        ko_all=ko_all, genotype_all=genotype_all, sex_all=sex_all,
                        bw_all=bw_all, gwat_all=gwat_all, sc_all=sc_all)

if DEBUG:
    plt.clf()

    for i in range(len(area_q_all)):

        # plot
        if ko_all[i] == 'PAT':
            color = 'g'
        elif ko_all[i] == 'MAT':
            color = 'r'
        else:
            raise ValueError('Unknown ko value: ' + ko)

        if sex_all[i] == 'f':
            plt.subplot(121)
            plt.plot(quantiles, area_q_all[i] * 1e12 * 1e-3, color=color)
        elif sex_all[i] == 'm':
            plt.subplot(122)
            plt.plot(quantiles, area_q_all[i] * 1e12 * 1e-3, color=color)
        else:
            raise ValueError('Unknown sex value: ' + sex)


    legend_f = [i + ' ' + j.replace('KLF14-KO:', '') for i, j
                in zip(id_all[sex_all == 'f'], genotype_all[sex_all == 'f'])]
    legend_m = [i + ' ' + j.replace('KLF14-KO:', '') for i, j
                in zip(id_all[sex_all == 'm'], genotype_all[sex_all == 'm'])]
    plt.subplot(121)
    plt.title('Female', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.xlabel('Quantile', fontsize=14)
    plt.ylabel('Area ($10^{3}\ \mu m^2$)', fontsize=14)
    plt.legend(legend_f, fontsize=12)
    plt.subplot(122)
    plt.title('Male', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.xlabel('Quantile', fontsize=14)
    plt.legend(legend_m, fontsize=12)

# DEBUG:
# area_q_all = np.vstack((area_q_all, area_q_all))
# id_all = np.hstack((id_all, id_all))
# ko_all = np.hstack((ko_all, ko_all))
# genotype_all = np.hstack((genotype_all, genotype_all))
# sex_all = np.hstack((sex_all, sex_all))

# compute variability of area values for each quantile
area_q_f_pat = area_q_all[(sex_all == 'f') * (ko_all == 'PAT'), :]
area_q_m_pat = area_q_all[(sex_all == 'm') * (ko_all == 'PAT'), :]
area_q_f_mat = area_q_all[(sex_all == 'f') * (ko_all == 'MAT'), :]
area_q_m_mat = area_q_all[(sex_all == 'm') * (ko_all == 'MAT'), :]
area_interval_f_pat = scipy.stats.mstats.hdquantiles(area_q_f_pat, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
area_interval_m_pat = scipy.stats.mstats.hdquantiles(area_q_m_pat, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
area_interval_f_mat = scipy.stats.mstats.hdquantiles(area_q_f_mat, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
area_interval_m_mat = scipy.stats.mstats.hdquantiles(area_q_m_mat, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)

area_q_f_pat_wt = area_q_all[(sex_all == 'f') * (ko_all == 'PAT') * (genotype_all == 'KLF14-KO:WT'), :]
area_q_m_pat_wt = area_q_all[(sex_all == 'm') * (ko_all == 'PAT') * (genotype_all == 'KLF14-KO:WT'), :]
area_q_f_mat_wt = area_q_all[(sex_all == 'f') * (ko_all == 'MAT') * (genotype_all == 'KLF14-KO:WT'), :]
area_q_m_mat_wt = area_q_all[(sex_all == 'm') * (ko_all == 'MAT') * (genotype_all == 'KLF14-KO:WT'), :]
area_interval_f_pat_wt = scipy.stats.mstats.hdquantiles(area_q_f_pat_wt, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
area_interval_m_pat_wt = scipy.stats.mstats.hdquantiles(area_q_m_pat_wt, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
area_interval_f_mat_wt = scipy.stats.mstats.hdquantiles(area_q_f_mat_wt, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
area_interval_m_mat_wt = scipy.stats.mstats.hdquantiles(area_q_m_mat_wt, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)

area_q_f_pat_het = area_q_all[(sex_all == 'f') * (ko_all == 'PAT') * (genotype_all == 'KLF14-KO:Het'), :]
area_q_m_pat_het = area_q_all[(sex_all == 'm') * (ko_all == 'PAT') * (genotype_all == 'KLF14-KO:Het'), :]
area_q_f_mat_het = area_q_all[(sex_all == 'f') * (ko_all == 'MAT') * (genotype_all == 'KLF14-KO:Het'), :]
area_q_m_mat_het = area_q_all[(sex_all == 'm') * (ko_all == 'MAT') * (genotype_all == 'KLF14-KO:Het'), :]
area_interval_f_pat_het = scipy.stats.mstats.hdquantiles(area_q_f_pat_het, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
area_interval_m_pat_het = scipy.stats.mstats.hdquantiles(area_q_m_pat_het, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
area_interval_f_mat_het = scipy.stats.mstats.hdquantiles(area_q_f_mat_het, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
area_interval_m_mat_het = scipy.stats.mstats.hdquantiles(area_q_m_mat_het, prob=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0)

n_f_pat_wt = area_q_f_pat_wt.shape[0]
n_m_pat_wt = area_q_m_pat_wt.shape[0]
n_f_mat_wt = area_q_f_mat_wt.shape[0]
n_m_mat_wt = area_q_m_mat_wt.shape[0]
n_f_pat_het = area_q_f_pat_het.shape[0]
n_m_pat_het = area_q_m_pat_het.shape[0]
n_f_mat_het = area_q_f_mat_het.shape[0]
n_m_mat_het = area_q_m_mat_het.shape[0]

if DEBUG:
    plt.clf()

    plt.subplot(121)
    plt.plot(quantiles * 100, area_interval_f_pat_wt[2, :] * 1e12 * 1e-3, 'C0', linewidth=3, label=str(n_f_pat_wt) + ' Female PAT WT')
    plt.fill_between(quantiles * 100, area_interval_f_pat_wt[1, :] * 1e12 * 1e-3, area_interval_f_pat_wt[3, :] * 1e12 * 1e-3,
                     facecolor='C0', alpha=0.3)
    # plt.plot(quantiles, area_interval_f_pat_wt[0, :] * 1e12 * 1e-3, 'C0:', linewidth=2, label='Female PAT WT 95%-interval')
    # plt.plot(quantiles, area_interval_f_pat_wt[4, :] * 1e12 * 1e-3, 'C0:', linewidth=2)

    plt.plot(quantiles * 100, area_interval_f_pat_het[2, :] * 1e12 * 1e-3, 'C1', linewidth=3, label=str(n_f_pat_het) + ' Female PAT Het')
    plt.fill_between(quantiles * 100, area_interval_f_pat_het[1, :] * 1e12 * 1e-3, area_interval_f_pat_het[3, :] * 1e12 * 1e-3,
                     facecolor='C1', alpha=0.3)

    # plt.title('Inguinal subcutaneous', fontsize=16)
    plt.xlabel('Cell population quantile (%)', fontsize=14)
    plt.ylabel('Area ($\cdot 10^3 \mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='upper left', prop={'size': 12})
    plt.ylim(0, 15)
    plt.tight_layout()

    plt.subplot(122)
    plt.plot(quantiles * 100, area_interval_f_mat_wt[2, :] * 1e12 * 1e-3, 'C2', linewidth=3, label=str(n_f_mat_wt) + ' Female MAT WT')
    plt.fill_between(quantiles * 100, area_interval_f_mat_wt[1, :] * 1e12 * 1e-3, area_interval_f_mat_wt[3, :] * 1e12 * 1e-3,
                     facecolor='C2', alpha=0.3)

    plt.plot(quantiles * 100, area_interval_f_mat_het[2, :] * 1e12 * 1e-3, 'C3', linewidth=3, label=str(n_f_mat_het) + ' Female MAT Het')
    plt.fill_between(quantiles * 100, area_interval_f_mat_het[1, :] * 1e12 * 1e-3, area_interval_f_mat_het[3, :] * 1e12 * 1e-3,
                     facecolor='C3', alpha=0.3)

    # plt.title('Inguinal subcutaneous', fontsize=16)
    plt.xlabel('Cell population quantile (%)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='upper left', prop={'size': 12})
    plt.ylim(0, 15)
    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'exp_0099_' + depot + '_cell_area_female_pat_vs_mat_bands.svg'))
    plt.savefig(os.path.join(figures_dir, 'exp_0099_' + depot + '_cell_area_female_pat_vs_mat_bands.png'))

if DEBUG:
    plt.clf()

    plt.subplot(121)
    plt.plot(quantiles * 100, area_interval_m_pat_wt[2, :] * 1e12 * 1e-3, 'C0', linewidth=3, label=str(n_m_pat_wt) + ' Male PAT WT')
    plt.fill_between(quantiles * 100, area_interval_m_pat_wt[1, :] * 1e12 * 1e-3, area_interval_m_pat_wt[3, :] * 1e12 * 1e-3,
                     facecolor='C0', alpha=0.3)
    # plt.plot(quantiles, area_interval_f_pat_wt[0, :] * 1e12 * 1e-3, 'C0:', linewidth=2, label='Female PAT WT 95%-interval')
    # plt.plot(quantiles, area_interval_f_pat_wt[4, :] * 1e12 * 1e-3, 'C0:', linewidth=2)

    plt.plot(quantiles * 100, area_interval_m_pat_het[2, :] * 1e12 * 1e-3, 'C1', linewidth=3, label=str(n_m_pat_het) + ' Male PAT Het')
    plt.fill_between(quantiles * 100, area_interval_m_pat_het[1, :] * 1e12 * 1e-3, area_interval_m_pat_het[3, :] * 1e12 * 1e-3,
                     facecolor='C1', alpha=0.3)

    # plt.title('Inguinal subcutaneous', fontsize=16)
    plt.xlabel('Cell population quantile (%)', fontsize=14)
    plt.ylabel('Area ($\cdot 10^3 \mu$m$^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='upper left', prop={'size': 12})
    plt.ylim(0, 16)
    plt.tight_layout()

    plt.subplot(122)
    plt.plot(quantiles * 100, area_interval_m_mat_wt[2, :] * 1e12 * 1e-3, 'C2', linewidth=3, label=str(n_m_mat_wt) + ' Male MAT WT')
    plt.fill_between(quantiles * 100, area_interval_m_mat_wt[1, :] * 1e12 * 1e-3, area_interval_m_mat_wt[3, :] * 1e12 * 1e-3,
                     facecolor='C2', alpha=0.3)

    plt.plot(quantiles * 100, area_interval_m_mat_het[2, :] * 1e12 * 1e-3, 'C3', linewidth=3, label=str(n_m_mat_het) + ' Male MAT Het')
    plt.fill_between(quantiles * 100, area_interval_m_mat_het[1, :] * 1e12 * 1e-3, area_interval_m_mat_het[3, :] * 1e12 * 1e-3,
                     facecolor='C3', alpha=0.3)

    # plt.title('Inguinal subcutaneous', fontsize=16)
    plt.xlabel('Cell population quantile (%)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='upper left', prop={'size': 12})
    plt.ylim(0, 16)
    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'exp_0099_' + depot + '_cell_area_male_pat_vs_mat_bands.svg'))
    plt.savefig(os.path.join(figures_dir, 'exp_0099_' + depot + '_cell_area_male_pat_vs_mat_bands.png'))

filename_pvals = os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_pvals_' + depot + '.npz')
if os.path.isfile(filename_pvals):

    aux = np.load(filename_pvals)
    pval_perc_f_pat2mat = aux['pval_perc_f_pat2mat']
    pval_perc_m_pat2mat = aux['pval_perc_m_pat2mat']
    pval_perc_f_pat_wt2het = aux['pval_perc_f_pat_wt2het']
    pval_perc_f_mat_wt2het = aux['pval_perc_f_mat_wt2het']
    pval_perc_m_pat_wt2het = aux['pval_perc_m_pat_wt2het']
    pval_perc_m_mat_wt2het = aux['pval_perc_m_mat_wt2het']
    permutation_sample_size = aux['permutation_sample_size']

else:

    # test whether the median values are different enough between two groups
    func = lambda x, y: np.abs(scipy.stats.mstats.hdquantiles(x, prob=0.5, axis=0).data[0]
                               - scipy.stats.mstats.hdquantiles(y, prob=0.5, axis=0).data[0])
    # func = lambda x, y: np.abs(np.mean(x) - np.mean(y))

    ## PAT vs. MAT

    # test whether the median values are different enough between PAT vs. MAT
    pval_perc_f_pat2mat = np.zeros(shape=(len(quantiles),))
    for i, q in enumerate(quantiles):
        pval_perc_f_pat2mat[i] = permutation_test(x=area_q_f_pat[:, i], y=area_q_f_mat[:, i],
                                                  func=func, seed=None,
                                                  method='approximate', num_rounds=math.factorial(permutation_sample_size))

    pval_perc_m_pat2mat = np.zeros(shape=(len(quantiles),))
    for i, q in enumerate(quantiles):
        pval_perc_m_pat2mat[i] = permutation_test(x=area_q_m_pat[:, i], y=area_q_m_mat[:, i],
                                                  func=func, seed=None,
                                                  method='approximate', num_rounds=math.factorial(permutation_sample_size))

    ## WT vs. Het

    # PAT Females
    pval_perc_f_pat_wt2het = np.zeros(shape=(len(quantiles),))
    for i, q in enumerate(quantiles):
        pval_perc_f_pat_wt2het[i] = permutation_test(x=area_q_f_pat_wt[:, i], y=area_q_f_pat_het[:, i],
                                                     func=func, seed=None,
                                                     method='approximate',
                                                     num_rounds=math.factorial(permutation_sample_size))

    # MAT Females
    pval_perc_f_mat_wt2het = np.zeros(shape=(len(quantiles),))
    for i, q in enumerate(quantiles):
        pval_perc_f_mat_wt2het[i] = permutation_test(x=area_q_f_mat_wt[:, i], y=area_q_f_mat_het[:, i],
                                                     func=func, seed=None,
                                                     method='approximate',
                                                     num_rounds=math.factorial(permutation_sample_size))

    # PAT Males
    pval_perc_m_pat_wt2het = np.zeros(shape=(len(quantiles),))
    for i, q in enumerate(quantiles):
        pval_perc_m_pat_wt2het[i] = permutation_test(x=area_q_m_pat_wt[:, i], y=area_q_m_pat_het[:, i],
                                                     func=func, seed=None,
                                                     method='approximate',
                                                     num_rounds=math.factorial(permutation_sample_size))

    # MAT Males
    pval_perc_m_mat_wt2het = np.zeros(shape=(len(quantiles),))
    for i, q in enumerate(quantiles):
        pval_perc_m_mat_wt2het[i] = permutation_test(x=area_q_m_mat_wt[:, i], y=area_q_m_mat_het[:, i],
                                                     func=func, seed=None,
                                                     method='approximate',
                                                     num_rounds=math.factorial(permutation_sample_size))

    np.savez_compressed(filename_pvals, permutation_sample_size=permutation_sample_size,
                        pval_perc_f_pat2mat=pval_perc_f_pat2mat, pval_perc_m_pat2mat=pval_perc_m_pat2mat,
                        pval_perc_f_pat_wt2het=pval_perc_f_pat_wt2het, pval_perc_f_mat_wt2het=pval_perc_f_mat_wt2het,
                        pval_perc_m_pat_wt2het=pval_perc_m_pat_wt2het, pval_perc_m_mat_wt2het=pval_perc_m_mat_wt2het)


# data has been loaded or computed

np.set_printoptions(precision=2)
print('PAT vs. MAT before multitest correction')
print('Female:')
print(pval_perc_f_pat2mat)
print('Male:')
print(pval_perc_m_pat2mat)
np.set_printoptions(precision=8)

# multitest correction using Hochberg a.k.a. Simes-Hochberg method
_, pval_perc_f_pat2mat, _, _ = multipletests(pval_perc_f_pat2mat, method='fdr_bh', alpha=0.05, returnsorted=False)
_, pval_perc_m_pat2mat, _, _ = multipletests(pval_perc_m_pat2mat, method='fdr_bh', alpha=0.05, returnsorted=False)

np.set_printoptions(precision=2)
print('PAT vs. MAT with multitest correction')
print('Female:')
print(pval_perc_f_pat2mat)
print('Male:')
print(pval_perc_m_pat2mat)
np.set_printoptions(precision=8)

np.set_printoptions(precision=3)
print('WT vs. Het before multitest correction')
print('Female:')
print(pval_perc_f_pat_wt2het)
print(pval_perc_f_mat_wt2het)
print('Male:')
print(pval_perc_m_pat_wt2het)
print(pval_perc_m_mat_wt2het)
np.set_printoptions(precision=8)

# multitest correction using Hochberg a.k.a. Simes-Hochberg method
_, pval_perc_f_pat_wt2het, _, _ = multipletests(pval_perc_f_pat_wt2het, method='fdr_bh', alpha=0.05, returnsorted=False)
_, pval_perc_f_mat_wt2het, _, _ = multipletests(pval_perc_f_mat_wt2het, method='fdr_bh', alpha=0.05, returnsorted=False)
_, pval_perc_m_pat_wt2het, _, _ = multipletests(pval_perc_m_pat_wt2het, method='fdr_bh', alpha=0.05, returnsorted=False)
_, pval_perc_m_mat_wt2het, _, _ = multipletests(pval_perc_m_mat_wt2het, method='fdr_bh', alpha=0.05, returnsorted=False)

np.set_printoptions(precision=3)
print('WT vs. Het with multitest correction')
print('Female:')
print(pval_perc_f_pat_wt2het)
print(pval_perc_f_mat_wt2het)
print('Male:')
print(pval_perc_m_pat_wt2het)
print(pval_perc_m_mat_wt2het)
np.set_printoptions(precision=8)

# # plot the median difference and the population quantiles at which the difference is significant
# if DEBUG:
#     plt.clf()
#     idx = pval_perc_f_pat2mat < 0.05
#     delta_a_f_pat2mat = (area_interval_f_mat[1, :] - area_interval_f_pat[1, :]) / area_interval_f_pat[1, :]
#     if np.any(idx):
#         plt.stem(quantiles[idx], 100 * delta_a_f_pat2mat[idx],
#                  markerfmt='.', linefmt='C6-', basefmt='C6',
#                  label='p-val$_{\mathrm{PAT}}$ < 0.05')
#
#     idx = pval_perc_m_pat2mat < 0.05
#     delta_a_m_pat2mat = (area_interval_m_mat[1, :] - area_interval_m_pat[1, :]) / area_interval_m_pat[1, :]
#     if np.any(idx):
#         plt.stem(quantiles[idx], 100 * delta_a_m_pat2mat[idx],
#                  markerfmt='.', linefmt='C7-', basefmt='C7', bottom=250,
#                  label='p-val$_{\mathrm{MAT}}$ < 0.05')
#
#     plt.plot(quantiles, 100 * delta_a_f_pat2mat, 'C6', linewidth=3, label='Female PAT to MAT')
#     plt.plot(quantiles, 100 * delta_a_m_pat2mat, 'C7', linewidth=3, label='Male PAT to MAT')
#
#     plt.xlabel('Cell population quantile', fontsize=14)
#     plt.ylabel('Area change (%)', fontsize=14)
#     plt.tick_params(axis='both', which='major', labelsize=14)
#     plt.legend(loc='lower right', prop={'size': 12})
#     plt.tight_layout()
#
#     plt.savefig(os.path.join(figures_dir, 'exp_0099_' + depot + '_cell_area_change_pat_2_mat.svg'))
#     plt.savefig(os.path.join(figures_dir, 'exp_0099_' + depot + '_cell_area_change_pat_2_mat.png'))

########################################################################################################################
## Linear models of body weight (BW), fat depots weight (SC and gWAT), and categorical variables (sex, ko, genotype)
########################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import pandas as pd
import statsmodels.api as sm
import re

# directories
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
annotations_dir = os.path.join(home, 'Software/AIDA/dist/data/annotations')
ndpi_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
metainfo_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')

DEBUG = False

quantiles = np.linspace(0, 1, 11)
quantiles = quantiles[1:-1]

# load metainfo file
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# make sure that in the boxplots PAT comes before MAT
metainfo['sex'] = metainfo['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
metainfo['ko_parent'] = metainfo['ko_parent'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
metainfo['genotype'] = metainfo['genotype'].astype(pd.api.types.CategoricalDtype(categories=['KLF14-KO:WT', 'KLF14-KO:Het'], ordered=True))

# add litter and mother ID columns to the data
for i, id in enumerate(metainfo['id']):
    litter = re.sub('[a-zA-Z]', '', id)
    metainfo.loc[i, 'litter'] = litter

    metainfo.loc[i, 'mother_id'] = litter.split('.')[0]

# add rough mask area column to the data
depot = 'sqwat'
filename_rough_mask_area = os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_rough_mask_area_' + depot + '.npz')

aux = np.load(filename_rough_mask_area)
id_all = aux['id_all']
rough_mask_area_all = aux['rough_mask_area_all']
for id, rough_mask_area in zip(id_all, rough_mask_area_all):
    metainfo.loc[metainfo['id'] == id, 'SC_rough_mask_area'] = rough_mask_area

# add rough mask area column to the data
depot = 'gwat'
filename_rough_mask_area = os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_rough_mask_area_' + depot + '.npz')

aux = np.load(filename_rough_mask_area)
aux = np.load(filename_rough_mask_area)
id_all = aux['id_all']
rough_mask_area_all = aux['rough_mask_area_all']
for id, rough_mask_area in zip(id_all, rough_mask_area_all):
    metainfo.loc[metainfo['id'] == id, 'gWAT_rough_mask_area'] = rough_mask_area


########################################################################################################################
### Plot SC vs. gWAT to look for outliers
########################################################################################################################

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

if DEBUG:
    idx = idx_not_nan
    plt.clf()
    plt.scatter(metainfo['SC'][idx], metainfo['gWAT'][idx], color='k', label='All mice')
    for i in idx:
        plt.annotate(metainfo['id'][i], (metainfo['SC'][i], metainfo['gWAT'][i]))
    plt.xlabel('SC', fontsize=14)
    plt.ylabel('gWAT', fontsize=14)
    plt.legend()

if DEBUG:
    plt.clf()
    plt.subplot(221)
    idx = np.where((metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT'))[0]
    plt.scatter(metainfo['SC'][idx], metainfo['gWAT'][idx], color='k', label='f PAT')
    for i in idx:
        plt.annotate(metainfo['id'][i], (metainfo['SC'][i], metainfo['gWAT'][i]))
    plt.xlabel('SC', fontsize=14)
    plt.ylabel('gWAT', fontsize=14)
    plt.legend()

    plt.subplot(222)
    idx = np.where((metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT'))[0]
    plt.scatter(metainfo['SC'][idx], metainfo['gWAT'][idx], color='k', label='f MAT')
    for i in idx:
        plt.annotate(metainfo['id'][i], (metainfo['SC'][i], metainfo['gWAT'][i]))
    plt.xlabel('SC', fontsize=14)
    plt.ylabel('gWAT', fontsize=14)
    plt.legend()

    plt.subplot(223)
    idx = np.where((metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT'))[0]
    plt.scatter(metainfo['SC'][idx], metainfo['gWAT'][idx], color='k', label='m PAT')
    for i in idx:
        plt.annotate(metainfo['id'][i], (metainfo['SC'][i], metainfo['gWAT'][i]))
    plt.xlabel('SC', fontsize=14)
    plt.ylabel('gWAT', fontsize=14)
    plt.legend()

    plt.subplot(224)
    idx = np.where((metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT'))[0]
    plt.scatter(metainfo['SC'][idx], metainfo['gWAT'][idx], color='k', label='m MAT')
    for i in idx:
        plt.annotate(metainfo['id'][i], (metainfo['SC'][i], metainfo['gWAT'][i]))
    plt.xlabel('SC', fontsize=14)
    plt.ylabel('gWAT', fontsize=14)
    plt.legend()

# 64 and 65 are outliers.

########################################################################################################################
### Plot SC and gWAT vs. corresponding rough mask areas
########################################################################################################################

if DEBUG:
    plt.clf()
    plt.subplot(121)
    plt.scatter(np.power(metainfo['SC_rough_mask_area'] * 1e6, 3/2), metainfo['SC'])
    # for i in range(metainfo.shape[0]):
    #     plt.annotate(metainfo['id'][i], (np.power(metainfo['SC_rough_mask_area'][i] * 1e6, 3/2), metainfo['SC'][i]))

    plt.xlabel('Subcutaneous vol from slice area mm$^3$', fontsize=14)
    plt.ylabel('m$_{SC}$ (g)', fontsize=14)
    plt.tick_params(labelsize=14)

    plt.subplot(122)
    plt.scatter(np.power(metainfo['gWAT_rough_mask_area'] * 1e6, 3/2), metainfo['gWAT'])
    # for i in range(metainfo.shape[0]):
    #     plt.annotate(metainfo['id'][i], (metainfo['gWAT_rough_mask_area'][i] * 1e6, metainfo['gWAT'][i]))

    plt.xlabel('Gonadal slice area mm$^2$', fontsize=14)
    plt.ylabel('m$_{G}$ (g)', fontsize=14)
    plt.tick_params(labelsize=14)

########################################################################################################################
# Explore mice dataset in terms of weights alive and culled
########################################################################################################################

# plot weight when alive vs. weight after death
idx = np.where(~np.isnan(metainfo['BW_alive']) * ~np.isnan(metainfo['BW']))[0]
plt.clf()
plt.scatter(metainfo['BW_alive'], metainfo['BW'])
plt.plot([20, 50], [20, 50])
for i in idx:
    plt.annotate(i, (metainfo['BW_alive'][i], metainfo['BW'][i]))

# plot weight evolution over time
plt.clf()
for i in idx:
    plt.plot([metainfo['BW_alive_date'][i], metainfo['cull_age'][i]], [metainfo['BW_alive'][i], metainfo['BW'][i]])

# plot SC vs. BW (females)
if DEBUG:
    plt.clf()
    plt.subplot(121)

    for litter in np.unique(metainfo['litter']):

        # index of animals from this litter
        idx = np.where((metainfo['litter'] == litter) * (metainfo['sex'] == 'f'))[0]

        if (len(idx) > 1):

            # order in order of increasing SC
            idx_sort = np.argsort(metainfo['SC'][idx])

            if np.array(metainfo['ko_parent'][idx])[0] == 'PAT':
                color = 'k'
            elif np.array(metainfo['ko_parent'][idx])[0] == 'MAT':
                color = 'r'

            plt.scatter(np.array(metainfo.loc[idx, 'SC'])[idx_sort], np.array(metainfo.loc[idx, 'BW'])[idx_sort], color=color)
            plt.plot(np.array(metainfo.loc[idx, 'SC'])[idx_sort], np.array(metainfo.loc[idx, 'BW'])[idx_sort], color=color)

            # if litter in ['19.1', '19.2']:
            for i in idx:
                plt.annotate(metainfo['id'][i], (metainfo['SC'][i], metainfo['BW'][i]))

        plt.title('Females')
        plt.xlabel('$m_{SC}$ (g)', fontsize=14)
        plt.ylabel('BW (g)', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.tight_layout()

    # plot SC vs. BW (males)
    plt.subplot(122)

    for litter in np.unique(metainfo['litter']):

        # index of animals from this litter
        idx = np.where((metainfo['litter'] == litter) * (metainfo['sex'] == 'm'))[0]

        if (len(idx) > 1):

            # order in order of increasing SC
            idx_sort = np.argsort(metainfo['SC'][idx])

            if np.array(metainfo['ko_parent'][idx])[0] == 'PAT':
                color = 'k'
            elif np.array(metainfo['ko_parent'][idx])[0] == 'MAT':
                color = 'r'

            plt.scatter(np.array(metainfo.loc[idx, 'SC'])[idx_sort], np.array(metainfo.loc[idx, 'BW'])[idx_sort], color=color)
            plt.plot(np.array(metainfo.loc[idx, 'SC'])[idx_sort], np.array(metainfo.loc[idx, 'BW'])[idx_sort], color=color)

            # if litter in ['19.1', '19.2']:
            for i in idx:
                plt.annotate(metainfo['id'][i], (metainfo['SC'][i], metainfo['BW'][i]))

        plt.title('Males')
        plt.xlabel('$m_{SC}$ (g)', fontsize=14)
        plt.ylabel('BW (g)', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.tight_layout()

# plot gWAT vs. BW (female)
plt.clf()

for litter in np.unique(metainfo['litter']):

    # index of animals from this litter
    idx = np.where((metainfo['litter'] == litter) * (metainfo['sex'] == 'f'))[0]

    if len(idx) > 1:

        # order in order of increasing gWAT
        idx_sort = np.argsort(metainfo['gWAT'][idx])

        if np.array(metainfo['ko_parent'][idx])[0] == 'PAT':
            color = 'k'
        elif np.array(metainfo['ko_parent'][idx])[0] == 'MAT':
            color = 'r'

        plt.scatter(np.array(metainfo.loc[idx, 'gWAT'])[idx_sort], np.array(metainfo.loc[idx, 'BW'])[idx_sort], color=color)
        plt.plot(np.array(metainfo.loc[idx, 'gWAT'])[idx_sort], np.array(metainfo.loc[idx, 'BW'])[idx_sort], color=color)

        # if (litter == '19.2'):
        for i in idx:
            plt.annotate(i, (metainfo['gWAT'][i], metainfo['BW'][i]))

    plt.xlabel('$m_{G}$ (g)', fontsize=14)
    plt.ylabel('BW (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

########################################################################################################################
### Model BW ~ (C(sex) + C(ko_parent) + C(genotype)) * (SC * gWAT)
### WARNING: This model is an example of having too many variables
########################################################################################################################

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]
model = sm.formula.ols('BW ~ (C(sex) + C(ko_parent) + C(genotype)) * (SC * gWAT)', data=metainfo, subset=idx_not_nan).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     BW   R-squared:                       0.816
# Model:                            OLS   Adj. R-squared:                  0.769
# Method:                 Least Squares   F-statistic:                     17.69
# Date:                Mon, 24 Feb 2020   Prob (F-statistic):           1.19e-16
# Time:                        15:32:07   Log-Likelihood:                -197.81
# No. Observations:                  76   AIC:                             427.6
# Df Residuals:                      60   BIC:                             464.9
# Df Model:                          15
# Covariance Type:            nonrobust
# =======================================================================================================
#                                           coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------
# Intercept                              20.8129      3.915      5.317      0.000      12.982      28.644
# C(sex)[T.m]                             8.2952      5.920      1.401      0.166      -3.547      20.138
# C(ko_parent)[T.MAT]                    -3.1038      4.417     -0.703      0.485     -11.938       5.731
# C(genotype)[T.KLF14-KO:Het]             0.4508      4.154      0.109      0.914      -7.858       8.759
# SC                                     -1.7885      8.152     -0.219      0.827     -18.096      14.519
# C(sex)[T.m]:SC                         14.4534      9.416      1.535      0.130      -4.382      33.289
# C(ko_parent)[T.MAT]:SC                 13.9682     12.136      1.151      0.254     -10.307      38.244
# C(genotype)[T.KLF14-KO:Het]:SC         -4.4397      8.417     -0.527      0.600     -21.277      12.397
# gWAT                                    4.6153      4.514      1.022      0.311      -4.414      13.644
# C(sex)[T.m]:gWAT                       -1.1889      5.786     -0.205      0.838     -12.763      10.385
# C(ko_parent)[T.MAT]:gWAT               11.1354      4.622      2.409      0.019       1.890      20.380
# C(genotype)[T.KLF14-KO:Het]:gWAT       -0.6653      4.608     -0.144      0.886      -9.883       8.553
# SC:gWAT                                 5.7281      7.709      0.743      0.460      -9.692      21.148
# C(sex)[T.m]:SC:gWAT                    -8.8374      8.256     -1.070      0.289     -25.352       7.678
# C(ko_parent)[T.MAT]:SC:gWAT           -20.5613      9.163     -2.244      0.029     -38.889      -2.234
# C(genotype)[T.KLF14-KO:Het]:SC:gWAT     1.2538      7.420      0.169      0.866     -13.589      16.096
# ==============================================================================
# Omnibus:                        0.893   Durbin-Watson:                   1.584
# Prob(Omnibus):                  0.640   Jarque-Bera (JB):                0.419
# Skew:                           0.138   Prob(JB):                        0.811
# Kurtosis:                       3.236   Cond. No.                         103.
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

# list of point with high influence (large residuals and leverage)
idx_influence = [35, 36, 37, 64, 65]
idx_no_influence = list(set(range(metainfo.shape[0])) - set(idx_influence))
print(metainfo.loc[idx_influence, ['id', 'ko_parent', 'sex', 'genotype', 'BW', 'SC', 'gWAT']])

model = sm.formula.ols('BW ~ (C(sex) + C(ko_parent) + C(genotype)) * (SC * gWAT)', data=metainfo, subset=idx_no_influence).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     BW   R-squared:                       0.817
# Model:                            OLS   Adj. R-squared:                  0.768
# Method:                 Least Squares   F-statistic:                     16.41
# Date:                Wed, 26 Feb 2020   Prob (F-statistic):           3.87e-15
# Time:                        14:08:33   Log-Likelihood:                -183.06
# No. Observations:                  71   AIC:                             398.1
# Df Residuals:                      55   BIC:                             434.3
# Df Model:                          15
# Covariance Type:            nonrobust
# =======================================================================================================
#                                           coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------
# Intercept                              22.4929      5.698      3.948      0.000      11.074      33.912
# C(sex)[T.m]                             7.2408      6.769      1.070      0.289      -6.324      20.806
# C(ko_parent)[T.MAT]                    -2.6859      4.448     -0.604      0.548     -11.600       6.229
# C(genotype)[T.KLF14-KO:Het]            -0.0971      4.440     -0.022      0.983      -8.996       8.802
# SC                                     -9.2532     23.095     -0.401      0.690     -55.537      37.031
# C(sex)[T.m]:SC                         21.7051     21.391      1.015      0.315     -21.164      64.574
# C(ko_parent)[T.MAT]:SC                 10.9030     13.041      0.836      0.407     -15.231      37.037
# C(genotype)[T.KLF14-KO:Het]:SC         -2.7711     11.164     -0.248      0.805     -25.145      19.603
# gWAT                                    2.5214      5.410      0.466      0.643      -8.321      13.364
# C(sex)[T.m]:gWAT                        0.0320      6.181      0.005      0.996     -12.356      12.420
# C(ko_parent)[T.MAT]:gWAT               11.2072      4.596      2.439      0.018       1.997      20.417
# C(genotype)[T.KLF14-KO:Het]:gWAT       -0.3474      4.714     -0.074      0.942      -9.795       9.100
# SC:gWAT                                13.9732     19.484      0.717      0.476     -25.074      53.020
# C(sex)[T.m]:SC:gWAT                   -16.5886     17.869     -0.928      0.357     -52.398      19.221
# C(ko_parent)[T.MAT]:SC:gWAT           -18.1201      9.764     -1.856      0.069     -37.687       1.447
# C(genotype)[T.KLF14-KO:Het]:SC:gWAT    -0.2622      9.087     -0.029      0.977     -18.472      17.948
# ==============================================================================
# Omnibus:                        1.715   Durbin-Watson:                   1.455
# Prob(Omnibus):                  0.424   Jarque-Bera (JB):                1.060
# Skew:                           0.245   Prob(JB):                        0.589
# Kurtosis:                       3.342   Cond. No.                         229.
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

print(model.pvalues)

########################################################################################################################
### Model BW ~ C(sex) + C(sex):gWAT  + C(ko_parent) + gWAT
### Deprecated model by the next one. This one is too simple, and forces the slopes of the lines to be parallel
########################################################################################################################

# don't use lines with NaNs (these should be removed by fit(), but just in case)
idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

# data that we are going to use
idx_subset = idx_not_nan

# fit linear model to data
model = sm.formula.ols('BW ~ C(sex) + C(sex):gWAT  + C(ko_parent) + gWAT', data=metainfo, subset=idx_subset).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     BW   R-squared:                       0.740
# Model:                            OLS   Adj. R-squared:                  0.726
# Method:                 Least Squares   F-statistic:                     50.60
# Date:                Tue, 25 Feb 2020   Prob (F-statistic):           4.45e-20
# Time:                        15:24:38   Log-Likelihood:                -210.82
# No. Observations:                  76   AIC:                             431.6
# Df Residuals:                      71   BIC:                             443.3
# Df Model:                           4
# Covariance Type:            nonrobust
# =======================================================================================
#                           coef    std err          t      P>|t|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# Intercept              19.4126      1.598     12.144      0.000      16.225      22.600
# C(sex)[T.m]            17.6887      2.981      5.934      0.000      11.745      23.633
# C(ko_parent)[T.MAT]     2.7743      0.927      2.991      0.004       0.925       4.624
# gWAT                    8.0439      1.721      4.674      0.000       4.612      11.476
# C(sex)[T.m]:gWAT       -7.2480      2.845     -2.548      0.013     -12.921      -1.575
# ==============================================================================
# Omnibus:                        4.048   Durbin-Watson:                   1.411
# Prob(Omnibus):                  0.132   Jarque-Bera (JB):                3.306
# Skew:                           0.378   Prob(JB):                        0.191
# Kurtosis:                       3.687   Cond. No.                         16.2
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# helper function to plot the different lines created by the model
def model_line(model, sex, ko_parent, gWAT):
    sex = np.float(sex == 'm')
    ko_parent = np.float(ko_parent == 'MAT')

    return model.params['Intercept'] +\
           model.params['C(sex)[T.m]'] * sex + \
           model.params['C(ko_parent)[T.MAT]'] * ko_parent + \
           model.params['gWAT'] * gWAT + \
           model.params['C(sex)[T.m]:gWAT'] * sex * gWAT


# plot BW as a function of gWAT
if DEBUG:

    annotate = False
    plt.clf()

    # f PAT
    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    plt.scatter(metainfo['gWAT'][idx], metainfo['BW'][idx], label='f PAT WT', color='C0', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    plt.scatter(metainfo['gWAT'][idx], metainfo['BW'][idx], label='f PAT Het', color='C0')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    BW = model_line(model, sex='f', ko_parent='PAT', gWAT=gWAT)
    plt.plot(gWAT, BW, color='C0', linewidth=3)

    # f MAT
    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    plt.scatter(metainfo['gWAT'][idx], metainfo['BW'][idx], label='f MAT WT', color='C2', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    plt.scatter(metainfo['gWAT'][idx], metainfo['BW'][idx], label='f MAT Het', color='C2')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    BW = model_line(model, sex='f', ko_parent='MAT', gWAT=gWAT)
    plt.plot(gWAT, BW, color='C2', linewidth=3)

    # m PAT
    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    plt.scatter(metainfo['gWAT'][idx], metainfo['BW'][idx], label='m PAT WT', color='k', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    plt.scatter(metainfo['gWAT'][idx], metainfo['BW'][idx], label='m PAT Het', color='k')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    BW = model_line(model, sex='m', ko_parent='PAT', gWAT=gWAT)
    plt.plot(gWAT, BW, color='k', linewidth=3)

    # m MAT
    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    plt.scatter(metainfo['gWAT'][idx], metainfo['BW'][idx], label='m MAT WT', color='C3', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    plt.scatter(metainfo['gWAT'][idx], metainfo['BW'][idx], label='m MAT Het', color='C3')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    BW = model_line(model, sex='m', ko_parent='MAT', gWAT=gWAT)
    plt.plot(gWAT, BW, color='C3', linewidth=3)

    plt.legend()

    plt.xlabel('m$_G$ (g)', fontsize=14)
    plt.ylabel('BW (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_bw_model_mG_simpler.png'), bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_bw_model_mG_simpler.svg'), bbox_inches='tight')


########################################################################################################################
### Model BW ~ C(sex) * C(ko_parent) * gWAT
### VALID model
########################################################################################################################

# don't use lines with NaNs (these should be removed by fit(), but just in case)
idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

# data that we are going to use
idx_subset = idx_not_nan

# Mixed-effects linear model
aux = metainfo.loc[idx_subset, :]
formula = 'BW ~ C(sex) * C(ko_parent) * gWAT'
vc = {'litter': '0 + C(litter)'}  # litter is a random effected nested inside mother_id
model = sm.formula.mixedlm(formula, vc_formula=vc, re_formula='1', groups='mother_id', data=aux).fit()
print(model.summary())


# fit linear model to data
formula = 'BW ~ C(sex) * C(ko_parent) * gWAT'
model = sm.formula.ols(formula, data=metainfo, subset=idx_subset).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     BW   R-squared:                       0.773
# Model:                            OLS   Adj. R-squared:                  0.749
# Method:                 Least Squares   F-statistic:                     33.01
# Date:                Thu, 27 Feb 2020   Prob (F-statistic):           1.65e-19
# Time:                        10:14:01   Log-Likelihood:                -205.77
# No. Observations:                  76   AIC:                             427.5
# Df Residuals:                      68   BIC:                             446.2
# Df Model:                           7
# Covariance Type:            nonrobust
# ========================================================================================================
#                                            coef    std err          t      P>|t|      [0.025      0.975]
# --------------------------------------------------------------------------------------------------------
# Intercept                               19.8526      1.975     10.053      0.000      15.912      23.793
# C(sex)[T.m]                             12.2959      3.705      3.318      0.001       4.902      19.690
# C(ko_parent)[T.MAT]                      1.8066      2.984      0.605      0.547      -4.148       7.762
# C(sex)[T.m]:C(ko_parent)[T.MAT]         12.8513      5.808      2.213      0.030       1.261      24.441
# gWAT                                     6.7074      2.246      2.986      0.004       2.225      11.189
# C(sex)[T.m]:gWAT                        -0.5933      3.647     -0.163      0.871      -7.870       6.684
# C(ko_parent)[T.MAT]:gWAT                 2.5961      3.308      0.785      0.435      -4.005       9.197
# C(sex)[T.m]:C(ko_parent)[T.MAT]:gWAT   -14.5795      5.526     -2.638      0.010     -25.607      -3.552
# ==============================================================================
# Omnibus:                        4.620   Durbin-Watson:                   1.636
# Prob(Omnibus):                  0.099   Jarque-Bera (JB):                4.493
# Skew:                           0.308   Prob(JB):                        0.106
# Kurtosis:                       4.019   Cond. No.                         40.4
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

print(model.pvalues)

# helper function to plot the different lines created by the model
def model_line(model, sex, ko_parent, gWAT):
    sex = np.float(sex == 'm')
    ko_parent = np.float(ko_parent == 'MAT')

    return model.params['Intercept'] +\
           model.params['C(sex)[T.m]'] * sex + \
           model.params['C(ko_parent)[T.MAT]'] * ko_parent + \
           model.params['C(sex)[T.m]:C(ko_parent)[T.MAT]'] * sex * ko_parent + \
           model.params['gWAT'] * gWAT + \
           model.params['C(sex)[T.m]:gWAT'] * sex * gWAT + \
           model.params['C(ko_parent)[T.MAT]:gWAT'] * ko_parent * gWAT + \
           model.params['C(sex)[T.m]:C(ko_parent)[T.MAT]:gWAT'] * sex * ko_parent * gWAT


# plot BW as a function of gWAT
if DEBUG:

    annotate = False
    plt.clf()

    # f PAT
    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    plt.scatter(metainfo['gWAT'][idx], metainfo['BW'][idx], label='f PAT WT', color='C0', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    plt.scatter(metainfo['gWAT'][idx], metainfo['BW'][idx], label='f PAT Het', color='C0')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    BW = model_line(model, sex='f', ko_parent='PAT', gWAT=gWAT)
    plt.plot(gWAT, BW, color='C0', linewidth=3)

    # f MAT
    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    plt.scatter(metainfo['gWAT'][idx], metainfo['BW'][idx], label='f MAT WT', color='C2', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    plt.scatter(metainfo['gWAT'][idx], metainfo['BW'][idx], label='f MAT Het', color='C2')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    BW = model_line(model, sex='f', ko_parent='MAT', gWAT=gWAT)
    plt.plot(gWAT, BW, color='C2', linewidth=3)

    # m PAT
    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    plt.scatter(metainfo['gWAT'][idx], metainfo['BW'][idx], label='m PAT WT', color='k', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    plt.scatter(metainfo['gWAT'][idx], metainfo['BW'][idx], label='m PAT Het', color='k')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    BW = model_line(model, sex='m', ko_parent='PAT', gWAT=gWAT)
    plt.plot(gWAT, BW, color='k', linewidth=3)

    # m MAT
    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    plt.scatter(metainfo['gWAT'][idx], metainfo['BW'][idx], label='m MAT WT', color='C3', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    plt.scatter(metainfo['gWAT'][idx], metainfo['BW'][idx], label='m MAT Het', color='C3')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT')
    gWAT = np.linspace(np.min(metainfo['gWAT'][idx]), np.max(metainfo['gWAT'][idx]))
    BW = model_line(model, sex='m', ko_parent='MAT', gWAT=gWAT)
    plt.plot(gWAT, BW, color='C3', linewidth=3)

    plt.legend()

    plt.xlabel('m$_G$ (g)', fontsize=14)
    plt.ylabel('BW (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_bw_model_mG.png'), bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_bw_model_mG.svg'), bbox_inches='tight')

########################################################################################################################
### Model BW ~ C(sex) * C(ko_parent) * gWAT
### Add nested random effects: mother_id -> litter -> mouse
# https://www.statsmodels.org/dev/generated/statsmodels.formula.api.mixedlm.html?highlight=mixedlm#statsmodels.formula.api.mixedlm
### EXPERIMENTAL model
########################################################################################################################

# don't use lines with NaNs (these should be removed by fit(), but just in case)
idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

# data that we are going to use
idx_subset = idx_not_nan

# Mixed-effects linear model
aux = metainfo.loc[idx_subset, :]
formula = 'BW ~ C(sex) * C(ko_parent) * gWAT'
vc = {'litter': '0 + C(litter)'}  # litter is a random effected nested inside mother_id
model = sm.formula.mixedlm(formula, vc_formula=vc, re_formula='1', groups='mother_id', data=aux).fit()
print(model.summary())

# /home/rcasero/.conda/envs/cytometer_tensorflow/lib/python3.6/site-packages/statsmodels/base/model.py:1286: RuntimeWarning: invalid value encountered in sqrt
#   bse_ = np.sqrt(np.diag(self.cov_params()))
#                       Mixed Linear Model Regression Results
# =================================================================================
# Model:                      MixedLM         Dependent Variable:         BW
# No. Observations:           76              Method:                     REML
# No. Groups:                 8               Scale:                      9.5401
# Min. group size:            1               Likelihood:                 -189.5050
# Max. group size:            20              Converged:                  Yes
# Mean group size:            9.5
# ---------------------------------------------------------------------------------
#                                       Coef.  Std.Err.   z    P>|z|  [0.025 0.975]
# ---------------------------------------------------------------------------------
# Intercept                             22.692    2.227 10.191 0.000  18.328 27.056
# C(sex)[T.m]                            9.604    3.253  2.953 0.003   3.229 15.980
# C(ko_parent)[T.MAT]                   -2.804    3.152 -0.889 0.374  -8.982  3.375
# C(sex)[T.m]:C(ko_parent)[T.MAT]        7.670    6.151  1.247 0.212  -4.386 19.727
# gWAT                                   4.437    2.195  2.022 0.043   0.136  8.739
# C(sex)[T.m]:gWAT                       1.656    3.157  0.524 0.600  -4.533  7.844
# C(ko_parent)[T.MAT]:gWAT               6.823    3.262  2.092 0.036   0.430 13.216
# C(sex)[T.m]:C(ko_parent)[T.MAT]:gWAT -10.814    5.674 -1.906 0.057 -21.935  0.308
# mother_id Var                          0.000
# litter Var                             7.659
# =================================================================================


########################################################################################################################
### BW ~ C(sex) * C(ko_parent) * SC
### VALID model, but bad fit because of 5 outliers (we remove them in the next model)
########################################################################################################################

# don't use lines with NaNs (these should be removed by fit(), but just in case)
idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

# SC outliers
idx_outliers = []

# data that we are going to use
idx_subset = list(set(idx_not_nan) - set(idx_outliers))

# fit linear model to data
model = sm.formula.ols('BW ~ C(sex) * C(ko_parent) * SC', data=metainfo, subset=idx_subset).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     BW   R-squared:                       0.706
# Model:                            OLS   Adj. R-squared:                  0.676
# Method:                 Least Squares   F-statistic:                     23.32
# Date:                Thu, 27 Feb 2020   Prob (F-statistic):           8.42e-16
# Time:                        14:56:57   Log-Likelihood:                -215.55
# No. Observations:                  76   AIC:                             447.1
# Df Residuals:                      68   BIC:                             465.7
# Df Model:                           7
# Covariance Type:            nonrobust
# ======================================================================================================
#                                          coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------------------------------
# Intercept                             23.4759      1.721     13.637      0.000      20.041      26.911
# C(sex)[T.m]                           10.4744      3.158      3.317      0.001       4.173      16.776
# C(ko_parent)[T.MAT]                    4.6399      2.384      1.946      0.056      -0.117       9.397
# C(sex)[T.m]:C(ko_parent)[T.MAT]        4.6208      4.142      1.116      0.269      -3.645      12.887
# SC                                     2.9539      2.518      1.173      0.245      -2.071       7.979
# C(sex)[T.m]:SC                         3.0082      4.049      0.743      0.460      -5.072      11.088
# C(ko_parent)[T.MAT]:SC                 1.0103      4.400      0.230      0.819      -7.770       9.790
# C(sex)[T.m]:C(ko_parent)[T.MAT]:SC   -12.2497      6.355     -1.928      0.058     -24.931       0.432
# ==============================================================================
# Omnibus:                        3.016   Durbin-Watson:                   1.576
# Prob(Omnibus):                  0.221   Jarque-Bera (JB):                2.498
# Skew:                           0.440   Prob(JB):                        0.287
# Kurtosis:                       3.118   Cond. No.                         27.7
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

print(model.pvalues)

def model_line(model, sex, ko_parent, SC):
    sex = np.float(sex == 'm')
    ko_parent = np.float(ko_parent == 'MAT')

    return model.params['Intercept'] + \
           model.params['C(sex)[T.m]'] * sex + \
           model.params['C(ko_parent)[T.MAT]'] * ko_parent + \
           model.params['C(sex)[T.m]:C(ko_parent)[T.MAT]'] * sex * ko_parent + \
           model.params['SC'] * SC + \
           model.params['C(sex)[T.m]:SC'] * sex * SC + \
           model.params['C(ko_parent)[T.MAT]:SC'] * ko_parent * SC + \
           model.params['C(sex)[T.m]:C(ko_parent)[T.MAT]:SC'] * sex * ko_parent * SC

# plot BW as a function of SC
if DEBUG:

    annotate = False
    plt.clf()

    # f PAT
    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='f PAT WT', color='C0', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='f PAT Het', color='C0')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    BW = model_line(model, sex='f', ko_parent='PAT', SC=SC)
    plt.plot(SC, BW, color='C0', linewidth=3)

    # f MAT
    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='f MAT WT', color='C2', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='f MAT Het', color='C2')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    BW = model_line(model, sex='f', ko_parent='MAT', SC=SC)
    plt.plot(SC, BW, color='C2', linewidth=3)

    # m PAT
    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='m PAT WT', color='k', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='m PAT Het', color='k')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    BW = model_line(model, sex='m', ko_parent='PAT', SC=SC)
    plt.plot(SC, BW, color='k', linewidth=3)

    # m MAT
    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='m MAT WT', color='C3', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='m MAT Het', color='C3')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    BW = model_line(model, sex='m', ko_parent='MAT', SC=SC)
    plt.plot(SC, BW, color='C3', linewidth=3)

    plt.legend()

    plt.xlabel('m$_{SC}$ (g)', fontsize=14)
    plt.ylabel('BW (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_bw_model_mSC.png'), bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_bw_model_mSC.svg'), bbox_inches='tight')


########################################################################################################################
### BW ~ C(sex) * C(ko_parent) * SC
### VALID model, 5 outliers removed
########################################################################################################################

# don't use lines with NaNs (these should be removed by fit(), but just in case)
idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

# SC outliers
idx_outliers = [35, 36, 37, 64, 65]

# data that we are going to use
idx_subset = list(set(idx_not_nan) - set(idx_outliers))

# fit linear model to data
model = sm.formula.ols('BW ~ C(sex) * C(ko_parent) * SC', data=metainfo, subset=idx_subset).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     BW   R-squared:                       0.754
# Model:                            OLS   Adj. R-squared:                  0.727
# Method:                 Least Squares   F-statistic:                     27.62
# Date:                Thu, 27 Feb 2020   Prob (F-statistic):           6.14e-17
# Time:                        15:14:50   Log-Likelihood:                -193.59
# No. Observations:                  71   AIC:                             403.2
# Df Residuals:                      63   BIC:                             421.3
# Df Model:                           7
# Covariance Type:            nonrobust
# ======================================================================================================
#                                          coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------------------------------
# Intercept                             21.6675      1.715     12.634      0.000      18.240      25.095
# C(sex)[T.m]                           12.2828      2.936      4.184      0.000       6.416      18.150
# C(ko_parent)[T.MAT]                    0.2006      3.291      0.061      0.952      -6.375       6.777
# C(sex)[T.m]:C(ko_parent)[T.MAT]        9.0601      4.486      2.020      0.048       0.095      18.025
# SC                                     8.4782      2.989      2.837      0.006       2.506      14.450
# C(sex)[T.m]:SC                        -2.5161      4.132     -0.609      0.545     -10.774       5.742
# C(ko_parent)[T.MAT]:SC                20.4707     10.549      1.941      0.057      -0.609      41.550
# C(sex)[T.m]:C(ko_parent)[T.MAT]:SC   -31.7102     11.327     -2.799      0.007     -54.346      -9.074
# ==============================================================================
# Omnibus:                        5.918   Durbin-Watson:                   1.687
# Prob(Omnibus):                  0.052   Jarque-Bera (JB):                5.075
# Skew:                           0.575   Prob(JB):                       0.0790
# Kurtosis:                       3.626   Cond. No.                         53.2
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

print(model.pvalues)

# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

def model_line(model, sex, ko_parent, SC):
    sex = np.float(sex == 'm')
    ko_parent = np.float(ko_parent == 'MAT')

    return model.params['Intercept'] + \
           model.params['C(sex)[T.m]'] * sex + \
           model.params['C(ko_parent)[T.MAT]'] * ko_parent + \
           model.params['C(sex)[T.m]:C(ko_parent)[T.MAT]'] * sex * ko_parent + \
           model.params['SC'] * SC + \
           model.params['C(sex)[T.m]:SC'] * sex * SC + \
           model.params['C(ko_parent)[T.MAT]:SC'] * ko_parent * SC + \
           model.params['C(sex)[T.m]:C(ko_parent)[T.MAT]:SC'] * sex * ko_parent * SC

# plot BW as a function of SC
if DEBUG:

    annotate = False
    plt.clf()

    # f PAT
    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='f PAT WT', color='C0', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='f PAT Het', color='C0')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    BW = model_line(model, sex='f', ko_parent='PAT', SC=SC)
    plt.plot(SC, BW, color='C0', linewidth=3)

    # f MAT
    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='f MAT WT', color='C2', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='f MAT Het', color='C2')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    BW = model_line(model, sex='f', ko_parent='MAT', SC=SC)
    plt.plot(SC, BW, color='C2', linewidth=3)

    # m PAT
    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='m PAT WT', color='k', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='m PAT Het', color='k')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    BW = model_line(model, sex='m', ko_parent='PAT', SC=SC)
    plt.plot(SC, BW, color='k', linewidth=3)

    # m MAT
    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='m MAT WT', color='C3', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='m MAT Het', color='C3')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    BW = model_line(model, sex='m', ko_parent='MAT', SC=SC)
    plt.plot(SC, BW, color='C3', linewidth=3)

    plt.legend()

    plt.xlabel('m$_{SC}$ (g)', fontsize=14)
    plt.ylabel('BW (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_bw_model_mSC_no_outliers.png'), bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_bw_model_mSC_no_outliers.svg'), bbox_inches='tight')


########################################################################################################################
### BW ~ C(sex) * C(ko_parent) * SC
### VALID model, 2 outliers removed
########################################################################################################################

# don't use lines with NaNs (these should be removed by fit(), but just in case)
idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

# SC outliers
idx_outliers = [35, 36, 65]

# data that we are going to use
idx_subset = list(set(idx_not_nan) - set(idx_outliers))

# fit linear model to data
model = sm.formula.ols('BW ~ C(sex) * C(ko_parent) * SC', data=metainfo, subset=idx_subset).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     BW   R-squared:                       0.743
# Model:                            OLS   Adj. R-squared:                  0.715
# Method:                 Least Squares   F-statistic:                     26.85
# Date:                Wed, 04 Mar 2020   Prob (F-statistic):           6.67e-17
# Time:                        02:52:03   Log-Likelihood:                -201.43
# No. Observations:                  73   AIC:                             418.9
# Df Residuals:                      65   BIC:                             437.2
# Df Model:                           7
# Covariance Type:            nonrobust
# ======================================================================================================
#                                          coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------------------------------
# Intercept                             21.7064      1.769     12.273      0.000      18.174      25.239
# C(sex)[T.m]                           12.2438      3.028      4.044      0.000       6.197      18.291
# C(ko_parent)[T.MAT]                    3.6077      2.575      1.401      0.166      -1.535       8.751
# C(sex)[T.m]:C(ko_parent)[T.MAT]        5.6530      4.064      1.391      0.169      -2.464      13.770
# SC                                     7.4582      3.031      2.460      0.017       1.404      13.512
# C(sex)[T.m]:SC                        -1.4961      4.225     -0.354      0.724      -9.934       6.942
# C(ko_parent)[T.MAT]:SC                 7.5800      6.199      1.223      0.226      -4.801      19.961
# C(sex)[T.m]:C(ko_parent)[T.MAT]:SC   -18.8194      7.520     -2.503      0.015     -33.838      -3.801
# ==============================================================================
# Omnibus:                        5.698   Durbin-Watson:                   1.666
# Prob(Omnibus):                  0.058   Jarque-Bera (JB):                4.862
# Skew:                           0.571   Prob(JB):                       0.0879
# Kurtosis:                       3.544   Cond. No.                         34.6
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


print(model.pvalues)

# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

def model_line(model, sex, ko_parent, SC):
    sex = np.float(sex == 'm')
    ko_parent = np.float(ko_parent == 'MAT')

    return model.params['Intercept'] + \
           model.params['C(sex)[T.m]'] * sex + \
           model.params['C(ko_parent)[T.MAT]'] * ko_parent + \
           model.params['C(sex)[T.m]:C(ko_parent)[T.MAT]'] * sex * ko_parent + \
           model.params['SC'] * SC + \
           model.params['C(sex)[T.m]:SC'] * sex * SC + \
           model.params['C(ko_parent)[T.MAT]:SC'] * ko_parent * SC + \
           model.params['C(sex)[T.m]:C(ko_parent)[T.MAT]:SC'] * sex * ko_parent * SC

# plot BW as a function of SC
if DEBUG:

    annotate = False
    plt.clf()

    # f PAT
    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='f PAT WT', color='C0', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='f PAT Het', color='C0')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    BW = model_line(model, sex='f', ko_parent='PAT', SC=SC)
    plt.plot(SC, BW, color='C0', linewidth=3)

    # f MAT
    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='f MAT WT', color='C2', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='f MAT Het', color='C2')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    BW = model_line(model, sex='f', ko_parent='MAT', SC=SC)
    plt.plot(SC, BW, color='C2', linewidth=3)

    # m PAT
    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='m PAT WT', color='k', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='m PAT Het', color='k')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    BW = model_line(model, sex='m', ko_parent='PAT', SC=SC)
    plt.plot(SC, BW, color='k', linewidth=3)

    # m MAT
    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='m MAT WT', color='C3', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='m MAT Het', color='C3')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    BW = model_line(model, sex='m', ko_parent='MAT', SC=SC)
    plt.plot(SC, BW, color='C3', linewidth=3)

    plt.legend()

    plt.xlabel('m$_{SC}$ (g)', fontsize=14)
    plt.ylabel('BW (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_bw_model_mSC_3_outliers.png'), bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_bw_model_mSC_3_outliers.svg'), bbox_inches='tight')


########################################################################################################################
### Model BW ~ C(sex) * C(ko_parent) * gWAT
### Add nested random effects: mother_id -> litter -> mouse
# https://www.statsmodels.org/dev/generated/statsmodels.formula.api.mixedlm.html?highlight=mixedlm#statsmodels.formula.api.mixedlm
### EXPERIMENTAL model
########################################################################################################################

# don't use lines with NaNs (these should be removed by fit(), but just in case)
idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

# data that we are going to use
idx_subset = idx_not_nan

# Mixed-effects linear model
aux = metainfo.loc[idx_subset, :]
formula = 'BW ~ C(sex) * C(ko_parent) * SC'
vc = {'litter': '0 + C(litter)'}  # litter is a random effected nested inside mother_id
model = sm.formula.mixedlm(formula, vc_formula=vc, re_formula='1', groups='mother_id', data=aux).fit()
print(model.summary())

# /home/rcasero/.conda/envs/cytometer_tensorflow/lib/python3.6/site-packages/pandas/core/computation/expressions.py:183: UserWarning: evaluating in Python space because the '*' operator is not supported by numexpr for the bool dtype, use '&' instead
#   .format(op=op_str, alt_op=unsupported[op_str]))
#                      Mixed Linear Model Regression Results
# ===============================================================================
# Model:                    MixedLM         Dependent Variable:         BW
# No. Observations:         76              Method:                     REML
# No. Groups:               8               Scale:                      13.6930
# Min. group size:          1               Likelihood:                 -199.2320
# Max. group size:          20              Converged:                  Yes
# Mean group size:          9.5
# -------------------------------------------------------------------------------
#                                     Coef.  Std.Err.   z    P>|z|  [0.025 0.975]
# -------------------------------------------------------------------------------
# Intercept                           25.506    2.129 11.981 0.000  21.333 29.678
# C(sex)[T.m]                          8.618    2.990  2.882 0.004   2.757 14.479
# C(ko_parent)[T.MAT]                  2.191    3.433  0.638 0.523  -4.539  8.920
# C(sex)[T.m]:C(ko_parent)[T.MAT]      4.530    4.099  1.105 0.269  -3.504 12.565
# SC                                   1.437    2.447  0.587 0.557  -3.360  6.233
# C(sex)[T.m]:SC                       4.643    3.661  1.268 0.205  -2.531 11.818
# C(ko_parent)[T.MAT]:SC               4.991    5.019  0.994 0.320  -4.847 14.829
# C(sex)[T.m]:C(ko_parent)[T.MAT]:SC -13.179    6.603 -1.996 0.046 -26.120 -0.238
# mother_id Var                        2.805    3.306
# litter Var                           4.728    2.638
# ===============================================================================

def model_line(model, sex, ko_parent, SC):
    sex = np.float(sex == 'm')
    ko_parent = np.float(ko_parent == 'MAT')

    return model.params['Intercept'] + \
           model.params['C(sex)[T.m]'] * sex + \
           model.params['C(ko_parent)[T.MAT]'] * ko_parent + \
           model.params['C(sex)[T.m]:C(ko_parent)[T.MAT]'] * sex * ko_parent + \
           model.params['SC'] * SC + \
           model.params['C(sex)[T.m]:SC'] * sex * SC + \
           model.params['C(ko_parent)[T.MAT]:SC'] * ko_parent * SC + \
           model.params['C(sex)[T.m]:C(ko_parent)[T.MAT]:SC'] * sex * ko_parent * SC

# plot BW as a function of SC
if DEBUG:

    annotate = False
    plt.clf()

    # f PAT
    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='f PAT WT', color='C0', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='f PAT Het', color='C0')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    BW = model_line(model, sex='f', ko_parent='PAT', SC=SC)
    plt.plot(SC, BW, color='C0', linewidth=3)

    # f MAT
    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='f MAT WT', color='C2', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='f MAT Het', color='C2')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    BW = model_line(model, sex='f', ko_parent='MAT', SC=SC)
    plt.plot(SC, BW, color='C2', linewidth=3)

    # m PAT
    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='m PAT WT', color='k', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='m PAT Het', color='k')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    BW = model_line(model, sex='m', ko_parent='PAT', SC=SC)
    plt.plot(SC, BW, color='k', linewidth=3)

    # m MAT
    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='m MAT WT', color='C3', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    plt.scatter(metainfo['SC'][idx], metainfo['BW'][idx], label='m MAT Het', color='C3')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC'][i], metainfo['BW'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT')
    SC = np.linspace(np.min(metainfo['SC'][idx]), np.max(metainfo['SC'][idx]))
    BW = model_line(model, sex='m', ko_parent='MAT', SC=SC)
    plt.plot(SC, BW, color='C3', linewidth=3)

    plt.legend()

    plt.xlabel('m$_{SC}$ (g)', fontsize=14)
    plt.ylabel('BW (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_bw_mixed_model_mSC.png'), bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_bw_mixed_model_mSC.svg'), bbox_inches='tight')


########################################################################################################################
## Load SQWAT and gWAT quantile data computed in a previous section
### USED IN PAPER
########################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.multitest import multipletests
import scipy.stats

# directories
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
annotations_dir = os.path.join(home, 'Software/AIDA/dist/data/annotations')
ndpi_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
metainfo_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')

DEBUG = False

quantiles = np.linspace(0, 1, 11)
quantiles = quantiles[1:-1]

# load metainfo file
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# make sure that in the boxplots PAT comes before MAT
metainfo['sex'] = metainfo['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
metainfo['ko_parent'] = metainfo['ko_parent'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
metainfo['genotype'] = metainfo['genotype'].astype(pd.api.types.CategoricalDtype(categories=['KLF14-KO:WT', 'KLF14-KO:Het'], ordered=True))

# load SQWAT data
depot = 'sqwat'
filename_quantiles = os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_area_quantiles_' + depot + '.npz')

aux = np.load(filename_quantiles)
area_mean_sqwat = aux['area_mean_all']
area_q_sqwat = aux['area_q_all']
id_sqwat = aux['id_all']
ko_sqwat = aux['ko_all']
genotype_sqwat = aux['genotype_all']
sex_sqwat = aux['sex_all']

# load gWAT data
depot = 'gwat'
filename_quantiles = os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_area_quantiles_' + depot + '.npz')

aux = np.load(filename_quantiles)
area_mean_gwat = aux['area_mean_all']
area_q_gwat = aux['area_q_all']
id_gwat = aux['id_all']
ko_gwat = aux['ko_all']
genotype_gwat = aux['genotype_all']
sex_gwat = aux['sex_all']

# add a new column to the metainfo frame with the mean cell area for SQWAT
metainfo_idx = [np.where(metainfo['id'] == x)[0][0] for x in id_sqwat]
metainfo.loc[metainfo_idx, 'SC_area_mean'] = area_mean_sqwat  # m^2
metainfo['SC_area_mean_1e12'] = metainfo['SC_area_mean'] * 1e12  # um^2

# add a new column to the metainfo frame with the mean cell area for gWAT
metainfo_idx = [np.where(metainfo['id'] == x)[0][0] for x in id_gwat]
metainfo.loc[metainfo_idx, 'gWAT_area_mean'] = area_mean_gwat  # m^2
metainfo['gWAT_area_mean_1e12'] = metainfo['gWAT_area_mean'] * 1e12  # um^2


# volume sphere with same radius as given circle
def vol_sphere(area_circle):
    return (4 / 3 / np.sqrt(np.pi)) * np.power(area_circle, 3/2)

# add a new column to the metainfo frame with the mean cell volume for SQWAT
metainfo.loc[metainfo_idx, 'SC_vol_mean'] = vol_sphere(metainfo['SC_area_mean'])  # m^3
metainfo['SC_vol_mean_1e12'] = metainfo['SC_vol_mean'] * 1e12  # mean(gWAT_vol_mean_1e12) = 0.2013305044030984
metainfo['SC_vol_mean_1e18'] = metainfo['SC_vol_mean'] * 1e18  # mean(gWAT_vol_mean_1e18) = 201331 um^3

# add a new column to the metainfo frame with the mean cell volume for gWAT
metainfo.loc[metainfo_idx, 'gWAT_vol_mean'] = vol_sphere(metainfo['gWAT_area_mean'])  # m^3
metainfo['gWAT_vol_mean_1e12'] = metainfo['gWAT_vol_mean'] * 1e12 # mean(gWAT_vol_mean_1e12) = 0.3281878940926992
metainfo['gWAT_vol_mean_1e18'] = metainfo['gWAT_vol_mean'] * 1e18 # mean(gWAT_vol_mean_1e18) = 328188 um^3

# add a new column to the metainfo frame with the median cell volume for SQWAT
metainfo_idx = [np.where(metainfo['id'] == x)[0][0] for x in id_sqwat]
metainfo['SC_vol_median'] = np.NaN
metainfo.loc[metainfo_idx, 'SC_vol_median'] = vol_sphere(area_q_sqwat[:, 4])  # m^3
metainfo['SC_vol_median_1e12'] = metainfo['SC_vol_median'] * 1e12

# add a new column to the metainfo frame with the median cell volume for gWAT
metainfo_idx = [np.where(metainfo['id'] == x)[0][0] for x in id_gwat]
metainfo['gWAT_vol_median'] = np.NaN
metainfo.loc[metainfo_idx, 'gWAT_vol_median'] = vol_sphere(area_q_gwat[:, 4])  # m^3
metainfo['gWAT_vol_median_1e12'] = metainfo['gWAT_vol_median'] * 1e12

# add new column with fat depot weights normalised by body weight
metainfo['SC_BW'] = metainfo['SC'] / metainfo['BW']
metainfo['gWAT_BW'] = metainfo['gWAT'] / metainfo['BW']

# add new column with estimates number of cells
fat_density_SC = 0.9038  # g / cm^3
fat_density_gWAT = 0.9029  # g / cm^3
metainfo['SC_kN'] = metainfo['SC'] / (fat_density_SC * 1e6 * metainfo['SC_vol_mean'])
metainfo['gWAT_kN'] = metainfo['gWAT'] / (fat_density_gWAT * 1e6 * metainfo['gWAT_vol_mean'])

# shorthand for subgroups
idx_f = metainfo['sex'] == 'f'
idx_m = metainfo['sex'] == 'm'

idx_pat = metainfo['ko_parent'] == 'PAT'
idx_mat = metainfo['ko_parent'] == 'MAT'

idx_wt = metainfo['genotype'] == 'KLF14-KO:WT'
idx_het = metainfo['genotype'] == 'KLF14-KO:Het'

print('Females N = ' + str(np.count_nonzero(idx_f)))
print('Males N = ' + str(np.count_nonzero(idx_m)))

print('PAT N = ' + str(np.count_nonzero(idx_pat)))
print('MAT N = ' + str(np.count_nonzero(idx_mat)))

print('WT N = ' + str(np.count_nonzero(idx_wt)))
print('Het N = ' + str(np.count_nonzero(idx_het)))

# number of animals by group
# SC_area_mean female, SC_area_mean male
print(
    str(np.count_nonzero(~np.isnan(metainfo['SC_area_mean']) & idx_f & idx_pat & idx_wt)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['SC_area_mean']) & idx_f & idx_pat & idx_het)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['SC_area_mean']) & idx_f & idx_mat & idx_wt)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['SC_area_mean']) & idx_f & idx_mat & idx_het)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['SC_area_mean']) & idx_m & idx_pat & idx_wt)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['SC_area_mean']) & idx_m & idx_pat & idx_het))  + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['SC_area_mean']) & idx_m & idx_mat & idx_wt)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['SC_area_mean']) & idx_m & idx_mat & idx_het))
)

# gWAT_area_mean female, gWAT_area_mean male
print(
    str(np.count_nonzero(~np.isnan(metainfo['gWAT_area_mean']) & idx_f & idx_pat & idx_wt)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['gWAT_area_mean']) & idx_f & idx_pat & idx_het)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['gWAT_area_mean']) & idx_f & idx_mat & idx_wt)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['gWAT_area_mean']) & idx_f & idx_mat & idx_het)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['gWAT_area_mean']) & idx_m & idx_pat & idx_wt)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['gWAT_area_mean']) & idx_m & idx_pat & idx_het))  + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['gWAT_area_mean']) & idx_m & idx_mat & idx_wt)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['gWAT_area_mean']) & idx_m & idx_mat & idx_het))
)

# SC_kN female, SC_area_mean male
print(
    str(np.count_nonzero(~np.isnan(metainfo['SC_kN']) & idx_f & idx_pat & idx_wt)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['SC_kN']) & idx_f & idx_pat & idx_het)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['SC_kN']) & idx_f & idx_mat & idx_wt)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['SC_kN']) & idx_f & idx_mat & idx_het)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['SC_kN']) & idx_m & idx_pat & idx_wt)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['SC_kN']) & idx_m & idx_pat & idx_het))  + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['SC_kN']) & idx_m & idx_mat & idx_wt)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['SC_kN']) & idx_m & idx_mat & idx_het))
)

# gWAT_area_mean female, gWAT_area_mean male
print(
    str(np.count_nonzero(~np.isnan(metainfo['gWAT_kN']) & idx_f & idx_pat & idx_wt)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['gWAT_kN']) & idx_f & idx_pat & idx_het)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['gWAT_kN']) & idx_f & idx_mat & idx_wt)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['gWAT_kN']) & idx_f & idx_mat & idx_het)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['gWAT_kN']) & idx_m & idx_pat & idx_wt)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['gWAT_kN']) & idx_m & idx_pat & idx_het))  + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['gWAT_kN']) & idx_m & idx_mat & idx_wt)) + ', ' +
    str(np.count_nonzero(~np.isnan(metainfo['gWAT_kN']) & idx_m & idx_mat & idx_het))
)



if DEBUG:
    plt.clf()
    plt.scatter(metainfo['gWAT_vol_mean'], metainfo['gWAT_vol_median'])
    plt.clf()
    plt.scatter(metainfo['SC_vol_mean'], metainfo['SC_vol_median'])

if DEBUG:
    # compare SQWAT to gWAT
    plt.clf()
    plt.scatter(metainfo['SC_vol_median'], metainfo['gWAT_vol_median'])
    plt.scatter(metainfo['SC_vol_mean'], metainfo['gWAT_vol_median'])

    # BW vs. SQWAT
    plt.clf()
    plt.scatter(metainfo['SC_vol_median'], metainfo['BW'])
    plt.scatter(metainfo['SC_vol_mean'], metainfo['BW'])

    # BW vs. gWAT
    plt.clf()
    plt.scatter(metainfo['gWAT_vol_median'], metainfo['BW'])
    plt.scatter(metainfo['gWAT_vol_mean'], metainfo['BW'])

if DEBUG:
    print(np.min(metainfo['SC_vol_mean']) * 1e12)  # cm^3
    print(np.max(metainfo['SC_vol_mean']) * 1e12)  # cm^3
    print(np.min(metainfo['gWAT_vol_mean']) * 1e12)  # cm^3
    print(np.max(metainfo['gWAT_vol_mean']) * 1e12)  # cm^3

    print(np.min(metainfo['SC_kN']))
    print(np.max(metainfo['SC_kN']))
    print(np.min(metainfo['gWAT_kN']))
    print(np.max(metainfo['gWAT_kN']))


########################################################################################################################
### Statistical tests comparing sex, ko_parent and genotype for SC and gWAT mean areas
### All possible combinations between: PAT WT, PAT Het, MAT WT, MAT Het
### NOT USED IN PAPER
########################################################################################################################

def make_groups(metainfo, indices, groups):
    metainfo['groups'] = np.nan
    indices_acc = set({})
    for index, group in zip(indices, groups):
        if len(indices_acc.intersection(set(index))) > 0:
            raise ValueError('Index in more than one group')
        metainfo.loc[index, 'groups'] = group


# ANOVA: sex groups
model = sm.formula.ols('SC_area_mean_1e12 ~ C(sex)', data=metainfo).fit()
print(model.summary())

# Tukey's HSD post-hoc comparison: SC: sex+genotype groups
idx_not_nan = ~np.isnan(metainfo['SC_area_mean'])
make_groups(metainfo, (idx_f & idx_wt, idx_f & idx_het, idx_m & idx_wt, idx_m & idx_het), ('f_WT', 'f_Het', 'm_WT', 'm_Het'))
idx = idx_not_nan
mc = MultiComparison(metainfo.loc[idx, 'SC_area_mean'] * 1e12, metainfo.loc[idx, 'groups'])
mc_results = mc.tukeyhsd()
print(mc_results)

if DEBUG:
    plt.clf()
    mc_results.plot_simultaneous()

    plt.clf()
    plt.boxplot(
        (metainfo.loc[idx_f & idx_wt & idx_not_nan, 'SC_area_mean'] * 1e12,
         metainfo.loc[idx_f & idx_het & idx_not_nan, 'SC_area_mean'] * 1e12,
         metainfo.loc[idx_m & idx_wt & idx_not_nan, 'SC_area_mean'] * 1e12,
         metainfo.loc[idx_m & idx_het & idx_not_nan, 'SC_area_mean'] * 1e12),
        notch=True,
        labels=('f WT', 'f Het', 'm WT', 'm Het')
    )
    plt.ylabel('Area ($\mu m^2$)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

# Tukey's HSD post-hoc comparison: Female SC: ko_parent + genotype groups
idx_not_nan = ~np.isnan(metainfo['SC_area_mean'])
make_groups(metainfo,
            (idx_f & idx_pat & idx_wt, idx_f & idx_pat & idx_het, idx_f & idx_mat & idx_wt, idx_f & idx_mat & idx_het),
            ('PAT_WT', 'PAT_Het', 'MAT_WT', 'MAT_Het'))
idx = idx_not_nan & idx_f
mc = MultiComparison(metainfo.loc[idx, 'SC_area_mean'] * 1e12, metainfo.loc[idx, 'groups'])
mc_results = mc.tukeyhsd()
print(mc_results)

if DEBUG:
    mc_results.plot_simultaneous()

    plt.clf()
    plt.boxplot(
        (metainfo['SC_area_mean'][idx_f & idx_pat & idx_wt & idx_not_nan] * 1e12,
         metainfo['SC_area_mean'][idx_f & idx_pat & idx_het & idx_not_nan] * 1e12,
         metainfo['SC_area_mean'][idx_f & idx_mat & idx_wt & idx_not_nan] * 1e12,
         metainfo['SC_area_mean'][idx_f & idx_mat & idx_het & idx_not_nan] * 1e12),
        notch=False,
        labels=('PAT WT', 'PAT Het', 'MAT WT', 'MAT Het')
    )
    plt.ylabel('Area ($\mu m^2$)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

# Tukey's HSD post-hoc comparison: Female gWAT: ko_parent + genotype groups
idx_not_nan = ~np.isnan(metainfo['gWAT_area_mean'])
make_groups(metainfo,
            (idx_f & idx_pat & idx_wt, idx_f & idx_pat & idx_het, idx_f & idx_mat & idx_wt, idx_f & idx_mat & idx_het),
            ('PAT_WT', 'PAT_Het', 'MAT_WT', 'MAT_Het'))
idx = idx_not_nan & idx_f
mc = MultiComparison(metainfo.loc[idx, 'gWAT_area_mean'] * 1e12, metainfo.loc[idx, 'groups'])
mc_results = mc.tukeyhsd()
print(mc_results)

if DEBUG:
    mc_results.plot_simultaneous()

    plt.clf()
    plt.boxplot(
        (metainfo['gWAT_area_mean'][idx_f & idx_pat & idx_wt & idx_not_nan] * 1e12,
         metainfo['gWAT_area_mean'][idx_f & idx_pat & idx_het & idx_not_nan] * 1e12,
         metainfo['gWAT_area_mean'][idx_f & idx_mat & idx_wt & idx_not_nan] * 1e12,
         metainfo['gWAT_area_mean'][idx_f & idx_mat & idx_het & idx_not_nan] * 1e12),
        notch=False,
        labels=('PAT WT', 'PAT Het', 'MAT WT', 'MAT Het')
    )
    plt.ylabel('Area ($\mu m^2$)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

# Tukey's HSD post-hoc comparison: Male SC: ko_parent + genotype groups
idx_not_nan = ~np.isnan(metainfo['SC_area_mean'])
make_groups(metainfo,
            (idx_m & idx_pat & idx_wt, idx_m & idx_pat & idx_het, idx_m & idx_mat & idx_wt, idx_m & idx_mat & idx_het),
            ('PAT_WT', 'PAT_Het', 'MAT_WT', 'MAT_Het'))
idx = idx_not_nan * idx_m
mc = MultiComparison(metainfo.loc[idx, 'SC_area_mean'] * 1e12, metainfo.loc[idx, 'groups'])
mc_results = mc.tukeyhsd()
print(mc_results)

idx_not_nan = ~np.isnan(metainfo['SC_kN'])
make_groups(metainfo,
            (idx_m & idx_pat & idx_wt, idx_m & idx_pat & idx_het, idx_m & idx_mat & idx_wt, idx_m & idx_mat & idx_het),
            ('PAT_WT', 'PAT_Het', 'MAT_WT', 'MAT_Het'))
idx = idx_not_nan * idx_m
mc = MultiComparison(metainfo.loc[idx, 'SC_kN'], metainfo.loc[idx, 'groups'])
mc_results = mc.tukeyhsd()
print(mc_results)

if DEBUG:
    mc_results.plot_simultaneous()

    plt.clf()
    plt.boxplot(
        (metainfo['SC_kN'][idx_m & idx_pat & idx_wt & idx_not_nan],
         metainfo['SC_kN'][idx_m & idx_pat & idx_het & idx_not_nan],
         metainfo['SC_kN'][idx_m & idx_mat & idx_wt & idx_not_nan],
         metainfo['SC_kN'][idx_m & idx_mat & idx_het & idx_not_nan]),
        notch=False,
        labels=('PAT WT', 'PAT Het', 'MAT WT', 'MAT Het')
    )
    plt.ylabel('Weighted cell count', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

# Tukey's HSD post-hoc comparison: Male gWAT: ko_parent + genotype groups
idx_not_nan = ~np.isnan(metainfo['gWAT_area_mean'])
make_groups(metainfo,
            (idx_m & idx_pat & idx_wt, idx_m & idx_pat & idx_het, idx_m & idx_mat & idx_wt, idx_m & idx_mat & idx_het),
            ('PAT_WT', 'PAT_Het', 'MAT_WT', 'MAT_Het'))
idx = idx_not_nan * idx_m
mc = MultiComparison(metainfo.loc[idx, 'gWAT_area_mean'] * 1e12, metainfo.loc[idx, 'groups'])
mc_results = mc.tukeyhsd()
print(mc_results)

idx_not_nan = ~np.isnan(metainfo['gWAT_kN'])
make_groups(metainfo,
            (idx_m & idx_pat & idx_wt, idx_m & idx_pat & idx_het, idx_m & idx_mat & idx_wt, idx_m & idx_mat & idx_het),
            ('PAT_WT', 'PAT_Het', 'MAT_WT', 'MAT_Het'))
idx = idx_not_nan * idx_m
mc = MultiComparison(metainfo.loc[idx, 'gWAT_kN'], metainfo.loc[idx, 'groups'])
mc_results = mc.tukeyhsd()
print(mc_results)

if DEBUG:
    mc_results.plot_simultaneous()

    plt.clf()
    plt.boxplot(
        (metainfo['gWAT_kN'][idx_f & idx_pat & idx_wt & idx_not_nan],
         metainfo['gWAT_kN'][idx_f & idx_pat & idx_het & idx_not_nan],
         metainfo['gWAT_kN'][idx_f & idx_mat & idx_wt & idx_not_nan],
         metainfo['gWAT_kN'][idx_f & idx_mat & idx_het & idx_not_nan]),
        notch=False,
        labels=('PAT WT', 'PAT Het', 'MAT WT', 'MAT Het')
    )
    plt.ylabel('Weighted cell count', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

########################################################################################################################
### Female and male SC and gWAT mean cell area stratified in MAT and PAT
### Only compare: PAT WT vs. PAT Het and MAT WT vs. MAT Het
### USED IN PAPER
########################################################################################################################

pval = []
idx_not_nan = ~np.isnan(metainfo['SC_area_mean'])

# Female PAT, WT vs Het
group_0 = metainfo['SC_area_mean'][idx_not_nan & idx_f & idx_pat & idx_wt]
group_1 = metainfo['SC_area_mean'][idx_not_nan & idx_f & idx_pat & idx_het]
results = scipy.stats.f_oneway(group_0, group_1)
pval.append(results.pvalue)
print('SC Female PAT, WT vs Het: p = ' + str(results.pvalue)
      + ', F = ' + str(results.statistic)
      + ', meandiff = ' + str((np.mean(group_1) - np.mean(group_0)) * 1e12))


# Female MAT, WT vs Het
group_0 = metainfo['SC_area_mean'][idx_not_nan & idx_f & idx_mat & idx_wt]
group_1 = metainfo['SC_area_mean'][idx_not_nan & idx_f & idx_mat & idx_het]
results = scipy.stats.f_oneway(group_0, group_1)
pval.append(results.pvalue)
print('SC Female MAT, WT vs Het: p = ' + str(results.pvalue)
      + ', F = ' + str(results.statistic)
      + ', meandiff = ' + str((np.mean(group_1) - np.mean(group_0)) * 1e12))

# Male PAT, WT vs Het
group_0 = metainfo['SC_area_mean'][idx_not_nan & idx_m & idx_pat & idx_wt]
group_1 = metainfo['SC_area_mean'][idx_not_nan & idx_m & idx_pat & idx_het]
results = scipy.stats.f_oneway(group_0, group_1)
pval.append(results.pvalue)
print('SC Male PAT, WT vs Het: p = ' + str(results.pvalue)
      + ', F = ' + str(results.statistic)
      + ', meandiff = ' + str((np.mean(group_1) - np.mean(group_0)) * 1e12))

# Male MAT, WT vs Het
group_0 = metainfo['SC_area_mean'][idx_not_nan & idx_m & idx_mat & idx_wt]
group_1 = metainfo['SC_area_mean'][idx_not_nan & idx_m & idx_mat & idx_het]
results = scipy.stats.f_oneway(group_0, group_1)
pval.append(results.pvalue)
print('SC Male MAT, WT vs Het: p = ' + str(results.pvalue)
      + ', F = ' + str(results.statistic)
      + ', meandiff = ' + str((np.mean(group_1) - np.mean(group_0)) * 1e12))

# Benjamini/Hochberg tests
reject_h0, pval_adj, _, _ = multipletests(pval, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
print('Adjusted pvalues: ' + str(pval_adj))

pval = []
idx_not_nan = ~np.isnan(metainfo['gWAT_area_mean'])

# Female PAT, WT vs Het
group_0 = metainfo['gWAT_area_mean'][idx_not_nan & idx_f & idx_pat & idx_wt]
group_1 = metainfo['gWAT_area_mean'][idx_not_nan & idx_f & idx_pat & idx_het]
results = scipy.stats.f_oneway(group_0, group_1)
pval.append(results.pvalue)
print('gWAT Female PAT, WT vs Het: p = ' + str(results.pvalue)
      + ', F = ' + str(results.statistic)
      + ', meandiff = ' + str((np.mean(group_1) - np.mean(group_0)) * 1e12))

# Female MAT, WT vs Het
group_0 = metainfo['gWAT_area_mean'][idx_not_nan & idx_f & idx_mat & idx_wt]
group_1 = metainfo['gWAT_area_mean'][idx_not_nan & idx_f & idx_mat & idx_het]
results = scipy.stats.f_oneway(group_0, group_1)
pval.append(results.pvalue)
print('gWAT Female MAT, WT vs Het: p = ' + str(results.pvalue)
      + ', F = ' + str(results.statistic)
      + ', meandiff = ' + str((np.mean(group_1) - np.mean(group_0)) * 1e12))

# Male PAT, WT vs Het
group_0 = metainfo['gWAT_area_mean'][idx_not_nan & idx_m & idx_pat & idx_wt]
group_1 = metainfo['gWAT_area_mean'][idx_not_nan & idx_m & idx_pat & idx_het]
results = scipy.stats.f_oneway(group_0, group_1)
pval.append(results.pvalue)
print('gWAT Male PAT, WT vs Het: p = ' + str(results.pvalue)
      + ', F = ' + str(results.statistic)
      + ', meandiff = ' + str((np.mean(group_1) - np.mean(group_0)) * 1e12))

# Male MAT, WT vs Het
group_0 = metainfo['gWAT_area_mean'][idx_not_nan & idx_m & idx_mat & idx_wt]
group_1 = metainfo['gWAT_area_mean'][idx_not_nan & idx_m & idx_mat & idx_het]
results = scipy.stats.f_oneway(group_0, group_1)
pval.append(results.pvalue)
print('gWAT Male MAT, WT vs Het: p = ' + str(results.pvalue)
      + ', F = ' + str(results.statistic)
      + ', meandiff = ' + str((np.mean(group_1) - np.mean(group_0)) * 1e12))

# Benjamini/Hochberg tests
reject_h0, pval_adj, _, _ = multipletests(pval, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
print('Adjusted pvalues: ' + str(pval_adj))

########################################################################################################################
### Female and male SC and gWAT cell count (k N) stratified in MAT and PAT
### Only compare: PAT WT vs. PAT Het and MAT WT vs. MAT Het
### USED IN PAPER
########################################################################################################################

pval = []
idx_not_nan = ~np.isnan(metainfo['SC_kN'])

# Female PAT, WT vs Het
group_0 = metainfo['SC_kN'][idx_not_nan & idx_f & idx_pat & idx_wt]
group_1 = metainfo['SC_kN'][idx_not_nan & idx_f & idx_pat & idx_het]
results = scipy.stats.f_oneway(group_0, group_1)
pval.append(results.pvalue)
print('SC Female PAT, WT vs Het: p = ' + str(results.pvalue)
      + ', F = ' + str(results.statistic)
      + ', meandiff = ' + str((np.mean(group_1) - np.mean(group_0))))

# Female MAT, WT vs Het
group_0 = metainfo['SC_kN'][idx_not_nan & idx_f & idx_mat & idx_wt]
group_1 = metainfo['SC_kN'][idx_not_nan & idx_f & idx_mat & idx_het]
results = scipy.stats.f_oneway(group_0, group_1)
pval.append(results.pvalue)
print('SC Female MAT, WT vs Het: p = ' + str(results.pvalue)
      + ', F = ' + str(results.statistic)
      + ', meandiff = ' + str((np.mean(group_1) - np.mean(group_0))))

# Male PAT, WT vs Het
group_0 = metainfo['SC_kN'][idx_not_nan & idx_m & idx_pat & idx_wt]
group_1 = metainfo['SC_kN'][idx_not_nan & idx_m & idx_pat & idx_het]
results = scipy.stats.f_oneway(group_0, group_1)
pval.append(results.pvalue)
print('SC Male PAT, WT vs Het: p = ' + str(results.pvalue)
      + ', F = ' + str(results.statistic)
      + ', meandiff = ' + str((np.mean(group_1) - np.mean(group_0))))

# Male MAT, WT vs Het
group_0 = metainfo['SC_kN'][idx_not_nan & idx_m & idx_mat & idx_wt]
group_1 = metainfo['SC_kN'][idx_not_nan & idx_m & idx_mat & idx_het]
results = scipy.stats.f_oneway(group_0, group_1)
pval.append(results.pvalue)
print('SC Male MAT, WT vs Het: p = ' + str(results.pvalue)
      + ', F = ' + str(results.statistic)
      + ', meandiff = ' + str((np.mean(group_1) - np.mean(group_0))))

# Benjamini/Hochberg tests
reject_h0, pval_adj, _, _ = multipletests(pval, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
print('Adjusted pvalues: ' + str(pval_adj))

pval = []
idx_not_nan = ~np.isnan(metainfo['gWAT_kN'])

# Female PAT, WT vs Het
group_0 = metainfo['gWAT_kN'][idx_not_nan & idx_f & idx_pat & idx_wt]
group_1 = metainfo['gWAT_kN'][idx_not_nan & idx_f & idx_pat & idx_het]
results = scipy.stats.f_oneway(group_0, group_1)
pval.append(results.pvalue)
print('gWAT Female PAT, WT vs Het: p = ' + str(results.pvalue)
      + ', F = ' + str(results.statistic)
      + ', meandiff = ' + str((np.mean(group_1) - np.mean(group_0))))

# Female MAT, WT vs Het
group_0 = metainfo['gWAT_kN'][idx_not_nan & idx_f & idx_mat & idx_wt]
group_1 = metainfo['gWAT_kN'][idx_not_nan & idx_f & idx_mat & idx_het]
results = scipy.stats.f_oneway(group_0, group_1)
pval.append(results.pvalue)
print('gWAT Female MAT, WT vs Het: p = ' + str(results.pvalue)
      + ', F = ' + str(results.statistic)
      + ', meandiff = ' + str((np.mean(group_1) - np.mean(group_0))))

# Male PAT, WT vs Het
group_0 = metainfo['gWAT_kN'][idx_not_nan & idx_m & idx_pat & idx_wt]
group_1 = metainfo['gWAT_kN'][idx_not_nan & idx_m & idx_pat & idx_het]
results = scipy.stats.f_oneway(group_0, group_1)
pval.append(results.pvalue)
print('gWAT Male PAT, WT vs Het: p = ' + str(results.pvalue)
      + ', F = ' + str(results.statistic)
      + ', meandiff = ' + str((np.mean(group_1) - np.mean(group_0))))

# Male MAT, WT vs Het
group_0 = metainfo['gWAT_kN'][idx_not_nan & idx_m & idx_mat & idx_wt]
group_1 = metainfo['gWAT_kN'][idx_not_nan & idx_m & idx_mat & idx_het]
results = scipy.stats.f_oneway(group_0, group_1)
pval.append(results.pvalue)
print('gWAT Male MAT, WT vs Het: p = ' + str(results.pvalue)
      + ', F = ' + str(results.statistic)
      + ', meandiff = ' + str((np.mean(group_1) - np.mean(group_0))))

# Benjamini/Hochberg tests
reject_h0, pval_adj, _, _ = multipletests(pval, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
print('Adjusted pvalues: ' + str(pval_adj))

########################################################################################################################
### Control for BW
### Female and male SC and gWAT mean cell area stratified in MAT and PAT
### Only compare: PAT WT vs. PAT Het and MAT WT vs. MAT Het
### USED IN PAPER
########################################################################################################################

pval = []
idx_not_nan = ~np.isnan(metainfo['SC_vol_mean'])

# Female PAT, WT vs Het
idx_subset = idx_not_nan & idx_f & idx_pat
model = sm.formula.ols('SC_vol_mean ~  C(genotype) + BW', data=metainfo, subset=idx_subset).fit()
# print(model.summary())
pval.append(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
print('SC Female PAT, WT vs Het: p = ' + str(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
      + ', F = ' + str(model.fvalue)
      + ', meandiff = ' + str(1e18 * model.params['C(genotype)[T.KLF14-KO:Het]']))

# Female MAT, WT vs Het
idx_subset = idx_not_nan & idx_f & idx_mat
model = sm.formula.ols('SC_vol_mean ~  C(genotype) + BW', data=metainfo, subset=idx_subset).fit()
# print(model.summary())
pval.append(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
print('SC Female MAT, WT vs Het: p = ' + str(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
      + ', F = ' + str(model.fvalue)
      + ', meandiff = ' + str(1e18 * model.params['C(genotype)[T.KLF14-KO:Het]']))

# Male PAT, WT vs Het
idx_subset = idx_not_nan & idx_m & idx_pat
model = sm.formula.ols('SC_vol_mean ~  C(genotype) + BW', data=metainfo, subset=idx_subset).fit()
# print(model.summary())
pval.append(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
print('SC male PAT, WT vs Het: p = ' + str(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
      + ', F = ' + str(model.fvalue)
      + ', meandiff = ' + str(1e18 * model.params['C(genotype)[T.KLF14-KO:Het]']))

# Male MAT, WT vs Het
idx_subset = idx_not_nan & idx_m & idx_mat
model = sm.formula.ols('SC_vol_mean ~  C(genotype) + BW', data=metainfo, subset=idx_subset).fit()
# print(model.summary())
pval.append(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
print('SC male MAT, WT vs Het: p = ' + str(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
      + ', F = ' + str(model.fvalue)
      + ', meandiff = ' + str(1e18 * model.params['C(genotype)[T.KLF14-KO:Het]']))

# Benjamini/Hochberg tests
reject_h0, pval_adj, _, _ = multipletests(pval, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
print('Adjusted pvalues: ' + str(pval_adj))

pval = []
idx_not_nan = ~np.isnan(metainfo['gWAT_vol_mean'])

# Female PAT, WT vs Het
idx_subset = idx_not_nan & idx_f & idx_pat
model = sm.formula.ols('gWAT_vol_mean ~  C(genotype) + BW', data=metainfo, subset=idx_subset).fit()
# print(model.summary())
pval.append(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
print('gWAT Female PAT, WT vs Het: p = ' + str(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
      + ', F = ' + str(model.fvalue)
      + ', meandiff = ' + str(1e18 * model.params['C(genotype)[T.KLF14-KO:Het]']))

# Female MAT, WT vs Het
idx_subset = idx_not_nan & idx_f & idx_mat
model = sm.formula.ols('gWAT_vol_mean ~  C(genotype) + BW', data=metainfo, subset=idx_subset).fit()
# print(model.summary())
pval.append(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
print('gWAT Female MAT, WT vs Het: p = ' + str(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
      + ', F = ' + str(model.fvalue)
      + ', meandiff = ' + str(1e18 * model.params['C(genotype)[T.KLF14-KO:Het]']))

# Male PAT, WT vs Het
idx_subset = idx_not_nan & idx_m & idx_pat
model = sm.formula.ols('gWAT_vol_mean ~  C(genotype) + BW', data=metainfo, subset=idx_subset).fit()
# print(model.summary())
pval.append(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
print('gWAT male PAT, WT vs Het: p = ' + str(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
      + ', F = ' + str(model.fvalue)
      + ', meandiff = ' + str(1e18 * model.params['C(genotype)[T.KLF14-KO:Het]']))

# Male MAT, WT vs Het
idx_subset = idx_not_nan & idx_m & idx_mat
model = sm.formula.ols('gWAT_vol_mean ~  C(genotype) + BW', data=metainfo, subset=idx_subset).fit()
# print(model.summary())
pval.append(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
print('gWAT male MAT, WT vs Het: p = ' + str(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
      + ', F = ' + str(model.fvalue)
      + ', meandiff = ' + str(1e18 * model.params['C(genotype)[T.KLF14-KO:Het]']))

# Benjamini/Hochberg tests
reject_h0, pval_adj, _, _ = multipletests(pval, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
print('Adjusted pvalues: ' + str(pval_adj))

########################################################################################################################
### Adjust for BW
### Female and male SC and gWAT cell count (k N) stratified in MAT and PAT
### Only compare: PAT WT vs. PAT Het and MAT WT vs. MAT Het
### USED IN PAPER
########################################################################################################################

pval = []
idx_not_nan = ~np.isnan(metainfo['SC_kN'])

# Female PAT, WT vs Het
idx_subset = idx_not_nan & idx_f & idx_pat
model = sm.formula.ols('SC_kN ~  C(genotype) + BW', data=metainfo, subset=idx_subset).fit()
# print(model.summary())
pval.append(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
print('SC Female PAT, WT vs Het: p = ' + str(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
      + ', F = ' + str(model.fvalue)
      + ', meandiff = ' + str(model.params['C(genotype)[T.KLF14-KO:Het]']))

# Female MAT, WT vs Het
idx_subset = idx_not_nan & idx_f & idx_mat
model = sm.formula.ols('SC_kN ~  C(genotype) + BW', data=metainfo, subset=idx_subset).fit()
# print(model.summary())
pval.append(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
print('SC Female MAT, WT vs Het: p = ' + str(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
      + ', F = ' + str(model.fvalue)
      + ', meandiff = ' + str(model.params['C(genotype)[T.KLF14-KO:Het]']))

# Male PAT, WT vs Het
idx_subset = idx_not_nan & idx_m & idx_pat
model = sm.formula.ols('SC_kN ~  C(genotype) + BW', data=metainfo, subset=idx_subset).fit()
# print(model.summary())
pval.append(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
print('SC male PAT, WT vs Het: p = ' + str(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
      + ', F = ' + str(model.fvalue)
      + ', meandiff = ' + str(model.params['C(genotype)[T.KLF14-KO:Het]']))

# Male MAT, WT vs Het
idx_subset = idx_not_nan & idx_m & idx_mat
model = sm.formula.ols('SC_kN ~  C(genotype) + BW', data=metainfo, subset=idx_subset).fit()
# print(model.summary())
pval.append(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
print('SC male MAT, WT vs Het: p = ' + str(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
      + ', F = ' + str(model.fvalue)
      + ', meandiff = ' + str(model.params['C(genotype)[T.KLF14-KO:Het]']))

# Benjamini/Hochberg tests
reject_h0, pval_adj, _, _ = multipletests(pval, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
print('Adjusted pvalues: ' + str(pval_adj))

pval = []
idx_not_nan = ~np.isnan(metainfo['gWAT_kN'])

# Female PAT, WT vs Het
idx_subset = idx_not_nan & idx_f & idx_pat
model = sm.formula.ols('gWAT_kN ~  C(genotype) + BW', data=metainfo, subset=idx_subset).fit()
# print(model.summary())
pval.append(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
print('gWAT Female PAT, WT vs Het: p = ' + str(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
      + ', F = ' + str(model.fvalue)
      + ', meandiff = ' + str(model.params['C(genotype)[T.KLF14-KO:Het]']))

# Female MAT, WT vs Het
idx_subset = idx_not_nan & idx_f & idx_mat
model = sm.formula.ols('gWAT_kN ~  C(genotype) + BW', data=metainfo, subset=idx_subset).fit()
# print(model.summary())
pval.append(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
print('gWAT Female MAT, WT vs Het: p = ' + str(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
      + ', F = ' + str(model.fvalue)
      + ', meandiff = ' + str(model.params['C(genotype)[T.KLF14-KO:Het]']))

# Male PAT, WT vs Het
idx_subset = idx_not_nan & idx_m & idx_pat
model = sm.formula.ols('gWAT_kN ~  C(genotype) + BW', data=metainfo, subset=idx_subset).fit()
# print(model.summary())
pval.append(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
print('gWAT male PAT, WT vs Het: p = ' + str(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
      + ', F = ' + str(model.fvalue)
      + ', meandiff = ' + str(model.params['C(genotype)[T.KLF14-KO:Het]']))

# Male MAT, WT vs Het
idx_subset = idx_not_nan & idx_m & idx_mat
model = sm.formula.ols('gWAT_kN ~  C(genotype) + BW', data=metainfo, subset=idx_subset).fit()
# print(model.summary())
pval.append(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
print('gWAT male MAT, WT vs Het: p = ' + str(model.pvalues['C(genotype)[T.KLF14-KO:Het]'])
      + ', F = ' + str(model.fvalue)
      + ', meandiff = ' + str(model.params['C(genotype)[T.KLF14-KO:Het]']))

# Benjamini/Hochberg tests
reject_h0, pval_adj, _, _ = multipletests(pval, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
print('Adjusted pvalues: ' + str(pval_adj))


########################################################################################################################
### Model gWAT ~ gWAT_vol_mean_1e12 * C(ko_parent) * C(sex)
### INVALID model: No significance for ko_parent
### NOT USED IN PAPER
########################################################################################################################

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

# data that we are going to use
idx_subset = idx_not_nan

model = sm.formula.ols('gWAT ~ gWAT_vol_mean_1e12 * C(ko_parent) * C(sex)', data=metainfo, subset=idx_subset).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                   gWAT   R-squared:                       0.386
# Model:                            OLS   Adj. R-squared:                  0.316
# Method:                 Least Squares   F-statistic:                     5.557
# Date:                Fri, 28 Feb 2020   Prob (F-statistic):           5.36e-05
# Time:                        19:36:00   Log-Likelihood:                -13.303
# No. Observations:                  70   AIC:                             42.61
# Df Residuals:                      62   BIC:                             60.59
# Df Model:                           7
# Covariance Type:            nonrobust
# ======================================================================================================================
#                                                          coef    std err          t      P>|t|      [0.025      0.975]
# ----------------------------------------------------------------------------------------------------------------------
# Intercept                                              0.2711      0.179      1.513      0.135      -0.087       0.629
# C(ko_parent)[T.MAT]                                    0.0546      0.255      0.214      0.831      -0.455       0.564
# C(sex)[T.m]                                            0.1560      0.606      0.258      0.798      -1.054       1.366
# C(ko_parent)[T.MAT]:C(sex)[T.m]                        1.6161      0.858      1.883      0.064      -0.099       3.331
# gWAT_vol_mean_1e12                                     2.5752      0.839      3.068      0.003       0.897       4.253
# gWAT_vol_mean_1e12:C(ko_parent)[T.MAT]                -0.7311      1.031     -0.709      0.481      -2.792       1.329
# gWAT_vol_mean_1e12:C(sex)[T.m]                        -1.0670      1.574     -0.678      0.500      -4.214       2.080
# gWAT_vol_mean_1e12:C(ko_parent)[T.MAT]:C(sex)[T.m]    -3.1432      2.181     -1.441      0.154      -7.502       1.216
# ==============================================================================
# Omnibus:                        5.047   Durbin-Watson:                   1.692
# Prob(Omnibus):                  0.080   Jarque-Bera (JB):                4.181
# Skew:                           0.550   Prob(JB):                        0.124
# Kurtosis:                       3.472   Cond. No.                         105.
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

def model_line(model, sex, ko_parent, gWAT_vol_mean_1e12):
    sex = np.float(sex == 'm')
    ko_parent = np.float(ko_parent == 'MAT')

    return model.params['Intercept'] + \
           model.params['C(ko_parent)[T.MAT]'] * ko_parent + \
           model.params['C(sex)[T.m]'] * sex + \
           model.params['C(ko_parent)[T.MAT]:C(sex)[T.m]'] * ko_parent * sex + \
           model.params['gWAT_vol_mean_1e12'] * gWAT_vol_mean_1e12 + \
           model.params['gWAT_vol_mean_1e12:C(ko_parent)[T.MAT]'] * gWAT_vol_mean_1e12 * ko_parent + \
           model.params['gWAT_vol_mean_1e12:C(sex)[T.m]'] * gWAT_vol_mean_1e12 * sex + \
           model.params['gWAT_vol_mean_1e12:C(ko_parent)[T.MAT]:C(sex)[T.m]'] * gWAT_vol_mean_1e12 * ko_parent * sex


# plot BW as a function of gWAT
if DEBUG:

    annotate = False
    plt.clf()

    # f PAT
    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    gWAT_vol_mean_1e12 = np.linspace(np.min(metainfo['gWAT_vol_mean_1e12'][idx]), np.max(metainfo['gWAT_vol_mean_1e12'][idx]))
    plt.scatter(metainfo['gWAT_vol_mean_1e12'][idx], metainfo['gWAT'][idx], label='f PAT WT', color='C0', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT_vol_mean_1e12'][i], metainfo['gWAT'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    gWAT_vol_mean_1e12 = np.linspace(np.min(metainfo['gWAT_vol_mean_1e12'][idx]), np.max(metainfo['gWAT_vol_mean_1e12'][idx]))
    plt.scatter(metainfo['gWAT_vol_mean_1e12'][idx], metainfo['gWAT'][idx], label='f PAT Het', color='C0')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT_vol_mean_1e12'][i], metainfo['gWAT'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT')
    gWAT_vol_mean_1e12 = np.linspace(np.min(metainfo['gWAT_vol_mean_1e12'][idx]), np.max(metainfo['gWAT_vol_mean_1e12'][idx]))
    gWAT = model_line(model, sex='f', ko_parent='PAT', gWAT_vol_mean_1e12=gWAT_vol_mean_1e12)
    plt.plot(gWAT_vol_mean_1e12, gWAT, color='C0', linewidth=3)

    # f MAT
    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    gWAT_vol_mean_1e12 = np.linspace(np.min(metainfo['gWAT_vol_mean_1e12'][idx]), np.max(metainfo['gWAT_vol_mean_1e12'][idx]))
    plt.scatter(metainfo['gWAT_vol_mean_1e12'][idx], metainfo['gWAT'][idx], label='f MAT WT', color='C2', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT_vol_mean_1e12'][i], metainfo['gWAT'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    gWAT_vol_mean_1e12 = np.linspace(np.min(metainfo['gWAT_vol_mean_1e12'][idx]), np.max(metainfo['gWAT_vol_mean_1e12'][idx]))
    plt.scatter(metainfo['gWAT_vol_mean_1e12'][idx], metainfo['gWAT'][idx], label='f MAT Het', color='C2')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT_vol_mean_1e12'][i], metainfo['gWAT'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT')
    gWAT_vol_mean_1e12 = np.linspace(np.min(metainfo['gWAT_vol_mean_1e12'][idx]), np.max(metainfo['gWAT_vol_mean_1e12'][idx]))
    gWAT = model_line(model, sex='f', ko_parent='MAT', gWAT_vol_mean_1e12=gWAT_vol_mean_1e12)
    plt.plot(gWAT_vol_mean_1e12, gWAT, color='C2', linewidth=3)

    # m PAT
    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    gWAT_vol_mean_1e12 = np.linspace(np.min(metainfo['gWAT_vol_mean_1e12'][idx]), np.max(metainfo['gWAT_vol_mean_1e12'][idx]))
    plt.scatter(metainfo['gWAT_vol_mean_1e12'][idx], metainfo['gWAT'][idx], label='m PAT WT', color='k', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT_vol_mean_1e12'][i], metainfo['gWAT'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    gWAT_vol_mean_1e12 = np.linspace(np.min(metainfo['gWAT_vol_mean_1e12'][idx]), np.max(metainfo['gWAT_vol_mean_1e12'][idx]))
    plt.scatter(metainfo['gWAT_vol_mean_1e12'][idx], metainfo['gWAT'][idx], label='m PAT Het', color='k')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT_vol_mean_1e12'][i], metainfo['gWAT'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT')
    gWAT_vol_mean_1e12 = np.linspace(np.min(metainfo['gWAT_vol_mean_1e12'][idx]), np.max(metainfo['gWAT_vol_mean_1e12'][idx]))
    gWAT = model_line(model, sex='m', ko_parent='PAT', gWAT_vol_mean_1e12=gWAT_vol_mean_1e12)
    plt.plot(gWAT_vol_mean_1e12, gWAT, color='k', linewidth=3)

    # m MAT
    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    gWAT_vol_mean_1e12 = np.linspace(np.min(metainfo['gWAT_vol_mean_1e12'][idx]), np.max(metainfo['gWAT_vol_mean_1e12'][idx]))
    plt.scatter(metainfo['gWAT_vol_mean_1e12'][idx], metainfo['gWAT'][idx], label='m MAT WT', color='C3', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT_vol_mean_1e12'][i], metainfo['gWAT'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    gWAT_vol_mean_1e12 = np.linspace(np.min(metainfo['gWAT_vol_mean_1e12'][idx]), np.max(metainfo['gWAT_vol_mean_1e12'][idx]))
    plt.scatter(metainfo['gWAT_vol_mean_1e12'][idx], metainfo['gWAT'][idx], label='m MAT Het', color='C3')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['gWAT_vol_mean_1e12'][i], metainfo['gWAT'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT')
    gWAT_vol_mean_1e12 = np.linspace(np.min(metainfo['gWAT_vol_mean_1e12'][idx]), np.max(metainfo['gWAT_vol_mean_1e12'][idx]))
    gWAT = model_line(model, sex='m', ko_parent='MAT', gWAT_vol_mean_1e12=gWAT_vol_mean_1e12)
    plt.plot(gWAT_vol_mean_1e12, gWAT, color='C3', linewidth=3)

    plt.legend()

    plt.xlabel(r'$\overline{V}_{gWAT}$ (g)', fontsize=14)
    plt.ylabel('m$_{gWAT}$ (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_gWAT_model_Vm.png'), bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_gWAT_model_Vm.svg'), bbox_inches='tight')


########################################################################################################################
### Model SC ~ SC_vol_mean_1e12 * C(ko_parent) * C(sex)
### INVALID model, like previous
### NOT USED IN PAPER
########################################################################################################################

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

# data that we are going to use
idx_subset = idx_not_nan

model = sm.formula.ols('SC ~ SC_vol_mean_1e12 * C(ko_parent) * C(sex)', data=metainfo, subset=idx_subset).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     SC   R-squared:                       0.362
# Model:                            OLS   Adj. R-squared:                  0.294
# Method:                 Least Squares   F-statistic:                     5.347
# Date:                Thu, 27 Feb 2020   Prob (F-statistic):           7.01e-05
# Time:                        16:08:51   Log-Likelihood:                -12.523
# No. Observations:                  74   AIC:                             41.05
# Df Residuals:                      66   BIC:                             59.48
# Df Model:                           7
# Covariance Type:            nonrobust
# ====================================================================================================================
#                                                        coef    std err          t      P>|t|      [0.025      0.975]
# --------------------------------------------------------------------------------------------------------------------
# Intercept                                            0.0901      0.165      0.546      0.587      -0.239       0.419
# C(ko_parent)[T.MAT]                                  0.1587      0.218      0.729      0.468      -0.276       0.593
# C(sex)[T.m]                                         -0.1723      0.324     -0.532      0.597      -0.819       0.475
# C(ko_parent)[T.MAT]:C(sex)[T.m]                      0.5270      0.455      1.159      0.251      -0.381       1.435
# SC_vol_mean_1e12                                     4.1312      1.309      3.156      0.002       1.518       6.744
# SC_vol_mean_1e12:C(ko_parent)[T.MAT]                -3.4677      1.480     -2.342      0.022      -6.424      -0.512
# SC_vol_mean_1e12:C(sex)[T.m]                        -0.8534      1.670     -0.511      0.611      -4.189       2.482
# SC_vol_mean_1e12:C(ko_parent)[T.MAT]:C(sex)[T.m]     0.0342      2.143      0.016      0.987      -4.245       4.314
# ==============================================================================
# Omnibus:                       22.663   Durbin-Watson:                   1.433
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):               30.557
# Skew:                           1.358   Prob(JB):                     2.31e-07
# Kurtosis:                       4.593   Cond. No.                         119.
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

def model_line(model, sex, ko_parent, SC_vol_mean_1e12):
    sex = np.float(sex == 'm')
    ko_parent = np.float(ko_parent == 'MAT')

    return model.params['Intercept'] + \
           model.params['C(ko_parent)[T.MAT]'] * ko_parent + \
           model.params['C(sex)[T.m]'] * sex + \
           model.params['C(ko_parent)[T.MAT]:C(sex)[T.m]'] * ko_parent * sex + \
           model.params['SC_vol_mean_1e12'] * SC_vol_mean_1e12 + \
           model.params['SC_vol_mean_1e12:C(ko_parent)[T.MAT]'] * SC_vol_mean_1e12 * ko_parent + \
           model.params['SC_vol_mean_1e12:C(sex)[T.m]'] * SC_vol_mean_1e12 * sex + \
           model.params['SC_vol_mean_1e12:C(ko_parent)[T.MAT]:C(sex)[T.m]'] * SC_vol_mean_1e12 * ko_parent * sex


# plot BW as a function of SC
if DEBUG:

    annotate = False
    plt.clf()

    # f PAT
    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC_vol_mean_1e12 = np.linspace(np.min(metainfo['SC_vol_mean_1e12'][idx]), np.max(metainfo['SC_vol_mean_1e12'][idx]))
    plt.scatter(metainfo['SC_vol_mean_1e12'][idx], metainfo['SC'][idx], label='f PAT WT', color='C0', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC_vol_mean_1e12'][i], metainfo['SC'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC_vol_mean_1e12 = np.linspace(np.min(metainfo['SC_vol_mean_1e12'][idx]), np.max(metainfo['SC_vol_mean_1e12'][idx]))
    plt.scatter(metainfo['SC_vol_mean_1e12'][idx], metainfo['SC'][idx], label='f PAT Het', color='C0')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC_vol_mean_1e12'][i], metainfo['SC'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'PAT')
    SC_vol_mean_1e12 = np.linspace(np.min(metainfo['SC_vol_mean_1e12'][idx]), np.max(metainfo['SC_vol_mean_1e12'][idx]))
    SC = model_line(model, sex='f', ko_parent='PAT', SC_vol_mean_1e12=SC_vol_mean_1e12)
    plt.plot(SC_vol_mean_1e12, SC, color='C0', linewidth=3)

    # f MAT
    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC_vol_mean_1e12 = np.linspace(np.min(metainfo['SC_vol_mean_1e12'][idx]), np.max(metainfo['SC_vol_mean_1e12'][idx]))
    plt.scatter(metainfo['SC_vol_mean_1e12'][idx], metainfo['SC'][idx], label='f MAT WT', color='C2', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC_vol_mean_1e12'][i], metainfo['SC'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC_vol_mean_1e12 = np.linspace(np.min(metainfo['SC_vol_mean_1e12'][idx]), np.max(metainfo['SC_vol_mean_1e12'][idx]))
    plt.scatter(metainfo['SC_vol_mean_1e12'][idx], metainfo['SC'][idx], label='f MAT Het', color='C2')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC_vol_mean_1e12'][i], metainfo['SC'][i]))

    idx = (metainfo['sex'] == 'f') * (metainfo['ko_parent'] == 'MAT')
    SC_vol_mean_1e12 = np.linspace(np.min(metainfo['SC_vol_mean_1e12'][idx]), np.max(metainfo['SC_vol_mean_1e12'][idx]))
    SC = model_line(model, sex='f', ko_parent='MAT', SC_vol_mean_1e12=SC_vol_mean_1e12)
    plt.plot(SC_vol_mean_1e12, SC, color='C2', linewidth=3)

    # m PAT
    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC_vol_mean_1e12 = np.linspace(np.min(metainfo['SC_vol_mean_1e12'][idx]), np.max(metainfo['SC_vol_mean_1e12'][idx]))
    plt.scatter(metainfo['SC_vol_mean_1e12'][idx], metainfo['SC'][idx], label='m PAT WT', color='k', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC_vol_mean_1e12'][i], metainfo['SC'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC_vol_mean_1e12 = np.linspace(np.min(metainfo['SC_vol_mean_1e12'][idx]), np.max(metainfo['SC_vol_mean_1e12'][idx]))
    plt.scatter(metainfo['SC_vol_mean_1e12'][idx], metainfo['SC'][idx], label='m PAT Het', color='k')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC_vol_mean_1e12'][i], metainfo['SC'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'PAT')
    SC_vol_mean_1e12 = np.linspace(np.min(metainfo['SC_vol_mean_1e12'][idx]), np.max(metainfo['SC_vol_mean_1e12'][idx]))
    SC = model_line(model, sex='m', ko_parent='PAT', SC_vol_mean_1e12=SC_vol_mean_1e12)
    plt.plot(SC_vol_mean_1e12, SC, color='k', linewidth=3)

    # m MAT
    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:WT')
    SC_vol_mean_1e12 = np.linspace(np.min(metainfo['SC_vol_mean_1e12'][idx]), np.max(metainfo['SC_vol_mean_1e12'][idx]))
    plt.scatter(metainfo['SC_vol_mean_1e12'][idx], metainfo['SC'][idx], label='m MAT WT', color='C3', facecolor='none')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC_vol_mean_1e12'][i], metainfo['SC'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT') * (metainfo['genotype'] == 'KLF14-KO:Het')
    SC_vol_mean_1e12 = np.linspace(np.min(metainfo['SC_vol_mean_1e12'][idx]), np.max(metainfo['SC_vol_mean_1e12'][idx]))
    plt.scatter(metainfo['SC_vol_mean_1e12'][idx], metainfo['SC'][idx], label='m MAT Het', color='C3')
    if annotate:
        for i in np.where(idx)[0]:
            plt.annotate(i, (metainfo['SC_vol_mean_1e12'][i], metainfo['SC'][i]))

    idx = (metainfo['sex'] == 'm') * (metainfo['ko_parent'] == 'MAT')
    SC_vol_mean_1e12 = np.linspace(np.min(metainfo['SC_vol_mean_1e12'][idx]), np.max(metainfo['SC_vol_mean_1e12'][idx]))
    SC = model_line(model, sex='m', ko_parent='MAT', SC_vol_mean_1e12=SC_vol_mean_1e12)
    plt.plot(SC_vol_mean_1e12, SC, color='C3', linewidth=3)

    plt.legend()

    plt.xlabel(r'$\overline{V}_{SC}$ (g)', fontsize=14)
    plt.ylabel('m$_{SC}$ (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_SC_model_Vm.png'), bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0099_SC_model_Vm.svg'), bbox_inches='tight')


########################################################################################################################
### Model gWAT_vol_mean_1e18 ~ C(ko_parent) * C(sex) * C(genotype)
### VALID model.
### NOT USED IN PAPER
########################################################################################################################

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

# data that we are going to use
idx_subset = idx_not_nan

model = sm.formula.ols('gWAT_vol_mean_1e18 ~ C(ko_parent) * C(sex) * C(genotype)', data=metainfo, subset=idx_subset).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:     gWAT_vol_mean_1e18   R-squared:                       0.596
# Model:                            OLS   Adj. R-squared:                  0.551
# Method:                 Least Squares   F-statistic:                     13.08
# Date:                Wed, 04 Mar 2020   Prob (F-statistic):           3.30e-10
# Time:                        04:15:59   Log-Likelihood:                -891.51
# No. Observations:                  70   AIC:                             1799.
# Df Residuals:                      62   BIC:                             1817.
# Df Model:                           7
# Covariance Type:            nonrobust
# ===============================================================================================================================
#                                                                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                    1.724e+05   2.91e+04      5.920      0.000    1.14e+05    2.31e+05
# C(ko_parent)[T.MAT]                                           1.38e+05   4.01e+04      3.440      0.001    5.78e+04    2.18e+05
# C(sex)[T.m]                                                  2.433e+05   4.12e+04      5.909      0.000    1.61e+05    3.26e+05
# C(genotype)[T.KLF14-KO:Het]                                  4.539e+04   4.24e+04      1.070      0.289   -3.94e+04     1.3e+05
# C(ko_parent)[T.MAT]:C(sex)[T.m]                              -1.23e+05   5.75e+04     -2.138      0.036   -2.38e+05   -8018.527
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]             -1.132e+05   5.84e+04     -1.938      0.057    -2.3e+05    3571.181
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                     -1.171e+04   6.11e+04     -0.192      0.849   -1.34e+05    1.11e+05
# C(ko_parent)[T.MAT]:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]  4.928e+04   8.39e+04      0.587      0.559   -1.19e+05    2.17e+05
# ==============================================================================
# Omnibus:                        1.120   Durbin-Watson:                   1.766
# Prob(Omnibus):                  0.571   Jarque-Bera (JB):                1.193
# Skew:                           0.260   Prob(JB):                        0.551
# Kurtosis:                       2.628   Cond. No.                         17.8
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


print(model.pvalues)

# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")


########################################################################################################################
### Model SC_vol_mean_1e18 ~ C(ko_parent) * C(sex) * C(genotype)
### NOT USED IN PAPER
########################################################################################################################

formula = 'SC_vol_mean_1e18 ~ C(ko_parent) * C(sex) * C(genotype)'

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

model = sm.formula.ols(formula, data=metainfo, subset=idx_not_nan).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:       SC_vol_mean_1e18   R-squared:                       0.404
# Model:                            OLS   Adj. R-squared:                  0.341
# Method:                 Least Squares   F-statistic:                     6.404
# Date:                Wed, 04 Mar 2020   Prob (F-statistic):           9.26e-06
# Time:                        04:26:28   Log-Likelihood:                -931.85
# No. Observations:                  74   AIC:                             1880.
# Df Residuals:                      66   BIC:                             1898.
# Df Model:                           7
# Covariance Type:            nonrobust
# ===============================================================================================================================
#                                                                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                     1.28e+05   2.85e+04      4.490      0.000    7.11e+04    1.85e+05
# C(ko_parent)[T.MAT]                                          6.585e+04   3.72e+04      1.772      0.081   -8362.656     1.4e+05
# C(sex)[T.m]                                                  1.231e+05   3.72e+04      3.311      0.002    4.88e+04    1.97e+05
# C(genotype)[T.KLF14-KO:Het]                                 -2.858e+04    3.8e+04     -0.752      0.455   -1.04e+05    4.73e+04
# C(ko_parent)[T.MAT]:C(sex)[T.m]                             -4.697e+04   5.02e+04     -0.936      0.353   -1.47e+05    5.32e+04
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]              1541.4338   5.08e+04      0.030      0.976   -9.99e+04    1.03e+05
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                      4.829e+04   5.22e+04      0.925      0.358   -5.59e+04    1.53e+05
# C(ko_parent)[T.MAT]:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het] -7.957e+04   7.07e+04     -1.125      0.265   -2.21e+05    6.16e+04
# ==============================================================================
# Omnibus:                        2.630   Durbin-Watson:                   1.580
# Prob(Omnibus):                  0.268   Jarque-Bera (JB):                2.575
# Skew:                           0.437   Prob(JB):                        0.276
# Kurtosis:                       2.731   Cond. No.                         19.2
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


print(model.pvalues)

# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

# list of point with high influence (large residuals and leverage)
idx_influence = [7, 20, 6, 26]

# list of data points to use in the model
idx_for_model = (set(range(metainfo.shape[0])) - set(idx_influence)) & set(idx_not_nan)
idx_for_model = list(idx_for_model)

model = sm.formula.ols(formula, data=metainfo, subset=idx_for_model).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:       SC_vol_mean_1e18   R-squared:                       0.563
# Model:                            OLS   Adj. R-squared:                  0.513
# Method:                 Least Squares   F-statistic:                     11.39
# Date:                Wed, 04 Mar 2020   Prob (F-statistic):           3.43e-09
# Time:                        04:30:42   Log-Likelihood:                -867.28
# No. Observations:                  70   AIC:                             1751.
# Df Residuals:                      62   BIC:                             1769.
# Df Model:                           7
# Covariance Type:            nonrobust
# ===============================================================================================================================
#                                                                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                     1.28e+05   2.34e+04      5.482      0.000    8.13e+04    1.75e+05
# C(ko_parent)[T.MAT]                                          2.746e+04    3.2e+04      0.859      0.394   -3.65e+04    9.14e+04
# C(sex)[T.m]                                                  1.231e+05   3.04e+04      4.042      0.000    6.22e+04    1.84e+05
# C(genotype)[T.KLF14-KO:Het]                                 -2.858e+04   3.11e+04     -0.918      0.362   -9.08e+04    3.37e+04
# C(ko_parent)[T.MAT]:C(sex)[T.m]                             -8580.3806   4.23e+04     -0.203      0.840   -9.31e+04    7.59e+04
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]             -4572.5179   4.39e+04     -0.104      0.917   -9.22e+04    8.31e+04
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                      4.829e+04   4.28e+04      1.129      0.263   -3.72e+04    1.34e+05
# C(ko_parent)[T.MAT]:C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het] -7.346e+04   5.95e+04     -1.234      0.222   -1.92e+05    4.56e+04
# ==============================================================================
# Omnibus:                        6.511   Durbin-Watson:                   1.528
# Prob(Omnibus):                  0.039   Jarque-Bera (JB):                2.547
# Skew:                          -0.006   Prob(JB):                        0.280
# Kurtosis:                       2.066   Cond. No.                         19.2
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


print(model.pvalues)

# ANOVA table
aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)

########################################################################################################################
### Model gWAT_vol_mean ~ C(sex) * C(ko_parent) * C(genotype)
### NOT USED IN PAPER
########################################################################################################################

formula = 'gWAT_vol_mean ~ C(sex) * C(ko_parent) * C(genotype)'

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

model = sm.formula.ols(formula, data=metainfo, subset=idx_not_nan).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:          gWAT_vol_mean   R-squared:                       0.596
# Model:                            OLS   Adj. R-squared:                  0.551
# Method:                 Least Squares   F-statistic:                     13.08
# Date:                Fri, 21 Feb 2020   Prob (F-statistic):           3.30e-10
# Time:                        16:56:33   Log-Likelihood:                 2009.7
# No. Observations:                  70   AIC:                            -4003.
# Df Residuals:                      62   BIC:                            -3986.
# Df Model:                           7
# Covariance Type:            nonrobust
# ===============================================================================================================================
#                                                                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                    1.724e-13   2.91e-14      5.920      0.000    1.14e-13    2.31e-13
# C(sex)[T.m]                                                  2.433e-13   4.12e-14      5.909      0.000    1.61e-13    3.26e-13
# C(ko_parent)[T.MAT]                                           1.38e-13   4.01e-14      3.440      0.001    5.78e-14    2.18e-13
# C(genotype)[T.KLF14-KO:Het]                                  4.539e-14   4.24e-14      1.070      0.289   -3.94e-14     1.3e-13
# C(sex)[T.m]:C(ko_parent)[T.MAT]                              -1.23e-13   5.75e-14     -2.138      0.036   -2.38e-13   -8.02e-15
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                     -1.171e-14   6.11e-14     -0.192      0.849   -1.34e-13    1.11e-13
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]             -1.132e-13   5.84e-14     -1.938      0.057    -2.3e-13    3.57e-15
# C(sex)[T.m]:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]  4.928e-14   8.39e-14      0.587      0.559   -1.19e-13    2.17e-13
# ==============================================================================
# Omnibus:                        1.120   Durbin-Watson:                   1.766
# Prob(Omnibus):                  0.571   Jarque-Bera (JB):                1.193
# Skew:                           0.260   Prob(JB):                        0.551
# Kurtosis:                       2.628   Cond. No.                         17.8
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

# list of point with high influence (large residuals and leverage)
idx_influence = [37, 7, 0]

# list of data points to use in the model
idx_for_model = (set(range(metainfo.shape[0])) - set(idx_influence)) & set(idx_not_nan)
idx_for_model = list(idx_for_model)

model = sm.formula.ols(formula, data=metainfo, subset=idx_for_model).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:          gWAT_vol_mean   R-squared:                       0.689
# Model:                            OLS   Adj. R-squared:                  0.652
# Method:                 Least Squares   F-statistic:                     18.66
# Date:                Fri, 21 Feb 2020   Prob (F-statistic):           7.43e-13
# Time:                        16:57:05   Log-Likelihood:                 1933.1
# No. Observations:                  67   AIC:                            -3850.
# Df Residuals:                      59   BIC:                            -3833.
# Df Model:                           7
# Covariance Type:            nonrobust
# ===============================================================================================================================
#                                                                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                    1.724e-13   2.54e-14      6.798      0.000    1.22e-13    2.23e-13
# C(sex)[T.m]                                                  2.433e-13   3.59e-14      6.785      0.000    1.72e-13    3.15e-13
# C(ko_parent)[T.MAT]                                              9e-14    3.7e-14      2.435      0.018     1.6e-14    1.64e-13
# C(genotype)[T.KLF14-KO:Het]                                  4.539e-14    3.7e-14      1.228      0.224   -2.86e-14    1.19e-13
# C(sex)[T.m]:C(ko_parent)[T.MAT]                             -7.491e-14   5.15e-14     -1.455      0.151   -1.78e-13    2.81e-14
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                     -1.171e-14   5.32e-14     -0.220      0.827   -1.18e-13    9.48e-14
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]             -8.848e-14    5.3e-14     -1.668      0.101   -1.95e-13    1.76e-14
# C(sex)[T.m]:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]  2.457e-14   7.46e-14      0.329      0.743   -1.25e-13    1.74e-13
# ==============================================================================
# Omnibus:                        3.101   Durbin-Watson:                   1.777
# Prob(Omnibus):                  0.212   Jarque-Bera (JB):                1.925
# Skew:                           0.177   Prob(JB):                        0.382
# Kurtosis:                       2.249   Cond. No.                         17.9
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

print(model.pvalues)

# ANOVA table
aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)

########################################################################################################################
### Model SC_kN ~ C(sex) * C(ko_parent) * C(genotype)
### NOT USED IN PAPER
########################################################################################################################

formula = 'SC_kN ~ C(sex) * C(ko_parent) * C(genotype)'

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

model = sm.formula.ols(formula, data=metainfo, subset=idx_not_nan).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                  SC_kN   R-squared:                       0.282
# Model:                            OLS   Adj. R-squared:                  0.206
# Method:                 Least Squares   F-statistic:                     3.708
# Date:                Wed, 04 Mar 2020   Prob (F-statistic):            0.00193
# Time:                        05:00:26   Log-Likelihood:                -1184.4
# No. Observations:                  74   AIC:                             2385.
# Df Residuals:                      66   BIC:                             2403.
# Df Model:                           7
# Covariance Type:            nonrobust
# ===============================================================================================================================
#                                                                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                    4.263e+06   8.65e+05      4.930      0.000    2.54e+06    5.99e+06
# C(sex)[T.m]                                                 -1.426e+06   1.13e+06     -1.265      0.210   -3.68e+06    8.25e+05
# C(ko_parent)[T.MAT]                                         -9.607e+05   1.13e+06     -0.852      0.397   -3.21e+06    1.29e+06
# C(genotype)[T.KLF14-KO:Het]                                  2.268e+06   1.15e+06      1.967      0.053   -3.37e+04    4.57e+06
# C(sex)[T.m]:C(ko_parent)[T.MAT]                              3.072e+05   1.52e+06      0.202      0.841   -2.73e+06    3.35e+06
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                     -1.371e+06   1.58e+06     -0.866      0.390   -4.53e+06    1.79e+06
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]             -3.702e+06   1.54e+06     -2.402      0.019   -6.78e+06   -6.25e+05
# C(sex)[T.m]:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]  4.165e+06   2.14e+06      1.942      0.056   -1.17e+05    8.45e+06
# ==============================================================================
# Omnibus:                       32.335   Durbin-Watson:                   1.792
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):               70.647
# Skew:                           1.499   Prob(JB):                     4.56e-16
# Kurtosis:                       6.731   Cond. No.                         19.2
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

# list of point with high influence (large residuals and leverage)
idx_influence = [65, 53, 35, 31, 62, 42]

# list of data points to use in the model
idx_for_model = (set(range(metainfo.shape[0])) - set(idx_influence)) & set(idx_not_nan)
idx_for_model = list(idx_for_model)

model = sm.formula.ols(formula, data=metainfo, subset=idx_for_model).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:               SC_kN   R-squared:                       0.347
# Model:                            OLS   Adj. R-squared:                  0.271
# Method:                 Least Squares   F-statistic:                     4.562
# Date:                Fri, 21 Feb 2020   Prob (F-statistic):           0.000390
# Time:                        17:53:50   Log-Likelihood:                -1048.4
# No. Observations:                  68   AIC:                             2113.
# Df Residuals:                      60   BIC:                             2131.
# Df Model:                           7
# Covariance Type:            nonrobust
# ===============================================================================================================================
#                                                                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                     3.55e+06   5.22e+05      6.802      0.000    2.51e+06    4.59e+06
# C(sex)[T.m]                                                 -7.126e+05    6.6e+05     -1.079      0.285   -2.03e+06    6.08e+05
# C(ko_parent)[T.MAT]                                          -9.44e+05   6.74e+05     -1.401      0.166   -2.29e+06    4.04e+05
# C(genotype)[T.KLF14-KO:Het]                                  1.494e+06   7.38e+05      2.024      0.047    1.72e+04    2.97e+06
# C(sex)[T.m]:C(ko_parent)[T.MAT]                              2.905e+05   8.84e+05      0.329      0.743   -1.48e+06    2.06e+06
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                     -5.965e+05   9.55e+05     -0.625      0.535   -2.51e+06    1.31e+06
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]             -2.231e+06   9.43e+05     -2.366      0.021   -4.12e+06   -3.45e+05
# C(sex)[T.m]:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]  2.021e+06   1.27e+06      1.597      0.116   -5.11e+05    4.55e+06
# ==============================================================================
# Omnibus:                       21.123   Durbin-Watson:                   1.610
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):               35.834
# Skew:                           1.105   Prob(JB):                     1.65e-08
# Kurtosis:                       5.786   Cond. No.                         20.1
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


print(model.pvalues)

# ANOVA table
aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)

########################################################################################################################
### Model gWAT_kN ~ C(sex) * C(ko_parent) * C(genotype)
### NOT USED IN PAPER
########################################################################################################################

formula = 'gWAT_kN ~ C(sex) * C(ko_parent) * C(genotype)'

idx_not_nan = np.where(~np.isnan(metainfo['SC']) * ~np.isnan(metainfo['gWAT']) * ~np.isnan(metainfo['BW']))[0]

model = sm.formula.ols(formula, data=metainfo, subset=idx_not_nan).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                gWAT_kN   R-squared:                       0.135
# Model:                            OLS   Adj. R-squared:                  0.038
# Method:                 Least Squares   F-statistic:                     1.388
# Date:                Wed, 04 Mar 2020   Prob (F-statistic):              0.226
# Time:                        05:01:51   Log-Likelihood:                -1121.8
# No. Observations:                  70   AIC:                             2260.
# Df Residuals:                      62   BIC:                             2278.
# Df Model:                           7
# Covariance Type:            nonrobust
# ===============================================================================================================================
#                                                                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                     4.73e+06   7.82e+05      6.049      0.000    3.17e+06    6.29e+06
# C(sex)[T.m]                                                 -1.716e+06   1.11e+06     -1.551      0.126   -3.93e+06    4.95e+05
# C(ko_parent)[T.MAT]                                         -7.536e+05   1.08e+06     -0.699      0.487   -2.91e+06     1.4e+06
# C(genotype)[T.KLF14-KO:Het]                                  4.282e+05   1.14e+06      0.376      0.708   -1.85e+06    2.71e+06
# C(sex)[T.m]:C(ko_parent)[T.MAT]                              4.602e+05   1.54e+06      0.298      0.767   -2.63e+06    3.55e+06
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                     -9.345e+05   1.64e+06     -0.569      0.571   -4.22e+06    2.35e+06
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]             -1.323e+06   1.57e+06     -0.843      0.402   -4.46e+06    1.81e+06
# C(sex)[T.m]:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]  2.516e+06   2.25e+06      1.116      0.269   -1.99e+06    7.02e+06
# ==============================================================================
# Omnibus:                       66.663   Durbin-Watson:                   1.398
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):              432.785
# Skew:                           2.837   Prob(JB):                     1.05e-94
# Kurtosis:                      13.779   Cond. No.                         17.8
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



# partial regression and influence plots
if DEBUG:
    sm.graphics.plot_partregress_grid(model)
    sm.graphics.influence_plot(model, criterion="cooks")

# list of point with high influence (large residuals and leverage)
idx_influence = [63, 62, 35]

# list of data points to use in the model
idx_for_model = (set(range(metainfo.shape[0])) - set(idx_influence)) & set(idx_not_nan)
idx_for_model = list(idx_for_model)

model = sm.formula.ols(formula, data=metainfo, subset=idx_for_model).fit()
print(model.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:             gWAT_kN   R-squared:                       0.109
# Model:                            OLS   Adj. R-squared:                  0.004
# Method:                 Least Squares   F-statistic:                     1.034
# Date:                Fri, 21 Feb 2020   Prob (F-statistic):              0.418
# Time:                        17:56:47   Log-Likelihood:                -1022.2
# No. Observations:                  67   AIC:                             2060.
# Df Residuals:                      59   BIC:                             2078.
# Df Model:                           7
# Covariance Type:            nonrobust
# ===============================================================================================================================
#                                                                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                    3.616e+06   3.85e+05      9.382      0.000    2.84e+06    4.39e+06
# C(sex)[T.m]                                                 -6.021e+05    5.3e+05     -1.137      0.260   -1.66e+06    4.58e+05
# C(ko_parent)[T.MAT]                                         -3.454e+05    5.3e+05     -0.652      0.517   -1.41e+06    7.15e+05
# C(genotype)[T.KLF14-KO:Het]                                 -2.249e+04   5.64e+05     -0.040      0.968   -1.15e+06    1.11e+06
# C(sex)[T.m]:C(ko_parent)[T.MAT]                              5.197e+04   7.38e+05      0.070      0.944   -1.42e+06    1.53e+06
# C(sex)[T.m]:C(genotype)[T.KLF14-KO:Het]                     -4.838e+05   7.88e+05     -0.614      0.541   -2.06e+06    1.09e+06
# C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]             -1.668e+05   7.63e+05     -0.219      0.828   -1.69e+06    1.36e+06
# C(sex)[T.m]:C(ko_parent)[T.MAT]:C(genotype)[T.KLF14-KO:Het]   1.36e+06   1.07e+06      1.269      0.209   -7.84e+05     3.5e+06
# ==============================================================================
# Omnibus:                        5.089   Durbin-Watson:                   1.635
# Prob(Omnibus):                  0.079   Jarque-Bera (JB):                4.244
# Skew:                           0.583   Prob(JB):                        0.120
# Kurtosis:                       3.402   Cond. No.                         18.4
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

print(model.pvalues)

# ANOVA table
aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)

########################################################################################################################
### ANOVA testing of gWAT Vm w.r.t. ko_parent and genotype
### NOT USED IN PAPER
########################################################################################################################

idx_not_nan = ~np.isnan(metainfo['gWAT_vol_mean_1e18'])
idx_f = metainfo['sex'] == 'f'
idx_m = metainfo['sex'] == 'm'

# gWAT female PAT, MAT, WT, Het: TUKEY’S HSD POST-HOC COMPARISON
idx = np.where(idx_not_nan * idx_f)[0]
mc = MultiComparison(metainfo.loc[idx, 'gWAT_vol_mean_1e18'], metainfo.loc[idx, 'ko_parent_genotype'])
mc_results = mc.tukeyhsd()
print(mc_results)

# gWAT male PAT, MAT, WT, Het: TUKEY’S HSD POST-HOC COMPARISON
idx = np.where(idx_not_nan * idx_m)[0]
mc = MultiComparison(metainfo.loc[idx, 'gWAT_vol_mean_1e18'], metainfo.loc[idx, 'ko_parent_genotype'])
mc_results = mc.tukeyhsd()
print(mc_results)

# gWAT female, PAT vs MAT: TUKEY’S HSD POST-HOC COMPARISON
idx = np.where(idx_not_nan * idx_f)[0]
mc = MultiComparison(metainfo.loc[idx, 'gWAT_vol_mean_1e18'], metainfo.loc[idx, 'ko_parent'])
mc_results = mc.tukeyhsd()
print(mc_results)

# gWAT male, PAT vs MAT: TUKEY’S HSD POST-HOC COMPARISON
idx = np.where(idx_not_nan * idx_m)[0]
mc = MultiComparison(metainfo.loc[idx, 'gWAT_vol_mean_1e18'], metainfo.loc[idx, 'ko_parent'])
mc_results = mc.tukeyhsd()
print(mc_results)

########################################################################################################################
### ANOVA testing of SC Vm w.r.t. ko_parent and genotype
### NOT USED IN PAPER
########################################################################################################################

idx_not_nan = ~np.isnan(metainfo['SC_vol_mean_1e18'])
idx_f = metainfo['sex'] == 'f'
idx_m = metainfo['sex'] == 'm'

# SC female PAT, MAT, WT, Het: TUKEY’S HSD POST-HOC COMPARISON
idx = np.where(idx_not_nan * idx_f)[0]
mc = MultiComparison(metainfo.loc[idx, 'SC_vol_mean_1e18'], metainfo.loc[idx, 'ko_parent_genotype'])
mc_results = mc.tukeyhsd()
print(mc_results)

# SC male PAT, MAT, WT, Het: TUKEY’S HSD POST-HOC COMPARISON
idx = np.where(idx_not_nan * idx_m)[0]
mc = MultiComparison(metainfo.loc[idx, 'SC_vol_mean_1e18'], metainfo.loc[idx, 'ko_parent_genotype'])
mc_results = mc.tukeyhsd()
print(mc_results)

# SC female, PAT vs MAT: TUKEY’S HSD POST-HOC COMPARISON
idx = np.where(idx_not_nan * idx_f)[0]
mc = MultiComparison(metainfo.loc[idx, 'SC_vol_mean_1e18'], metainfo.loc[idx, 'ko_parent'])
mc_results = mc.tukeyhsd()
print(mc_results)

# SC male, PAT vs MAT: TUKEY’S HSD POST-HOC COMPARISON
idx = np.where(idx_not_nan * idx_m)[0]
mc = MultiComparison(metainfo.loc[idx, 'SC_vol_mean_1e18'], metainfo.loc[idx, 'ko_parent'])
mc_results = mc.tukeyhsd()
print(mc_results)

