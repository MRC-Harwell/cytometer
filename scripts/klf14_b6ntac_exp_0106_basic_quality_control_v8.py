"""
Check that all histology files have been segmented by the pipeline
"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0106_basic_quality_control_v8'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import sys
if os.path.join(home, 'Software/cytometer') not in sys.path:
    sys.path.extend([os.path.join(home, 'Software/cytometer')])

# Filter out INFO & WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# limit number of GPUs
if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
    print('Limiting visible CUDA devices to: ' + os.environ['CUDA_VISIBLE_DEVICES'])

# force tensorflow environment
os.environ['KERAS_BACKEND'] = 'tensorflow'

import warnings
import openslide
import numpy as np
import matplotlib.pyplot as plt

# import tensorflow as tf
# if tf.test.is_gpu_available():
#     print('GPU available')
# else:
#     raise SystemError('GPU is not available')

DEBUG = False
SAVE_FIGS = False

# data paths
histology_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
histology_ext = '.ndpi'
area2quantile_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v8')
saved_models_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v8')
annotations_dir = os.path.join(home, 'bit/cytometer_data/aida_data_Klf14_v8/annotations')

# k-folds file
saved_extra_kfolds_filename = 'klf14_b6ntac_exp_0094_generate_extra_training_images.pickle'

# full resolution image window and network expected receptive field parameters
fullres_box_size = np.array([2751, 2751])
receptive_field = np.array([131, 131])


########################################################################################################################
# dictionary of images and the folds they will be processing under
########################################################################################################################

# files used for training and testing, all from SCWAT. Each slide is assigned to the same fold as used for training
histo_files_list = {}
histo_files_list['KLF14-B6NTAC-MAT-18.2b  58-16 C1 - 2016-02-03 11.10.52'] = 0
histo_files_list['KLF14-B6NTAC-MAT-18.2d  60-16 C1 - 2016-02-03 13.13.57'] = 0
histo_files_list['KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38'] = 1
histo_files_list['KLF14-B6NTAC-MAT-17.2c  66-16 C1 - 2016-02-04 11.46.39'] = 1
histo_files_list['KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04'] = 2
histo_files_list['KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53'] = 2
histo_files_list['KLF14-B6NTAC-37.1c PAT 108-16 C1 - 2016-02-15 14.49.45'] = 3
histo_files_list['KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46'] = 3
histo_files_list['KLF14-B6NTAC-37.1d PAT 109-16 C1 - 2016-02-15 15.19.08'] = 4
histo_files_list['KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52'] = 4
histo_files_list['KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38'] = 5
histo_files_list['KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06'] = 5
histo_files_list['KLF14-B6NTAC-MAT-18.1a  50-16 C1 - 2016-02-02 09.12.41'] = 6
histo_files_list['KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11'] = 6
histo_files_list['KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00'] = 7
histo_files_list['KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32'] = 7
histo_files_list['KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33'] = 8
histo_files_list['KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52'] = 8
histo_files_list['KLF14-B6NTAC-MAT-17.2f  68-16 C1 - 2016-02-04 15.05.54'] = 9
histo_files_list['KLF14-B6NTAC-MAT-18.2g  63-16 C1 - 2016-02-03 16.58.52'] = 9

# add GWAT slices not seen at training, from animals seen at training. We assign images to a fold so that an image won't
# be segmented by a pipeline that has seen the same slides from the same animal at training)
histo_files_list['KLF14-B6NTAC-MAT-18.2b  58-16 B1 - 2016-02-03 09.58.06'] = 0
histo_files_list['KLF14-B6NTAC-MAT-18.2d  60-16 B1 - 2016-02-03 12.56.49'] = 0
histo_files_list['KLF14-B6NTAC 36.1i PAT 104-16 B1 - 2016-02-12 11.37.56'] = 1
histo_files_list['KLF14-B6NTAC-MAT-17.2c  66-16 B1 - 2016-02-04 11.14.28'] = 1
histo_files_list['KLF14-B6NTAC-MAT-17.1c  46-16 B1 - 2016-02-01 13.01.30'] = 2
histo_files_list['KLF14-B6NTAC-MAT-18.3d  224-16 B1 - 2016-02-26 10.48.56'] = 2
histo_files_list['KLF14-B6NTAC-37.1c PAT 108-16 B1 - 2016-02-15 12.33.10'] = 3
histo_files_list['KLF14-B6NTAC-MAT-16.2d  214-16 B1 - 2016-02-17 15.43.57'] = 3
histo_files_list['KLF14-B6NTAC-37.1d PAT 109-16 B1 - 2016-02-15 15.03.44'] = 4
histo_files_list['KLF14-B6NTAC-PAT-37.2g  415-16 B1 - 2016-03-16 11.04.45'] = 4
histo_files_list['KLF14-B6NTAC-36.1a PAT 96-16 B1 - 2016-02-10 15.32.31'] = 5
histo_files_list['KLF14-B6NTAC-36.1b PAT 97-16 B1 - 2016-02-10 17.15.16'] = 5
histo_files_list['KLF14-B6NTAC-MAT-18.1a  50-16 B1 - 2016-02-02 08.49.06'] = 6
histo_files_list['KLF14-B6NTAC-PAT-36.3d  416-16 B1 - 2016-03-16 14.22.04'] = 6
histo_files_list['KLF14-B6NTAC-36.1c PAT 98-16 B1 - 2016-02-10 18.32.40'] = 7
histo_files_list['KLF14-B6NTAC-PAT-37.4a  417-16 B1 - 2016-03-16 15.25.38'] = 7
histo_files_list['KLF14-B6NTAC-MAT-18.1e  54-16 B1 - 2016-02-02 15.06.05'] = 8
histo_files_list['KLF14-B6NTAC-MAT-18.3b  223-16 B1 - 2016-02-25 16.53.42'] = 8
histo_files_list['KLF14-B6NTAC-MAT-17.2f  68-16 B1 - 2016-02-04 14.01.40'] = 9
histo_files_list['KLF14-B6NTAC-MAT-18.2g  63-16 B1 - 2016-02-03 16.40.37'] = 9

# add SCWAT slices not seen at training, from animals not seen at training. We randomly assign images to a fold
# males
histo_files_list['KLF14-B6NTAC 36.1d PAT 99-16 C1 - 2016-02-11 11.48.31'] = 3
histo_files_list['KLF14-B6NTAC 36.1e PAT 100-16 C1 - 2016-02-11 14.06.56'] = 7
histo_files_list['KLF14-B6NTAC 36.1f PAT 101-16 C1 - 2016-02-11 15.23.06'] = 5
histo_files_list['KLF14-B6NTAC 36.1g PAT 102-16 C1 - 2016-02-11 17.20.14'] = 1
histo_files_list['KLF14-B6NTAC 36.1h PAT 103-16 C1 - 2016-02-12 10.15.22'] = 4
histo_files_list['KLF14-B6NTAC 36.1j PAT 105-16 C1 - 2016-02-12 14.33.33'] = 1
histo_files_list['KLF14-B6NTAC-PAT-37.2h  418-16 C1 - 2016-03-16 17.01.17'] = 8
histo_files_list['KLF14-B6NTAC-PAT-37.4b  419-16 C1 - 2016-03-17 10.22.54'] = 1
histo_files_list['KLF14-B6NTAC-PAT-39.2d  454-16 C1 - 2016-03-17 14.33.38'] = 2
histo_files_list['KLF14-B6NTAC-37.1h PAT 113-16 C1 - 2016-02-16 15.14.09'] = 5
histo_files_list['KLF14-B6NTAC-38.1e PAT 94-16 C1 - 2016-02-10 12.13.10'] = 1
histo_files_list['KLF14-B6NTAC-38.1f PAT 95-16 C1 - 2016-02-10 14.41.44'] = 0
histo_files_list['KLF14-B6NTAC-MAT-16.2b  212-16 C1 - 2016-02-17 12.49.00'] = 3
histo_files_list['KLF14-B6NTAC-MAT-16.2c  213-16 C1 - 2016-02-17 14.51.18'] = 8
histo_files_list['KLF14-B6NTAC-MAT-16.2e  215-16 C1 - 2016-02-18 09.19.26'] = 4
histo_files_list['KLF14-B6NTAC-MAT-16.2f  216-16 C1 - 2016-02-18 10.28.27'] = 5
histo_files_list['KLF14-B6NTAC-MAT-17.1e  48-16 C1 - 2016-02-01 16.27.05'] = 5
histo_files_list['KLF14-B6NTAC-MAT-17.1f  49-16 C1 - 2016-02-01 17.51.46'] = 6
histo_files_list['KLF14-B6NTAC-MAT-17.2g  69-16 C1 - 2016-02-04 16.15.05'] = 6
histo_files_list['KLF14-B6NTAC-MAT-18.1d  53-16 C1 - 2016-02-02 14.32.03'] = 7
histo_files_list['KLF14-B6NTAC-MAT-18.1f  55-16 C1 - 2016-02-02 16.14.30'] = 4
histo_files_list['KLF14-B6NTAC-MAT-18.2f  62-16 C1 - 2016-02-03 15.46.15'] = 9
histo_files_list['KLF14-B6NTAC-MAT-18.3c  218-16 C1 - 2016-02-18 13.12.09'] = 9
histo_files_list['KLF14-B6NTAC-MAT-19.1a  56-16 C1 - 2016-02-02 17.23.31'] = 8
histo_files_list['KLF14-B6NTAC-MAT-19.2f  217-16 C1 - 2016-02-18 11.48.16'] = 8
histo_files_list['KLF14-B6NTAC-MAT-19.2g  222-16 C1 - 2016-02-25 15.13.00'] = 8

# females
histo_files_list['KLF14-B6NTAC 37.1a PAT 106-16 C1 - 2016-02-12 16.21.00'] = 6
histo_files_list['KLF14-B6NTAC-37.1b PAT 107-16 C1 - 2016-02-15 11.43.31'] = 4
histo_files_list['KLF14-B6NTAC-37.1e PAT 110-16 C1 - 2016-02-15 17.33.11'] = 3
histo_files_list['KLF14-B6NTAC-37.1g PAT 112-16 C1 - 2016-02-16 13.33.09'] = 9
histo_files_list['KLF14-B6NTAC-PAT-36.3a  409-16 C1 - 2016-03-15 10.18.46'] = 7
histo_files_list['KLF14-B6NTAC-PAT-36.3b  412-16 C1 - 2016-03-15 14.37.55'] = 7
histo_files_list['KLF14-B6NTAC-PAT-37.2a  406-16 C1 - 2016-03-14 12.01.56'] = 3
histo_files_list['KLF14-B6NTAC-PAT-37.2b  410-16 C1 - 2016-03-15 11.24.20'] = 8
histo_files_list['KLF14-B6NTAC-PAT-37.2c  407-16 C1 - 2016-03-14 14.13.54'] = 0
histo_files_list['KLF14-B6NTAC-PAT-37.2d  411-16 C1 - 2016-03-15 12.42.26'] = 9
histo_files_list['KLF14-B6NTAC-PAT-37.2e  408-16 C1 - 2016-03-14 16.23.30'] = 7
histo_files_list['KLF14-B6NTAC-PAT-37.2f  405-16 C1 - 2016-03-14 10.58.34'] = 1
histo_files_list['KLF14-B6NTAC-PAT-37.3a  413-16 C1 - 2016-03-15 15.54.12'] = 6
histo_files_list['KLF14-B6NTAC-PAT-37.3c  414-16 C1 - 2016-03-15 17.15.41'] = 7
histo_files_list['KLF14-B6NTAC-PAT-39.1h  453-16 C1 - 2016-03-17 11.38.04'] = 6
histo_files_list['KLF14-B6NTAC-MAT-16.2a  211-16 C1 - 2016-02-17 11.46.42'] = 8
histo_files_list['KLF14-B6NTAC-MAT-17.1a  44-16 C1 - 2016-02-01 11.14.17'] = 4
histo_files_list['KLF14-B6NTAC-MAT-17.1b  45-16 C1 - 2016-02-01 12.23.50'] = 3
histo_files_list['KLF14-B6NTAC-MAT-17.1d  47-16 C1 - 2016-02-01 15.25.53'] = 6
histo_files_list['KLF14-B6NTAC-MAT-17.2a  64-16 C1 - 2016-02-04 09.17.52'] = 9
histo_files_list['KLF14-B6NTAC-MAT-17.2b  65-16 C1 - 2016-02-04 10.24.22'] = 1
histo_files_list['KLF14-B6NTAC-MAT-17.2d  67-16 C1 - 2016-02-04 12.34.32'] = 9
histo_files_list['KLF14-B6NTAC-MAT-18.1b  51-16 C1 - 2016-02-02 09.59.16'] = 9
histo_files_list['KLF14-B6NTAC-MAT-18.1c  52-16 C1 - 2016-02-02 12.26.58'] = 1
histo_files_list['KLF14-B6NTAC-MAT-18.2a  57-16 C1 - 2016-02-03 09.10.17'] = 6
histo_files_list['KLF14-B6NTAC-MAT-18.2c  59-16 C1 - 2016-02-03 11.56.52'] = 2
histo_files_list['KLF14-B6NTAC-MAT-18.2e  61-16 C1 - 2016-02-03 14.19.35'] = 0
histo_files_list['KLF14-B6NTAC-MAT-19.2b  219-16 C1 - 2016-02-18 15.41.38'] = 5
histo_files_list['KLF14-B6NTAC-MAT-19.2c  220-16 C1 - 2016-02-18 17.03.38'] = 0
histo_files_list['KLF14-B6NTAC-MAT-19.2e  221-16 C1 - 2016-02-25 14.00.14'] = 3

# add GWAT slices not seen at training, from animals not seen at training. We randomly assign images to a fold
histo_files_list['KLF14-B6NTAC 36.1d PAT 99-16 B1 - 2016-02-11 11.29.55'] = 9
histo_files_list['KLF14-B6NTAC 36.1e PAT 100-16 B1 - 2016-02-11 12.51.11'] = 4
histo_files_list['KLF14-B6NTAC 36.1f PAT 101-16 B1 - 2016-02-11 14.57.03'] = 9
histo_files_list['KLF14-B6NTAC 36.1g PAT 102-16 B1 - 2016-02-11 16.12.01'] = 4
histo_files_list['KLF14-B6NTAC 36.1h PAT 103-16 B1 - 2016-02-12 09.51.08'] = 0
histo_files_list['KLF14-B6NTAC 36.1j PAT 105-16 B1 - 2016-02-12 14.08.19'] = 4
histo_files_list['KLF14-B6NTAC 37.1a PAT 106-16 B1 - 2016-02-12 15.33.02'] = 2
histo_files_list['KLF14-B6NTAC-37.1b PAT 107-16 B1 - 2016-02-15 11.25.20'] = 0
histo_files_list['KLF14-B6NTAC-37.1e PAT 110-16 B1 - 2016-02-15 16.16.06'] = 1
histo_files_list['KLF14-B6NTAC-37.1g PAT 112-16 B1 - 2016-02-16 12.02.07'] = 4
histo_files_list['KLF14-B6NTAC-37.1h PAT 113-16 B1 - 2016-02-16 14.53.02'] = 3
histo_files_list['KLF14-B6NTAC-38.1e PAT 94-16 B1 - 2016-02-10 11.35.53'] = 4
histo_files_list['KLF14-B6NTAC-38.1f PAT 95-16 B1 - 2016-02-10 14.16.55'] = 1
histo_files_list['KLF14-B6NTAC-MAT-16.2a  211-16 B1 - 2016-02-17 11.21.54'] = 1
histo_files_list['KLF14-B6NTAC-MAT-16.2b  212-16 B1 - 2016-02-17 12.33.18'] = 0
histo_files_list['KLF14-B6NTAC-MAT-16.2c  213-16 B1 - 2016-02-17 14.01.06'] = 9
histo_files_list['KLF14-B6NTAC-MAT-16.2e  215-16 B1 - 2016-02-17 17.14.16'] = 4
histo_files_list['KLF14-B6NTAC-MAT-16.2f  216-16 B1 - 2016-02-18 10.05.52'] = 4
histo_files_list['KLF14-B6NTAC-MAT-17.1a  44-16 B1 - 2016-02-01 09.19.20'] = 1
histo_files_list['KLF14-B6NTAC-MAT-17.1b  45-16 B1 - 2016-02-01 12.05.15'] = 5
histo_files_list['KLF14-B6NTAC-MAT-17.1d  47-16 B1 - 2016-02-01 15.11.42'] = 8
histo_files_list['KLF14-B6NTAC-MAT-17.1e  48-16 B1 - 2016-02-01 16.01.09'] = 8
histo_files_list['KLF14-B6NTAC-MAT-17.1f  49-16 B1 - 2016-02-01 17.12.31'] = 0
histo_files_list['KLF14-B6NTAC-MAT-17.2a  64-16 B1 - 2016-02-04 08.57.34'] = 4
histo_files_list['KLF14-B6NTAC-MAT-17.2b  65-16 B1 - 2016-02-04 10.06.00'] = 5
histo_files_list['KLF14-B6NTAC-MAT-17.2d  67-16 B1 - 2016-02-04 12.20.20'] = 4
histo_files_list['KLF14-B6NTAC-MAT-17.2g  69-16 B1 - 2016-02-04 15.52.52'] = 0
histo_files_list['KLF14-B6NTAC-MAT-18.1b  51-16 B1 - 2016-02-02 09.46.31'] = 5
histo_files_list['KLF14-B6NTAC-MAT-18.1c  52-16 B1 - 2016-02-02 11.24.31'] = 7
histo_files_list['KLF14-B6NTAC-MAT-18.1d  53-16 B1 - 2016-02-02 14.11.37'] = 9
histo_files_list['KLF14-B6NTAC-MAT-18.2a  57-16 B1 - 2016-02-03 08.54.27'] = 6
histo_files_list['KLF14-B6NTAC-MAT-18.2c  59-16 B1 - 2016-02-03 11.41.32'] = 3
histo_files_list['KLF14-B6NTAC-MAT-18.2e  61-16 B1 - 2016-02-03 14.02.25'] = 8
histo_files_list['KLF14-B6NTAC-MAT-18.2f  62-16 B1 - 2016-02-03 15.00.17'] = 5
histo_files_list['KLF14-B6NTAC-MAT-18.3c  218-16 B1 - 2016-02-18 12.51.46'] = 8
histo_files_list['KLF14-B6NTAC-MAT-19.1a  56-16 B1 - 2016-02-02 16.57.46'] = 1
histo_files_list['KLF14-B6NTAC-MAT-19.2b  219-16 B1 - 2016-02-18 14.21.50'] = 5
histo_files_list['KLF14-B6NTAC-MAT-19.2c  220-16 B1 - 2016-02-18 16.40.48'] = 1
histo_files_list['KLF14-B6NTAC-MAT-19.2e  221-16 B1 - 2016-02-25 13.15.27'] = 2
histo_files_list['KLF14-B6NTAC-MAT-19.2f  217-16 B1 - 2016-02-18 11.23.22'] = 7
histo_files_list['KLF14-B6NTAC-MAT-19.2g  222-16 B1 - 2016-02-25 14.51.57'] = 9
histo_files_list['KLF14-B6NTAC-PAT-36.3a  409-16 B1 - 2016-03-15 09.24.54'] = 8
histo_files_list['KLF14-B6NTAC-PAT-36.3b  412-16 B1 - 2016-03-15 14.11.47'] = 7
histo_files_list['KLF14-B6NTAC-PAT-37.2a  406-16 B1 - 2016-03-14 11.46.47'] = 4
histo_files_list['KLF14-B6NTAC-PAT-37.2b  410-16 B1 - 2016-03-15 11.12.01'] = 9
histo_files_list['KLF14-B6NTAC-PAT-37.2c  407-16 B1 - 2016-03-14 12.54.55'] = 7
histo_files_list['KLF14-B6NTAC-PAT-37.2d  411-16 B1 - 2016-03-15 12.01.13'] = 9
histo_files_list['KLF14-B6NTAC-PAT-37.2e  408-16 B1 - 2016-03-14 16.06.43'] = 1
histo_files_list['KLF14-B6NTAC-PAT-37.2f  405-16 B1 - 2016-03-14 09.49.45'] = 9
histo_files_list['KLF14-B6NTAC-PAT-37.2h  418-16 B1 - 2016-03-16 16.42.16'] = 7
histo_files_list['KLF14-B6NTAC-PAT-37.3a  413-16 B1 - 2016-03-15 15.31.26'] = 3
histo_files_list['KLF14-B6NTAC-PAT-37.3c  414-16 B1 - 2016-03-15 16.49.22'] = 3
histo_files_list['KLF14-B6NTAC-PAT-37.4b  419-16 B1 - 2016-03-17 09.10.42'] = 7
histo_files_list['KLF14-B6NTAC-PAT-38.1a  90-16 B1 - 2016-02-04 17.27.42'] = 6
histo_files_list['KLF14-B6NTAC-PAT-39.1h  453-16 B1 - 2016-03-17 11.15.50'] = 0
histo_files_list['KLF14-B6NTAC-PAT-39.2d  454-16 B1 - 2016-03-17 12.16.06'] = 7

# add missing slice. Random fold
histo_files_list['KLF14-B6NTAC-37.1f PAT 111-16 C2 - 2016-02-16 11.26 (1)'] = 5

# add recut slices. Random fold
histo_files_list['KLF14-B6NTAC-PAT 37.2b 410-16 C4 - 2020-02-14 10.27.23'] = 8
histo_files_list['KLF14-B6NTAC-PAT 37.2c 407-16 C4 - 2020-02-14 10.15.57'] = 0
histo_files_list['KLF14-B6NTAC-PAT 37.2d 411-16 C4 - 2020-02-14 10.34.10'] = 9


########################################################################################################################
## Segmentation loop
########################################################################################################################

file_with_problems_count = 0
# DEBUG: i_file = 9; histo_file = list(histo_files_list.keys())[i_file]
for i_file, histo_file in enumerate(histo_files_list.keys()):

    # get fold for this image
    i_fold = histo_files_list[histo_file]

    # add extension to file
    histo_file += histology_ext

    print('File ' + str(i_file) + '/' + str(len(histo_files_list) - 1) + ' (fold ' + str(i_fold) + ')' + ': ' + histo_file)

    # make full path to histo file
    histo_file = os.path.join(histology_dir, histo_file)

    # check whether there's a lock on this file
    lock_file = os.path.basename(histo_file).replace(histology_ext, '.lock')
    lock_file = os.path.join(annotations_dir, lock_file)
    if not os.path.isfile(lock_file):
        print('\tMissing lock file')

    # name of file to save annotations to
    annotations_auto_file = os.path.basename(histo_file)
    annotations_auto_file = os.path.splitext(annotations_auto_file)[0]
    annotations_auto_file = os.path.join(annotations_dir, annotations_auto_file + '_exp_0106_auto.json')

    annotations_corrected_file = os.path.basename(histo_file)
    annotations_corrected_file = os.path.splitext(annotations_corrected_file)[0]
    annotations_corrected_file = os.path.join(annotations_dir, annotations_corrected_file + '_exp_0106_corrected.json')

    # name of soft link to display annotations on AIDA
    symlink_name = os.path.basename(histo_file).replace('.ndpi', '.json')
    symlink_name = os.path.join(annotations_dir, symlink_name)

    # # create soft link to annotations file, for easy visual evaluation
    # if os.path.islink(symlink_name):
    #     os.remove(symlink_name)
    # os.symlink(os.path.basename(annotations_auto_file), symlink_name)
    # # os.symlink(os.path.basename(annotations_corrected_file), symlink_name)

    # name of file to save rough mask, current mask, and time steps
    coarse_mask_file = os.path.basename(histo_file)
    coarse_mask_file = coarse_mask_file.replace(histology_ext, '_coarse_mask.npz')
    coarse_mask_file = os.path.join(annotations_dir, coarse_mask_file)

    # if the rough mask has been pre-computed, just load it
    if os.path.isfile(coarse_mask_file):

        with np.load(coarse_mask_file) as aux:
            lores_istissue = aux['lores_istissue']
            lores_istissue0 = aux['lores_istissue0']
            im_downsampled = aux['im_downsampled']
            step = aux['step'].item()
            perc_completed_all = list(aux['perc_completed_all'])
            time_step_all = list(aux['time_step_all'])
            prev_first_row = aux['prev_first_row'].item()
            prev_last_row = aux['prev_last_row'].item()
            prev_first_col = aux['prev_first_col'].item()
            prev_last_col = aux['prev_last_col'].item()

        # if perc_completed_all[-1] < 100 and os.path.isfile(lock_file):
        #     print('DELETE LOCK')
        #     os.remove(lock_file)
        #     count += 1

        if perc_completed_all[-1] < 100:
            print('\tUnfinished segmentation')
            file_with_problems_count += 1

    else:
        print('\tMissing mask')
        file_with_problems_count += 1

print('Total files with problems: ' + str(file_with_problems_count))
