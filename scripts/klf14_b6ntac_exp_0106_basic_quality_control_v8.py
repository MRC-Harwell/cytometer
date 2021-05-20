"""
Check that all histology files have been segmented by the pipeline
"""

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
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

import numpy as np
import matplotlib.pyplot as plt
import cytometer.data

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

    # load contours and their confidence measure from annotation file
    cells, props = cytometer.data.aida_get_contours(annotations_auto_file, layer_name='White adipocyte.*', return_props=True)
    if len(cells) < 1000:
        print('-------> Cells Auto: ' + str(len(cells)))
    else:
        print('Cells Auto: ' + str(len(cells)))
    cells, props = cytometer.data.aida_get_contours(annotations_corrected_file, layer_name='White adipocyte.*', return_props=True)
    if len(cells) < 1000:
        print('-------> Cells Corrected: ' + str(len(cells)))
    else:
        print('Cells Corrected: ' + str(len(cells)))

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

# File 0/155 (fold 0): KLF14-B6NTAC-MAT-18.2b  58-16 C1 - 2016-02-03 11.10.52.ndpi
# Cells Auto: 34727
# Cells Corrected: 34727
# File 1/155 (fold 0): KLF14-B6NTAC-MAT-18.2d  60-16 C1 - 2016-02-03 13.13.57.ndpi
# Cells Auto: 51313
# Cells Corrected: 51313
# File 2/155 (fold 1): KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38.ndpi
# Cells Auto: 57400
# Cells Corrected: 57400
# File 3/155 (fold 1): KLF14-B6NTAC-MAT-17.2c  66-16 C1 - 2016-02-04 11.46.39.ndpi
# Cells Auto: 41228
# Cells Corrected: 41228
# File 4/155 (fold 2): KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04.ndpi
# Cells Auto: 27250
# Cells Corrected: 27250
# File 5/155 (fold 2): KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53.ndpi
# Cells Auto: 43615
# Cells Corrected: 43615
# File 6/155 (fold 3): KLF14-B6NTAC-37.1c PAT 108-16 C1 - 2016-02-15 14.49.45.ndpi
# Cells Auto: 46159
# Cells Corrected: 46159
# File 7/155 (fold 3): KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46.ndpi
# Cells Auto: 19022
# Cells Corrected: 19022
# File 8/155 (fold 4): KLF14-B6NTAC-37.1d PAT 109-16 C1 - 2016-02-15 15.19.08.ndpi
# Cells Auto: 41242
# Cells Corrected: 41242
# File 9/155 (fold 4): KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52.ndpi
# Cells Auto: 75279
# Cells Corrected: 75279
# File 10/155 (fold 5): KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38.ndpi
# Cells Auto: 75077
# Cells Corrected: 75077
# File 11/155 (fold 5): KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06.ndpi
# Cells Auto: 40888
# Cells Corrected: 40888
# File 12/155 (fold 6): KLF14-B6NTAC-MAT-18.1a  50-16 C1 - 2016-02-02 09.12.41.ndpi
# Cells Auto: 32633
# Cells Corrected: 32633
# File 13/155 (fold 6): KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11.ndpi
# Cells Auto: 62949
# Cells Corrected: 62949
# File 14/155 (fold 7): KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00.ndpi
# Cells Auto: 73643
# Cells Corrected: 73643
# File 15/155 (fold 7): KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32.ndpi
# Cells Auto: 43404
# Cells Corrected: 43404
# File 16/155 (fold 8): KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33.ndpi
# Cells Auto: 45463
# Cells Corrected: 45463
# File 17/155 (fold 8): KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52.ndpi
# Cells Auto: 37408
# Cells Corrected: 37408
# File 18/155 (fold 9): KLF14-B6NTAC-MAT-17.2f  68-16 C1 - 2016-02-04 15.05.54.ndpi
# Cells Auto: 58947
# Cells Corrected: 58947
# File 19/155 (fold 9): KLF14-B6NTAC-MAT-18.2g  63-16 C1 - 2016-02-03 16.58.52.ndpi
# Cells Auto: 81583
# Cells Corrected: 81583
# File 20/155 (fold 0): KLF14-B6NTAC-MAT-18.2b  58-16 B1 - 2016-02-03 09.58.06.ndpi
# Cells Auto: 45255
# Cells Corrected: 45255
# File 21/155 (fold 0): KLF14-B6NTAC-MAT-18.2d  60-16 B1 - 2016-02-03 12.56.49.ndpi
# Cells Auto: 42968
# Cells Corrected: 42968
# File 22/155 (fold 1): KLF14-B6NTAC 36.1i PAT 104-16 B1 - 2016-02-12 11.37.56.ndpi
# Cells Auto: 57830
# Cells Corrected: 57830
# File 23/155 (fold 1): KLF14-B6NTAC-MAT-17.2c  66-16 B1 - 2016-02-04 11.14.28.ndpi
# Cells Auto: 20980
# Cells Corrected: 20980
# File 24/155 (fold 2): KLF14-B6NTAC-MAT-17.1c  46-16 B1 - 2016-02-01 13.01.30.ndpi
# Cells Auto: 37461
# Cells Corrected: 37461
# File 25/155 (fold 2): KLF14-B6NTAC-MAT-18.3d  224-16 B1 - 2016-02-26 10.48.56.ndpi
# Cells Auto: 72311
# Cells Corrected: 72311
# File 26/155 (fold 3): KLF14-B6NTAC-37.1c PAT 108-16 B1 - 2016-02-15 12.33.10.ndpi
# Cells Auto: 44401
# Cells Corrected: 44401
# File 27/155 (fold 3): KLF14-B6NTAC-MAT-16.2d  214-16 B1 - 2016-02-17 15.43.57.ndpi
# Cells Auto: 34952
# Cells Corrected: 34952
# File 28/155 (fold 4): KLF14-B6NTAC-37.1d PAT 109-16 B1 - 2016-02-15 15.03.44.ndpi
# Cells Auto: 47615
# Cells Corrected: 47615
# File 29/155 (fold 4): KLF14-B6NTAC-PAT-37.2g  415-16 B1 - 2016-03-16 11.04.45.ndpi
# Cells Auto: 36093
# Cells Corrected: 36093
# File 30/155 (fold 5): KLF14-B6NTAC-36.1a PAT 96-16 B1 - 2016-02-10 15.32.31.ndpi
# Cells Auto: 53453
# Cells Corrected: 53453
# File 31/155 (fold 5): KLF14-B6NTAC-36.1b PAT 97-16 B1 - 2016-02-10 17.15.16.ndpi
# Cells Auto: 69048
# Cells Corrected: 69048
# File 32/155 (fold 6): KLF14-B6NTAC-MAT-18.1a  50-16 B1 - 2016-02-02 08.49.06.ndpi
# Cells Auto: 55566
# Cells Corrected: 55566
# File 33/155 (fold 6): KLF14-B6NTAC-PAT-36.3d  416-16 B1 - 2016-03-16 14.22.04.ndpi
# Cells Auto: 46578
# Cells Corrected: 46578
# File 34/155 (fold 7): KLF14-B6NTAC-36.1c PAT 98-16 B1 - 2016-02-10 18.32.40.ndpi
# Cells Auto: 68349
# Cells Corrected: 68317
# File 35/155 (fold 7): KLF14-B6NTAC-PAT-37.4a  417-16 B1 - 2016-03-16 15.25.38.ndpi
# Cells Auto: 40531
# Cells Corrected: 40531
# File 36/155 (fold 8): KLF14-B6NTAC-MAT-18.1e  54-16 B1 - 2016-02-02 15.06.05.ndpi
# Cells Auto: 40512
# Cells Corrected: 40512
# File 37/155 (fold 8): KLF14-B6NTAC-MAT-18.3b  223-16 B1 - 2016-02-25 16.53.42.ndpi
# Cells Auto: 23310
# Cells Corrected: 23310
# File 38/155 (fold 9): KLF14-B6NTAC-MAT-17.2f  68-16 B1 - 2016-02-04 14.01.40.ndpi
# Cells Auto: 5118
# Cells Corrected: 5118
# File 39/155 (fold 9): KLF14-B6NTAC-MAT-18.2g  63-16 B1 - 2016-02-03 16.40.37.ndpi
# Cells Auto: 44076
# Cells Corrected: 44076
# File 40/155 (fold 3): KLF14-B6NTAC 36.1d PAT 99-16 C1 - 2016-02-11 11.48.31.ndpi
# -------> Cells Auto: 44
# -------> Cells Corrected: 44
# File 41/155 (fold 7): KLF14-B6NTAC 36.1e PAT 100-16 C1 - 2016-02-11 14.06.56.ndpi
# Cells Auto: 41996
# Cells Corrected: 41996
# File 42/155 (fold 5): KLF14-B6NTAC 36.1f PAT 101-16 C1 - 2016-02-11 15.23.06.ndpi
# Cells Auto: 43968
# Cells Corrected: 43968
# File 43/155 (fold 1): KLF14-B6NTAC 36.1g PAT 102-16 C1 - 2016-02-11 17.20.14.ndpi
# Cells Auto: 64754
# Cells Corrected: 64754
# File 44/155 (fold 4): KLF14-B6NTAC 36.1h PAT 103-16 C1 - 2016-02-12 10.15.22.ndpi
# Cells Auto: 48658
# Cells Corrected: 48658
# File 45/155 (fold 1): KLF14-B6NTAC 36.1j PAT 105-16 C1 - 2016-02-12 14.33.33.ndpi
# Cells Auto: 71598
# Cells Corrected: 71598
# File 46/155 (fold 8): KLF14-B6NTAC-PAT-37.2h  418-16 C1 - 2016-03-16 17.01.17.ndpi
# Cells Auto: 17909
# Cells Corrected: 17909
# File 47/155 (fold 1): KLF14-B6NTAC-PAT-37.4b  419-16 C1 - 2016-03-17 10.22.54.ndpi
# Cells Auto: 51717
# Cells Corrected: 51717
# File 48/155 (fold 2): KLF14-B6NTAC-PAT-39.2d  454-16 C1 - 2016-03-17 14.33.38.ndpi
# Cells Auto: 41140
# Cells Corrected: 41140
# File 49/155 (fold 5): KLF14-B6NTAC-37.1h PAT 113-16 C1 - 2016-02-16 15.14.09.ndpi
# Cells Auto: 61631
# Cells Corrected: 61631
# File 50/155 (fold 1): KLF14-B6NTAC-38.1e PAT 94-16 C1 - 2016-02-10 12.13.10.ndpi
# Cells Auto: 70998
# Cells Corrected: 70998
# File 51/155 (fold 0): KLF14-B6NTAC-38.1f PAT 95-16 C1 - 2016-02-10 14.41.44.ndpi
# Cells Auto: 66854
# Cells Corrected: 66854
# File 52/155 (fold 3): KLF14-B6NTAC-MAT-16.2b  212-16 C1 - 2016-02-17 12.49.00.ndpi
# Cells Auto: 45619
# Cells Corrected: 45619
# File 53/155 (fold 8): KLF14-B6NTAC-MAT-16.2c  213-16 C1 - 2016-02-17 14.51.18.ndpi
# Cells Auto: 40064
# Cells Corrected: 40064
# File 54/155 (fold 4): KLF14-B6NTAC-MAT-16.2e  215-16 C1 - 2016-02-18 09.19.26.ndpi
# Cells Auto: 40599
# Cells Corrected: 40599
# File 55/155 (fold 5): KLF14-B6NTAC-MAT-16.2f  216-16 C1 - 2016-02-18 10.28.27.ndpi
# Cells Auto: 45491
# Cells Corrected: 45491
# File 56/155 (fold 5): KLF14-B6NTAC-MAT-17.1e  48-16 C1 - 2016-02-01 16.27.05.ndpi
# Cells Auto: 30463
# Cells Corrected: 30463
# File 57/155 (fold 6): KLF14-B6NTAC-MAT-17.1f  49-16 C1 - 2016-02-01 17.51.46.ndpi
# Cells Auto: 22115
# Cells Corrected: 22115
# File 58/155 (fold 6): KLF14-B6NTAC-MAT-17.2g  69-16 C1 - 2016-02-04 16.15.05.ndpi
# Cells Auto: 42324
# Cells Corrected: 42324
# File 59/155 (fold 7): KLF14-B6NTAC-MAT-18.1d  53-16 C1 - 2016-02-02 14.32.03.ndpi
# Cells Auto: 42692
# Cells Corrected: 42692
# File 60/155 (fold 4): KLF14-B6NTAC-MAT-18.1f  55-16 C1 - 2016-02-02 16.14.30.ndpi
# Cells Auto: 38481
# Cells Corrected: 38481
# File 61/155 (fold 9): KLF14-B6NTAC-MAT-18.2f  62-16 C1 - 2016-02-03 15.46.15.ndpi
# Cells Auto: 40778
# Cells Corrected: 40778
# File 62/155 (fold 9): KLF14-B6NTAC-MAT-18.3c  218-16 C1 - 2016-02-18 13.12.09.ndpi
# Cells Auto: 21875
# Cells Corrected: 21875
# File 63/155 (fold 8): KLF14-B6NTAC-MAT-19.1a  56-16 C1 - 2016-02-02 17.23.31.ndpi
# Cells Auto: 31769
# Cells Corrected: 31769
# File 64/155 (fold 8): KLF14-B6NTAC-MAT-19.2f  217-16 C1 - 2016-02-18 11.48.16.ndpi
# Cells Auto: 55272
# Cells Corrected: 55272
# File 65/155 (fold 8): KLF14-B6NTAC-MAT-19.2g  222-16 C1 - 2016-02-25 15.13.00.ndpi
# Cells Auto: 79967
# Cells Corrected: 79967
# File 66/155 (fold 6): KLF14-B6NTAC 37.1a PAT 106-16 C1 - 2016-02-12 16.21.00.ndpi
# Cells Auto: 30570
# Cells Corrected: 30570
# File 67/155 (fold 4): KLF14-B6NTAC-37.1b PAT 107-16 C1 - 2016-02-15 11.43.31.ndpi
# Cells Auto: 57791
# Cells Corrected: 57791
# File 68/155 (fold 3): KLF14-B6NTAC-37.1e PAT 110-16 C1 - 2016-02-15 17.33.11.ndpi
# Cells Auto: 51477
# Cells Corrected: 51477
# File 69/155 (fold 9): KLF14-B6NTAC-37.1g PAT 112-16 C1 - 2016-02-16 13.33.09.ndpi
# Cells Auto: 58015
# Cells Corrected: 58015
# File 70/155 (fold 7): KLF14-B6NTAC-PAT-36.3a  409-16 C1 - 2016-03-15 10.18.46.ndpi
# Cells Auto: 79184
# Cells Corrected: 79184
# File 71/155 (fold 7): KLF14-B6NTAC-PAT-36.3b  412-16 C1 - 2016-03-15 14.37.55.ndpi
# Cells Auto: 129461
# Cells Corrected: 129461
# File 72/155 (fold 3): KLF14-B6NTAC-PAT-37.2a  406-16 C1 - 2016-03-14 12.01.56.ndpi
# Cells Auto: 59170
# Cells Corrected: 59170
# File 73/155 (fold 8): KLF14-B6NTAC-PAT-37.2b  410-16 C1 - 2016-03-15 11.24.20.ndpi
# Cells Auto: 49110
# Cells Corrected: 49110
# File 74/155 (fold 0): KLF14-B6NTAC-PAT-37.2c  407-16 C1 - 2016-03-14 14.13.54.ndpi
# Cells Auto: 24729
# Cells Corrected: 24729
# File 75/155 (fold 9): KLF14-B6NTAC-PAT-37.2d  411-16 C1 - 2016-03-15 12.42.26.ndpi
# Cells Auto: 32277
# Cells Corrected: 32277
# File 76/155 (fold 7): KLF14-B6NTAC-PAT-37.2e  408-16 C1 - 2016-03-14 16.23.30.ndpi
# Cells Auto: 70360
# Cells Corrected: 70360
# File 77/155 (fold 1): KLF14-B6NTAC-PAT-37.2f  405-16 C1 - 2016-03-14 10.58.34.ndpi
# Cells Auto: 61841
# Cells Corrected: 61841
# File 78/155 (fold 6): KLF14-B6NTAC-PAT-37.3a  413-16 C1 - 2016-03-15 15.54.12.ndpi
# Cells Auto: 49749
# Cells Corrected: 49749
# File 79/155 (fold 7): KLF14-B6NTAC-PAT-37.3c  414-16 C1 - 2016-03-15 17.15.41.ndpi
# Cells Auto: 77309
# Cells Corrected: 77309
# File 80/155 (fold 6): KLF14-B6NTAC-PAT-39.1h  453-16 C1 - 2016-03-17 11.38.04.ndpi
# Cells Auto: 54170
# Cells Corrected: 54170
# File 81/155 (fold 8): KLF14-B6NTAC-MAT-16.2a  211-16 C1 - 2016-02-17 11.46.42.ndpi
# Cells Auto: 26253
# Cells Corrected: 26253
# File 82/155 (fold 4): KLF14-B6NTAC-MAT-17.1a  44-16 C1 - 2016-02-01 11.14.17.ndpi
# Cells Auto: 23963
# Cells Corrected: 23963
# File 83/155 (fold 3): KLF14-B6NTAC-MAT-17.1b  45-16 C1 - 2016-02-01 12.23.50.ndpi
# Cells Auto: 29678
# Cells Corrected: 29678
# File 84/155 (fold 6): KLF14-B6NTAC-MAT-17.1d  47-16 C1 - 2016-02-01 15.25.53.ndpi
# Cells Auto: 42811
# Cells Corrected: 42811
# File 85/155 (fold 9): KLF14-B6NTAC-MAT-17.2a  64-16 C1 - 2016-02-04 09.17.52.ndpi
# Cells Auto: 45858
# Cells Corrected: 45858
# File 86/155 (fold 1): KLF14-B6NTAC-MAT-17.2b  65-16 C1 - 2016-02-04 10.24.22.ndpi
# Cells Auto: 57611
# Cells Corrected: 57611
# File 87/155 (fold 9): KLF14-B6NTAC-MAT-17.2d  67-16 C1 - 2016-02-04 12.34.32.ndpi
# Cells Auto: 36618
# Cells Corrected: 36618
# File 88/155 (fold 9): KLF14-B6NTAC-MAT-18.1b  51-16 C1 - 2016-02-02 09.59.16.ndpi
# Cells Auto: 40963
# Cells Corrected: 40963
# File 89/155 (fold 1): KLF14-B6NTAC-MAT-18.1c  52-16 C1 - 2016-02-02 12.26.58.ndpi
# Cells Auto: 38994
# Cells Corrected: 38994
# File 90/155 (fold 6): KLF14-B6NTAC-MAT-18.2a  57-16 C1 - 2016-02-03 09.10.17.ndpi
# Cells Auto: 68169
# Cells Corrected: 68169
# File 91/155 (fold 2): KLF14-B6NTAC-MAT-18.2c  59-16 C1 - 2016-02-03 11.56.52.ndpi
# Cells Auto: 29556
# Cells Corrected: 29556
# File 92/155 (fold 0): KLF14-B6NTAC-MAT-18.2e  61-16 C1 - 2016-02-03 14.19.35.ndpi
# Cells Auto: 52549
# Cells Corrected: 52549
# File 93/155 (fold 5): KLF14-B6NTAC-MAT-19.2b  219-16 C1 - 2016-02-18 15.41.38.ndpi
# Cells Auto: 25357
# Cells Corrected: 25357
# File 94/155 (fold 0): KLF14-B6NTAC-MAT-19.2c  220-16 C1 - 2016-02-18 17.03.38.ndpi
# Cells Auto: 20208
# Cells Corrected: 20208
# File 95/155 (fold 3): KLF14-B6NTAC-MAT-19.2e  221-16 C1 - 2016-02-25 14.00.14.ndpi
# Cells Auto: 47560
# Cells Corrected: 47560
# File 96/155 (fold 9): KLF14-B6NTAC 36.1d PAT 99-16 B1 - 2016-02-11 11.29.55.ndpi
# Cells Auto: 34616
# Cells Corrected: 34616
# File 97/155 (fold 4): KLF14-B6NTAC 36.1e PAT 100-16 B1 - 2016-02-11 12.51.11.ndpi
# Cells Auto: 62641
# Cells Corrected: 62641
# File 98/155 (fold 9): KLF14-B6NTAC 36.1f PAT 101-16 B1 - 2016-02-11 14.57.03.ndpi
# Cells Auto: 88384
# Cells Corrected: 88384
# File 99/155 (fold 4): KLF14-B6NTAC 36.1g PAT 102-16 B1 - 2016-02-11 16.12.01.ndpi
# Cells Auto: 43754
# Cells Corrected: 43754
# File 100/155 (fold 0): KLF14-B6NTAC 36.1h PAT 103-16 B1 - 2016-02-12 09.51.08.ndpi
# Cells Auto: 57363
# Cells Corrected: 57363
# File 101/155 (fold 4): KLF14-B6NTAC 36.1j PAT 105-16 B1 - 2016-02-12 14.08.19.ndpi
# Cells Auto: 61945
# Cells Corrected: 61945
# File 102/155 (fold 2): KLF14-B6NTAC 37.1a PAT 106-16 B1 - 2016-02-12 15.33.02.ndpi
# Cells Auto: 46644
# Cells Corrected: 46644
# File 103/155 (fold 0): KLF14-B6NTAC-37.1b PAT 107-16 B1 - 2016-02-15 11.25.20.ndpi
# Cells Auto: 38382
# Cells Corrected: 38382
# File 104/155 (fold 1): KLF14-B6NTAC-37.1e PAT 110-16 B1 - 2016-02-15 16.16.06.ndpi
# Cells Auto: 24854
# Cells Corrected: 24854
# File 105/155 (fold 4): KLF14-B6NTAC-37.1g PAT 112-16 B1 - 2016-02-16 12.02.07.ndpi
# Cells Auto: 60167
# Cells Corrected: 60167
# File 106/155 (fold 3): KLF14-B6NTAC-37.1h PAT 113-16 B1 - 2016-02-16 14.53.02.ndpi
# Cells Auto: 57526
# Cells Corrected: 57311
# File 107/155 (fold 4): KLF14-B6NTAC-38.1e PAT 94-16 B1 - 2016-02-10 11.35.53.ndpi
# Cells Auto: 68987
# Cells Corrected: 68987
# File 108/155 (fold 1): KLF14-B6NTAC-38.1f PAT 95-16 B1 - 2016-02-10 14.16.55.ndpi
# Cells Auto: 61609
# Cells Corrected: 61609
# File 109/155 (fold 1): KLF14-B6NTAC-MAT-16.2a  211-16 B1 - 2016-02-17 11.21.54.ndpi
# Cells Auto: 23912
# Cells Corrected: 23912
# File 110/155 (fold 0): KLF14-B6NTAC-MAT-16.2b  212-16 B1 - 2016-02-17 12.33.18.ndpi
# Cells Auto: 28053
# Cells Corrected: 28053
# File 111/155 (fold 9): KLF14-B6NTAC-MAT-16.2c  213-16 B1 - 2016-02-17 14.01.06.ndpi
# Cells Auto: 33477
# Cells Corrected: 33477
# File 112/155 (fold 4): KLF14-B6NTAC-MAT-16.2e  215-16 B1 - 2016-02-17 17.14.16.ndpi
# Cells Auto: 32205
# Cells Corrected: 32205
# File 113/155 (fold 4): KLF14-B6NTAC-MAT-16.2f  216-16 B1 - 2016-02-18 10.05.52.ndpi
# Cells Auto: 36954
# Cells Corrected: 36954
# File 114/155 (fold 1): KLF14-B6NTAC-MAT-17.1a  44-16 B1 - 2016-02-01 09.19.20.ndpi
# Cells Auto: 30744
# Cells Corrected: 30744
# File 115/155 (fold 5): KLF14-B6NTAC-MAT-17.1b  45-16 B1 - 2016-02-01 12.05.15.ndpi
# Cells Auto: 33384
# Cells Corrected: 33384
# File 116/155 (fold 8): KLF14-B6NTAC-MAT-17.1d  47-16 B1 - 2016-02-01 15.11.42.ndpi
# Cells Auto: 30442
# Cells Corrected: 30442
# File 117/155 (fold 8): KLF14-B6NTAC-MAT-17.1e  48-16 B1 - 2016-02-01 16.01.09.ndpi
# Cells Auto: 26149
# Cells Corrected: 26149
# File 118/155 (fold 0): KLF14-B6NTAC-MAT-17.1f  49-16 B1 - 2016-02-01 17.12.31.ndpi
# Cells Auto: 32475
# Cells Corrected: 32475
# File 119/155 (fold 4): KLF14-B6NTAC-MAT-17.2a  64-16 B1 - 2016-02-04 08.57.34.ndpi
# Cells Auto: 32376
# Cells Corrected: 32376
# File 120/155 (fold 5): KLF14-B6NTAC-MAT-17.2b  65-16 B1 - 2016-02-04 10.06.00.ndpi
# Cells Auto: 40152
# Cells Corrected: 40152
# File 121/155 (fold 4): KLF14-B6NTAC-MAT-17.2d  67-16 B1 - 2016-02-04 12.20.20.ndpi
# Cells Auto: 52949
# Cells Corrected: 52949
# File 122/155 (fold 0): KLF14-B6NTAC-MAT-17.2g  69-16 B1 - 2016-02-04 15.52.52.ndpi
# Cells Auto: 38922
# Cells Corrected: 38922
# File 123/155 (fold 5): KLF14-B6NTAC-MAT-18.1b  51-16 B1 - 2016-02-02 09.46.31.ndpi
# Cells Auto: 36102
# Cells Corrected: 36102
# File 124/155 (fold 7): KLF14-B6NTAC-MAT-18.1c  52-16 B1 - 2016-02-02 11.24.31.ndpi
# Cells Auto: 33473
# Cells Corrected: 33473
# File 125/155 (fold 9): KLF14-B6NTAC-MAT-18.1d  53-16 B1 - 2016-02-02 14.11.37.ndpi
# Cells Auto: 40161
# Cells Corrected: 40161
# File 126/155 (fold 6): KLF14-B6NTAC-MAT-18.2a  57-16 B1 - 2016-02-03 08.54.27.ndpi
# Cells Auto: 34576
# Cells Corrected: 34576
# File 127/155 (fold 3): KLF14-B6NTAC-MAT-18.2c  59-16 B1 - 2016-02-03 11.41.32.ndpi
# Cells Auto: 48861
# Cells Corrected: 48861
# File 128/155 (fold 8): KLF14-B6NTAC-MAT-18.2e  61-16 B1 - 2016-02-03 14.02.25.ndpi
# Cells Auto: 51255
# Cells Corrected: 51255
# File 129/155 (fold 5): KLF14-B6NTAC-MAT-18.2f  62-16 B1 - 2016-02-03 15.00.17.ndpi
# Cells Auto: 26691
# Cells Corrected: 26691
# File 130/155 (fold 8): KLF14-B6NTAC-MAT-18.3c  218-16 B1 - 2016-02-18 12.51.46.ndpi
# Cells Auto: 27258
# Cells Corrected: 27258
# File 131/155 (fold 1): KLF14-B6NTAC-MAT-19.1a  56-16 B1 - 2016-02-02 16.57.46.ndpi
# Cells Auto: 33949
# Cells Corrected: 33949
# File 132/155 (fold 5): KLF14-B6NTAC-MAT-19.2b  219-16 B1 - 2016-02-18 14.21.50.ndpi
# Cells Auto: 62371
# Cells Corrected: 62371
# File 133/155 (fold 1): KLF14-B6NTAC-MAT-19.2c  220-16 B1 - 2016-02-18 16.40.48.ndpi
# Cells Auto: 60904
# Cells Corrected: 60904
# File 134/155 (fold 2): KLF14-B6NTAC-MAT-19.2e  221-16 B1 - 2016-02-25 13.15.27.ndpi
# Cells Auto: 14366
# Cells Corrected: 14366
# File 135/155 (fold 7): KLF14-B6NTAC-MAT-19.2f  217-16 B1 - 2016-02-18 11.23.22.ndpi
# Cells Auto: 27475
# Cells Corrected: 27475
# File 136/155 (fold 9): KLF14-B6NTAC-MAT-19.2g  222-16 B1 - 2016-02-25 14.51.57.ndpi
# Cells Auto: 25319
# Cells Corrected: 25319
# File 137/155 (fold 8): KLF14-B6NTAC-PAT-36.3a  409-16 B1 - 2016-03-15 09.24.54.ndpi
# Cells Auto: 91030
# Cells Corrected: 91030
# File 138/155 (fold 7): KLF14-B6NTAC-PAT-36.3b  412-16 B1 - 2016-03-15 14.11.47.ndpi
# Cells Auto: 70324
# Cells Corrected: 70324
# File 139/155 (fold 4): KLF14-B6NTAC-PAT-37.2a  406-16 B1 - 2016-03-14 11.46.47.ndpi
# Cells Auto: 37403
# Cells Corrected: 37403
# File 140/155 (fold 9): KLF14-B6NTAC-PAT-37.2b  410-16 B1 - 2016-03-15 11.12.01.ndpi
# Cells Auto: 68826
# Cells Corrected: 68826
# File 141/155 (fold 7): KLF14-B6NTAC-PAT-37.2c  407-16 B1 - 2016-03-14 12.54.55.ndpi
# Cells Auto: 62110
# Cells Corrected: 62110
# File 142/155 (fold 9): KLF14-B6NTAC-PAT-37.2d  411-16 B1 - 2016-03-15 12.01.13.ndpi
# Cells Auto: 49499
# Cells Corrected: 49499
# File 143/155 (fold 1): KLF14-B6NTAC-PAT-37.2e  408-16 B1 - 2016-03-14 16.06.43.ndpi
# Cells Auto: 31110
# Cells Corrected: 31110
# File 144/155 (fold 9): KLF14-B6NTAC-PAT-37.2f  405-16 B1 - 2016-03-14 09.49.45.ndpi
# Cells Auto: 24243
# Cells Corrected: 24243
# File 145/155 (fold 7): KLF14-B6NTAC-PAT-37.2h  418-16 B1 - 2016-03-16 16.42.16.ndpi
# Cells Auto: 40175
# Cells Corrected: 40175
# File 146/155 (fold 3): KLF14-B6NTAC-PAT-37.3a  413-16 B1 - 2016-03-15 15.31.26.ndpi
# Cells Auto: 48746
# Cells Corrected: 48746
# File 147/155 (fold 3): KLF14-B6NTAC-PAT-37.3c  414-16 B1 - 2016-03-15 16.49.22.ndpi
# Cells Auto: 78602
# Cells Corrected: 78602
# File 148/155 (fold 7): KLF14-B6NTAC-PAT-37.4b  419-16 B1 - 2016-03-17 09.10.42.ndpi
# Cells Auto: 47384
# Cells Corrected: 47384
# File 149/155 (fold 6): KLF14-B6NTAC-PAT-38.1a  90-16 B1 - 2016-02-04 17.27.42.ndpi
# Cells Auto: 30607
# Cells Corrected: 30607
# File 150/155 (fold 0): KLF14-B6NTAC-PAT-39.1h  453-16 B1 - 2016-03-17 11.15.50.ndpi
# Cells Auto: 75834
# Cells Corrected: 75834
# File 151/155 (fold 7): KLF14-B6NTAC-PAT-39.2d  454-16 B1 - 2016-03-17 12.16.06.ndpi
# Cells Auto: 52165
# Cells Corrected: 52165
# File 152/155 (fold 5): KLF14-B6NTAC-37.1f PAT 111-16 C2 - 2016-02-16 11.26 (1).ndpi
# Cells Auto: 58067
# Cells Corrected: 58067
# File 153/155 (fold 8): KLF14-B6NTAC-PAT 37.2b 410-16 C4 - 2020-02-14 10.27.23.ndpi
# Cells Auto: 46827
# Cells Corrected: 46827
# File 154/155 (fold 0): KLF14-B6NTAC-PAT 37.2c 407-16 C4 - 2020-02-14 10.15.57.ndpi
# Cells Auto: 25102
# Cells Corrected: 25102
# File 155/155 (fold 9): KLF14-B6NTAC-PAT 37.2d 411-16 C4 - 2020-02-14 10.34.10.ndpi
# Cells Auto: 37320
# Cells Corrected: 37320
# Total files with problems: 0
