"""
Processing full slides of pipeline v7:

 * data generation
   * training images (*0076*)
   * non-overlap training images (*0077*)
   * augmented training images (*0078*)
   * k-folds + extra "other" for classifier (*0094*)
 * segmentation
   * dmap (*0086*)
   * contour from dmap (*0091*)
 * classifier (*0095*)
 * segmentation correction (*0089*) networks"
 * validation (*0096*)
"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0097_full_slide_pipeline_v7'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
from pathlib import Path
import sys
import pickle
import ujson
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import cytometer.utils
import cytometer.data

# Filter out INFO & WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # limit number of GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ['KERAS_BACKEND'] = 'tensorflow'

import time
import openslide
import numpy as np
import matplotlib.pyplot as plt
import glob
from cytometer.utils import rough_foreground_mask, bspline_resample
import PIL
import tensorflow as tf
import keras
from keras import backend as K
from skimage.measure import regionprops
import shutil
import itertools
from shapely.geometry import Polygon

# # limit GPU memory used
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# set_session(tf.Session(config=config))

DEBUG = False
SAVE_FIGS = False

root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
data_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
training_dir = os.path.join(home, root_data_dir, 'klf14_b6ntac_training')
seg_dir = os.path.join(home, root_data_dir, 'klf14_b6ntac_seg')
figures_dir = os.path.join(root_data_dir, 'figures')
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
fullres_box_size = np.array([2001, 2001])  # rescomp servers have less GPU memory than titan
receptive_field = np.array([131, 131])

# rough_foreground_mask() parameters
downsample_factor = 8.0
dilation_size = 25
component_size_threshold = 50e3
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

# loop the folds to get the ndpi files that correspond to testing of each fold.
# Training SCWAT slices.
# i_file = [0, 19]
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

# add more SCWAT slices. E.g. if 'KLF14-B6NTAC-MAT-18.2b  58-16 C1 - 2016-02-03 11.10.52' is in the list, we want to add
# the C2 and C3 cuts too
# i_file = [20, 59]
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.2b  58-16 C2 - 2016-02-03 11.15.14'] = 0
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.2b  58-16 C3 - 2016-02-03 11.19.28'] = 0
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.2d  60-16 C2 - 2016-02-03 13.19.18'] = 0
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.2d  60-16 C3 - 2016-02-03 13.25.12'] = 0
ndpi_files_test_list['KLF14-B6NTAC 36.1i PAT 104-16 C2 - 2016-02-12 12.22.20'] = 1
ndpi_files_test_list['KLF14-B6NTAC 36.1i PAT 104-16 C3 - 2016-02-12 12.29.22'] = 1
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.2c  66-16 C2 - 2016-02-04 11.51.43'] = 1
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.2c  66-16 C3 - 2016-02-04 11.56.51'] = 1
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.1c  46-16 C2 - 2016-02-01 14.08.04'] = 2
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.1c  46-16 C3 - 2016-02-01 14.14.08'] = 2
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.3d  224-16 C2 - 2016-02-26 11.19.06'] = 2
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.3d  224-16 C3 - 2016-02-26 11.24.28'] = 2
ndpi_files_test_list['KLF14-B6NTAC-37.1c PAT 108-16 C2 - 2016-02-15 13.01.29'] = 3
ndpi_files_test_list['KLF14-B6NTAC-37.1c PAT 108-16 C3 - 2016-02-15 12.57.56'] = 3
ndpi_files_test_list['KLF14-B6NTAC-MAT-16.2d  214-16 C2 - 2016-02-17 16.05.58'] = 3
ndpi_files_test_list['KLF14-B6NTAC-MAT-16.2d  214-16 C3 - 2016-02-17 16.53.30'] = 3
ndpi_files_test_list['KLF14-B6NTAC-37.1d PAT 109-16 C2 - 2016-02-15 15.22.53'] = 4
ndpi_files_test_list['KLF14-B6NTAC-37.1d PAT 109-16 C3 - 2016-02-15 15.26.39'] = 4
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.2g  415-16 C2 - 2016-03-16 11.56.14'] = 4
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.2g  415-16 C3 - 2016-03-16 12.05.08'] = 4
ndpi_files_test_list['KLF14-B6NTAC-36.1a PAT 96-16 C2 - 2016-02-10 16.05.02'] = 5
ndpi_files_test_list['KLF14-B6NTAC-36.1a PAT 96-16 C3 - 2016-02-10 15.58.00'] = 5
ndpi_files_test_list['KLF14-B6NTAC-36.1b PAT 97-16 C2 - 2016-02-10 17.42.35'] = 5
ndpi_files_test_list['KLF14-B6NTAC-36.1b PAT 97-16 C3 - 2016-02-10 17.47.13'] = 5
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.1a  50-16 C2 - 2016-02-02 09.17.36'] = 6
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.1a  50-16 C3 - 2016-02-02 09.22.47'] = 6
ndpi_files_test_list['KLF14-B6NTAC-PAT-36.3d  416-16 C2 - 2016-03-16 14.51.38'] = 6
ndpi_files_test_list['KLF14-B6NTAC-PAT-36.3d  416-16 C3 - 2016-03-16 14.59.33'] = 6
ndpi_files_test_list['KLF14-B6NTAC 36.1c PAT 98-16 C2 - 2016-02-11 10.50.59'] = 7
ndpi_files_test_list['KLF14-B6NTAC 36.1c PAT 98-16 C3 - 2016-02-11 10.57.24'] = 7
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.4a  417-16 C2 - 2016-03-16 16.00.21'] = 7
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.4a  417-16 C3 - 2016-03-16 16.06.30'] = 7
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.1e  54-16 C2 - 2016-02-02 15.32.37'] = 8
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.1e  54-16 C3 - 2016-02-02 15.38.38'] = 8
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.3b  223-16 C1 - 2016-02-26 09.18.44'] = 8
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.3b  223-16 C3 - 2016-02-26 09.29.11'] = 8
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.2f  68-16 C2 - 2016-02-04 15.11.37'] = 9
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.2f  68-16 C3 - 2016-02-04 15.18.41'] = 9
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.2g  63-16 C2 - 2016-02-03 17.05.57'] = 9
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.2g  63-16 C3 - 2016-02-03 17.12.44'] = 9

# add slices from GWAT, but only one slice per animal
# i_file = [60, 79]
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.2b  58-16 B1 - 2016-02-03 09.58.06'] = 0
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.2d  60-16 B1 - 2016-02-03 12.56.49'] = 0
ndpi_files_test_list['KLF14-B6NTAC 36.1i PAT 104-16 B1 - 2016-02-12 11.37.56'] = 1
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.2c  66-16 B1 - 2016-02-04 11.14.28'] = 1
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.1c  46-16 B1 - 2016-02-01 13.01.30'] = 2
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.3d  224-16 B1 - 2016-02-26 10.48.56'] = 2
ndpi_files_test_list['KLF14-B6NTAC-37.1c PAT 108-16 B1 - 2016-02-15 12.33.10'] = 3
ndpi_files_test_list['KLF14-B6NTAC-MAT-16.2d  214-16 B1 - 2016-02-17 15.43.57'] = 3
ndpi_files_test_list['KLF14-B6NTAC-37.1d PAT 109-16 B1 - 2016-02-15 15.03.44'] = 4
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.2g  415-16 B1 - 2016-03-16 11.04.45'] = 4
ndpi_files_test_list['KLF14-B6NTAC-36.1a PAT 96-16 B1 - 2016-02-10 15.32.31'] = 5
ndpi_files_test_list['KLF14-B6NTAC-36.1b PAT 97-16 B1 - 2016-02-10 17.15.16'] = 5
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.1a  50-16 B1 - 2016-02-02 08.49.06'] = 6
ndpi_files_test_list['KLF14-B6NTAC-PAT-36.3d  416-16 B1 - 2016-03-16 14.22.04'] = 6
ndpi_files_test_list['KLF14-B6NTAC-36.1c PAT 98-16 B1 - 2016-02-10 18.32.40'] = 7
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.4a  417-16 B1 - 2016-03-16 15.25.38'] = 7
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.1e  54-16 B1 - 2016-02-02 15.06.05'] = 8
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.3b  223-16 B1 - 2016-02-25 16.53.42'] = 8
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.2f  68-16 B1 - 2016-02-04 14.01.40'] = 9
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.2g  63-16 B1 - 2016-02-03 16.40.37'] = 9

# add slices from SCWAT, slices from other animals not from training
# i_file = [80, 135]
# males
ndpi_files_test_list['KLF14-B6NTAC 36.1d PAT 99-16 C1 - 2016-02-11 11.48.31'] = 3
ndpi_files_test_list['KLF14-B6NTAC 36.1e PAT 100-16 C1 - 2016-02-11 14.06.56'] = 7
ndpi_files_test_list['KLF14-B6NTAC 36.1f PAT 101-16 C1 - 2016-02-11 15.23.06'] = 5
ndpi_files_test_list['KLF14-B6NTAC 36.1g PAT 102-16 C1 - 2016-02-11 17.20.14'] = 1
ndpi_files_test_list['KLF14-B6NTAC 36.1h PAT 103-16 C1 - 2016-02-12 10.15.22'] = 4
ndpi_files_test_list['KLF14-B6NTAC 36.1j PAT 105-16 C1 - 2016-02-12 14.33.33'] = 1
# females
ndpi_files_test_list['KLF14-B6NTAC 37.1a PAT 106-16 C1 - 2016-02-12 16.21.00'] = 6
ndpi_files_test_list['KLF14-B6NTAC-37.1b PAT 107-16 C1 - 2016-02-15 11.43.31'] = 4
ndpi_files_test_list['KLF14-B6NTAC-37.1e PAT 110-16 C1 - 2016-02-15 17.33.11'] = 3
ndpi_files_test_list['KLF14-B6NTAC-37.1g PAT 112-16 C1 - 2016-02-16 13.33.09'] = 9
# females
ndpi_files_test_list['KLF14-B6NTAC-PAT-36.3a  409-16 C1 - 2016-03-15 10.18.46'] = 7
ndpi_files_test_list['KLF14-B6NTAC-PAT-36.3b  412-16 C1 - 2016-03-15 14.37.55'] = 7
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.2a  406-16 C1 - 2016-03-14 12.01.56'] = 3
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.2b  410-16 C1 - 2016-03-15 11.24.20'] = 8
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.2c  407-16 C1 - 2016-03-14 14.13.54'] = 0
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.2d  411-16 C1 - 2016-03-15 12.42.26'] = 9
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.2e  408-16 C1 - 2016-03-14 16.23.30'] = 7
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.2f  405-16 C1 - 2016-03-14 10.58.34'] = 1
# male
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.2h  418-16 C1 - 2016-03-16 17.01.17'] = 8
# female
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.3a  413-16 C1 - 2016-03-15 15.54.12'] = 6
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.3c  414-16 C1 - 2016-03-15 17.15.41'] = 7
# male
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.4b  419-16 C1 - 2016-03-17 10.22.54'] = 1
# female
ndpi_files_test_list['KLF14-B6NTAC-PAT-39.1h  453-16 C1 - 2016-03-17 11.38.04'] = 6
# male
ndpi_files_test_list['KLF14-B6NTAC-PAT-39.2d  454-16 C1 - 2016-03-17 14.33.38'] = 2
ndpi_files_test_list['KLF14-B6NTAC-37.1h PAT 113-16 C1 - 2016-02-16 15.14.09'] = 5
ndpi_files_test_list['KLF14-B6NTAC-38.1e PAT 94-16 C1 - 2016-02-10 12.13.10'] = 1
ndpi_files_test_list['KLF14-B6NTAC-38.1f PAT 95-16 C1 - 2016-02-10 14.41.44'] = 0

# female
ndpi_files_test_list['KLF14-B6NTAC-MAT-16.2a  211-16 C1 - 2016-02-17 11.46.42'] = 8
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.1a  44-16 C1 - 2016-02-01 11.14.17'] = 4
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.1b  45-16 C1 - 2016-02-01 12.23.50'] = 3
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.1d  47-16 C1 - 2016-02-01 15.25.53'] = 6
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.2a  64-16 C1 - 2016-02-04 09.17.52'] = 9
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.2b  65-16 C1 - 2016-02-04 10.24.22'] = 1
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.2d  67-16 C1 - 2016-02-04 12.34.32'] = 9
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.1b  51-16 C1 - 2016-02-02 09.59.16'] = 9
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.1c  52-16 C1 - 2016-02-02 12.26.58'] = 1
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.2a  57-16 C1 - 2016-02-03 09.10.17'] = 6
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.2c  59-16 C1 - 2016-02-03 11.56.52'] = 2
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.2e  61-16 C1 - 2016-02-03 14.19.35'] = 0
ndpi_files_test_list['KLF14-B6NTAC-MAT-19.2b  219-16 C1 - 2016-02-18 15.41.38'] = 5
ndpi_files_test_list['KLF14-B6NTAC-MAT-19.2c  220-16 C1 - 2016-02-18 17.03.38'] = 0
ndpi_files_test_list['KLF14-B6NTAC-MAT-19.2e  221-16 C1 - 2016-02-25 14.00.14'] = 3
# male
ndpi_files_test_list['KLF14-B6NTAC-MAT-16.2b  212-16 C1 - 2016-02-17 12.49.00'] = 3
ndpi_files_test_list['KLF14-B6NTAC-MAT-16.2c  213-16 C1 - 2016-02-17 14.51.18'] = 8
ndpi_files_test_list['KLF14-B6NTAC-MAT-16.2e  215-16 C1 - 2016-02-18 09.19.26'] = 4
ndpi_files_test_list['KLF14-B6NTAC-MAT-16.2f  216-16 C1 - 2016-02-18 10.28.27'] = 5
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.1e  48-16 C1 - 2016-02-01 16.27.05'] = 5
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.1f  49-16 C1 - 2016-02-01 17.51.46'] = 6
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.2g  69-16 C1 - 2016-02-04 16.15.05'] = 6
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.1d  53-16 C1 - 2016-02-02 14.32.03'] = 7
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.1f  55-16 C1 - 2016-02-02 16.14.30'] = 4
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.2f  62-16 C1 - 2016-02-03 15.46.15'] = 9
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.3c  218-16 C1 - 2016-02-18 13.12.09'] = 9
ndpi_files_test_list['KLF14-B6NTAC-MAT-19.1a  56-16 C1 - 2016-02-02 17.23.31'] = 8
ndpi_files_test_list['KLF14-B6NTAC-MAT-19.2f  217-16 C1 - 2016-02-18 11.48.16'] = 8
ndpi_files_test_list['KLF14-B6NTAC-MAT-19.2g  222-16 C1 - 2016-02-25 15.13.00'] = 8

# add slices from GWAT that haven't been added already
# i_file = [136, 191]
ndpi_files_test_list['KLF14-B6NTAC 36.1d PAT 99-16 B1 - 2016-02-11 11.29.55'] = 9
ndpi_files_test_list['KLF14-B6NTAC 36.1e PAT 100-16 B1 - 2016-02-11 12.51.11'] = 4
ndpi_files_test_list['KLF14-B6NTAC 36.1f PAT 101-16 B1 - 2016-02-11 14.57.03'] = 9
ndpi_files_test_list['KLF14-B6NTAC 36.1g PAT 102-16 B1 - 2016-02-11 16.12.01'] = 4
ndpi_files_test_list['KLF14-B6NTAC 36.1h PAT 103-16 B1 - 2016-02-12 09.51.08'] = 0
ndpi_files_test_list['KLF14-B6NTAC 36.1j PAT 105-16 B1 - 2016-02-12 14.08.19'] = 4
ndpi_files_test_list['KLF14-B6NTAC 37.1a PAT 106-16 B1 - 2016-02-12 15.33.02'] = 2
ndpi_files_test_list['KLF14-B6NTAC-37.1b PAT 107-16 B1 - 2016-02-15 11.25.20'] = 0
ndpi_files_test_list['KLF14-B6NTAC-37.1e PAT 110-16 B1 - 2016-02-15 16.16.06'] = 1
ndpi_files_test_list['KLF14-B6NTAC-37.1g PAT 112-16 B1 - 2016-02-16 12.02.07'] = 4
ndpi_files_test_list['KLF14-B6NTAC-37.1h PAT 113-16 B1 - 2016-02-16 14.53.02'] = 3
ndpi_files_test_list['KLF14-B6NTAC-38.1e PAT 94-16 B1 - 2016-02-10 11.35.53'] = 4
ndpi_files_test_list['KLF14-B6NTAC-38.1f PAT 95-16 B1 - 2016-02-10 14.16.55'] = 1
ndpi_files_test_list['KLF14-B6NTAC-MAT-16.2a  211-16 B1 - 2016-02-17 11.21.54'] = 1
ndpi_files_test_list['KLF14-B6NTAC-MAT-16.2b  212-16 B1 - 2016-02-17 12.33.18'] = 0
ndpi_files_test_list['KLF14-B6NTAC-MAT-16.2c  213-16 B1 - 2016-02-17 14.01.06'] = 9
ndpi_files_test_list['KLF14-B6NTAC-MAT-16.2e  215-16 B1 - 2016-02-17 17.14.16'] = 4
ndpi_files_test_list['KLF14-B6NTAC-MAT-16.2f  216-16 B1 - 2016-02-18 10.05.52'] = 4
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.1a  44-16 B1 - 2016-02-01 09.19.20'] = 1
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.1b  45-16 B1 - 2016-02-01 12.05.15'] = 5
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.1d  47-16 B1 - 2016-02-01 15.11.42'] = 8
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.1e  48-16 B1 - 2016-02-01 16.01.09'] = 8
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.1f  49-16 B1 - 2016-02-01 17.12.31'] = 0
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.2a  64-16 B1 - 2016-02-04 08.57.34'] = 4
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.2b  65-16 B1 - 2016-02-04 10.06.00'] = 5
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.2d  67-16 B1 - 2016-02-04 12.20.20'] = 4
ndpi_files_test_list['KLF14-B6NTAC-MAT-17.2g  69-16 B1 - 2016-02-04 15.52.52'] = 0
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.1b  51-16 B1 - 2016-02-02 09.46.31'] = 5
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.1c  52-16 B1 - 2016-02-02 11.24.31'] = 7
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.1d  53-16 B1 - 2016-02-02 14.11.37'] = 9
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.2a  57-16 B1 - 2016-02-03 08.54.27'] = 6
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.2c  59-16 B1 - 2016-02-03 11.41.32'] = 3
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.2e  61-16 B1 - 2016-02-03 14.02.25'] = 8
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.2f  62-16 B1 - 2016-02-03 15.00.17'] = 5
ndpi_files_test_list['KLF14-B6NTAC-MAT-18.3c  218-16 B1 - 2016-02-18 12.51.46'] = 8
ndpi_files_test_list['KLF14-B6NTAC-MAT-19.1a  56-16 B1 - 2016-02-02 16.57.46'] = 1
ndpi_files_test_list['KLF14-B6NTAC-MAT-19.2b  219-16 B1 - 2016-02-18 14.21.50'] = 5
ndpi_files_test_list['KLF14-B6NTAC-MAT-19.2c  220-16 B1 - 2016-02-18 16.40.48'] = 1
ndpi_files_test_list['KLF14-B6NTAC-MAT-19.2e  221-16 B1 - 2016-02-25 13.15.27'] = 2
ndpi_files_test_list['KLF14-B6NTAC-MAT-19.2f  217-16 B1 - 2016-02-18 11.23.22'] = 7
ndpi_files_test_list['KLF14-B6NTAC-MAT-19.2g  222-16 B1 - 2016-02-25 14.51.57'] = 9
ndpi_files_test_list['KLF14-B6NTAC-PAT-36.3a  409-16 B1 - 2016-03-15 09.24.54'] = 8
ndpi_files_test_list['KLF14-B6NTAC-PAT-36.3b  412-16 B1 - 2016-03-15 14.11.47'] = 7
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.2a  406-16 B1 - 2016-03-14 11.46.47'] = 4
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.2b  410-16 B1 - 2016-03-15 11.12.01'] = 9
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.2c  407-16 B1 - 2016-03-14 12.54.55'] = 7
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.2d  411-16 B1 - 2016-03-15 12.01.13'] = 9
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.2e  408-16 B1 - 2016-03-14 16.06.43'] = 1
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.2f  405-16 B1 - 2016-03-14 09.49.45'] = 9
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.2h  418-16 B1 - 2016-03-16 16.42.16'] = 7
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.3a  413-16 B1 - 2016-03-15 15.31.26'] = 3
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.3c  414-16 B1 - 2016-03-15 16.49.22'] = 3
ndpi_files_test_list['KLF14-B6NTAC-PAT-37.4b  419-16 B1 - 2016-03-17 09.10.42'] = 7
ndpi_files_test_list['KLF14-B6NTAC-PAT-38.1a  90-16 B1 - 2016-02-04 17.27.42'] = 6
ndpi_files_test_list['KLF14-B6NTAC-PAT-39.1h  453-16 B1 - 2016-03-17 11.15.50'] = 0
ndpi_files_test_list['KLF14-B6NTAC-PAT-39.2d  454-16 B1 - 2016-03-17 12.16.06'] = 7
# add missing slices
# i_file = [192]
ndpi_files_test_list['KLF14-B6NTAC-37.1f PAT 111-16 C2 - 2016-02-16 11.26 (1)'] = 5
# add recut slices
# i_file = [193, 195]
ndpi_files_test_list['KLF14-B6NTAC-PAT 37.2b 410-16 C4 - 2020-02-14 10.27.23'] = 8
ndpi_files_test_list['KLF14-B6NTAC-PAT 37.2c 407-16 C4 - 2020-02-14 10.15.57'] = 0
ndpi_files_test_list['KLF14-B6NTAC-PAT 37.2d 411-16 C4 - 2020-02-14 10.34.10'] = 9


if DEBUG:
    for i, key in enumerate(ndpi_files_test_list.keys()):
        print('File ' + str(i) + ': Fold ' + str(ndpi_files_test_list[key]) + ': ' + key)

########################################################################################################################
## Colourmap for AIDA
########################################################################################################################

# loop files with hand traced contours
manual_areas_all = []
for i, file_svg in enumerate(file_svg_list):

    print('file ' + str(i) + '/' + str(len(file_svg_list) - 1) + ': ' + os.path.basename(file_svg))

    # change file extension from .svg to .tif
    file_tif = file_svg.replace('.svg', '.tif')

    # open histology training image
    im = PIL.Image.open(file_tif)

    # read pixel size information
    xres = 0.0254 / im.info['dpi'][0]  # m
    yres = 0.0254 / im.info['dpi'][1]  # m

    # read the ground truth cell contours in the SVG file. This produces a list [contour_0, ..., contour_N-1]
    # where each contour_i = [(X_0, Y_0), ..., (X_P-1, X_P-1)]
    contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False,
                                                       minimum_npoints=3)

    # compute cell area
    manual_areas_all.append([Polygon(c).area * xres * yres for c in contours])  # (um^2)

manual_areas_all = list(itertools.chain.from_iterable(manual_areas_all))

# compute function to map between cell areas and [0.0, 1.0], that we can use to sample the colourmap uniformly according
# to area quantiles
f_area2quantile = cytometer.data.area2quantile(manual_areas_all)

########################################################################################################################
## Segmentation loop
########################################################################################################################

# DEBUG: i_file = 0; ndpi_file_kernel = list(ndpi_files_test_list.keys())[i_file]
# for i_file, ndpi_file_kernel in reversed(list(enumerate(ndpi_files_test_list))):
for i_file in list(range(193, 196)):

    # name of the slice to analyse
    ndpi_file_kernel = list(ndpi_files_test_list.keys())[i_file]

    # fold  where the current .ndpi image was not used for training
    i_fold = ndpi_files_test_list[ndpi_file_kernel]

    print('File ' + str(i_file) + '/' + str(len(ndpi_files_test_list) - 1) + ': ' + ndpi_file_kernel
          + '. Fold = ' + str(i_fold))

    # make full path to ndpi file
    ndpi_file = os.path.join(data_dir, ndpi_file_kernel + '.ndpi')

    # check whether there's a lock on this file
    lock_file = os.path.basename(ndpi_file).replace('.ndpi', '.lock')
    lock_file = os.path.join(annotations_dir, lock_file)
    if os.path.isfile(lock_file):
        print('Lock on file, skipping')
        continue
    else:
        # create an empty lock file to prevent other other instances of the script to process the same .ndpi file
        Path(lock_file).touch()

    contour_model_file = os.path.join(saved_models_dir, contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    dmap_model_file = os.path.join(saved_models_dir, dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    classifier_model_file = os.path.join(saved_models_dir,
                                         classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    correction_model_file = os.path.join(saved_models_dir,
                                         correction_model_basename + '_model_fold_' + str(i_fold) + '.h5')

    # name of file to save annotations to
    annotations_file = os.path.basename(ndpi_file)
    annotations_file = os.path.splitext(annotations_file)[0]
    annotations_file = os.path.join(annotations_dir, annotations_file + '_exp_0097_no_overlap.json')

    annotations_corrected_file = os.path.basename(ndpi_file)
    annotations_corrected_file = os.path.splitext(annotations_corrected_file)[0]
    annotations_corrected_file = os.path.join(annotations_dir, annotations_corrected_file + '_exp_0097_corrected.json')

    # name of file to save rough mask, current mask, and time steps
    rough_mask_file = os.path.basename(ndpi_file)
    rough_mask_file = rough_mask_file.replace('.ndpi', '_rough_mask.npz')
    rough_mask_file = os.path.join(annotations_dir, rough_mask_file)

    # check whether we continue previous execution, or we start a new one
    continue_previous = os.path.isfile(rough_mask_file)

    # open full resolution histology slide
    im = openslide.OpenSlide(ndpi_file)

    # pixel size
    assert(im.properties['tiff.ResolutionUnit'] == 'centimeter')
    xres = 1e-2 / float(im.properties['tiff.XResolution'])
    yres = 1e-2 / float(im.properties['tiff.YResolution'])

    # if the rough mask has been pre-computed, just load it
    if continue_previous:

        aux = np.load(rough_mask_file)
        lores_istissue = aux['lores_istissue']
        lores_istissue0 = aux['lores_istissue0']
        im_downsampled = aux['im_downsampled']
        step = aux['step']
        perc_completed_all = list(aux['perc_completed_all'])
        time_step_all = list(aux['time_step_all'])
        del aux

    else:

        time_prev = time.time()

        # rough segmentation of the tissue in the image
        if (i_file <= 19) and (os.path.basename(ndpi_file) != 'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52.ndpi'):

            # the original 20 images were thresholded with mode - std, except for one image
            lores_istissue0, im_downsampled = rough_foreground_mask(ndpi_file, downsample_factor=downsample_factor,
                                                                    dilation_size=dilation_size,
                                                                    component_size_threshold=component_size_threshold,
                                                                    hole_size_treshold=hole_size_treshold,
                                                                    return_im=True)

        elif (i_file <= 19) and (os.path.basename(ndpi_file) == 'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52.ndpi'):

            # special case for an image that has very low contrast, with strong bright pink and purple areas of other
            # tissue. We threshold with mode - 0.25 std
            lores_istissue0, im_downsampled = rough_foreground_mask(ndpi_file, downsample_factor=downsample_factor,
                                                                    dilation_size=dilation_size,
                                                                    component_size_threshold=component_size_threshold,
                                                                    hole_size_treshold=hole_size_treshold, std_k=0.25,
                                                                    return_im=True)

        elif os.path.basename(ndpi_file) in {
            'KLF14-B6NTAC-36.1c PAT 98-16 B1 - 2016-02-10 18.32.40.ndpi',
            'KLF14-B6NTAC-MAT-18.3b  223-16 B1 - 2016-02-25 16.53.42.ndpi',
            'KLF14-B6NTAC-MAT-17.2f  68-16 B1 - 2016-02-04 14.01.40.ndpi',
            'KLF14-B6NTAC-MAT-18.2b  58-16 B1 - 2016-02-03 09.58.06.ndpi',
            'KLF14-B6NTAC-MAT-18.2d  60-16 B1 - 2016-02-03 12.56.49.ndpi',
            'KLF14-B6NTAC-MAT-17.2c  66-16 B1 - 2016-02-04 11.14.28.ndpi',
            'KLF14-B6NTAC-MAT-17.1c  46-16 B1 - 2016-02-01 13.01.30.ndpi',
            'KLF14-B6NTAC-37.1c PAT 108-16 B1 - 2016-02-15 12.33.10.ndpi'}:

            # some of the posterior images also work with mode - std
            lores_istissue0, im_downsampled = rough_foreground_mask(ndpi_file, downsample_factor=downsample_factor,
                                                                    dilation_size=dilation_size,
                                                                    component_size_threshold=component_size_threshold,
                                                                    hole_size_treshold=hole_size_treshold,
                                                                    return_im=True)

        else:  # any other case

            lores_istissue0, im_downsampled = rough_foreground_mask(ndpi_file, downsample_factor=downsample_factor,
                                                                    dilation_size=dilation_size,
                                                                    component_size_threshold=component_size_threshold,
                                                                    hole_size_treshold=hole_size_treshold, std_k=0.25,
                                                                    return_im=True)

        # segmentation copy, to keep track of what's left to do
        lores_istissue = lores_istissue0.copy()

        # initialize block algorithm variables
        step = 0
        perc_completed_all = [float(0.0),]
        time_step = time.time() - time_prev
        time_step_all = [time_step,]

        # save to the rough mask file
        np.savez_compressed(rough_mask_file, lores_istissue=lores_istissue, lores_istissue0=lores_istissue0,
                            im_downsampled=im_downsampled, step=step, perc_completed_all=perc_completed_all,
                            time_step_all=time_step_all)


    # checkpoint: here the rough tissue mask has either been loaded or computed
    time_step = time_step_all[-1]
    time_total = np.sum(time_step_all)
    print('File ' + str(i_file) + '/' + str(len(ndpi_files_test_list) - 1) + ': step ' +
          str(step) + ': ' +
          str(np.count_nonzero(lores_istissue)) + '/' + str(np.count_nonzero(lores_istissue0)) + ': ' +
          "{0:.1f}".format(100.0 - np.count_nonzero(lores_istissue) / np.count_nonzero(lores_istissue0) * 100) +
          '% completed: ' +
          'time step ' + "{0:.2f}".format(time_step) + ' s' +
          ', total time ' + "{0:.2f}".format(time_total) + ' s')

    if DEBUG:
            plt.clf()
            plt.subplot(211)
            plt.imshow(im_downsampled)
            plt.contour(lores_istissue0, colors='k')
            plt.subplot(212)
            plt.imshow(lores_istissue0)

            plt.clf()
            plt.subplot(211)
            plt.imshow(im_downsampled)
            plt.contour(lores_istissue, colors='k')
            plt.subplot(212)
            plt.imshow(lores_istissue)

    # keep extracting histology windows until we have finished
    while np.count_nonzero(lores_istissue) > 0:

        time_prev = time.time()

        # next step (it starts from 0)
        step += 1

        # get indices for the next histology window to process
        (first_row, last_row, first_col, last_col), \
        (lores_first_row, lores_last_row, lores_first_col, lores_last_col) = \
            cytometer.utils.get_next_roi_to_process(lores_istissue, downsample_factor=downsample_factor,
                                                    max_window_size=fullres_box_size,
                                                    border=np.round((receptive_field-1)/2))

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

        # if no cells found, wipe out current window from tissue segmentation, and go to next iteration. Otherwise we'd
        # enter an infinite loop
        if len(index_list) == 0:
            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0
            continue

        if DEBUG:
            j = 4
            plt.clf()
            plt.subplot(221)
            plt.imshow(tile[:, :, :])
            plt.title('Histology', fontsize=16)
            plt.axis('off')
            plt.subplot(222)
            plt.imshow(tile[:, :, :])
            plt.contour(labels, levels=np.unique(labels), colors='C0')
            plt.contourf(todo_edge, colors='C2', levels=[0.5, 1])
            plt.title('Full segmentation', fontsize=16)
            plt.axis('off')
            plt.subplot(212)
            plt.imshow(window_im[j, :, :, :])
            plt.contour(window_labels[j, :, :], colors='C0')
            plt.contour(window_labels_corrected[j, :, :], colors='C1')
            plt.title('Crop around object and corrected segmentation', fontsize=16)
            plt.axis('off')
            plt.tight_layout()

        # downsample "to do" mask so that the rough tissue segmentation can be updated
        lores_todo_edge = PIL.Image.fromarray(todo_edge.astype(np.uint8))
        lores_todo_edge = lores_todo_edge.resize((lores_last_col - lores_first_col,
                                                  lores_last_row - lores_first_row),
                                                 resample=PIL.Image.NEAREST)
        lores_todo_edge = np.array(lores_todo_edge)

        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col])
            plt.title('Low res tissue mask', fontsize=16)
            plt.axis('off')
            plt.subplot(222)
            plt.imshow(istissue_tile)
            plt.title('Full res tissue mask', fontsize=16)
            plt.axis('off')
            plt.subplot(223)
            plt.imshow(todo_edge)
            plt.title('Full res left over tissue', fontsize=16)
            plt.axis('off')
            plt.subplot(224)
            plt.imshow(lores_todo_edge)
            plt.title('Low res left over tissue', fontsize=16)
            plt.axis('off')
            plt.tight_layout()

        # convert overlap labels in cropped images to contours (points), and add cropping window offset so that the
        # contours are in the tile-window coordinates
        offset_xy = index_list[:, [2, 3]]  # index_list: [i, lab, x0, y0, xend, yend]
        contours = cytometer.utils.labels2contours(window_labels, offset_xy=offset_xy,
                                                   scaling_factor_xy=scaling_factor_list)
        contours_corrected = cytometer.utils.labels2contours(window_labels_corrected, offset_xy=offset_xy,
                                                             scaling_factor_xy=scaling_factor_list)

        if DEBUG:
            # no overlap
            plt.clf()
            plt.imshow(tile)
            for j in range(len(contours)):
                plt.fill(contours[j][:, 0], contours[j][:, 1], edgecolor='C0', fill=False)
                # plt.text(contours[j][0, 0], contours[j][0, 1], str(j))

            # overlap
            plt.clf()
            plt.imshow(tile)
            for j in range(len(contours_corrected)):
                plt.fill(contours_corrected[j][:, 0], contours_corrected[j][:, 1], edgecolor='C0', fill=False)
                # plt.text(contours_corrected[j][0, 0], contours_corrected[j][0, 1], str(j))

        # downsample contours for AIDA annotations file
        lores_contours = []
        for c in contours:
            lores_c = bspline_resample(c, factor=contour_downsample_factor, min_n=10, k=bspline_k, is_closed=True)
            lores_contours.append(lores_c)

        lores_contours_corrected = []
        for c in contours_corrected:
            lores_c = bspline_resample(c, factor=contour_downsample_factor, min_n=10, k=bspline_k, is_closed=True)
            lores_contours_corrected.append(lores_c)

        if DEBUG:
            # no overlap
            plt.clf()
            plt.imshow(tile)
            for j in range(len(contours)):
                plt.fill(lores_contours[j][:, 0], lores_contours[j][:, 1], edgecolor='C1', fill=False)

            # overlap
            plt.clf()
            plt.imshow(tile)
            for j in range(len(contours_corrected)):
                plt.fill(lores_contours_corrected[j][:, 0], lores_contours_corrected[j][:, 1], edgecolor='C1', fill=False)

        # add tile offset, so that contours are in full slide coordinates
        for j in range(len(contours)):
            lores_contours[j][:, 0] += first_col
            lores_contours[j][:, 1] += first_row

        for j in range(len(contours_corrected)):
            lores_contours_corrected[j][:, 0] += first_col
            lores_contours_corrected[j][:, 1] += first_row

        # convert non-overlap contours to AIDA items
        contour_items = cytometer.data.aida_contour_items(lores_contours, f_area2quantile, xres=xres, yres=yres)
        rectangle = (first_col, first_row, last_col - first_col, last_row - first_row)  # (x0, y0, width, height)
        rectangle_item = cytometer.data.aida_rectangle_items([rectangle,])

        if step == 0:
            # in the first step, overwrite previous annotations file, or create new one
            cytometer.data.aida_write_new_items(annotations_file, rectangle_item, mode='w')
            cytometer.data.aida_write_new_items(annotations_file, contour_items, mode='append_new_layer')
        else:
            # in next steps, add contours to previous layer
            cytometer.data.aida_write_new_items(annotations_file, rectangle_item, mode='append_to_last_layer')
            cytometer.data.aida_write_new_items(annotations_file, contour_items, mode='append_new_layer')

        # convert corrected contours to AIDA items
        contour_items_corrected = cytometer.data.aida_contour_items(lores_contours_corrected, f_area2quantile, xres=xres, yres=yres)

        if step == 0:
            # in the first step, overwrite previous annotations file, or create new one
            cytometer.data.aida_write_new_items(annotations_corrected_file, rectangle_item, mode='w')
            cytometer.data.aida_write_new_items(annotations_corrected_file, contour_items_corrected, mode='append_new_layer')
        else:
            # in next steps, add contours to previous layer
            cytometer.data.aida_write_new_items(annotations_corrected_file, rectangle_item, mode='append_to_last_layer')
            cytometer.data.aida_write_new_items(annotations_corrected_file, contour_items_corrected, mode='append_new_layer')

        # update the tissue segmentation mask with the current window
        if np.all(lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] == lores_todo_edge):
            # if the mask remains identical, wipe out the whole window, as otherwise we'd have an
            # infinite loop
            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0
        else:
            # if the mask has been updated, use it to update the total tissue segmentation
            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = lores_todo_edge

        perc_completed = 100.0 - np.count_nonzero(lores_istissue) / np.count_nonzero(lores_istissue0) * 100
        perc_completed_all.append(perc_completed)
        time_step = time.time() - time_prev
        time_step_all.append(time_step)
        time_total = np.sum(time_step_all)

        print('File ' + str(i_file) + '/' + str(len(ndpi_files_test_list) - 1) + ': step ' +
              str(step) + ': ' +
              str(np.count_nonzero(lores_istissue)) + '/' + str(np.count_nonzero(lores_istissue0)) + ': ' +
              "{0:.1f}".format(perc_completed) +
              '% completed: ' +
              'time step ' + "{0:.2f}".format(time_step) + ' s' +
              ', total time ' + "{0:.2f}".format(time_total) + ' s')

        # save to the rough mask file
        np.savez_compressed(rough_mask_file, lores_istissue=lores_istissue, lores_istissue0=lores_istissue0,
                            im_downsampled=im_downsampled, step=step, perc_completed_all=perc_completed_all,
                            time_step_all=time_step_all)

# end of "keep extracting histology windows until we have finished"

# if we run the script with qsub on the cluster, the standard output is in file
# klf14_b6ntac_exp_0001_cnn_dmap_contour.sge.sh.oPID where PID is the process ID
# Save it to saved_models directory
log_filename = os.path.join(saved_models_dir, experiment_id + '.log')
stdout_filename = os.path.join(home, 'Software', 'cytometer', 'scripts', experiment_id + '.sge.sh.o*')
stdout_filename = glob.glob(stdout_filename)[0]
if stdout_filename and os.path.isfile(stdout_filename):
    shutil.copy2(stdout_filename, log_filename)
else:
    # if we ran the script with nohup in linux, the standard output is in file nohup.out.
    # Save it to saved_models directory
    log_filename = os.path.join(saved_models_dir, experiment_id + '.log')
    nohup_filename = os.path.join(home, 'Software', 'cytometer', 'scripts', 'nohup.out')
    if os.path.isfile(nohup_filename):
        shutil.copy2(nohup_filename, log_filename)


