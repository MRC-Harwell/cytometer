"""
Processing full slides of KLF14 WAT histology with pipeline v8:

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

Difference with pipeline v7:
  * Contrast enhancement to compute rough tissue mask
  * Colour correction to match the median colour of the training data for segmentation
  * All segmented objects are saved, together with the white adipocyte probability score. That way, we can decide later
    which ones we want to keep, and which ones we want to reject.
  * If a processing window overlaps the previous one by 90% or more, we wipe it out.

Difference with rreb1_tm1b_exp_0003_full_slide_pipeline_v8.py:
  * Only the images it is applied to.

Difference with fus_delta_exp_0001_full_slide_pipeline_v8.py:
  * Change to GTEx data.

Differences with gtex_exp_0001_full_slide_pipeline_v8.py:
  * Change to KLF14 data.

You can run this script limiting it to one GPU with:

    export CUDA_VISIBLE_DEVICES=0 && python klf14_b6ntac_exp_0106_full_slide_pipeline_v8.py

 Requirements for this script to work:

 1) Upload the cytometer project directory to ~/Software in the server where you are going to process the data.

 2) Run ./install_dependencies.sh in cytometer.

 3) Mount the network share //scan-srv2/cox/"Maz Yon" on ~/scan_srv2_cox with CIFS so that we have access to
    KLF14 .ndpi files. You can do it by creating an empty directory

    mkdir ~/scan_srv2_cox

    and adding a line like this to /etc/fstab in the server (adjusting the uid and gid to your own).

    //scan-srv2/cox /home/rcasero/scan_srv2_cox cifs credentials=/home/rcasero/.smbcredentials,vers=3.0,domain=MRCH,sec=ntlmv2,uid=1003,gid=1003,user 0 0

    Then

    mount ~/scan_srv2_cox

 4) Convert the .ndpi files to AIDA .dzi files, so that we can see the results of the segmentation.
    You need to go to the server that's going to process the slides, add a list of the files you want to process to
    ~/Software/cytometer/tools/klf14_full_histology_ndpi_to_dzi.sh

    and run

    cd ~/Software/cytometer/tools
    ./klf14_full_histology_ndpi_to_dzi.sh

 5) You need to have the models for the 10-folds of the pipeline that were trained on the KLF14 data in
    ~/Data/cytometer_data/klf14/saved_models.

 6) To monitor the segmentation as it's being processed, you need to have AIDA running

    cd ~/Software/AIDA/dist/
    node aidaLocal.js &

    You also need to create a soft link per .dzi file to the annotations you want to visualise for that file, whether
    the non-overlapping ones, or the corrected ones. E.g.

    ln -s 'KLF14-B6NTAC-PAT-39.2d  454-16 C1 - 2016-03-17 14.33.38_exp_0106_corrected.json' 'KLF14-B6NTAC-PAT-39.2d  454-16 C1 - 2016-03-17 14.33.38.ndpi'

    Then you can use a browser to open the AIDA web interface by visiting the URL (note that you need to be on the MRC
    VPN, or connected from inside the office to get access to the titanrtx server)

    http://titanrtx:3000/dashboard

    You can use the interface to open a .dzi file that corresponds to an .ndpi file being segmented, and see the
    annotations (segmentation) being created for it.

"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0106_full_slide_pipeline_v8'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import sys
if os.path.join(home, 'Software/cytometer') not in sys.path:
    sys.path.extend([os.path.join(home, 'Software/cytometer')])
import cytometer.utils
import cytometer.data

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
import time
import openslide
import numpy as np
import matplotlib.pyplot as plt
from cytometer.utils import rough_foreground_mask, bspline_resample
import PIL
from keras import backend as K
import scipy.stats
from shapely.geometry import Polygon
import pandas as pd

import tensorflow as tf
if tf.test.is_gpu_available():
    print('GPU available')
else:
    raise SystemError('GPU is not available')

# # limit GPU memory used
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# set_session(tf.Session(config=config))

DEBUG = False
SAVE_FIGS = False

# data paths
histology_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
histology_ext = '.ndpi'
area2quantile_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v8')
saved_models_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v8')
annotations_dir = os.path.join(home, 'bit/cytometer_data/aida_data_Klf14_v8/annotations')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
metainfo_dir = os.path.join(home, 'Data/cytometer_data/klf14')

# file with area->quantile map precomputed from all automatically segmented slides in klf14_b6ntac_exp_0098_full_slide_size_analysis_v7.py
filename_area2quantile = os.path.join(area2quantile_dir, 'klf14_b6ntac_exp_0098_filename_area2quantile.npz')

# file with RGB modes from all training data
klf14_training_colour_histogram_file = os.path.join(saved_models_dir, 'klf14_training_colour_histogram.npz')

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
downsample_factor_goal = 16  # approximate value, that may vary a bit in each histology file
dilation_size = 25
component_size_threshold = 50e3
hole_size_treshold = 8000
std_k = 0.25
enhance_contrast = 4.0
ignore_white_threshold = 253

# contour parameters
contour_downsample_factor = 0.1
bspline_k = 1

# block_split() parameters in downsampled image
block_len = np.ceil((fullres_box_size - receptive_field) / downsample_factor_goal)
block_overlap = np.ceil((receptive_field - 1) / 2 / downsample_factor_goal).astype(np.int)
window_overlap_fraction_max = 0.9

# segmentation parameters
min_cell_area = 0  # pixel; we want all small objects
max_cell_area = 200e3  # pixel
min_mask_overlap = 0.8
phagocytosis = True
min_class_prop = 0.0  # we want all the objects
correction_window_len = 401
correction_smoothing = 11
batch_size = 16

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

if DEBUG:
    for i, key in enumerate(histo_files_list.keys()):
        print('File ' + str(i) + ': Fold ' + str(histo_files_list[key]) + ': ' + key)

########################################################################################################################
# target background colour
########################################################################################################################

# statistical mode of the background colour (typical colour value) from the training dataset
with np.load(klf14_training_colour_histogram_file) as data:
    mode_r_target = data['mode_r']
    mode_g_target = data['mode_g']
    mode_b_target = data['mode_b']

# colourmap for AIDA, based on KLF14 automatically segmented data
if os.path.isfile(filename_area2quantile):
    with np.load(filename_area2quantile, allow_pickle=True) as aux:
        f_area2quantile_f = aux['f_area2quantile_f'].item()
        f_area2quantile_m = aux['f_area2quantile_m'].item()
else:
    raise FileNotFoundError('Cannot find file with area->quantile map precomputed from all automatically segmented' +
                            ' slides in klf14_b6ntac_exp_0098_full_slide_size_analysis_v7.py')

########################################################################################################################
## Segmentation loop
########################################################################################################################

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
    if os.path.isfile(lock_file):
        print('Lock on file, skipping')
        continue
    else:
        # create an empty lock file to prevent other other instances of the script to process the same histo file
        Path(lock_file).touch()

    contour_model_file = os.path.join(saved_models_dir, contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    dmap_model_file = os.path.join(saved_models_dir, dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    classifier_model_file = os.path.join(saved_models_dir,
                                         classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    correction_model_file = os.path.join(saved_models_dir,
                                         correction_model_basename + '_model_fold_' + str(i_fold) + '.h5')

    # name of file to save annotations to
    annotations_file = os.path.basename(histo_file)
    annotations_file = os.path.splitext(annotations_file)[0]
    annotations_file = os.path.join(annotations_dir, annotations_file + '_exp_0106_auto.json')

    annotations_corrected_file = os.path.basename(histo_file)
    annotations_corrected_file = os.path.splitext(annotations_corrected_file)[0]
    annotations_corrected_file = os.path.join(annotations_dir, annotations_corrected_file + '_exp_0106_corrected.json')

    # name of file to save rough mask, current mask, and time steps
    coarse_mask_file = os.path.basename(histo_file)
    coarse_mask_file = coarse_mask_file.replace(histology_ext, '_coarse_mask.npz')
    coarse_mask_file = os.path.join(annotations_dir, coarse_mask_file)

    # open full resolution histology slide
    im = openslide.OpenSlide(histo_file)

    # pixel size
    xres = float(im.properties['openslide.mpp-x']) # um/pixel
    yres = float(im.properties['openslide.mpp-y']) # um/pixel

    # check whether we continue previous execution, or we start a new one
    continue_previous = os.path.isfile(coarse_mask_file)

    # true downsampled factor as reported by histology file
    level_actual = np.abs(np.array(im.level_downsamples) - downsample_factor_goal).argmin()
    downsample_factor_actual = im.level_downsamples[level_actual]
    if np.abs(downsample_factor_actual - downsample_factor_goal) > 1:
        warnings.warn('The histology file has no downsample level close enough to the target downsample level')
        continue

    # if the rough mask has been pre-computed, just load it
    if continue_previous:

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

    else:

        time_prev = time.time()

        # compute the rough foreground mask of tissue vs. background
        lores_istissue0, im_downsampled = \
            rough_foreground_mask(histo_file, downsample_factor=downsample_factor_actual,
                                  dilation_size=dilation_size,
                                  component_size_threshold=component_size_threshold,
                                  hole_size_treshold=hole_size_treshold, std_k=std_k,
                                  return_im=True, enhance_contrast=enhance_contrast,
                                  ignore_white_threshold=ignore_white_threshold)

        if DEBUG:
            enhancer = PIL.ImageEnhance.Contrast(PIL.Image.fromarray(im_downsampled))
            im_downsampled_enhanced = np.array(enhancer.enhance(enhance_contrast))
            plt.clf()
            plt.subplot(211)
            plt.imshow(im_downsampled_enhanced)
            plt.axis('off')
            plt.subplot(212)
            plt.imshow(im_downsampled_enhanced)
            plt.contour(lores_istissue0)
            plt.axis('off')

        # segmentation copy, to keep track of what's left to do
        lores_istissue = lores_istissue0.copy()

        # initialize block algorithm variables
        step = 0
        perc_completed_all = [float(0.0),]
        time_step = time.time() - time_prev
        time_step_all = [time_step,]
        (prev_first_row, prev_last_row, prev_first_col, prev_last_col) = (0, 0, 0, 0)

        # save to the rough mask file
        np.savez_compressed(coarse_mask_file, lores_istissue=lores_istissue, lores_istissue0=lores_istissue0,
                            im_downsampled=im_downsampled, step=step, perc_completed_all=perc_completed_all,
                            prev_first_row=prev_first_row, prev_last_row=prev_last_row,
                            prev_first_col=prev_first_col, prev_last_col=prev_last_col,
                            time_step_all=time_step_all)

        # end "computing the rough foreground mask"

    # checkpoint: here the rough tissue mask has either been loaded or computed
    time_step = time_step_all[-1]
    time_total = np.sum(time_step_all)
    print('File ' + str(i_file) + '/' + str(len(histo_files_list) - 1) + ': step ' +
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

            plt.cla()
            plt.imshow(lores_istissue)

    # estimate the colour mode of the downsampled image, so that we can correct the image tint to match the KLF14
    # training dataset. We apply the same correction to each tile, to avoid that a tile with e.g. only muscle gets
    # overcorrected
    mode_r_tile = scipy.stats.mode(im_downsampled[:, :, 0], axis=None).mode[0]
    mode_g_tile = scipy.stats.mode(im_downsampled[:, :, 1], axis=None).mode[0]
    mode_b_tile = scipy.stats.mode(im_downsampled[:, :, 2], axis=None).mode[0]

    # keep extracting histology windows until we have finished
    while np.count_nonzero(lores_istissue) > 0:

        time_prev = time.time()

        # next step (it starts from 1 here, because step 0 is the rough mask computation)
        step += 1

        # get indices for the next histology window to process
        (first_row, last_row, first_col, last_col), \
        (lores_first_row, lores_last_row, lores_first_col, lores_last_col) = \
            cytometer.utils.get_next_roi_to_process(lores_istissue, downsample_factor=downsample_factor_actual,
                                                    max_window_size=fullres_box_size,
                                                    border=np.round((receptive_field-1)/2))

        # overlap between current and previous window, as a fraction of current window area
        current_window = Polygon([(first_col, first_row), (last_col, first_row),
                                  (last_col, last_row), (first_col, last_row)])
        prev_window = Polygon([(prev_first_col, prev_first_row), (prev_last_col, prev_first_row),
                               (prev_last_col, prev_last_row), (prev_first_col, prev_last_row)])
        window_overlap_fraction = current_window.intersection(prev_window).area / current_window.area

        # check that we are not trying to process almost the same window
        if window_overlap_fraction > window_overlap_fraction_max:
            # if we are trying to process almost the same window as in the previous step, what's probably happening is
            # that we have some big labels on the edges that are not white adipocytes, and the segmentation algorithm is
            # also finding one or more spurious labels within the window. That prevents the whole lores_istissue window
            # from being wiped out, and the big edge labels keep the window selection being almost the same. Thus, we
            # wipe it out and move to another tissue area
            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0
            continue

        else:
            # remember processed window for next step
            (prev_first_row, prev_last_row, prev_first_col, prev_last_col) = (first_row, last_row, first_col, last_col)

        # load window from full resolution slide
        tile = im.read_region(location=(first_col, first_row), level=0,
                              size=(last_col - first_col, last_row - first_row))
        tile = np.array(tile)
        tile = tile[:, :, 0:3]

        # correct tint of the tile to match KLF14 training data
        tile[:, :, 0] = tile[:, :, 0] + (mode_r_target - mode_r_tile)
        tile[:, :, 1] = tile[:, :, 1] + (mode_g_target - mode_g_tile)
        tile[:, :, 2] = tile[:, :, 2] + (mode_b_target - mode_b_tile)

        # interpolate coarse tissue segmentation to full resolution
        istissue_tile = lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col]
        istissue_tile = cytometer.utils.resize(istissue_tile, size=(last_col - first_col, last_row - first_row),
                                               resample=PIL.Image.NEAREST)

        if DEBUG:
            plt.clf()
            plt.imshow(tile)
            plt.imshow(istissue_tile, alpha=0.5)
            plt.contour(istissue_tile, colors='k')
            plt.title('Yellow: Tissue mask. Purple: Background')
            plt.axis('off')

        # segment histology, split into individual objects, and apply segmentation correction
        labels, labels_class, todo_edge, \
        window_im, window_labels, window_labels_corrected, window_labels_class, index_list, scaling_factor_list \
            = cytometer.utils.segmentation_pipeline6(im=tile,
                                                     dmap_model=dmap_model_file,
                                                     contour_model=contour_model_file,
                                                     correction_model=correction_model_file,
                                                     classifier_model=classifier_model_file,
                                                     min_cell_area=min_cell_area,
                                                     max_cell_area=max_cell_area,
                                                     mask=istissue_tile,
                                                     min_mask_overlap=min_mask_overlap,
                                                     phagocytosis=phagocytosis,
                                                     min_class_prop=min_class_prop,
                                                     correction_window_len=correction_window_len,
                                                     correction_smoothing=correction_smoothing,
                                                     return_bbox=True, return_bbox_coordinates='xy',
                                                     batch_size=batch_size)


        # compute the "white adipocyte" probability for each object
        if len(window_labels) > 0:
            window_white_adipocyte_prob = np.sum(window_labels * window_labels_class, axis=(1, 2)) \
                                          / np.sum(window_labels, axis=(1, 2))
            window_white_adipocyte_prob_corrected = np.sum(window_labels_corrected * window_labels_class, axis=(1, 2)) \
                                                    / np.sum(window_labels_corrected, axis=(1, 2))
        else:
            window_white_adipocyte_prob = np.array([])
            window_white_adipocyte_prob_corrected = np.array([])

        # if no cells found, wipe out current window from tissue segmentation, and go to next iteration. Otherwise we'd
        # enter an infinite loop
        if len(index_list) == 0:  # empty segmentation

            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0

        else:  # there's at least one object in the segmentation

            if DEBUG:
                j = 0
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
                plt.imshow(todo_edge.astype(np.uint8))
                plt.title('Full res left over tissue', fontsize=16)
                plt.axis('off')
                plt.subplot(224)
                plt.imshow(lores_todo_edge.astype(np.uint8))
                plt.title('Low res left over tissue', fontsize=16)
                plt.axis('off')
                plt.tight_layout()

            # convert labels in cropped images to contours (points), and add cropping window offset so that the
            # contours are in the whole slide coordinates
            offset_xy = index_list[:, [2, 3]]  # index_list: [i, lab, x0, y0, xend, yend]
            contours = cytometer.utils.labels2contours(window_labels, offset_xy=offset_xy,
                                                       scaling_factor_xy=scaling_factor_list)
            contours_corrected = cytometer.utils.labels2contours(window_labels_corrected, offset_xy=offset_xy,
                                                                 scaling_factor_xy=scaling_factor_list)

            if DEBUG:
                enhancer = PIL.ImageEnhance.Contrast(PIL.Image.fromarray(tile))
                tile_enhanced = np.array(enhancer.enhance(enhance_contrast))
                # no overlap
                plt.clf()
                plt.imshow(tile_enhanced)
                for j in range(len(contours)):
                    plt.fill(contours[j][:, 0], contours[j][:, 1], edgecolor='C0', fill=False)
                    plt.text(contours[j][0, 0], contours[j][0, 1], str(j))

                # overlap
                plt.clf()
                plt.imshow(tile_enhanced)
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
                enhancer = PIL.ImageEnhance.Contrast(PIL.Image.fromarray(tile))
                tile_enhanced = np.array(enhancer.enhance(enhance_contrast))
                # no overlap
                plt.clf()
                plt.imshow(tile_enhanced)
                for j in range(len(contours)):
                    plt.fill(lores_contours[j][:, 0], lores_contours[j][:, 1], edgecolor='C0', fill=False)

                # overlap
                plt.clf()
                plt.imshow(tile_enhanced)
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
            # TODO: check whether the mouse is male or female, and use corresponding f_area2quantile
            contour_items = cytometer.data.aida_contour_items(lores_contours, f_area2quantile_m,
                                                              cell_prob=window_white_adipocyte_prob,
                                                              xres=xres, yres=yres)
            rectangle = (first_col, first_row, last_col - first_col, last_row - first_row)  # (x0, y0, width, height)
            rectangle_item = cytometer.data.aida_rectangle_items([rectangle,])

            # sometimes the network filesystem's server gives a ConnectionResetError. If that's the case, we try 
            if step == 1:
                # in the first step, overwrite previous annotations file, or create new one
                cytometer.data.aida_write_new_items(annotations_file, rectangle_item, mode='w', number_of_attempts=5)
                cytometer.data.aida_write_new_items(annotations_file, contour_items, mode='append_new_layer', number_of_attempts=5)
            else:
                # in next steps, add contours to previous layer
                cytometer.data.aida_write_new_items(annotations_file, rectangle_item, mode='append_to_last_layer', number_of_attempts=5)
                cytometer.data.aida_write_new_items(annotations_file, contour_items, mode='append_new_layer', number_of_attempts=5)

            # convert corrected contours to AIDA items
            contour_items_corrected = cytometer.data.aida_contour_items(lores_contours_corrected, f_area2quantile_m,
                                                                        cell_prob=window_white_adipocyte_prob_corrected,
                                                                        xres=xres, yres=yres)

            if step == 1:
                # in the first step, overwrite previous annotations file, or create new one
                cytometer.data.aida_write_new_items(annotations_corrected_file, rectangle_item, mode='w', number_of_attempts=5)
                cytometer.data.aida_write_new_items(annotations_corrected_file, contour_items_corrected, mode='append_new_layer', number_of_attempts=5)
            else:
                # in next steps, add contours to previous layer
                cytometer.data.aida_write_new_items(annotations_corrected_file, rectangle_item, mode='append_to_last_layer', number_of_attempts=5)
                cytometer.data.aida_write_new_items(annotations_corrected_file, contour_items_corrected, mode='append_new_layer', number_of_attempts=5)

            # update the tissue segmentation mask with the current window
            if np.all(lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] == lores_todo_edge):
                # if the mask remains identical, wipe out the whole window, as otherwise we'd have an
                # infinite loop
                lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0
            else:
                # if the mask has been updated, use it to update the total tissue segmentation
                lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = lores_todo_edge

        # end of "if len(index_list) == 0:"
        # Thus, regardless of whether there were any objects in the segmentation or not, here we continue the execution
        # of the program

        perc_completed = 100.0 - np.count_nonzero(lores_istissue) / np.count_nonzero(lores_istissue0) * 100
        perc_completed_all.append(perc_completed)
        time_step = time.time() - time_prev
        time_step_all.append(time_step)
        time_total = np.sum(time_step_all)

        print('File ' + str(i_file) + '/' + str(len(histo_files_list) - 1) + ': step ' +
              str(step) + ': ' +
              str(np.count_nonzero(lores_istissue)) + '/' + str(np.count_nonzero(lores_istissue0)) + ': ' +
              "{0:.1f}".format(perc_completed) +
              '% completed: ' +
              'time step ' + "{0:.2f}".format(time_step) + ' s' +
              ', total time ' + "{0:.2f}".format(time_total) + ' s')

        # save to the rough mask file
        np.savez_compressed(coarse_mask_file, lores_istissue=lores_istissue, lores_istissue0=lores_istissue0,
                            im_downsampled=im_downsampled, step=step, perc_completed_all=   perc_completed_all,
                            time_step_all=time_step_all,
                            prev_first_row=prev_first_row, prev_last_row=prev_last_row,
                            prev_first_col=prev_first_col, prev_last_col=prev_last_col)

        # clear keras session to prevent each segmentation iteration from getting slower. Note that this forces us to
        # reload the models every time, but that's not too slow
        K.clear_session()

    # end of "keep extracting histology windows until we have finished"

########################################################################################################################
## Compute colourmap for AIDA (using all automatically segmented data)
########################################################################################################################

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# auxiliary file to keep all Corrected areas
filename_corrected_areas = os.path.join(figures_dir, 'klf14_b6ntac_exp_0106_corrected_areas.npz')

if os.path.isfile(filename_corrected_areas):

    with np.load(filename_corrected_areas, allow_pickle=True) as aux:
        areas_corrected_f = aux['areas_corrected_f']
        areas_corrected_m = aux['areas_corrected_m']

else:

    areas_corrected_f = []
    areas_corrected_m = []
    for i_file, histo_file in enumerate(histo_files_list.keys()):

        print('file ' + str(i_file) + '/' + str(len(histo_files_list.keys()) - 1) + ': ' + os.path.basename(histo_file))

        # change file extension from .svg to .tif
        file_ndpi = histo_file + histology_ext
        file_ndpi = os.path.join(histology_dir, file_ndpi)

        # open histology training image
        im = openslide.OpenSlide(file_ndpi)

        # pixel size
        assert (im.properties['tiff.ResolutionUnit'] == 'centimeter')
        xres = 1e-2 / float(im.properties['tiff.XResolution']) * 1e6  # um^2
        yres = 1e-2 / float(im.properties['tiff.YResolution']) * 1e6  # um^2

        # create dataframe for this image
        df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(histo_file),
                                                              values=[i_file,], values_tag='i',
                                                              tags_to_keep=['id', 'ko_parent', 'sex'])

        # mouse ID as a string
        id = df_common['id'].values[0]
        sex = df_common['sex'].values[0]
        ko = df_common['ko_parent'].values[0]

        # load list of contours in Corrected segmentations
        json_file_corrected = os.path.join(annotations_dir, histo_file + '_exp_0106_corrected.json')
        contours_corrected = cytometer.data.aida_get_contours(json_file_corrected, layer_name='White adipocyte.*')

        # loop items (one contour per item)
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
    # areas_corrected_f = np.array(areas_corrected_f)
    # areas_corrected_m = np.array(areas_corrected_m)
    np.savez(filename_corrected_areas, areas_corrected_f=areas_corrected_f, areas_corrected_m=areas_corrected_m)

# file to store quantile-to-area functions
filename_area2quantile = os.path.join(area2quantile_dir, 'klf14_b6ntac_exp_0106_filename_area2quantile_v8.npz')

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
