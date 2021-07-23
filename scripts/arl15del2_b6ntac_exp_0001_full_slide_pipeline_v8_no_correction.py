"""
Processing full slides of Ying Bai's  with pipeline v8, without segmentation correction:

 * data generation
   * training images (*0076*)
   * non-overlap training images (*0077*)
   * augmented training images (*0078*)
   * k-folds + extra "other" for classifier (*0094*)
 * segmentation
   * dmap (*0086*)
   * contour from dmap (*0091*)
 * classifier (*0095*)
 * validation (*0096*)

Difference with pipeline v7:
  * Contrast enhancement to compute coarse tissue mask
  * Colour correction to match the median colour of the training data for segmentation
  * All segmented objects are saved, together with the white adipocyte probability score. That way, we can decide later
    which ones we want to keep, and which ones we want to reject.

Difference with rreb1_tm1b_exp_0003_pilot_full_slide_pipeline_v8.py:
  * No segmentation correction.

Difference with rreb1_tm1b_exp_0004_pilot_full_slide_pipeline_v8_no_correction.py:
  * Change filenames and dirs to point at Ying's files instead of Grace's.
  * Use colourmap in klf14_b6ntac_exp_0106_filename_area2quantile_v8.npz instead of klf14_b6ntac_exp_0098_filename_area2quantile.npz.

 Requirements for this script to work:

 1) Upload the cytometer project directory to ~/Software in the server where you are going to process the data.

 2) Run ./install_dependencies_machine.sh in cytometer.

 3) Mount the network share with the histology slides onto ~/scan_srv2_cox.

 4) Convert the .ndpi files to AIDA .dzi files, so that we can see the results of the segmentation.
    You need to go to the server that's going to process the slides, add a list of the files you want to process to
    ~/Software/cytometer/tools/arl15del2_full_histology_ndpi_to_dzi.sh

    and run

    cd ~/Software/cytometer/tools
    ./arl15del2_full_histology_ndpi_to_dzi.sh

 5) You need to have the models for the 10-folds of the pipeline that were trained on the KLF14 data in
    ~/Data/cytometer_data/klf14/saved_models.

 6) To monitor the segmentation as it's being processed, you need to have AIDA running

    cd ~/Software/AIDA/dist/
    node aidaLocal.js &

    You also need to create a soft link per .dzi file to the annotations you want to visualise for that file, whether
    the non-overlapping ones, or the corrected ones. E.g.

    ln -s 'APL15-DEL2-EM1-B6N 31.1a 696-19 Gwat 1 - 2019-08-19 12.44.49_exp_0097_corrected.json' 'APL15-DEL2-EM1-B6N 31.1a 696-19 Gwat 1 - 2019-08-19 12.44.49_exp_0001.json'

    Then you can use a browser to open the AIDA web interface by visiting the URL (note that you need to be on the MRC
    VPN, or connected from inside the office to get access to the titanrtx server)

    http://titanrtx:3000/dashboard

    You can use the interface to open a .dzi file that corresponds to an .ndpi file being segmented, and see the
    annotations (segmentation) being created for it.

"""

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

# script name to identify this experiment
experiment_id = 'apl15_del2_exp_0001_full_slide_pipeline_v8_no_correction'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
from pathlib import Path
import sys
if os.path.join(home, 'Software/cytometer') not in sys.path:
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
from cytometer.utils import rough_foreground_mask, bspline_resample
import PIL
from keras import backend as K
import scipy.stats
from shapely.geometry import Polygon

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

histology_dir = os.path.join(home, 'scan_srv2_cox/Ying Bai/For Ramon')
area2quantile_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v8')
saved_models_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v8')
annotations_dir = os.path.join(home, 'Data/cytometer_data/aida_data_Arl15_del2/annotations')

# file with area->quantile map precomputed from all automatically segmented slides in klf14_b6ntac_exp_0106_filename_area2quantile_v8.py
filename_area2quantile = os.path.join(area2quantile_dir, 'klf14_b6ntac_exp_0106_filename_area2quantile_v8.npz')

# file with RGB modes from all training data
klf14_training_colour_histogram_file = os.path.join(saved_models_dir, 'klf14_training_colour_histogram.npz')

# model names
dmap_model_basename = 'klf14_b6ntac_exp_0086_cnn_dmap'
contour_model_basename = 'klf14_b6ntac_exp_0091_cnn_contour_after_dmap'
classifier_model_basename = 'klf14_b6ntac_exp_0095_cnn_tissue_classifier_fcn'

# full resolution image window and network expected receptive field parameters
fullres_box_size = np.array([2751, 2751])
receptive_field = np.array([131, 131])

# rough_foreground_mask() parameters
downsample_factor = 8.0
dilation_size = 25
component_size_threshold = 50e3
hole_size_treshold = 8000
std_k = 1.00
enhance_contrast = 4.0

# contour parameters
contour_downsample_factor = 0.1
bspline_k = 1

# block_split() parameters in downsampled image
block_len = np.ceil((fullres_box_size - receptive_field) / downsample_factor)
block_overlap = np.ceil((receptive_field - 1) / 2 / downsample_factor).astype(int)
window_overlap_fraction_max = 0.9

# segmentation parameters
min_cell_area = 200  # pixel
max_cell_area = 200e3  # pixel
min_mask_overlap = 0.8
phagocytosis = True
min_class_prop = 0.0  # we accept all segmented objects here, as that gives as a chance to filter with different thresholds later
correction_window_len = 401
correction_smoothing = 11
batch_size = 16

# list of NDPI files to process
histo_files_list = [
'APL15-DEL2-EM1-B6N 31.1a 696-19 Gwat 1 - 2019-08-19 12.44.49.ndpi',
'APL15-DEL2-EM1-B6N 31.1a 696-19 Gwat 3 - 2019-08-19 12.57.53.ndpi',
'APL15-DEL2-EM1-B6N 31.1a 696-19 Gwat 6 - 2019-08-19 13.16.35.ndpi',
'APL15-DEL2-EM1-B6N 31.1a 696-19 Iwat 1 - 2019-08-19 13.54.09.ndpi',
'APL15-DEL2-EM1-B6N 31.1a 696-19 Iwat 3 - 2019-08-19 15.29.54.ndpi',
'APL15-DEL2-EM1-B6N 31.1a 696-19 Iwat 6 - 2019-08-19 15.52.25.ndpi',
'APL15-DEL2-EM1-B6N 32.2b 695-19 Gwat 1 - 2019-08-16 13.04.34.ndpi',
'APL15-DEL2-EM1-B6N 32.2b 695-19 Gwat 3 - 2019-08-16 13.18.24.ndpi',
'APL15-DEL2-EM1-B6N 32.2b 695-19 Gwat 6 - 2019-08-16 13.37.00.ndpi',
'APL15-DEL2-EM1-B6N 32.2b 695-19 Iwat 1 - 2019-08-19 12.01.13.ndpi',
'APL15-DEL2-EM1-B6N 32.2b 695-19 Iwat 3 - 2019-08-19 12.13.50.ndpi',
'APL15-DEL2-EM1-B6N 32.2b 695-19 Iwat 6 - 2019-08-19 12.31.59.ndpi',
'APL15-DEL2-EM1-B6N 34.1b 693-19 Gwat 1 - 2019-08-15 11.11.40.ndpi',
'APL15-DEL2-EM1-B6N 34.1b 693-19 Gwat 3 - 2019-08-15 11.24.16.ndpi',
'APL15-DEL2-EM1-B6N 34.1b 693-19 Gwat 6 - 2019-08-15 11.45.59.ndpi',
'APL15-DEL2-EM1-B6N 34.1b 693-19 Iwat 1 - 2019-08-15 11.55.14.ndpi',
'APL15-DEL2-EM1-B6N 34.1b 693-19 Iwat 3 - 2019-08-15 13.31.37.ndpi',
'APL15-DEL2-EM1-B6N 34.1b 693-19 Iwat 6 - 2019-08-15 12.22.16.ndpi',
'APL15-DEL2-EM1-B6N 35.2b 701-19 Gwat 1 - 2019-08-22 09.47.48.ndpi',
'APL15-DEL2-EM1-B6N 35.2b 701-19 Gwat 3 - 2019-08-22 10.59.34.ndpi',
'APL15-DEL2-EM1-B6N 35.2b 701-19 Gwat 6 - 2019-08-22 10.15.02.ndpi',
'APL15-DEL2-EM1-B6N 35.2b 701-19 Iwat 1 - 2019-08-22 11.12.17.ndpi',
'APL15-DEL2-EM1-B6N 35.2b 701-19 Iwat 3 - 2019-08-22 11.24.19.ndpi',
'APL15-DEL2-EM1-B6N 35.2b 701-19 Iwat 6 - 2019-08-22 11.41.44.ndpi',
'APL15-DEL2-EM1-B6N 35.2c 694-19 Gwat 1 - 2019-08-15 13.41.17.ndpi',
'APL15-DEL2-EM1-B6N 35.2c 694-19 Gwat 3 - 2019-08-15 13.48.15.ndpi',
'APL15-DEL2-EM1-B6N 35.2c 694-19 Gwat 6 - 2019-08-15 14.03.44.ndpi',
'APL15-DEL2-EM1-B6N 35.2c 694-19 Iwat 1 - 2019-08-16 11.13.41.ndpi',
'APL15-DEL2-EM1-B6N 35.2c 694-19 Iwat 3 - 2019-08-16 11.25.11.ndpi',
'APL15-DEL2-EM1-B6N 35.2c 694-19 Iwat 6 - 2019-08-16 11.42.18.ndpi',
'APL15-DEL2-EM1-B6N 37.1b 697-19 Gwat 1 - 2019-08-20 11.40.02.ndpi',
'APL15-DEL2-EM1-B6N 37.1b 697-19 Gwat 3 - 2019-08-20 12.44.02.ndpi',
'APL15-DEL2-EM1-B6N 37.1b 697-19 Gwat 6 - 2019-08-20 13.03.14.ndpi',
'APL15-DEL2-EM1-B6N 37.1b 697-19 Iwat 1 - 2019-08-20 14.08.10.ndpi',
'APL15-DEL2-EM1-B6N 37.1b 697-19 Iwat 3 - 2019-08-20 14.22.06.ndpi',
'APL15-DEL2-EM1-B6N 37.1b 697-19 Iwat 6 - 2019-08-20 14.41.48.ndpi',
'APL15-DEL2-EM1-B6N 38.1c 700-19 Gwat 1 - 2019-08-21 11.59.38.ndpi',
'APL15-DEL2-EM1-B6N 38.1c 700-19 Gwat 3 - 2019-08-21 16.23.02.ndpi',
'APL15-DEL2-EM1-B6N 38.1c 700-19 Gwat 6 - 2019-08-21 16.43.02.ndpi',
'APL15-DEL2-EM1-B6N 38.1c 700-19 Iwat 1 - 2019-08-22 08.36.58.ndpi',
'APL15-DEL2-EM1-B6N 38.1c 700-19 Iwat 3 - 2019-08-22 08.49.27.ndpi',
'APL15-DEL2-EM1-B6N 38.1c 700-19 Iwat 6 - 2019-08-22 09.08.29.ndpi',
'APL15-DEL2-EM1-B6N 38.1d 698-19 Gwat 1 - 2019-08-20 15.05.02.ndpi',
'APL15-DEL2-EM1-B6N 38.1d 698-19 Gwat 3 - 2019-08-20 15.18.30.ndpi',
'APL15-DEL2-EM1-B6N 38.1d 698-19 Gwat 6 - 2019-08-20 15.37.51.ndpi',
'APL15-DEL2-EM1-B6N 38.1d 698-19 Iwat 1 - 2019-08-21 08.59.51.ndpi',
'APL15-DEL2-EM1-B6N 38.1d 698-19 Iwat 3 - 2019-08-21 09.13.35.ndpi',
'APL15-DEL2-EM1-B6N 38.1d 698-19 Iwat 6 - 2019-08-21 09.32.24.ndpi',
'APL15-DEL2-EM1-B6N 38.1e 699-19 Gwat 1 - 2019-08-21 09.44.01.ndpi',
'APL15-DEL2-EM1-B6N 38.1e 699-19 Gwat 3 - 2019-08-21 09.57.22.ndpi',
'APL15-DEL2-EM1-B6N 38.1e 699-19 Gwat 6 - 2019-08-21 10.15.39.ndpi',
'APL15-DEL2-EM1-B6N 38.1e 699-19 Iwat 1 - 2019-08-21 11.16.47.ndpi',
'APL15-DEL2-EM1-B6N 38.1e 699-19 Iwat 3 - 2019-08-21 11.29.58.ndpi',
'APL15-DEL2-EM1-B6N 38.1e 699-19 Iwat 6 - 2019-08-21 11.49.53.ndpi',
'APL15-DEL2_EM1_B6N 34.1e 692-19 Gwat 1 - 2019-08-14 16.48.21.ndpi',
'APL15-DEL2_EM1_B6N 34.1e 692-19 Gwat 3 - 2019-08-14 17.09.59.ndpi',
'APL15-DEL2_EM1_B6N 34.1e 692-19 Gwat 6 - 2019-08-14 17.28.37.ndpi',
'APL15-DEL2_EM1_B6N 34.1e 692-19 Iwat 1 - 2019-08-15 09.49.41.ndpi',
'APL15-DEL2_EM1_B6N 34.1e 692-19 Iwat 3 - 2019-08-15 10.02.15.ndpi',
'APL15-DEL2_EM1_B6N 34.1e 692-19 Iwat 6 - 2019-08-15 10.20.19.ndpi',
'APL15-DEL2_EM1_B6N 35.1a 690-19 Gwat 1 - 2019-08-14 12.06.19.ndpi',
'APL15-DEL2_EM1_B6N 35.1a 690-19 Gwat 3 - 2019-08-14 12.22.40.ndpi',
'APL15-DEL2_EM1_B6N 35.1a 690-19 Gwat 6 - 2019-08-14 12.42.00.ndpi',
'APL15-DEL2_EM1_B6N 35.1a 690-19 Iwat 1 - 2019-08-14 12.53.04.ndpi',
'APL15-DEL2_EM1_B6N 35.1a 690-19 Iwat 3 - 2019-08-14 13.04.25.ndpi',
'APL15-DEL2_EM1_B6N 35.1a 690-19 Iwat 6 - 2019-08-14 13.22.36.ndpi',
'APL15-DEL2_EM1_B6N 38.1a 691-19 Gwat 1 - 2019-08-14 13.48.53.ndpi',
'APL15-DEL2_EM1_B6N 38.1a 691-19 Gwat 3 - 2019-08-14 14.02.42.ndpi',
'APL15-DEL2_EM1_B6N 38.1a 691-19 Gwat 6 - 2019-08-14 14.24.10.ndpi',
'APL15-DEL2_EM1_B6N 38.1a 691-19 Iwat 1 - 2019-08-14 15.34.14.ndpi',
'APL15-DEL2_EM1_B6N 38.1a 691-19 Iwat 3 - 2019-08-14 15.44.04.ndpi',
'APL15-DEL2_EM1_B6N 38.1a 691-19 Iwat 6 - 2019-08-14 15.59.53.ndpi',
'ARL15-DEL2-EM1-B6N-29.1E Gwat 689-19 1 - 2019-08-08 11.51.43.ndpi',
'ARL15-DEL2-EM1-B6N-29.1E Gwat 689-19 3 - 2019-08-08 12.04.20.ndpi',
'ARL15-DEL2-EM1-B6N-29.1E Gwat 689-19 6 - 2019-08-08 12.24.13.ndpi',
'ARL15-DEL2-EM1-B6N-29.1E Iwat 689-19 1 - 2019-08-08 12.33.17.ndpi',
'ARL15-DEL2-EM1-B6N-29.1E Iwat 689-19 3 - 2019-08-12 16.08.24.ndpi',
'ARL15-DEL2-EM1-B6N-29.1E Iwat 689-19 6 - 2019-08-12 16.30.33.ndpi',
'ARL15-DEL2-EM1-B6N-32.1f Gwat 688-19 1 - 2019-08-08 10.07.15.ndpi',
'ARL15-DEL2-EM1-B6N-32.1f Gwat 688-19 3 - 2019-08-08 10.23.58.ndpi',
'ARL15-DEL2-EM1-B6N-32.1f Gwat 688-19 6 - 2019-08-08 10.46.56.ndpi',
'ARL15-DEL2-EM1-B6N-32.1f Iwat 688-19 1 - 2019-08-08 11.13.12.ndpi',
'ARL15-DEL2-EM1-B6N-32.1f Iwat 688-19 3 - 2019-08-08 11.31.23.ndpi',
'ARL15-DEL2-EM1-B6N-32.1f Iwat 688-19 6 - 2019-08-08 11.43.11.ndpi'
]

# load colour modes of the KLF14 training dataset
with np.load(klf14_training_colour_histogram_file) as data:
    mode_r_klf14 = data['mode_r']
    mode_g_klf14 = data['mode_g']
    mode_b_klf14 = data['mode_b']

########################################################################################################################
## Colourmap for AIDA, based on KLF14 automatically segmented data
########################################################################################################################

if os.path.isfile(filename_area2quantile):
    with np.load(filename_area2quantile, allow_pickle=True) as aux:
        f_area2quantile_f = aux['f_area2quantile_f']
        f_area2quantile_m = aux['f_area2quantile_m']
else:
    raise FileNotFoundError('Cannot find file with area->quantile map precomputed from all automatically segmented' +
                            ' slides in klf14_b6ntac_exp_0106_filename_area2quantile_v8.py')

# load AIDA's colourmap
cm = cytometer.data.aida_colourmap()

########################################################################################################################
## Segmentation loop
########################################################################################################################

for i_file, histo_file in enumerate(histo_files_list):

    print('File ' + str(i_file) + '/' + str(len(histo_files_list) - 1) + ': ' + histo_file)

    # make full path to ndpi file
    histo_file = os.path.join(histology_dir, histo_file)

    # check whether there's a lock on this file
    lock_file = os.path.basename(histo_file).replace('.ndpi', '.lock')
    lock_file = os.path.join(annotations_dir, lock_file)
    if os.path.isfile(lock_file):
        print('Lock on file, skipping')
        continue
    else:
        # create an empty lock file to prevent other other instances of the script to process the same .ndpi file
        Path(lock_file).touch()

    # choose a random fold for this image
    np.random.seed(i_file)
    i_fold = np.random.randint(0, 10)

    contour_model_file = os.path.join(saved_models_dir, contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    dmap_model_file = os.path.join(saved_models_dir, dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    classifier_model_file = os.path.join(saved_models_dir,
                                         classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')

    # name of file to save annotations to
    annotations_file = os.path.basename(histo_file)
    annotations_file = os.path.splitext(annotations_file)[0]
    annotations_file = os.path.join(annotations_dir, annotations_file + '_exp_0004_auto.json')

    # name of file to save coarse mask, current mask, and time steps
    coarse_mask_file = os.path.basename(histo_file)
    coarse_mask_file = coarse_mask_file.replace('.ndpi', '_coarse_mask.npz')
    coarse_mask_file = os.path.join(annotations_dir, coarse_mask_file)

    # open full resolution histology slide
    im = openslide.OpenSlide(histo_file)

    # pixel size
    assert(im.properties['tiff.ResolutionUnit'] == 'centimeter')
    xres = 1e-2 / float(im.properties['tiff.XResolution'])
    yres = 1e-2 / float(im.properties['tiff.YResolution'])

    # check whether we continue previous execution, or we start a new one
    continue_previous = os.path.isfile(coarse_mask_file)

    # if the coarse mask has been pre-computed, just load it
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

        # compute the coarse foreground mask of tissue vs. background
        lores_istissue0, im_downsampled = rough_foreground_mask(histo_file, downsample_factor=downsample_factor,
                                                                dilation_size=dilation_size,
                                                                component_size_threshold=component_size_threshold,
                                                                hole_size_treshold=hole_size_treshold, std_k=std_k,
                                                                return_im=True, enhance_contrast=enhance_contrast,
                                                                ignore_white_threshold=253)

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

        # save to the coarse mask file
        np.savez_compressed(coarse_mask_file, lores_istissue=lores_istissue, lores_istissue0=lores_istissue0,
                            im_downsampled=im_downsampled, step=step, perc_completed_all=perc_completed_all,
                            prev_first_row=prev_first_row, prev_last_row=prev_last_row,
                            prev_first_col=prev_first_col, prev_last_col=prev_last_col,
                            time_step_all=time_step_all)

        # end "computing the coarse foreground mask"

    # checkpoint: here the coarse tissue mask has either been loaded or computed
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

    # estimate the colour mode of the downsampled image, so that we can correct the image tint to match the KLF14
    # training dataset. We apply the same correction to each tile, to avoid that a tile with e.g. only muscle gets
    # overcorrected
    mode_r_rrbe1 = scipy.stats.mode(im_downsampled[:, :, 0], axis=None).mode[0]
    mode_g_rrbe1 = scipy.stats.mode(im_downsampled[:, :, 1], axis=None).mode[0]
    mode_b_rrbe1 = scipy.stats.mode(im_downsampled[:, :, 2], axis=None).mode[0]

    # keep extracting histology windows until we have finished
    while np.count_nonzero(lores_istissue) > 0:

        time_prev = time.time()

        # next step (it starts from 1 here, because step 0 is the coarse mask computation)
        step += 1

        # get indices for the next histology window to process
        (first_row, last_row, first_col, last_col), \
        (lores_first_row, lores_last_row, lores_first_col, lores_last_col) = \
            cytometer.utils.get_next_roi_to_process(lores_istissue, downsample_factor=downsample_factor,
                                                    max_window_size=fullres_box_size,
                                                    border=np.round((receptive_field-1)/2),
                                                    version='old')

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
        tile[:, :, 0] = tile[:, :, 0] + (mode_r_klf14 - mode_r_rrbe1)
        tile[:, :, 1] = tile[:, :, 1] + (mode_g_klf14 - mode_g_rrbe1)
        tile[:, :, 2] = tile[:, :, 2] + (mode_b_klf14 - mode_b_rrbe1)

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
                                                     correction_model=None,
                                                     classifier_model=classifier_model_file,
                                                     min_cell_area=0,
                                                     max_cell_area=np.inf,
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
        else:
            window_white_adipocyte_prob = np.array([])

        # if no cells found, wipe out current window from tissue segmentation, and go to next iteration. Otherwise we'd
        # enter an infinite loop
        if len(index_list) == 0:  # empty segmentation

            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0

        else:  # there's at least one object in the segmentation

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

            # downsample "to do" mask so that the coarse tissue segmentation can be updated
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

            if DEBUG:
                # no overlap
                plt.clf()
                plt.imshow(tile)
                for j in range(len(contours)):
                    plt.fill(contours[j][:, 0], contours[j][:, 1], edgecolor='C0', fill=False)
                    # plt.text(contours[j][0, 0], contours[j][0, 1], str(j))

            # downsample contours for AIDA annotations file
            lores_contours = []
            for c in contours:
                lores_c = bspline_resample(c, factor=contour_downsample_factor, min_n=10, k=bspline_k, is_closed=True)
                lores_contours.append(lores_c)

            if DEBUG:
                # no overlap
                plt.clf()
                plt.imshow(tile)
                for j in range(len(contours)):
                    plt.fill(lores_contours[j][:, 0], lores_contours[j][:, 1], edgecolor='C1', fill=False)

            # add tile offset, so that contours are in full slide coordinates
            for j in range(len(contours)):
                lores_contours[j][:, 0] += first_col
                lores_contours[j][:, 1] += first_row

            # convert non-overlap contours to AIDA items
            # TODO: check whether the mouse is male or female, and use corresponding f_area2quantile
            contour_items = cytometer.data.aida_contour_items(lores_contours, f_area2quantile_m.item(),
                                                              cell_prob=window_white_adipocyte_prob,
                                                              xres=xres*1e6, yres=yres*1e6)
            rectangle = (first_col, first_row, last_col - first_col, last_row - first_row)  # (x0, y0, width, height)
            rectangle_item = cytometer.data.aida_rectangle_items([rectangle,])

            if step == 1:
                # in the first step, overwrite previous annotations file, or create new one
                cytometer.data.aida_write_new_items(annotations_file, rectangle_item, mode='w')
                cytometer.data.aida_write_new_items(annotations_file, contour_items, mode='append_new_layer')
            else:
                # in next steps, add contours to previous layer
                cytometer.data.aida_write_new_items(annotations_file, rectangle_item, mode='append_to_last_layer')
                cytometer.data.aida_write_new_items(annotations_file, contour_items, mode='append_new_layer')

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

        # save to the coarse mask file
        np.savez_compressed(coarse_mask_file, lores_istissue=lores_istissue, lores_istissue0=lores_istissue0,
                            im_downsampled=im_downsampled, step=step, perc_completed_all=perc_completed_all,
                            time_step_all=time_step_all,
                            prev_first_row=prev_first_row, prev_last_row=prev_last_row,
                            prev_first_col=prev_first_col, prev_last_col=prev_last_col)

        # clear keras session to prevent each segmentation iteration from getting slower. Note that this forces us to
        # reload the models every time, but that's not too slow
        K.clear_session()

    # end of "keep extracting histology windows until we have finished"
