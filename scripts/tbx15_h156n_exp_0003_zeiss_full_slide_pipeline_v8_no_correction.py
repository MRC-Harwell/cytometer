"""
Processing Louisa Zolkiewski's Zeiss scanner TBX15-H156N-IC whole slides with pipeline v8, without segmentation correction:

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
  * Contrast enhancement to compute rough tissue mask
  * Colour correction to match the median colour of the training data for segmentation
  * All segmented objects are saved, together with the white adipocyte probability score. That way, we can decide later
    which ones we want to keep, and which ones we want to reject.

Difference with rreb1_tm1b_exp_0003_pilot_full_slide_pipeline_v8.py:
  * No segmentation correction.

Difference with rreb1_tm1b_exp_0004_pilot_full_slide_pipeline_v8_no_correction.py:
  * Applied to .tif files converted from Zeiss .czi files instead of Hamamatsu .ndpi files.

Difference with rreb1_tm1b_exp_0007_zeiss_full_slide_pipeline_v8_no_correction.py:
  * Applied to Louisa's data instead of Grace Yu's data.
  * Louisa's images are scanned with 10x instead of 5x magnification, so we use 32x downsampling for the coarse image
    segmentation and 2x downsampling for segmentation. That corresponds to pixel sizes similar to what we've done with
    Klf14 data.

 Requirements for this script to work:

 1) Upload the cytometer project directory to ~/Software in the server where you are going to process the data.

 2) Run ./install_dependencies_machine.sh in cytometer.

 3) Mount the network share that contains the histology to ~/coxgroup_zeiss_test.

 4) Convert the .czi files to .dzi and .tif files, so that we can see the results of the segmentation and open the
    images with openslide, respectively.
    You need to go to the server that's going to process the slides, add a list of the files you want to process to
    ~/Software/cytometer/scripts/tbx15_h156n_exp_0001_convert_zeiss_histology_to_deepzoom.py

    and run

    cd ~/Software/cytometer/scripts
    ./rreb1_tm1b_exp_0006_convert_zeiss_histology_to_deepzoom.sh

 5) You need to have the models for the 10-folds of the pipeline that were trained on the KLF14 data in
    ~/Data/cytometer_data/deepcytometer_pipeline_v8.

 6) To monitor the segmentation as it's being processed, you need to have AIDA running

    cd ~/Software/AIDA/dist/
    node aidaLocal.js &

    You also need to create a soft link per .dzi file to the annotations you want to visualise for that file, whether
    the non-overlapping ones, or the corrected ones. E.g.

    ln -s 'RREB1-TM1B-B6N-IC-1.1a  1132-18 G1 - 2018-11-16 14.58.55_exp_0097_corrected.json' 'RREB1-TM1B-B6N-IC-1.1a  1132-18 G1 - 2018-11-16 14.58.55_exp_0097.json'

    Then you can use a browser to open the AIDA web interface by visiting the URL (note that you need to be on the MRC
    VPN, or connected from inside the office to get access to the titanrtx server)

    http://titanrtx:3000/dashboard

    You can use the interface to open a .dzi file that corresponds to a histology file being segmented, and see the
    annotations (segmentation) being created for it.

"""

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

# script name to identify this experiment
experiment_id = 'tbx15_h156n_exp_0003_zeiss_full_slide_pipeline_v8_no_correction'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import warnings
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
import cv2

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

histology_dir = os.path.join(home, 'coxgroup_zeiss_test/tif')
area2quantile_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v8')
saved_models_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v8')
annotations_dir = os.path.join(home, 'Data/cytometer_data/aida_data_Tbx15/annotations')

# file with functions that map ECDF probabilities to intensity values
klf14_training_colour_histogram_file = os.path.join(saved_models_dir, 'klf14_exp_0112_training_colour_histogram.npz')

# file with area->quantile map precomputed from all automatically segmented slides in klf14_b6ntac_exp_0106_filename_area2quantile_v8.py
filename_area2quantile = os.path.join(area2quantile_dir, 'klf14_b6ntac_exp_0106_filename_area2quantile_v8.npz')

# model names
dmap_model_basename = 'klf14_b6ntac_exp_0086_cnn_dmap'
contour_model_basename = 'klf14_b6ntac_exp_0091_cnn_contour_after_dmap'
classifier_model_basename = 'klf14_b6ntac_exp_0095_cnn_tissue_classifier_fcn'
correction_model_basename = 'klf14_b6ntac_exp_0089_cnn_segmentation_correction_overlapping_scaled_contours'

# full resolution image window and network expected receptive field parameters
fullres_box_size = np.array([2751, 2751])
receptive_field = np.array([131, 131])

# rough_foreground_mask() parameters
downsample_factor = 32.0
dilation_size = 25
component_size_threshold = 50e3
hole_size_treshold = 8000
std_k = 1.00
enhance_contrast = 4.0
ignore_white_threshold = 253

# contour parameters
contour_downsample_factor = 0.1
bspline_k = 1

# block_split() parameters in downsampled image
block_len = np.ceil((fullres_box_size - receptive_field) / downsample_factor)
block_overlap = np.ceil((receptive_field - 1) / 2 / downsample_factor).astype(np.int64)
window_overlap_fraction_max = 0.9

# segmentation parameters
min_cell_area = 200  # pixel; we want all small objects
max_cell_area = 200e3  # pixel
min_mask_overlap = 0.8
phagocytosis = True
min_class_prop = 0.0  # we accept all segmented objects here, as that gives as a chance to filter with different thresholds later
correction_window_len = 401
correction_smoothing = 11
batch_size = 8

# list of histology files to process
histo_files_list = [
    'TBX15-H156N-IC-0001-15012021.tif',
    'TBX15-H156N-IC-0002-15012021.tif',
    'TBX15-H156N-IC-0003-15012021.tif',
    'TBX15-H156N-IC-0004-15012021.tif',
    'TBX15-H156N-IC-0005-15012021.tif',
    'TBX15-H156N-IC-0006-15012021.tif',
    'TBX15-H156N-IC-0007-15012021.tif',
]

# load functions to map ECDF quantiles to intensity values for the Klf14 training dataset
with np.load(klf14_training_colour_histogram_file, allow_pickle=True) as data:
    f_ecdf_to_val_r_klf14 = data['f_ecdf_to_val_r_klf14'].item()
    f_ecdf_to_val_g_klf14 = data['f_ecdf_to_val_g_klf14'].item()
    f_ecdf_to_val_b_klf14 = data['f_ecdf_to_val_b_klf14'].item()

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

    # make full path to histology file
    histo_file = os.path.join(histology_dir, histo_file)

    # check whether there's a lock on this file
    lock_file = os.path.basename(histo_file).replace('.tif', '.lock')
    lock_file = os.path.join(annotations_dir, lock_file)
    if os.path.isfile(lock_file):
        print('Lock on file, skipping')
        continue
    else:
        # create an empty lock file to prevent other other instances of the script to process the same histology file
        Path(lock_file).touch()

    # choose a random fold for this image
    np.random.seed(i_file)
    i_fold = np.random.randint(0, 10)

    contour_model_file = os.path.join(saved_models_dir, contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    dmap_model_file = os.path.join(saved_models_dir, dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    classifier_model_file = os.path.join(saved_models_dir,
                                         classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    correction_model_file = os.path.join(saved_models_dir,
                                         correction_model_basename + '_model_fold_' + str(i_fold) + '.h5')

    # name of file to save annotations to
    annotations_file = os.path.basename(histo_file)
    annotations_file = os.path.splitext(annotations_file)[0]
    annotations_file = os.path.join(annotations_dir, annotations_file + '_exp_0007_auto.json')

    # name of file to save rough mask, current mask, and time steps
    rough_mask_file = os.path.basename(histo_file)
    rough_mask_file = rough_mask_file.replace('.tif', '_rough_mask.npz')
    rough_mask_file = os.path.join(annotations_dir, rough_mask_file)

    # open full resolution histology slide
    im = openslide.OpenSlide(histo_file)

    # pixel size at level 1 (the level at which we segment the histology)
    lvl = 1
    assert(im.properties['tiff.ResolutionUnit'] == 'centimeter')
    xres = 1e-2 / float(im.properties['tiff.XResolution']) * im.level_downsamples[lvl] # m, e.g. 4.4e-07
    yres = 1e-2 / float(im.properties['tiff.YResolution']) * im.level_downsamples[lvl]  # m, e.g. 4.4e-07
    # xres = 0.43972097231744584 um
    if (xres < 0.4e-6) or (xres > 0.5e-6):
        warnings.warn('xres is not in [0.4, 0.5] um')
    # yres = 0.43972097231744584 um
    if (yres < 0.4e-6) or (yres > 0.5e-6):
        warnings.warn('yres is not in [0.4, 0.5] um')

    # check whether we continue previous execution, or we start a new one
    continue_previous = os.path.isfile(rough_mask_file)

    # if the rough mask has been pre-computed, just load it
    if continue_previous:

        with np.load(rough_mask_file) as aux:
            tissue_mask_l5 = aux['lores_istissue']
            tissue_mask0_l5 = aux['lores_istissue0']
            im_l5 = aux['im_downsampled']
            step = aux['step'].item()
            perc_completed_all = list(aux['perc_completed_all'])
            time_step_all = list(aux['time_step_all'])
            prev_first_row_l5 = aux['prev_first_row'].item()
            prev_last_row_l5 = aux['prev_last_row'].item()
            prev_first_col_l5 = aux['prev_first_col'].item()
            prev_last_col_l5 = aux['prev_last_col'].item()

    else:

        time_prev = time.time()

        # compute the coarse foreground mask of tissue vs. background
        tissue_mask0_l5, im_l5 = rough_foreground_mask(histo_file, downsample_factor=downsample_factor,
                                                       dilation_size=dilation_size,
                                                       component_size_threshold=component_size_threshold,
                                                       hole_size_treshold=hole_size_treshold, std_k=std_k,
                                                       return_im=True, enhance_contrast=enhance_contrast,
                                                       clear_border=[1, 0, 1, 0],
                                                       ignore_white_threshold=ignore_white_threshold,
                                                       ignore_black_threshold=0,
                                                       ignore_violet_border=None)

        if DEBUG:
            # enhancer = PIL.ImageEnhance.Contrast(PIL.Image.fromarray(im_downsampled))
            # im_downsampled_enhanced = np.array(enhancer.enhance(enhance_contrast))
            plt.clf()
            plt.subplot(211)
            plt.imshow(im_l5)
            plt.axis('off')
            plt.subplot(212)
            plt.imshow(im_l5)
            plt.contour(tissue_mask0_l5)
            plt.axis('off')

        # segmentation copy, to keep track of what's left to do
        tissue_mask_l5 = tissue_mask0_l5.copy()

        # initialize block algorithm variables
        step = 0
        perc_completed_all = [float(0.0),]
        time_step = time.time() - time_prev
        time_step_all = [time_step,]
        (prev_first_row_l5, prev_last_row_l5, prev_first_col_l5, prev_last_col_l5) = (0, 0, 0, 0)

        # save to the rough mask file
        np.savez_compressed(rough_mask_file, lores_istissue=tissue_mask_l5, lores_istissue0=tissue_mask0_l5,
                            im_downsampled=im_l5, step=step, perc_completed_all=perc_completed_all,
                            prev_first_row=prev_first_row_l5, prev_last_row=prev_last_row_l5,
                            prev_first_col=prev_first_col_l5, prev_last_col=prev_last_col_l5,
                            time_step_all=time_step_all)

        # end "computing the rough foreground mask"

    # checkpoint: here the rough tissue mask has either been loaded or computed
    time_step = time_step_all[-1]
    time_total = np.sum(time_step_all)
    print('File ' + str(i_file) + '/' + str(len(histo_files_list) - 1) + ': step ' +
          str(step) + ': ' +
          str(np.count_nonzero(tissue_mask_l5)) + '/' + str(np.count_nonzero(tissue_mask0_l5)) + ': ' +
          "{0:.1f}".format(100.0 - np.count_nonzero(tissue_mask_l5) / np.count_nonzero(tissue_mask0_l5) * 100) +
          '% completed: ' +
          'time step ' + "{0:.2f}".format(time_step) + ' s' +
          ', total time ' + "{0:.2f}".format(time_total) + ' s')

    if DEBUG:
            plt.clf()
            plt.subplot(211)
            plt.imshow(im_l5)
            plt.contour(tissue_mask0_l5, colors='k')
            plt.subplot(212)
            plt.imshow(tissue_mask0_l5)

    # compute ECDF of whole slide intensity pixel colours (one ECDF per channel)
    # compute the intensity values that correspond to each quantile of an ECDF per colour channel
    non_black_mask = np.prod(im_l5 <= 0, axis=2) == 0
    p = np.linspace(0.0, 1.0, 101)
    k = 10  # step for subsampling the image data
    val_r_im_l5 = scipy.stats.mstats.hdquantiles(im_l5[:, :, 0][non_black_mask][::10], prob=p, axis=0)
    val_g_im_l5 = scipy.stats.mstats.hdquantiles(im_l5[:, :, 1][non_black_mask][::10], prob=p, axis=0)
    val_b_im_l5 = scipy.stats.mstats.hdquantiles(im_l5[:, :, 2][non_black_mask][::10], prob=p, axis=0)
    f_val_to_ecdf_r_im_l5 = scipy.interpolate.interp1d(val_r_im_l5, p, fill_value=(0.0, 1.0), bounds_error=False)
    f_val_to_ecdf_g_im_l5 = scipy.interpolate.interp1d(val_g_im_l5, p, fill_value=(0.0, 1.0), bounds_error=False)
    f_val_to_ecdf_b_im_l5 = scipy.interpolate.interp1d(val_b_im_l5, p, fill_value=(0.0, 1.0), bounds_error=False)

    # colour correction of slide
    im_l5_r_corrected = f_ecdf_to_val_r_klf14(f_val_to_ecdf_r_im_l5(im_l5[:, :, 0][non_black_mask]))
    im_l5_g_corrected = f_ecdf_to_val_g_klf14(f_val_to_ecdf_g_im_l5(im_l5[:, :, 1][non_black_mask]))
    im_l5_b_corrected = f_ecdf_to_val_b_klf14(f_val_to_ecdf_b_im_l5(im_l5[:, :, 2][non_black_mask]))
    val_r_im_l5_corrected = scipy.stats.mstats.hdquantiles(im_l5_r_corrected[::10], prob=p, axis=0)
    val_g_im_l5_corrected = scipy.stats.mstats.hdquantiles(im_l5_g_corrected[::10], prob=p, axis=0)
    val_b_im_l5_corrected = scipy.stats.mstats.hdquantiles(im_l5_b_corrected[::10], prob=p, axis=0)

    if DEBUG:
        with np.load(klf14_training_colour_histogram_file, allow_pickle=True) as data:
            p = data['p']
            val_r_klf14 = data['val_r_klf14']
            val_g_klf14 = data['val_g_klf14']
            val_b_klf14 = data['val_b_klf14']

        plt.clf()
        plt.subplot(311)
        plt.plot(val_r_klf14, p)
        plt.plot(val_g_klf14, p)
        plt.plot(val_b_klf14, p)
        plt.title('Klf14 training dataset')
        plt.subplot(312)
        plt.plot(val_r_im_l5, p)
        plt.plot(val_g_im_l5, p)
        plt.plot(val_b_im_l5, p)
        plt.title('Slide before colour correction')
        plt.subplot(313)
        plt.plot(val_r_im_l5_corrected, p)
        plt.plot(val_g_im_l5_corrected, p)
        plt.plot(val_b_im_l5_corrected, p)
        plt.title('Slide after colour correction')

    # keep extracting histology windows until we have finished
    while np.count_nonzero(tissue_mask_l5) > 0:

        time_prev = time.time()

        # next step (it starts from 1 here, because step 0 is the rough mask computation)
        step += 1

        # get location and size of ROI for each level, so that we can pass it to read_region()
        l1l5_downsample_factor = im.level_downsamples[5] / im.level_downsamples[1]
        location_all, size_all = \
            cytometer.utils.get_next_roi_to_process(tissue_mask_l5, im=im,
                                                    max_window_size=np.round(fullres_box_size / l1l5_downsample_factor),
                                                    border=np.ceil((receptive_field - 1) / 2 / l1l5_downsample_factor))

        # make notation clearer
        lvl = 0
        (first_col_l0, last_col_l0) = [location_all[lvl][0], location_all[lvl][0] + size_all[lvl][0]]
        (first_row_l0, last_row_l0) = [location_all[lvl][1], location_all[lvl][1] + size_all[lvl][1]]
        lvl = 1
        (first_col_l1, last_col_l1) = [location_all[lvl][0], location_all[lvl][0] + size_all[lvl][0]]
        (first_row_l1, last_row_l1) = [location_all[lvl][1], location_all[lvl][1] + size_all[lvl][1]]
        lvl = 5
        (first_col_l5, last_col_l5) = [location_all[lvl][0], location_all[lvl][0] + size_all[lvl][0]]
        (first_row_l5, last_row_l5) = [location_all[lvl][1], location_all[lvl][1] + size_all[lvl][1]]

        # overlap between current and previous window, as a fraction of current window area
        current_window = Polygon([(first_col_l5, first_row_l5), (last_col_l5, first_row_l5),
                                  (last_col_l5, last_row_l5), (first_col_l5, last_row_l5)])
        prev_window = Polygon([(prev_first_col_l5, prev_first_row_l5), (prev_last_col_l5, prev_first_row_l5),
                               (prev_last_col_l5, prev_last_row_l5), (prev_first_col_l5, prev_last_row_l5)])
        window_overlap_fraction = current_window.intersection(prev_window).area / current_window.area

        # check that we are not trying to process almost the same window
        if window_overlap_fraction > window_overlap_fraction_max:
            # if we are trying to process almost the same window as in the previous step, what's probably happening is
            # that we have some big labels on the edges that are not white adipocytes, and the segmentation algorithm is
            # also finding one or more spurious labels within the window. That prevents the whole lores_istissue window
            # from being wiped out, and the big edge labels keep the window selection being almost the same. Thus, we
            # wipe it out and move to another tissue area
            tissue_mask_l5[first_row_l5:last_row_l5, first_col_l5:last_col_l5] = 0
            continue

        else:
            # remember processed window for next step
            (prev_first_row_l5, prev_last_row_l5, prev_first_col_l5, prev_last_col_l5) = (first_row_l5, last_row_l5, first_col_l5, last_col_l5)

        # load histology window
        tile = im.read_region(location=(first_col_l0, first_row_l0), level=1, size=size_all[1])
        tile = np.array(tile)
        tile = tile[:, :, 0:3]

        if DEBUG:
            plt.clf()
            plt.subplot(211)
            plt.imshow(tissue_mask_l5[first_row_l5:last_row_l5, first_col_l5:last_col_l5])
            plt.subplot(212)
            plt.imshow(tile)

        # correct colours so that they match the training data
        non_black_mask = np.prod(tile <= 0, axis=2) == 0
        tile[:, :, 0][non_black_mask] = f_ecdf_to_val_r_klf14(f_val_to_ecdf_r_im_l5(tile[:, :, 0][non_black_mask]))
        tile[:, :, 1][non_black_mask] = f_ecdf_to_val_g_klf14(f_val_to_ecdf_g_im_l5(tile[:, :, 1][non_black_mask]))
        tile[:, :, 2][non_black_mask] = f_ecdf_to_val_b_klf14(f_val_to_ecdf_b_im_l5(tile[:, :, 2][non_black_mask]))

        if DEBUG:
            non_black_mask = np.prod(tile <= 0, axis=2) == 0
            plt.subplot(312)
            plt.cla()
            val_r_tile_corrected = scipy.stats.mstats.hdquantiles(tile[:, :, 0][non_black_mask], prob=p, axis=0)
            val_g_tile_corrected = scipy.stats.mstats.hdquantiles(tile[:, :, 1][non_black_mask], prob=p, axis=0)
            val_b_tile_corrected = scipy.stats.mstats.hdquantiles(tile[:, :, 2][non_black_mask], prob=p, axis=0)
            plt.plot(val_r_tile_corrected, p)
            plt.plot(val_g_tile_corrected, p)
            plt.plot(val_b_tile_corrected, p)

        # upsample tissue mask window to tile resolution
        tissue_mask_tile = tissue_mask_l5[first_row_l5:last_row_l5, first_col_l5:last_col_l5]
        tissue_mask_tile = cytometer.utils.resize(tissue_mask_tile, size=size_all[1], resample=PIL.Image.NEAREST)

        if DEBUG:
            plt.clf()
            plt.imshow(tile)
            plt.imshow(tissue_mask_tile, alpha=0.5)
            plt.contour(tissue_mask_tile, colors='k')
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
                                                     mask=tissue_mask_tile,
                                                     min_mask_overlap=min_mask_overlap,
                                                     phagocytosis=phagocytosis,
                                                     min_class_prop=min_class_prop,
                                                     correction_window_len=correction_window_len,
                                                     correction_smoothing=correction_smoothing,
                                                     remove_edge_labels=True,
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

            tissue_mask_l5[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0

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

            # downsample "to do" mask so that the rough tissue segmentation can be updated
            lores_todo_edge = PIL.Image.fromarray(todo_edge.astype(np.uint8))
            lores_todo_edge = lores_todo_edge.resize((lores_last_col - lores_first_col,
                                                      lores_last_row - lores_first_row),
                                                     resample=PIL.Image.NEAREST)
            lores_todo_edge = np.array(lores_todo_edge)

            if DEBUG:
                plt.clf()
                plt.subplot(221)
                plt.imshow(tissue_mask_l5[lores_first_row:lores_last_row, lores_first_col:lores_last_col])
                plt.title('Low res tissue mask', fontsize=16)
                plt.axis('off')
                plt.subplot(222)
                plt.imshow(tissue_mask_tile)
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
            if np.all(tissue_mask_l5[lores_first_row:lores_last_row, lores_first_col:lores_last_col] == lores_todo_edge):
                # if the mask remains identical, wipe out the whole window, as otherwise we'd have an
                # infinite loop
                tissue_mask_l5[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0
            else:
                # if the mask has been updated, use it to update the total tissue segmentation
                tissue_mask_l5[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = lores_todo_edge

        # end of "if len(index_list) == 0:"
        # Thus, regardless of whether there were any objects in the segmentation or not, here we continue the execution
        # of the program

        perc_completed = 100.0 - np.count_nonzero(tissue_mask_l5) / np.count_nonzero(tissue_mask0_l5) * 100
        perc_completed_all.append(perc_completed)
        time_step = time.time() - time_prev
        time_step_all.append(time_step)
        time_total = np.sum(time_step_all)

        print('File ' + str(i_file) + '/' + str(len(histo_files_list) - 1) + ': step ' +
              str(step) + ': ' +
              str(np.count_nonzero(tissue_mask_l5)) + '/' + str(np.count_nonzero(tissue_mask0_l5)) + ': ' +
              "{0:.1f}".format(perc_completed) +
              '% completed: ' +
              'time step ' + "{0:.2f}".format(time_step) + ' s' +
              ', total time ' + "{0:.2f}".format(time_total) + ' s')

        # save to the rough mask file
        np.savez_compressed(rough_mask_file, lores_istissue=tissue_mask_l5, lores_istissue0=tissue_mask0_l5,
                            im_downsampled=im_l5, step=step, perc_completed_all=perc_completed_all,
                            time_step_all=time_step_all,
                            prev_first_row=prev_first_row_l5, prev_last_row=prev_last_row_l5,
                            prev_first_col=prev_first_col_l5, prev_last_col=prev_last_col_l5)

        # clear keras session to prevent each segmentation iteration from getting slower. Note that this forces us to
        # reload the models every time, but that's not too slow
        K.clear_session()

    # end of "keep extracting histology windows until we have finished"
