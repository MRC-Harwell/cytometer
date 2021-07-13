"""
Processing full slides of RREB1-TM1B_B6N-IC with pipeline v7 (modfied with colour correction):

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
  * Constants added to colour channels so that the medians match the training data.

 Requirements for this script to work:

 1) Upload the cytometer project directory to ~/Software in the server where you are going to process the data.

 2) Upload the AIDA project directory to ~/Software too.

 3) Mount the network share with the histology slides onto ~/scan_srv2_cox.

 4) Convert the .ndpi files to AIDA .dzi files, so that we can see the results of the segmentation.
    You need to go to the server that's going to process the slides, add a list of the files you want to process to
    ~/Software/cytometer/tools/rebb1_pilot_full_histology_ndpi_to_dzi.sh

    and run

    cd ~/Software/cytometer/tools
    ./rebb1_pilot_full_histology_ndpi_to_dzi.sh

 5) You need to have the models for the 10-folds of the pipeline that were trained on the KLF14 data.

 6) To monitor the segmentation as it's being processed, you need to have AIDA running

    cd ~/Software/AIDA/dist/
    node aidaLocal.js &

    You also need to create a soft link per .dzi file to the annotations you want to visualise for that file, whether
    the non-overlapping ones, or the corrected ones. E.g.

    ln -s 'RREB1-TM1B-B6N-IC-1.1a  1132-18 G1 - 2018-11-16 14.58.55_exp_0097_corrected.json' 'RREB1-TM1B-B6N-IC-1.1a  1132-18 G1 - 2018-11-16 14.58.55_exp_0097.json'

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
experiment_id = 'rreb1_tm1b_exp_0001_pilot_full_slide_pipeline_v7.py'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
from pathlib import Path
import sys
import pickle
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
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from keras import backend as K
import itertools
from shapely.geometry import Polygon
import scipy.stats

# # limit GPU memory used
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# set_session(tf.Session(config=config))

DEBUG = False
SAVE_FIGS = False

pipeline_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
experiment_root_data_dir = os.path.join(home, 'Data/cytometer_data/rreb1')
data_dir = os.path.join(home, 'scan_srv2_cox/Liz Bentley/Grace')
figures_dir = os.path.join(experiment_root_data_dir, 'figures')
saved_models_dir = os.path.join(pipeline_root_data_dir, 'saved_models')
results_dir = os.path.join(experiment_root_data_dir, 'results')
annotations_dir = os.path.join(home, 'Data/cytometer_data/aida_data_Rreb1_tm1b/annotations')
klf14_training_colour_histogram_file = os.path.join(saved_models_dir, 'klf14_training_colour_histogram.npz')

# although we don't need k-folds here, we need this file to load the list of SVG contours that we compute the AIDA
# colourmap from
# TODO: just save a cell size - colour function, instead of having to recompute it every time
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

# list of NDPI files to process
ndpi_files_list = [
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 G1 - 2018-11-16 14.58.55.ndpi',
]

# load colour modes of the KLF14 training dataset
with np.load(klf14_training_colour_histogram_file) as data:
    mode_r_klf14 = data['mode_r']
    mode_g_klf14 = data['mode_g']
    mode_b_klf14 = data['mode_b']

########################################################################################################################
## Colourmap for AIDA
########################################################################################################################

# TODO: load a pre-computed colourmap, instead of having to compute cell sizes every time

# list of SVG contours
saved_kfolds_filename = os.path.join(saved_models_dir, saved_extra_kfolds_filename)
with open(saved_kfolds_filename, 'rb') as f:
    aux = pickle.load(f)
file_svg_list = aux['file_list']
file_svg_list = [x.replace('/users/rittscher/rcasero', home) for x in file_svg_list]
file_svg_list = [x.replace('/home/rcasero', home) for x in file_svg_list]

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

for i_file, ndpi_file in enumerate(ndpi_files_list):

    print('File ' + str(i_file) + '/' + str(len(ndpi_files_list) - 1) + ': ' + ndpi_file)

    # make full path to ndpi file
    ndpi_file = os.path.join(data_dir, ndpi_file)

    # check whether there's a lock on this file
    lock_file = os.path.basename(ndpi_file).replace('.ndpi', '.lock')
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
    correction_model_file = os.path.join(saved_models_dir,
                                         correction_model_basename + '_model_fold_' + str(i_fold) + '.h5')

    # name of file to save annotations to
    annotations_file = os.path.basename(ndpi_file)
    annotations_file = os.path.splitext(annotations_file)[0]
    annotations_file = os.path.join(annotations_dir, annotations_file + '_exp_0097_auto.json')

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

        # compute the rough foreground mask of tissue vs. background
        lores_istissue0, im_downsampled = rough_foreground_mask(ndpi_file, downsample_factor=downsample_factor,
                                                                dilation_size=dilation_size,
                                                                component_size_threshold=component_size_threshold,
                                                                hole_size_treshold=hole_size_treshold, std_k=1.00,
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

        # end computing the rough foreground mask

    # checkpoint: here the rough tissue mask has either been loaded or computed
    time_step = time_step_all[-1]
    time_total = np.sum(time_step_all)
    print('File ' + str(i_file) + '/' + str(len(ndpi_files_list) - 1) + ': step ' +
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

        # next step (it starts from 0)
        step += 1

        # get indices for the next histology window to process
        (first_row, last_row, first_col, last_col), \
        (lores_first_row, lores_last_row, lores_first_col, lores_last_col) = \
            cytometer.utils.get_next_roi_to_process(lores_istissue, downsample_factor=downsample_factor,
                                                    max_window_size=fullres_box_size,
                                                    border=np.round((receptive_field-1)/2), version='old')

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

        print('File ' + str(i_file) + '/' + str(len(ndpi_files_list) - 1) + ': step ' +
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
