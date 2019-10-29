"""
Processing full slides with Fold 0 of pipeline v6:

 * data generation
   * training images (*0076*)
   * non-overlap training images (*0077*)
   * augmented training images (*0078*)
   * k-folds (*0079*)
 * segmentation
   * dmap (*0086*)
   * contour from dmap (0091)
 * classifier (*0088*)
 * segmentation correction (0089) networks"
 * validation (0092)
"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0093_full_slide_pipeline_v6'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import cytometer.utils

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

os.environ['KERAS_BACKEND'] = 'tensorflow'

import time
import openslide
import numpy as np
import matplotlib.pyplot as plt
import glob
from cytometer.utils import rough_foreground_mask, bspline_resample
from cytometer.data import append_paths_to_aida_json_file, write_paths_to_aida_json_file
import PIL
import tensorflow as tf
import keras
from skimage.measure import regionprops
import shutil
import itertools

# limit GPU memory used
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
figures_dir = os.path.join(root_data_dir, 'figures')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')
results_dir = os.path.join(root_data_dir, 'klf14_b6ntac_results')
annotations_dir = os.path.join(home, 'Software/AIDA/dist/data/annotations')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')

# k-folds file
saved_kfolds_filename = 'klf14_b6ntac_exp_0079_generate_kfolds.pickle'

# model names
dmap_model_basename = 'klf14_b6ntac_exp_0086_cnn_dmap'
contour_model_basename = 'klf14_b6ntac_exp_0091_cnn_contour_after_dmap'
classifier_model_basename = 'klf14_b6ntac_exp_0088_cnn_tissue_classifier_fcn'
correction_model_basename = 'klf14_b6ntac_exp_0089_cnn_segmentation_correction_overlapping_scaled_contours'

# full resolution image window and network expected receptive field parameters
fullres_box_size = np.array([1751, 1751])
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
min_mask_overlap = 0.8
phagocytosis = True
correction_window_len = 401
correction_smoothing = 11
batch_size = 16

# segmentation correction parameters

# process all histology slices in the data directory
# files_list = glob.glob(os.path.join(data_dir, 'KLF14*.ndpi'))

# # process only histology slices that were used for the hand traced dataset
# files_list = glob.glob(os.path.join(training_augmented_dir, 'im_seed_nan_*.tif'))
# for i, file in enumerate(files_list):
#     file_parts = os.path.split(file)
#     # recover original .ndpi filename (e.g. from
#     #'im_seed_nan_KLF14-B6NTAC-37.1c PAT 108-16 C1 - 2016-02-15 14.49.45_row_007372_col_008556.tif'
#     # to
#     #
#     files_list[i] = os.path.join(data_dir, file_parts[1][12:66] + '.ndpi')

# HACK: only process four images
files_list = [
    os.path.join(data_dir, 'KLF14-B6NTAC 36.1j PAT 105-16 C1 - 2016-02-12 14.33.33.ndpi'),  # male PAT
    os.path.join(data_dir, 'KLF14-B6NTAC-MAT-17.2g  69-16 C1 - 2016-02-04 16.15.05.ndpi'),  # male MAT
    os.path.join(data_dir, 'KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38.ndpi'),   # female PAT
    os.path.join(data_dir, 'KLF14-B6NTAC-MAT-17.1b  45-16 C1 - 2016-02-01 12.23.50.ndpi')   # female MAT
]
files_list = files_list[::-1]

# select the models that correspond to current fold
fold_i = 0
contour_model_file = os.path.join(saved_models_dir, contour_model_basename + '_model_fold_' + str(fold_i) + '.h5')
dmap_model_file = os.path.join(saved_models_dir, dmap_model_basename + '_model_fold_' + str(fold_i) + '.h5')
classifier_model_file = os.path.join(saved_models_dir, classifier_model_basename + '_model_fold_' + str(fold_i) + '.h5')
correction_model_file = os.path.join(saved_models_dir, correction_model_basename + '_model_fold_' + str(fold_i) + '.h5')

# load models
contour_model = keras.models.load_model(contour_model_file)
dmap_model = keras.models.load_model(dmap_model_file)
classifier_model = keras.models.load_model(classifier_model_file)
correction_model = keras.models.load_model(correction_model_file)

# "KLF14-B6NTAC-MAT-18.2b  58-16 B3 - 2016-02-03 11.01.43.ndpi"
# file_i = 10; file = files_list[file_i]
# "KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38.ndpi"
# file_i = 331; file = files_list[file_i]
# "KLF14-B6NTAC-MAT-17.1b  45-16 C1 - 2016-02-01 12.23.50.ndpi"
# file_i = 55; file = files_list[file_i]
for i_file, file in enumerate(files_list):

    print('File ' + str(i_file) + '/' + str(len(files_list)) + ': ' + file)

    # name of file to save annotations
    annotations_file = os.path.basename(file)
    annotations_file = os.path.splitext(annotations_file)[0]
    annotations_file = os.path.join(annotations_dir, annotations_file + '_exp_0093.json')

    # name of file to save areas and contours
    results_file = os.path.basename(file)
    results_file = os.path.splitext(results_file)[0]
    results_file = os.path.join(results_dir, results_file + '_exp_0093.npz')

    # # delete annotations file, if an older one exists
    # if os.path.isfile(annotations_file):
    #     os.remove(annotations_file)

    # rough segmentation of the tissue in the image
    lores_istissue0, im_downsampled = rough_foreground_mask(file, downsample_factor=downsample_factor,
                                                            dilation_size=dilation_size,
                                                            component_size_threshold=component_size_threshold,
                                                            hole_size_treshold=hole_size_treshold,
                                                            return_im=True)

    if DEBUG:
        plt.clf()
        plt.subplot(211)
        plt.imshow(im_downsampled)
        plt.subplot(212)
        plt.imshow(lores_istissue0)

    # segmentation copy, to keep track of what's left to do
    lores_istissue = lores_istissue0.copy()

    # open full resolution histology slide
    im = openslide.OpenSlide(file)

    # pixel size
    assert(im.properties['tiff.ResolutionUnit'] == 'centimeter')
    xres = 1e-2 / float(im.properties['tiff.XResolution'])
    yres = 1e-2 / float(im.properties['tiff.YResolution'])

    # init empty list to store area values and contour coordinates
    areas_all = []
    contours_all = []

    # keep extracting histology windows until we have finished
    step = -1
    time0 = time.time()
    while np.count_nonzero(lores_istissue) > 0:

        # next step (it starts from 0)
        step += 1

        print('File ' + str(i_file) + '/' + str(len(files_list)) + ': step ' +
              str(step) + ': ' +
              str(np.count_nonzero(lores_istissue)) + '/' + str(np.count_nonzero(lores_istissue0)) + ': ' +
              "{0:.1f}".format(100.0 - np.count_nonzero(lores_istissue) / np.count_nonzero(lores_istissue0) * 100) +
              '% completed: time ' + "{0:.2f}".format(time.time() - time0) + ' s')

        # get indices for the next histology window to process
        (first_row, last_row, first_col, last_col), \
        (lores_first_row, lores_last_row, lores_first_col, lores_last_col) = \
            cytometer.utils.get_next_roi_to_process(lores_istissue, downsample_factor=downsample_factor,
                                                    max_window_size=fullres_box_size,
                                                    border=np.round((receptive_field-1)/2))

        # # DEBUG
        # first_row = int(3190 * downsample_factor)
        # last_row = first_row + 1001
        # first_col = int(3205 * downsample_factor)
        # last_col = first_col + 1001
        # lores_first_row = 3190
        # lores_last_row = lores_first_row + int(np.round(1001 / downsample_factor))
        # lores_first_col = 3205
        # lores_last_col = lores_first_col + int(np.round(1001 / downsample_factor))

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
            plt.contour(istissue_tile, colors='k')
            plt.axis('off')

        # segment histology, split into individual objects, and apply segmentation correction
        labels, labels_class, todo_edge, \
        window_im, window_labels, window_labels_corrected, window_labels_class, index_list, scaling_factor_list \
            = cytometer.utils.segmentation_pipeline6(tile,
                                                     dmap_model=dmap_model,
                                                     contour_model=contour_model,
                                                     correction_model=correction_model,
                                                     classifier_model=classifier_model,
                                                     min_cell_area=min_cell_area,
                                                     mask=istissue_tile,
                                                     min_mask_overlap=min_mask_overlap,
                                                     phagocytosis=phagocytosis,
                                                     correction_window_len=correction_window_len,
                                                     correction_smoothing=correction_smoothing,
                                                     return_bbox=True, return_bbox_coordinates='xy',
                                                     batch_size=batch_size)

        # if no cells found, wipe out current window from tissue segmentation, and go to next iteration. Otherwise we'd
        # enter an infinite loop
        if len(index_list) == 0:
            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0
            contours_all.append([])
            areas_all.append([])
            # np.savez(results_file, contours=contours_all, areas=areas_all, lores_istissue=lores_istissue)
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
        contours = cytometer.utils.labels2contours(window_labels_corrected, offset_xy=offset_xy,
                                                  scaling_factor_xy=scaling_factor_list)

        if DEBUG:
            plt.clf()
            plt.imshow(tile)
            for j in range(len(contours)):
                plt.fill(contours[j][:, 0], contours[j][:, 1], edgecolor='C0', fill=False)

        # compute non-overlap cell areas
        props = regionprops(labels)
        p_label = [p['label'] for p in props]
        p_area = np.array([p['area'] for p in props])
        areas = p_area * xres * yres  # (m^2)

        # downsample contours for AIDA annotations file
        lores_contours = []
        for c in contours:
            lores_c = bspline_resample(c, factor=contour_downsample_factor, k=bspline_k, is_closed=True)
            lores_contours.append(lores_c)

        if DEBUG:
            for j in range(len(contours)):
                plt.fill(lores_contours[j][:, 0], lores_contours[j][:, 1], edgecolor='C1', fill=False)

        # add tile offset, so that contours are in full slide coordinates
        for j in range(len(contours)):
            lores_contours[j][:, 0] += first_col
            lores_contours[j][:, 1] += first_row

        # give one of four colours to each output contour
        iter = itertools.cycle([0, 90, 180, 270])
        hue = []
        for j in range(len(lores_contours)):
            hue.append(next(iter))

        # add segmented contours to annotations file
        if os.path.isfile(annotations_file):
            append_paths_to_aida_json_file(annotations_file, lores_contours, hue=hue)
        elif len(contours) > 0:
            fp = open(annotations_file, 'w')
            write_paths_to_aida_json_file(fp, lores_contours, hue=hue)
            fp.close()

        # add contours to list of all contours for the image
        contours_all.append(lores_contours)
        areas_all.append(areas)

        # update the tissue segmentation mask with the current window
        if np.all(lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] == lores_todo_edge):
            # if the mask remains identical, wipe out the whole window, as otherwise we'd have an
            # infinite loop
            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0
        else:
            # if the mask has been updated, use it to update the total tissue segmentation
            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = lores_todo_edge

        # save results after every window computation
        np.savez(results_file, contours=contours_all, areas=areas_all, lores_istissue=lores_istissue)

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
