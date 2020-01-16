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
import sys
import pickle
import ujson
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import cytometer.utils

# Filter out INFO & WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

# DEBUG: i_file = 12; ndpi_file_kernel = list(ndpi_files_test_list.keys())[i_file]
for i_file, ndpi_file_kernel in enumerate(ndpi_files_test_list):

    print('    \'' + ndpi_file_kernel + '_exp_0097.json\'')

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
    annotations_file = os.path.join(annotations_dir, annotations_file + '_exp_0097_no_overlap.json')

    annotations_corrected_file = os.path.basename(ndpi_file)
    annotations_corrected_file = os.path.splitext(annotations_corrected_file)[0]
    annotations_corrected_file = os.path.join(annotations_dir, annotations_corrected_file + '_exp_0097_corrected.json')

    results_file = annotations_file.replace('_exp_0097_no_overlap.json', '_exp_0097_no_overlap_ujson.json')

    results_corrected_file = annotations_corrected_file.replace('_exp_0097_corrected.json', '_exp_0097_corrected_ujson.json')

    # # make a backup copy of the current annotations file
    # shutil.copy2(annotations_file, annotations_file + '.bak')

    # delete annotations file, if an older one exists
    if os.path.isfile(annotations_file):
        os.remove(annotations_file)

    # rough segmentation of the tissue in the image
    lores_istissue0, im_downsampled = rough_foreground_mask(ndpi_file, downsample_factor=downsample_factor,
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

        # compute cell areas
        areas = [Polygon(c).area * xres * yres for c in contours]  # (um^2)
        areas_corrected = [Polygon(c).area * xres * yres for c in contours_corrected]  # (um^2)

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

        # convert area values to quantiles
        q = f_area2quantile(np.array(areas))
        q_corrected = f_area2quantile(np.array(areas_corrected))

        # give a colour that is proportional to the area quantile
        hue = np.interp(q, [0.0, 1.0], [np.sqrt(20e3 * 1e-12), 315])
        hue_corrected = np.interp(q_corrected, [0.0, 1.0], [np.sqrt(20e3 * 1e-12), 315])

        # add segmented contours to annotations files
        if os.path.isfile(annotations_file):
            append_paths_to_aida_json_file(annotations_file, lores_contours, hue=hue, pretty_print=True)
            append_paths_to_aida_json_file(annotations_corrected_file, lores_contours_corrected, hue=hue_corrected, pretty_print=True)

        elif len(contours) > 0:
            with open(annotations_file, 'w') as fp:
                write_paths_to_aida_json_file(fp, lores_contours, hue=hue, pretty_print=True)
            with open(annotations_corrected_file, 'w') as fp:
                write_paths_to_aida_json_file(fp, lores_contours_corrected, hue=hue_corrected, pretty_print=True)

            aux = {'contours': [lores_contours[0],], 'areas': areas, 'area_units': 'um2', 'resolution': [xres, yres],
                   'tile_from_xy': [first_col, first_row], 'tile_to_xy': [last_col, last_row],
                   'total_time': (time.time() - time_0)}
            with open(results_file, mode='w') as fp:
                ujson.dump(aux, fp)
            ujson.dump(aux, results_file, 'a')
            ujson.dump(lores_contours, results_file, mode='a')

            foo_file = os.path.join(annotations_dir, 'KLF14-B6NTAC-MAT-18.2b  58-16 C1 - 2016-02-03 11.10.52_exp_0097.json')
            with open(foo_file) as fp:
                foo = ujson.load(fp)

            bar_file = os.path.join(annotations_dir,
                                    'foo.json')
            with open(bar_file, 'w') as fp:
                ujson.dump(foo, fp)

            foo['layers'][0]['name'] = 'Cell layer 2'
            with open(bar_file, 'a') as fp:
                ujson.dump(foo, fp)

            with open(bar_file) as fp:
                bar = ujson.load(fp)


        # update the tissue segmentation mask with the current window
        if np.all(lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] == lores_todo_edge):
            # if the mask remains identical, wipe out the whole window, as otherwise we'd have an
            # infinite loop
            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0
        else:
            # if the mask has been updated, use it to update the total tissue segmentation
            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = lores_todo_edge

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

