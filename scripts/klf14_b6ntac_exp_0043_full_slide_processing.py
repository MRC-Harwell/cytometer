# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import cytometer.utils

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

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
from skimage.measure import find_contours, regionprops
import shutil
import inspect

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

saved_contour_model_basename = 'klf14_b6ntac_exp_0034_cnn_contour'
saved_dmap_model_basename = 'klf14_b6ntac_exp_0035_cnn_dmap'
# saved_quality_model_basename = 'klf14_b6ntac_exp_0040_cnn_qualitynet_thresholded_sigmoid_masked_segmentation'
# saved_quality_model_basename = 'klf14_b6ntac_exp_0041_cnn_qualitynet_thresholded_sigmoid_pm_1_masked_segmentation'
saved_quality_model_basename = 'klf14_b6ntac_exp_0042_cnn_qualitynet_thresholded_sigmoid_pm_1_band_masked_segmentation'

contour_model_name = saved_contour_model_basename + '*.h5'
dmap_model_name = saved_dmap_model_basename + '*.h5'
quality_model_name = saved_quality_model_basename + '*.h5'

# script name to identify this experiment
experiment_id = inspect.getfile(inspect.currentframe())
if experiment_id == '<input>':
    experiment_id = 'unknownscript'
else:
    experiment_id = os.path.splitext(os.path.basename(experiment_id))[0]

# full resolution image window and network expected receptive field parameters
fullres_box_size = np.array([1751, 1751])
receptive_field = np.array([131, 131])

# rough_foreground_mask() parameters
downsample_factor = 8.0
dilation_size = 25
component_size_threshold = 1e5

# contour parameters
contour_downsample_factor = 0.1
bspline_k = 1

# block_split() parameters in downsampled image
block_len = np.ceil((fullres_box_size - receptive_field) / downsample_factor)
block_overlap = np.ceil((receptive_field - 1) / 2 / downsample_factor).astype(np.int)

files_list = glob.glob(os.path.join(data_dir, 'KLF14*.ndpi'))

# trained models for all folds
contour_model_files = sorted(glob.glob(os.path.join(saved_models_dir, contour_model_name)))
dmap_model_files = sorted(glob.glob(os.path.join(saved_models_dir, dmap_model_name)))
quality_model_files = sorted(glob.glob(os.path.join(saved_models_dir, quality_model_name)))

# select the models that correspond to current fold
fold_i = 0
contour_model_file = contour_model_files[fold_i]
dmap_model_file = dmap_model_files[fold_i]
quality_model_file = quality_model_files[fold_i]

# load models
contour_model = keras.models.load_model(contour_model_file)
dmap_model = keras.models.load_model(dmap_model_file)
quality_model = keras.models.load_model(quality_model_file)

# "KLF14-B6NTAC-MAT-18.2b  58-16 B3 - 2016-02-03 11.01.43.ndpi"
# file_i = 10; file = files_list[file_i]
# "KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38.ndpi"
# file_i = 331; file = files_list[file_i]
# "KLF14-B6NTAC-MAT-17.1b  45-16 C1 - 2016-02-01 12.23.50.ndpi"
# file_i = 55; file = files_list[file_i]
for file_i, file in enumerate(files_list):

    print('File ' + str(file_i) + '/' + str(len(files_list)) + ': ' + file)

    # name of file to save annotations
    annotations_file = os.path.basename(file)
    annotations_file = os.path.splitext(annotations_file)[0]
    annotations_file = os.path.join(annotations_dir, annotations_file + '.json')

    # name of file to save areas and contours
    results_file = os.path.basename(file)
    results_file = os.path.splitext(results_file)[0]
    results_file = os.path.join(results_dir, results_file + '.npz')

    # # delete annotations file, if an older one exists
    # if os.path.isfile(annotations_file):
    #     os.remove(annotations_file)

    # rough segmentation of the tissue in the image
    lores_istissue0, im_downsampled = rough_foreground_mask(file, downsample_factor=downsample_factor, dilation_size=dilation_size,
                                                            component_size_threshold=component_size_threshold, return_im=True)
    lores_istissue0 = (lores_istissue0 > 0).astype(np.uint8)

    if DEBUG:
        plt.clf()
        plt.subplot(121)
        plt.imshow(im_downsampled)
        plt.subplot(122)
        plt.imshow(lores_istissue0)

    # segmentation copy, to keep track of what's left to do
    lores_istissue = lores_istissue0.copy()

    # # save segmentation as a tiff file (with ZLIB compression)
    # outfilename = os.path.basename(file)
    # outfilename = os.path.splitext(outfilename)[0] + '_seg'
    # outfilename = os.path.join(seg_dir, outfilename + '.tif')
    # tifffile.imsave(outfilename, seg,
    #                 compress=9,
    #                 resolution=(int(im.properties["tiff.XResolution"]) / downsample_factor,
    #                             int(im.properties["tiff.YResolution"]) / downsample_factor,
    #                             im.properties["tiff.ResolutionUnit"].upper()))

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

        # next step
        step += 1

        print('File ' + str(file_i) + '/' + str(len(files_list)) + ': step ' +
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
        tile = np.reshape(tile, (1,) + tile.shape)
        tile = tile.astype(np.float32)
        tile /= 255

        # interpolate coarse tissue segmentation to full resolution
        istissue_tile = lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col]
        istissue_tile = PIL.Image.fromarray(istissue_tile)
        istissue_tile = istissue_tile.resize((last_col - first_col, last_row - first_row), resample=PIL.Image.NEAREST)
        istissue_tile = np.array(istissue_tile)
        istissue_tile = np.reshape(istissue_tile, newshape=(1,) + istissue_tile.shape)
        istissue_tile = (istissue_tile != 0).astype(np.uint8)

        # segment histology
        labels, labels_info = cytometer.utils.segmentation_pipeline(tile,
                                                                    contour_model, dmap_model, quality_model,
                                                                    quality_model_type='-1_1_band',
                                                                    mask=istissue_tile,
                                                                    smallest_cell_area=804)

        # if no cells found, wipe out current window from tissue segmentation, and go to next iteration
        if len(labels) == 0:
            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0
            contours_all.append([])
            areas_all.append([])
            np.savez(results_file, contours=contours_all, areas=areas_all, lores_istissue=lores_istissue)
            continue

        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(tile[0, :, :, :])
            plt.title('Histology', fontsize=16)
            plt.subplot(222)
            plt.imshow(tile[0, :, :, :])
            plt.contour(labels[0, :, :, 0], levels=labels_info['label'], colors='blue', linewidths=1)
            plt.title('Full segmentation', fontsize=16)
            plt.subplot(223)
            plt.boxplot(labels_info['quality'])
            plt.tick_params(labelbottom=False, bottom=False)
            plt.title('Quality values', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.subplot(224)
            aux = cytometer.utils.paint_labels(labels, labels_info['label'], labels_info['quality'] >= 0.9)
            plt.imshow(tile[0, :, :, :])
            plt.contour(aux[0, :, :, 0] * labels[0, :, :, 0],
                        levels=labels_info['label'], colors='blue', linewidths=1)
            plt.title('Labels with quality >= 0.9', fontsize=16)

        # list of cells that are on the edges
        edge_labels = cytometer.utils.edge_labels(labels[0, :, :, 0])

        # list of cells that are not on the edges
        non_edge_labels = np.setdiff1d(labels_info['label'], edge_labels)

        # list of cells that are OK'ed by quality network
        good_labels = labels_info['label'][labels_info['quality'] >= 0.9]

        # remove edge cells from the list of good cells
        good_labels = np.setdiff1d(good_labels, edge_labels)

        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(tile[0, :, :, :])
            plt.title('Histology', fontsize=16)
            plt.subplot(222)
            plt.imshow(tile[0, :, :, :])
            plt.contour(labels[0, :, :, 0], levels=labels_info['label'], colors='blue', linewidths=1)
            plt.title('Full segmentation', fontsize=16)
            plt.subplot(223)
            plt.boxplot(labels_info['quality'])
            plt.tick_params(labelbottom=False, bottom=False)
            plt.title('Quality values', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.subplot(224)
            aux = cytometer.utils.paint_labels(labels, labels_info['label'], np.isin(labels_info['label'], good_labels))
            plt.imshow(tile[0, :, :, :])
            plt.contour(aux[0, :, :, 0] * labels[0, :, :, 0],
                        levels=labels_info['label'], colors='blue', linewidths=1)
            plt.title('Labels with quality >= 0.9', fontsize=16)

        # # DEBUG
        # # remove ROI from segmentation
        # seg[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0
        #
        # if DEBUG:
        #     plt.clf()
        #     plt.subplot(121)
        #     plt.imshow(seg)
        #     plt.plot([lores_first_col, lores_last_col, lores_last_col, lores_first_col, lores_first_col],
        #              [lores_last_row, lores_last_row, lores_first_row, lores_first_row, lores_last_row], 'r')
        #     plt.xlim(700, 1500)
        #     plt.ylim(650, 300)
        #     plt.subplot(122)
        #     plt.imshow(imfuse(lores_istissue0, seg))
        #     plt.plot([lores_first_col, lores_last_col, lores_last_col, lores_first_col, lores_first_col],
        #              [lores_last_row, lores_last_row, lores_first_row, lores_first_row, lores_last_row], 'r')
        #     plt.xlim(700, 1500)
        #     plt.ylim(650, 300)
        #
        # if SAVE_FIGS:
        #     plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0043_get_next_roi_to_process_' +
        #                              str(step).zfill(2) + '.png'))

        # mark all edge cells as "to do"
        todo_labels = np.isin(labels[0, :, :, 0], edge_labels) * istissue_tile

        # downsample "to do"
        lores_todo_labels = PIL.Image.fromarray(todo_labels[0, :, :])
        lores_todo_labels = lores_todo_labels.resize((lores_last_col - lores_first_col,
                                                      lores_last_row - lores_first_row),
                                                     resample=PIL.Image.NEAREST)
        lores_todo_labels = np.array(lores_todo_labels)

        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col])
            plt.title('Low res tissue mask', fontsize=16)
            plt.subplot(222)
            plt.imshow(istissue_tile[0, :, :])
            plt.title('Full res tissue mask', fontsize=16)
            plt.subplot(223)
            plt.imshow(todo_labels[0, :, :])
            plt.title('Full res left over tissue', fontsize=16)
            plt.subplot(224)
            plt.imshow(lores_todo_labels)
            plt.title('Low res left over tissue', fontsize=16)

        # convert labels to contours (points) using marching squares
        # http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.find_contours
        contours = []
        for lab in good_labels:
            # load (row, col) coordinates
            aux = find_contours(labels[0, :, :, 0] == lab, 0.5,
                                fully_connected='low', positive_orientation='low')[0]
            # convert to (x, y) coordinates
            aux = aux[:, [1, 0]]
            if DEBUG:
                plt.plot(aux[:, 0], aux[:, 1])
            # add window offset
            aux[:, 0] = aux[:, 0] + first_col
            aux[:, 1] = aux[:, 1] + first_row
            # add to the list of contours
            contours.append(aux)

        # compute cell areas
        props = regionprops(labels)
        p_label = [p['label'] for p in props]
        p_area = np.array([p['area'] for p in props])
        areas = p_area[np.isin(p_label, good_labels)] * xres * yres  # (m^2)

        # downsample contours for AIDA annotations file
        lores_contours = []
        for c in contours:
            lores_c = bspline_resample(c, factor=contour_downsample_factor, k=bspline_k, is_closed=True)
            lores_contours.append(lores_c)
            if DEBUG:
                plt.plot(c[:, 0], c[:, 1], 'b')
                plt.plot(lores_c[:, 0], lores_c[:, 1], 'r')

        # add segmented contours to annotations file
        if os.path.isfile(annotations_file):
            append_paths_to_aida_json_file(annotations_file, lores_contours)
        elif len(contours) > 0:
            fp = open(annotations_file, 'w')
            write_paths_to_aida_json_file(fp, lores_contours)
            fp.close()

        # add contours to list of all contours for the image
        contours_all.append(lores_contours)
        areas_all.append(areas)

        # update the tissue segmentation mask with the current window
        if np.all(lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] == lores_todo_labels):
            # if the mask remains identical, wipe out the whole window, as otherwise we'd have an
            # infinite loop
            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0
        else:
            # if the mask has been updated, use it to update the total tissue segmentation
            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = lores_todo_labels

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


bar = 0
for foo1 in foo['areas']:
    bar += len(foo1)
