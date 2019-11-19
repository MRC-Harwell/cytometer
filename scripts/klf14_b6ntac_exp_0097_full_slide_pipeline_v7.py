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
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import cytometer.utils

# Filter out INFO & WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # limit number of GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

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

for i_file, ndpi_file_kernel in enumerate(ndpi_files_test_list):

    # fold  where the current .ndpi image was not used for training
    i_fold = ndpi_files_test_list[ndpi_file_kernel]

    # make full path to ndpi file
    ndpi_file = os.path.join(data_dir, ndpi_file_kernel + '.ndpi')

    print('File ' + str(i_file) + '/' + str(len(ndpi_files_test_list) - 1) + ': ' + ndpi_file)

    contour_model_file = os.path.join(saved_models_dir, contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    dmap_model_file = os.path.join(saved_models_dir, dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    classifier_model_file = os.path.join(saved_models_dir,
                                         classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    correction_model_file = os.path.join(saved_models_dir,
                                         correction_model_basename + '_model_fold_' + str(i_fold) + '.h5')

    # name of file to save annotations
    annotations_file = os.path.basename(ndpi_file)
    annotations_file = os.path.splitext(annotations_file)[0]
    annotations_file = os.path.join(annotations_dir, annotations_file + '_exp_0097.json')

    # name of file to save areas and contours
    results_file = os.path.basename(ndpi_file)
    results_file = os.path.splitext(results_file)[0]
    results_file = os.path.join(results_dir, results_file + '_exp_0097.npz')

    # # delete annotations file, if an older one exists
    # if os.path.isfile(annotations_file):
    #     os.remove(annotations_file)

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
            plt.contour(istissue_tile, colors='k')
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
            # contours_all.append([])
            # areas_all.append([])
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
            plt.clf()
            plt.imshow(tile)
            for j in range(len(contours)):
                plt.fill(lores_contours[j][:, 0], lores_contours[j][:, 1], edgecolor='C1', fill=False)

        # add tile offset, so that contours are in full slide coordinates
        for j in range(len(contours)):
            lores_contours[j][:, 0] += first_col
            lores_contours[j][:, 1] += first_row

        # give one of four colours to each output contour
        iter = itertools.cycle(np.linspace(0, 270, 10).astype(np.int))
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

        # # add contours to list of all contours for the image
        # contours_all.append(lores_contours)
        # areas_all.append(areas)

        # update the tissue segmentation mask with the current window
        if np.all(lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] == lores_todo_edge):
            # if the mask remains identical, wipe out the whole window, as otherwise we'd have an
            # infinite loop
            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0
        else:
            # if the mask has been updated, use it to update the total tissue segmentation
            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = lores_todo_edge

        # # save results after every window computation
        # np.savez(results_file, contours=contours_all, areas=areas_all, lores_istissue=lores_istissue)

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

# File 0/4: step 0: 10478432/10478432: 0.0% completed: step time 0.09 s, total time 0.09 s
# File 0/4: step 1: 10396261/10478432: 0.8% completed: step time 61.56 s, total time 61.65 s
# File 0/4: step 2: 10323102/10478432: 1.5% completed: step time 52.96 s, total time 114.62 s
# File 0/4: step 3: 10287518/10478432: 1.8% completed: step time 48.08 s, total time 162.70 s
# File 0/4: step 4: 10207625/10478432: 2.6% completed: step time 68.81 s, total time 231.51 s
# File 0/4: step 5: 10187510/10478432: 2.8% completed: step time 49.55 s, total time 281.06 s
# File 0/4: step 6: 10154468/10478432: 3.1% completed: step time 35.28 s, total time 316.34 s
# File 0/4: step 7: 10093579/10478432: 3.7% completed: step time 51.96 s, total time 368.30 s
# File 0/4: step 8: 10051734/10478432: 4.1% completed: step time 55.72 s, total time 424.02 s
# File 0/4: step 9: 9971783/10478432: 4.8% completed: step time 58.51 s, total time 482.53 s
# File 0/4: step 10: 9900003/10478432: 5.5% completed: step time 55.25 s, total time 537.78 s
# File 0/4: step 11: 9825230/10478432: 6.2% completed: step time 54.30 s, total time 592.08 s
# File 0/4: step 12: 9773352/10478432: 6.7% completed: step time 54.35 s, total time 646.43 s
# File 0/4: step 13: 9735863/10478432: 7.1% completed: step time 47.01 s, total time 693.44 s
# File 0/4: step 14: 9647837/10478432: 7.9% completed: step time 69.26 s, total time 762.71 s
# File 0/4: step 15: 9591684/10478432: 8.5% completed: step time 51.82 s, total time 814.53 s
# File 0/4: step 16: 9533809/10478432: 9.0% completed: step time 55.52 s, total time 870.05 s
# File 0/4: step 17: 9494164/10478432: 9.4% completed: step time 49.76 s, total time 919.81 s
# File 0/4: step 18: 9406384/10478432: 10.2% completed: step time 62.92 s, total time 982.72 s
# File 0/4: step 19: 9383867/10478432: 10.4% completed: step time 51.41 s, total time 1034.13 s
# File 0/4: step 20: 9328115/10478432: 11.0% completed: step time 51.27 s, total time 1085.40 s
# File 0/4: step 21: 9293967/10478432: 11.3% completed: step time 54.37 s, total time 1139.77 s
# File 0/4: step 22: 9259202/10478432: 11.6% completed: step time 48.15 s, total time 1187.91 s
# File 0/4: step 23: 9191239/10478432: 12.3% completed: step time 53.19 s, total time 1241.10 s
# File 0/4: step 24: 9145909/10478432: 12.7% completed: step time 54.67 s, total time 1295.78 s
# File 0/4: step 25: 9092098/10478432: 13.2% completed: step time 49.83 s, total time 1345.61 s
# File 0/4: step 26: 9041311/10478432: 13.7% completed: step time 57.44 s, total time 1403.05 s
# File 0/4: step 27: 8954729/10478432: 14.5% completed: step time 67.89 s, total time 1470.93 s
# File 0/4: step 28: 8901063/10478432: 15.1% completed: step time 52.05 s, total time 1522.99 s
# File 0/4: step 29: 8839234/10478432: 15.6% completed: step time 56.73 s, total time 1579.71 s
# File 0/4: step 30: 8750023/10478432: 16.5% completed: step time 61.01 s, total time 1640.73 s
# File 0/4: step 31: 8688045/10478432: 17.1% completed: step time 51.78 s, total time 1692.50 s
# File 0/4: step 32: 8600200/10478432: 17.9% completed: step time 71.84 s, total time 1764.34 s
# File 0/4: step 33: 8561527/10478432: 18.3% completed: step time 49.20 s, total time 1813.54 s
# File 0/4: step 34: 8537835/10478432: 18.5% completed: step time 49.57 s, total time 1863.11 s
# File 0/4: step 35: 8487997/10478432: 19.0% completed: step time 52.18 s, total time 1915.29 s
# File 0/4: step 36: 8408611/10478432: 19.8% completed: step time 61.06 s, total time 1976.35 s
# File 0/4: step 37: 8348623/10478432: 20.3% completed: step time 54.17 s, total time 2030.52 s
# File 0/4: step 38: 8282239/10478432: 21.0% completed: step time 55.46 s, total time 2085.98 s
# File 0/4: step 39: 8231218/10478432: 21.4% completed: step time 51.05 s, total time 2137.03 s
# File 0/4: step 40: 8154986/10478432: 22.2% completed: step time 54.90 s, total time 2191.92 s
# File 0/4: step 41: 8124402/10478432: 22.5% completed: step time 49.43 s, total time 2241.36 s
# File 0/4: step 42: 8058618/10478432: 23.1% completed: step time 52.89 s, total time 2294.25 s
# File 0/4: step 43: 8032798/10478432: 23.3% completed: step time 54.18 s, total time 2348.43 s
# File 0/4: step 44: 7986445/10478432: 23.8% completed: step time 55.85 s, total time 2404.28 s
# File 0/4: step 45: 7953875/10478432: 24.1% completed: step time 53.70 s, total time 2457.97 s
# File 0/4: step 46: 7900219/10478432: 24.6% completed: step time 51.99 s, total time 2509.97 s
# File 0/4: step 47: 7852198/10478432: 25.1% completed: step time 48.79 s, total time 2558.76 s
# File 0/4: step 48: 7796485/10478432: 25.6% completed: step time 56.07 s, total time 2614.84 s
# File 0/4: step 49: 7762859/10478432: 25.9% completed: step time 49.31 s, total time 2664.14 s
# File 0/4: step 50: 7690324/10478432: 26.6% completed: step time 60.18 s, total time 2724.33 s
# File 0/4: step 51: 7651749/10478432: 27.0% completed: step time 53.58 s, total time 2777.91 s
# File 0/4: step 52: 7608039/10478432: 27.4% completed: step time 49.30 s, total time 2827.21 s
# File 0/4: step 53: 7538981/10478432: 28.1% completed: step time 51.61 s, total time 2878.82 s
# File 0/4: step 54: 7468071/10478432: 28.7% completed: step time 55.77 s, total time 2934.60 s
# File 0/4: step 55: 7409501/10478432: 29.3% completed: step time 52.74 s, total time 2987.34 s
# File 0/4: step 56: 7325300/10478432: 30.1% completed: step time 61.31 s, total time 3048.65 s
# File 0/4: step 57: 7259725/10478432: 30.7% completed: step time 51.67 s, total time 3100.32 s
# File 0/4: step 58: 7179118/10478432: 31.5% completed: step time 60.37 s, total time 3160.68 s
# File 0/4: step 59: 7110753/10478432: 32.1% completed: step time 57.43 s, total time 3218.12 s
# File 0/4: step 60: 7077757/10478432: 32.5% completed: step time 53.76 s, total time 3271.88 s
# File 0/4: step 61: 6997976/10478432: 33.2% completed: step time 57.21 s, total time 3329.09 s
# File 0/4: step 62: 6976341/10478432: 33.4% completed: step time 45.85 s, total time 3374.94 s
# File 0/4: step 63: 6899083/10478432: 34.2% completed: step time 58.85 s, total time 3433.79 s
# File 0/4: step 64: 6850668/10478432: 34.6% completed: step time 50.89 s, total time 3484.68 s
# File 0/4: step 65: 6819309/10478432: 34.9% completed: step time 54.50 s, total time 3539.18 s
# File 0/4: step 66: 6788950/10478432: 35.2% completed: step time 53.35 s, total time 3592.53 s
# File 0/4: step 67: 6723502/10478432: 35.8% completed: step time 58.36 s, total time 3650.90 s
# File 0/4: step 68: 6677632/10478432: 36.3% completed: step time 51.77 s, total time 3702.67 s
# File 0/4: step 69: 6587778/10478432: 37.1% completed: step time 61.22 s, total time 3763.89 s
# File 0/4: step 70: 6526244/10478432: 37.7% completed: step time 55.52 s, total time 3819.41 s
# File 0/4: step 71: 6461837/10478432: 38.3% completed: step time 58.78 s, total time 3878.18 s
# File 0/4: step 72: 6434243/10478432: 38.6% completed: step time 48.54 s, total time 3926.73 s
# File 0/4: step 73: 6343820/10478432: 39.5% completed: step time 65.21 s, total time 3991.93 s
# File 0/4: step 74: 6273199/10478432: 40.1% completed: step time 53.99 s, total time 4045.92 s
# File 0/4: step 75: 6206442/10478432: 40.8% completed: step time 53.16 s, total time 4099.09 s
# File 0/4: step 76: 6149464/10478432: 41.3% completed: step time 56.71 s, total time 4155.79 s
# File 0/4: step 77: 6117230/10478432: 41.6% completed: step time 50.12 s, total time 4205.91 s
# File 0/4: step 78: 6035460/10478432: 42.4% completed: step time 59.38 s, total time 4265.29 s
# File 0/4: step 79: 5979284/10478432: 42.9% completed: step time 53.09 s, total time 4318.39 s
# File 0/4: step 80: 5899844/10478432: 43.7% completed: step time 62.43 s, total time 4380.81 s
# File 0/4: step 81: 5852436/10478432: 44.1% completed: step time 52.97 s, total time 4433.78 s
# File 0/4: step 82: 5771901/10478432: 44.9% completed: step time 60.73 s, total time 4494.51 s
# File 0/4: step 83: 5688351/10478432: 45.7% completed: step time 60.74 s, total time 4555.25 s
# File 0/4: step 84: 5602259/10478432: 46.5% completed: step time 63.33 s, total time 4618.57 s
# File 0/4: step 85: 5539124/10478432: 47.1% completed: step time 53.54 s, total time 4672.11 s
# File 0/4: step 86: 5501686/10478432: 47.5% completed: step time 49.90 s, total time 4722.01 s
# File 0/4: step 87: 5440040/10478432: 48.1% completed: step time 54.38 s, total time 4776.39 s
# File 0/4: step 88: 5356372/10478432: 48.9% completed: step time 64.18 s, total time 4840.57 s
# File 0/4: step 89: 5337938/10478432: 49.1% completed: step time 48.36 s, total time 4888.93 s
# File 0/4: step 90: 5306477/10478432: 49.4% completed: step time 49.10 s, total time 4938.03 s
# File 0/4: step 91: 5254922/10478432: 49.9% completed: step time 63.42 s, total time 5001.46 s
# File 0/4: step 92: 5191573/10478432: 50.5% completed: step time 56.18 s, total time 5057.64 s
# File 0/4: step 93: 5151074/10478432: 50.8% completed: step time 52.23 s, total time 5109.86 s
# File 0/4: step 94: 5108257/10478432: 51.2% completed: step time 52.21 s, total time 5162.08 s
# File 0/4: step 95: 5041544/10478432: 51.9% completed: step time 58.85 s, total time 5220.93 s
# File 0/4: step 96: 4984627/10478432: 52.4% completed: step time 55.62 s, total time 5276.55 s
# File 0/4: step 97: 4969761/10478432: 52.6% completed: step time 46.23 s, total time 5322.78 s
# File 0/4: step 98: 4922528/10478432: 53.0% completed: step time 66.98 s, total time 5389.76 s
# File 0/4: step 99: 4859130/10478432: 53.6% completed: step time 61.57 s, total time 5451.33 s
# File 0/4: step 100: 4797435/10478432: 54.2% completed: step time 54.23 s, total time 5505.56 s
# File 0/4: step 101: 4739814/10478432: 54.8% completed: step time 54.33 s, total time 5559.89 s
# File 0/4: step 102: 4672865/10478432: 55.4% completed: step time 58.65 s, total time 5618.53 s
# File 0/4: step 103: 4666978/10478432: 55.5% completed: step time 44.73 s, total time 5663.26 s
# File 0/4: step 104: 4576183/10478432: 56.3% completed: step time 64.96 s, total time 5728.22 s
# File 0/4: step 105: 4496530/10478432: 57.1% completed: step time 59.36 s, total time 5787.57 s
# File 0/4: step 106: 4407097/10478432: 57.9% completed: step time 59.84 s, total time 5847.41 s
# File 0/4: step 107: 4369208/10478432: 58.3% completed: step time 48.84 s, total time 5896.25 s
# File 0/4: step 108: 4313899/10478432: 58.8% completed: step time 59.62 s, total time 5955.87 s
# File 0/4: step 109: 4244004/10478432: 59.5% completed: step time 64.09 s, total time 6019.97 s
# File 0/4: step 110: 4207520/10478432: 59.8% completed: step time 50.01 s, total time 6069.98 s
# File 0/4: step 111: 4149142/10478432: 60.4% completed: step time 63.75 s, total time 6133.72 s
# File 0/4: step 112: 4059889/10478432: 61.3% completed: step time 62.40 s, total time 6196.12 s
# File 0/4: step 113: 4003632/10478432: 61.8% completed: step time 59.12 s, total time 6255.24 s
# File 0/4: step 114: 3951663/10478432: 62.3% completed: step time 55.56 s, total time 6310.80 s
# File 0/4: step 115: 3863863/10478432: 63.1% completed: step time 67.74 s, total time 6378.53 s
# File 0/4: step 116: 3796371/10478432: 63.8% completed: step time 61.65 s, total time 6440.18 s
# File 0/4: step 117: 3725456/10478432: 64.4% completed: step time 59.12 s, total time 6499.30 s
# File 0/4: step 118: 3712150/10478432: 64.6% completed: step time 46.82 s, total time 6546.12 s
# File 0/4: step 119: 3685271/10478432: 64.8% completed: step time 49.11 s, total time 6595.23 s
# File 0/4: step 120: 3629047/10478432: 65.4% completed: step time 53.33 s, total time 6648.56 s
# File 0/4: step 121: 3562235/10478432: 66.0% completed: step time 55.02 s, total time 6703.57 s
# File 0/4: step 122: 3531686/10478432: 66.3% completed: step time 50.93 s, total time 6754.51 s
# File 0/4: step 123: 3485588/10478432: 66.7% completed: step time 59.98 s, total time 6814.49 s
# File 0/4: step 124: 3432218/10478432: 67.2% completed: step time 57.87 s, total time 6872.36 s
# File 0/4: step 125: 3381132/10478432: 67.7% completed: step time 59.62 s, total time 6931.98 s
# File 0/4: step 126: 3336050/10478432: 68.2% completed: step time 53.42 s, total time 6985.41 s
# File 0/4: step 127: 3297056/10478432: 68.5% completed: step time 61.71 s, total time 7047.12 s
# File 0/4: step 128: 3261598/10478432: 68.9% completed: step time 53.42 s, total time 7100.54 s
# File 0/4: step 129: 3219303/10478432: 69.3% completed: step time 51.58 s, total time 7152.12 s
# File 0/4: step 130: 3205153/10478432: 69.4% completed: step time 34.30 s, total time 7186.42 s
# File 0/4: step 131: 3170064/10478432: 69.7% completed: step time 55.06 s, total time 7241.48 s
# File 0/4: step 132: 3106299/10478432: 70.4% completed: step time 56.12 s, total time 7297.60 s
# File 0/4: step 133: 3037133/10478432: 71.0% completed: step time 57.83 s, total time 7355.43 s
# File 0/4: step 134: 3017805/10478432: 71.2% completed: step time 50.20 s, total time 7405.64 s
# File 0/4: step 135: 2928219/10478432: 72.1% completed: step time 59.35 s, total time 7464.98 s
# File 0/4: step 136: 2912224/10478432: 72.2% completed: step time 49.06 s, total time 7514.04 s
# File 0/4: step 137: 2848199/10478432: 72.8% completed: step time 59.14 s, total time 7573.18 s
# File 0/4: step 138: 2814100/10478432: 73.1% completed: step time 49.82 s, total time 7623.01 s
# File 0/4: step 139: 2753319/10478432: 73.7% completed: step time 55.55 s, total time 7678.56 s
# File 0/4: step 140: 2696576/10478432: 74.3% completed: step time 59.44 s, total time 7738.00 s
# File 0/4: step 141: 2608063/10478432: 75.1% completed: step time 64.31 s, total time 7802.31 s
# File 0/4: step 142: 2526151/10478432: 75.9% completed: step time 59.20 s, total time 7861.51 s
# File 0/4: step 143: 2466291/10478432: 76.5% completed: step time 57.40 s, total time 7918.92 s
# File 0/4: step 144: 2465704/10478432: 76.5% completed: step time 43.80 s, total time 7962.71 s
# File 0/4: step 145: 2453342/10478432: 76.6% completed: step time 37.04 s, total time 7999.75 s
# File 0/4: step 146: 2422051/10478432: 76.9% completed: step time 51.03 s, total time 8050.79 s
# File 0/4: step 147: 2388129/10478432: 77.2% completed: step time 49.14 s, total time 8099.93 s
# File 0/4: step 148: 2345250/10478432: 77.6% completed: step time 47.62 s, total time 8147.55 s
# File 0/4: step 149: 2323130/10478432: 77.8% completed: step time 48.10 s, total time 8195.65 s
# File 0/4: step 150: 2290105/10478432: 78.1% completed: step time 44.17 s, total time 8239.82 s
# File 0/4: step 151: 2289454/10478432: 78.2% completed: step time 36.04 s, total time 8275.86 s
# File 0/4: step 152: 2235910/10478432: 78.7% completed: step time 60.31 s, total time 8336.17 s
# File 0/4: step 153: 2151102/10478432: 79.5% completed: step time 61.06 s, total time 8397.23 s
# File 0/4: step 154: 2085390/10478432: 80.1% completed: step time 55.18 s, total time 8452.41 s
# File 0/4: step 155: 2015597/10478432: 80.8% completed: step time 68.73 s, total time 8521.14 s
# File 0/4: step 156: 1976158/10478432: 81.1% completed: step time 51.57 s, total time 8572.71 s
# File 0/4: step 157: 1948205/10478432: 81.4% completed: step time 49.75 s, total time 8622.46 s
# File 0/4: step 158: 1922108/10478432: 81.7% completed: step time 41.23 s, total time 8663.69 s
# File 0/4: step 159: 1884149/10478432: 82.0% completed: step time 53.45 s, total time 8717.15 s
# File 0/4: step 160: 1835315/10478432: 82.5% completed: step time 54.52 s, total time 8771.66 s
# File 0/4: step 161: 1792727/10478432: 82.9% completed: step time 50.95 s, total time 8822.62 s
# File 0/4: step 162: 1743557/10478432: 83.4% completed: step time 55.90 s, total time 8878.51 s
# File 0/4: step 163: 1676688/10478432: 84.0% completed: step time 59.01 s, total time 8937.52 s
# File 0/4: step 164: 1635572/10478432: 84.4% completed: step time 52.65 s, total time 8990.17 s
# File 0/4: step 165: 1633866/10478432: 84.4% completed: step time 44.68 s, total time 9034.85 s
# File 0/4: step 166: 1571536/10478432: 85.0% completed: step time 58.37 s, total time 9093.22 s
# File 0/4: step 167: 1504832/10478432: 85.6% completed: step time 62.12 s, total time 9155.34 s
# File 0/4: step 168: 1470062/10478432: 86.0% completed: step time 53.26 s, total time 9208.60 s
# File 0/4: step 169: 1466914/10478432: 86.0% completed: step time 45.33 s, total time 9253.93 s
# File 0/4: step 170: 1410214/10478432: 86.5% completed: step time 54.29 s, total time 9308.22 s
# File 0/4: step 171: 1343347/10478432: 87.2% completed: step time 57.38 s, total time 9365.60 s
# File 0/4: step 172: 1278333/10478432: 87.8% completed: step time 59.76 s, total time 9425.36 s
# File 0/4: step 173: 1274466/10478432: 87.8% completed: step time 30.14 s, total time 9455.51 s
# File 0/4: step 174: 1208552/10478432: 88.5% completed: step time 56.79 s, total time 9512.29 s
# File 0/4: step 175: 1173206/10478432: 88.8% completed: step time 51.49 s, total time 9563.79 s
# File 0/4: step 176: 1168523/10478432: 88.8% completed: step time 27.13 s, total time 9590.92 s
# File 0/4: step 177: 1150087/10478432: 89.0% completed: step time 50.38 s, total time 9641.30 s
# File 0/4: step 178: 1146698/10478432: 89.1% completed: step time 28.04 s, total time 9669.34 s
# File 0/4: step 179: 1065253/10478432: 89.8% completed: step time 63.34 s, total time 9732.67 s
# File 0/4: step 180: 981413/10478432: 90.6% completed: step time 65.50 s, total time 9798.17 s
# File 0/4: step 181: 944140/10478432: 91.0% completed: step time 52.83 s, total time 9850.99 s
# File 0/4: step 182: 892485/10478432: 91.5% completed: step time 53.94 s, total time 9904.93 s
# File 0/4: step 183: 817747/10478432: 92.2% completed: step time 68.13 s, total time 9973.06 s
# File 0/4: step 184: 776427/10478432: 92.6% completed: step time 55.55 s, total time 10028.61 s
# File 0/4: step 185: 708256/10478432: 93.2% completed: step time 54.76 s, total time 10083.38 s
# File 0/4: step 186: 652393/10478432: 93.8% completed: step time 64.44 s, total time 10147.81 s
# File 0/4: step 187: 613737/10478432: 94.1% completed: step time 45.74 s, total time 10193.56 s
# File 0/4: step 188: 600771/10478432: 94.3% completed: step time 34.78 s, total time 10228.34 s
# File 0/4: step 189: 552695/10478432: 94.7% completed: step time 62.20 s, total time 10290.54 s
# File 0/4: step 190: 502881/10478432: 95.2% completed: step time 59.30 s, total time 10349.84 s
# File 0/4: step 191: 444949/10478432: 95.8% completed: step time 56.58 s, total time 10406.42 s
# File 0/4: step 192: 416795/10478432: 96.0% completed: step time 56.95 s, total time 10463.37 s
# File 0/4: step 193: 416802/10478432: 96.0% completed: step time 46.61 s, total time 10509.98 s
# File 0/4: step 194: 393910/10478432: 96.2% completed: step time 46.51 s, total time 10556.48 s
# File 0/4: step 195: 336477/10478432: 96.8% completed: step time 64.22 s, total time 10620.71 s
# File 0/4: step 196: 335624/10478432: 96.8% completed: step time 20.60 s, total time 10641.31 s
# File 0/4: step 197: 306793/10478432: 97.1% completed: step time 54.70 s, total time 10696.01 s
# File 0/4: step 198: 299480/10478432: 97.1% completed: step time 29.71 s, total time 10725.72 s
# File 0/4: step 199: 300678/10478432: 97.1% completed: step time 37.28 s, total time 10763.00 s
# File 0/4: step 200: 277188/10478432: 97.4% completed: step time 29.72 s, total time 10792.72 s
# File 0/4: step 201: 240060/10478432: 97.7% completed: step time 52.49 s, total time 10845.21 s
# File 0/4: step 202: 207745/10478432: 98.0% completed: step time 49.45 s, total time 10894.65 s
# File 0/4: step 203: 175616/10478432: 98.3% completed: step time 56.49 s, total time 10951.14 s
# File 0/4: step 204: 103846/10478432: 99.0% completed: step time 64.22 s, total time 11015.37 s
# File 0/4: step 207: 74385/10478432: 99.3% completed: step time 52.55 s, total time 11123.80 s
# File 0/4: step 208: 52597/10478432: 99.5% completed: step time 38.42 s, total time 11162.21 s
# File 0/4: step 209: 51583/10478432: 99.5% completed: step time 26.33 s, total time 11188.54 s
# File 0/4: step 210: 33196/10478432: 99.7% completed: step time 33.48 s, total time 11222.02 s
# File 0/4: step 211: 25349/10478432: 99.8% completed: step time 41.53 s, total time 11263.55 s
# File 0/4: step 212: 25169/10478432: 99.8% completed: step time 36.54 s, total time 11300.09 s
# File 0/4: step 213: 25331/10478432: 99.8% completed: step time 35.09 s, total time 11335.18 s
# File 0/4: step 214: 2635/10478432: 100.0% completed: step time 30.12 s, total time 11365.30 s
# File 0/4: step 215: 2407/10478432: 100.0% completed: step time 19.46 s, total time 11384.76 s
# File 0/4: step 216: 333/10478432: 100.0% completed: step time 20.64 s, total time 11405.39 s
# File 0/4: step 205: 92035/10478432: 99.1% completed: step time 36.32 s, total time 11051.69 s
# File 0/4: step 206: 91432/10478432: 99.1% completed: step time 19.55 s, total time 11071.24 s

