"""
Validate exp 0057 (tissue classifier).

Load testing data from each fold, and apply both t-SNE embedding and the classifier from exp 0057.
"""

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import pickle

# other imports
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from mahotas.features import haralick

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

import cytometer.utils
import cytometer.data
import tensorflow as tf
from sklearn import manifold

# limit GPU memory used
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False

# number of epochs for training
epochs = 25

# area (pixel**2) of the smallest object we accept as a cell (pi * (16 pixel)**2 = 804.2 pixel**2)
smallest_cell_area = 804

# training window length
training_window_len = 401

# remove from training cells that don't have a good enough overlap with a reference label
smallest_dice = 0.5

# segmentations with Dice >= threshold are accepted
dice_threshold = 0.9

# batch size for training
batch_size = 16


'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0058_inspect_testing_dataset'

# load k-folds training and testing data
kfold_info_filename = os.path.join(saved_models_dir, 'klf14_b6ntac_exp_0053_cnn_quality_network_fcn_overlapping_scaled_contours_kfold_info.pickle')
with open(kfold_info_filename, 'rb') as f:
    kfold_info = pickle.load(f)
file_list = kfold_info['file_list']
idx_test_all = kfold_info['idx_test']
idx_train_all = kfold_info['idx_train']
del kfold_info

# model names
contour_model_basename = 'klf14_b6ntac_exp_0055_cnn_contour_model'
dmap_model_basename = 'klf14_b6ntac_exp_0056_cnn_dmap_model'
quality_model_basename = 'klf14_b6ntac_exp_0053_cnn_quality_network_fcn_overlapping_scaled_contours_model'
classifier_model_basename = 'klf14_b6ntac_exp_0057_cnn_tissue_classifier_fcn_overlapping_scaled_contours_model'

# number of images
n_im = len(file_list)


'''Load the texture of each cell in each image
'''

# init output
contour_type_all = []
window_features_all = []
window_idx_all = []
window_seg_gtruth_all = []
window_im_all = []
window_masked_im_all = []

# loop files with hand traced contours
for i, file_svg in enumerate(file_list):

    print('file ' + str(i) + '/' + str(len(file_list) - 1))

    # change file extension from .svg to .tif
    file_tif = file_svg.replace('.svg', '.tif')

    # open histology training image
    im = Image.open(file_tif)

    # read pixel size information
    xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
    yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

    # make array copy
    im_array = np.array(im)

    if DEBUG:
        plt.clf()
        plt.imshow(im)

    # read the ground truth cell contours in the SVG file. This produces a list [contour_0, ..., contour_N-1]
    # where each contour_i = [(X_0, Y_0), ..., (X_P-1, X_P-1)]
    cell_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False,
                                                            minimum_npoints=3)
    other_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Other', add_offset_from_filename=False,
                                                             minimum_npoints=3)
    damaged_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Damaged', add_offset_from_filename=False,
                                                               minimum_npoints=3)

    # make a list with the type of cell each contour is classified as
    contour_type = [np.zeros(shape=(len(cell_contours),), dtype=np.uint8),
                    np.ones(shape=(len(other_contours),), dtype=np.uint8),
                    2 * np.ones(shape=(len(damaged_contours),), dtype=np.uint8)]
    contour_type = np.concatenate(contour_type)
    contour_type_all.append(contour_type)

    print('Cells: ' + str(len(cell_contours)))
    print('Other: ' + str(len(other_contours)))
    print('Damaged: ' + str(len(damaged_contours)))
    print('')

    # loop ground truth cell contours
    for j, contour in enumerate(cell_contours + other_contours + damaged_contours):

        if DEBUG:
            # centre of current cell
            xy_c = (np.mean([p[0] for p in contour]), np.mean([p[1] for p in contour]))

            plt.clf()
            plt.subplot(221)
            plt.imshow(im)
            plt.plot([p[0] for p in contour], [p[1] for p in contour])
            plt.scatter(xy_c[0], xy_c[1])

        # rasterise current ground truth segmentation
        cell_seg_gtruth = Image.new("1", im.size, "black")  # I = 32-bit signed integer pixels
        draw = ImageDraw.Draw(cell_seg_gtruth)
        draw.polygon(contour, outline="white", fill="white")
        cell_seg_gtruth = np.array(cell_seg_gtruth, dtype=np.uint8)

        # mask histology with segmentation mask
        cell_masked_im = cytometer.utils.quality_model_mask(cell_seg_gtruth, im=im_array, quality_model_type='0_1')
        cell_masked_im = cell_masked_im[0, :, :, :]

        if DEBUG:
            plt.subplot(222)
            plt.cla()
            plt.imshow(cell_masked_im)

        # compute bounding box that contains the mask, and leaves some margin
        bbox_x0, bbox_y0, bbox_xend, bbox_yend = \
            cytometer.utils.bounding_box_with_margin(cell_seg_gtruth, coordinates='xy', inc=1.00)
        bbox_r0, bbox_c0, bbox_rend, bbox_cend = \
            cytometer.utils.bounding_box_with_margin(cell_seg_gtruth, coordinates='rc', inc=1.00)

        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(im)
            plt.plot([p[0] for p in contour], [p[1] for p in contour])

            plt.subplot(222)
            plt.cla()
            plt.imshow(cell_seg_gtruth)
            plt.plot((bbox_x0, bbox_xend, bbox_xend, bbox_x0, bbox_x0),
                     (bbox_y0, bbox_y0, bbox_yend, bbox_yend, bbox_y0))

        # crop image and masks according to bounding box
        window_im = cytometer.utils.extract_bbox(im_array, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))
        window_seg_gtruth = cytometer.utils.extract_bbox(cell_seg_gtruth, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))

        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.cla()
            plt.imshow(im)
            plt.plot([p[0] for p in contour], [p[1] for p in contour])

            plt.subplot(222)
            plt.cla()
            plt.imshow(window_im)
            plt.contour(window_seg_gtruth, linewidths=1, levels=0.5, colors='blue')

        # input to the CNN: multiply histology by +1/-1 segmentation mask
        window_masked_im = \
            cytometer.utils.quality_model_mask(window_seg_gtruth.astype(np.float32), im=window_im.astype(np.float32),
                                               quality_model_type='-1_1')[0, :, :, :]

        # scaling factors for the training image
        training_size = (training_window_len, training_window_len)
        scaling_factor = np.array(training_size) / np.array(window_masked_im.shape[0:2])
        window_pixel_size = np.array([xres, yres]) / scaling_factor  # (um, um)

        # resize the images to training window size
        window_im = cytometer.utils.resize(window_im, size=training_size, resample=Image.LINEAR)
        window_masked_im = cytometer.utils.resize(window_masked_im, size=training_size, resample=Image.LINEAR)
        window_seg_gtruth = cytometer.utils.resize(window_seg_gtruth, size=training_size, resample=Image.NEAREST)

        # compute texture vectors per channel
        window_features = (haralick(window_masked_im[:, :, 0].astype(np.uint8), ignore_zeros=True),
                           haralick(window_masked_im[:, :, 1].astype(np.uint8), ignore_zeros=True),
                           haralick(window_masked_im[:, :, 2].astype(np.uint8), ignore_zeros=True))
        window_features = np.vstack(window_features)
        window_features = window_features.flatten()

        # add dummy dimensions for keras
        window_im = np.expand_dims(window_im, axis=0)
        window_masked_im = np.expand_dims(window_masked_im, axis=0)
        window_seg_gtruth = np.expand_dims(window_seg_gtruth, axis=0)

        # scale image values to float [0, 1]
        window_im = window_im.astype(np.float32)
        window_im /= 255
        window_masked_im = window_masked_im.astype(np.float32)
        window_masked_im /= 255

        # check sizes and types
        assert(window_im.ndim == 4 and window_im.dtype == np.float32)
        assert(window_masked_im.ndim == 4 and window_masked_im.dtype == np.float32)


        # append results to total vectors
        window_features_all.append(window_features)
        window_idx_all.append(np.array([i, j]))
        window_seg_gtruth_all.append(window_seg_gtruth)
        window_im_all.append(window_im)
        window_masked_im_all.append(window_masked_im)

# collapse lists into arrays
contour_type_all = np.concatenate(contour_type_all)
window_features_all = np.vstack(window_features_all)
window_idx_all = np.vstack(window_idx_all)
window_seg_gtruth_all = np.concatenate(window_seg_gtruth_all)
window_im_all = np.concatenate(window_im_all)
window_masked_im_all = np.concatenate(window_masked_im_all)

if DEBUG:
    np.savez('/tmp/foo.npz', contour_type_all=contour_type_all, window_features_all=window_features_all,
             window_idx_all=window_idx_all, window_seg_gtruth_all=window_seg_gtruth_all, window_im_all=window_im_all,
             window_masked_im_all=window_masked_im_all)

'''t-SNE embedding
'''

for i_fold in range(0, len(idx_test_all)):

    print('# Fold ' + str(i_fold) + '/' + str(len(idx_test_all) - 1))

    # test and training image indices
    idx_test = idx_test_all[i_fold]
    idx_train = idx_train_all[i_fold]

    # get cell indices for test and training, based on the image indices
    idx_test = np.where([x in idx_test for x in window_idx_all[:, 0]])[0]
    idx_train = np.where([x in idx_train for x in window_idx_all[:, 0]])[0]

    print('## len(idx_train) = ' + str(len(idx_train)))
    print('## len(idx_test) = ' + str(len(idx_test)))

    # embedding
    tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    window_features_embedding = tsne.fit_transform(window_features_all)

    if DEBUG:
        color = np.array(['C0', 'C1', 'C2'])
        plt.clf()
        plt.scatter(window_features_embedding[:, 0], window_features_embedding[:, 1], c=color[contour_type_all],
                    cmap=plt.cm.Spectral, s=2)

        plt.show()

'''Validate classifier network
'''

for i_fold in range(0, len(idx_test_all)):

    print('# Fold ' + str(i_fold) + '/' + str(len(idx_test_all) - 1))

    # test and training image indices
    idx_test = idx_test_all[i_fold]
    idx_train = idx_train_all[i_fold]

    # get cell indices for test and training, based on the image indices
    idx_test = np.where([x in idx_test for x in window_idx_all[:, 0]])[0]
    idx_train = np.where([x in idx_train for x in window_idx_all[:, 0]])[0]

    print('## len(idx_train) = ' + str(len(idx_train)))
    print('## len(idx_test) = ' + str(len(idx_test)))

    # extract test arrays
    contour_type_test = contour_type_all[idx_test]
    window_idx_test = window_idx_all[idx_test, :]
    window_seg_gtruth_test = window_seg_gtruth_all[idx_test, :, :]
    window_im_test = window_im_all[idx_test, :, :]

    # load classifier network
    classifier_model_filename = os.path.join(saved_models_dir,
                                             classifier_model_basename + '_fold_' + str(i_fold) + '.h5')
    classifier_model = keras.models.load_model(classifier_model_filename)

    # apply classification network to cell histology
    window_classifier_softmax = classifier_model.predict(window_im_test, batch_size=batch_size)

    # label each pixel as "Cell", "Other" or "Damaged" according to the channel with the maximum softmax output
    window_classifier_class = np.argmax(window_classifier_softmax, axis=3)

    if DEBUG:
        j = 0

        plt.clf()

        plt.subplot(221)
        plt.imshow(window_im_test[j, :, :, :])
        plt.contour(window_seg_gtruth_test[j, :, :], linewidths=1, levels=0.5, colors='blue')

        plt.subplot(222)
        plt.imshow(window_classifier_class[j, :, :])
        cbar = plt.colorbar(ticks=(0, 1, 2))
        cbar.ax.set_yticklabels(['Cell', 'Other', 'Damaged'])
        plt.contour(window_seg_gtruth_test[j, :, :], linewidths=2, levels=0.5, colors='white')

        plt.subplot(234)
        plt.imshow(window_classifier_softmax[j, :, :, 0])
        plt.title('Cell')

        plt.subplot(235)
        plt.imshow(window_classifier_softmax[j, :, :, 1])
        plt.title('Other')

        plt.subplot(236)
        plt.imshow(window_classifier_softmax[j, :, :, 2])
        plt.title('Damaged')

