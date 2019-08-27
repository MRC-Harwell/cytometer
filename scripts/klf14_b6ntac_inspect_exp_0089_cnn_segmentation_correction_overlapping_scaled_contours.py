"""
Segmentation Correction Network.

Use original hand traced cells (with overlaps) for training.

We create a ground truth segmentation, and then a series of eroded and dilated segmentations for training.

We create a bounding box to crop the images, and the scale everything to the same training window size, to remove
differences between cell sizes.

Training/test datasets created by random selection of individual cell images.

CHANGE: Here we assign cells to train or test sets grouped by image. This way, we guarantee that at testing time, the
network has not seen neighbour cells to the ones used for training.

Training for the CNN:
* Input: histology multiplied by segmentation.
* Output: mask = segmentation - ground truth.
* Other: mask for the loss function, to avoid looking too far from the cell.
"""

# script name to identify this experiment
original_experiment_id = 'klf14_b6ntac_exp_0089_cnn_segmentation_correction_overlapping_scaled_contours'
experiment_id = 'klf14_b6ntac_inspect_exp_0089_cnn_segmentation_correction_overlapping_scaled_contours'
print('Experiment ID: ' + experiment_id)

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import pickle
import json

# other imports
import datetime
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import time

# # limit number of GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

import cytometer.utils
import cytometer.data
import tensorflow as tf

# # limit GPU memory used
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# set_session(tf.Session(config=config))

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False

# area (pixel**2) of the smallest object we accept as a cell (pi * (16 pixel)**2 = 804.2 pixel**2)
smallest_cell_area = 804

# training window length
training_window_len = 201

# remove from training cells that don't have a good enough overlap with a reference label
smallest_dice = 0.5

# segmentations with Dice >= threshold are accepted
dice_threshold = 0.9

# batch size for training
batch_size = 12

# number of epochs for training
epochs = 100


'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

saved_kfolds_filename = 'klf14_b6ntac_exp_0079_generate_kfolds.pickle'


'''Load folds'''

# load list of images, and indices for training vs. testing indices
contour_model_kfold_filename = os.path.join(saved_models_dir, saved_kfolds_filename)
with open(contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
file_svg_list = aux['file_list']
idx_test_all = aux['idx_test']
idx_train_all = aux['idx_train']

# correct home directory
file_svg_list = [x.replace('/home/rcasero', home) for x in file_svg_list]

# number of images
n_im = len(file_svg_list)

# number of folds
n_folds = len(idx_test_all)

'''Inspect the data
'''

result = np.load(os.path.join(saved_models_dir, original_experiment_id + '_data.npz'))
window_im_all = result['window_im_all']
window_out_all = result['window_out_all']
window_mask_loss_all = result['window_mask_loss_all']
window_idx_all = result['window_idx_all']
del result

for i_fold in range(n_folds):

    print('# Fold ' + str(i_fold) + '/' + str(n_folds - 1))

    # test image indices
    idx_test = idx_test_all[i_fold]

    # test files
    file_svg_list_test = np.array(file_svg_list)[idx_test]

    for i, file_svg in enumerate(file_svg_list_test):

        print('file ' + str(i) + '/' + str(len(file_svg_list) - 1))

        # change file extension from .svg to .tif
        file_tif = file_svg.replace('.svg', '.tif')

        # open histology training image
        im = Image.open(file_tif)

        # make array copy
        im_array = np.array(im)

        # read the ground truth cell contours in the SVG file. This produces a list [contour_0, ..., contour_N-1]
        # where each contour_i = [(X_0, Y_0), ..., (X_P-1, X_P-1)]
        contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False)

        # loop ground truth cell contours
        for j, contour in enumerate(contours):

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

            if DEBUG:
                plt.subplot(222)
                plt.imshow(cell_seg_gtruth)

            # copy cell segmentation
            cell_seg = cell_seg_gtruth.copy()

            # compute bounding box that contains the mask, and leaves some margin
            bbox_x0, bbox_y0, bbox_xend, bbox_yend = \
                cytometer.utils.bounding_box_with_margin(cell_seg, coordinates='xy', inc=1.00)
            bbox_r0, bbox_c0, bbox_rend, bbox_cend = \
                cytometer.utils.bounding_box_with_margin(cell_seg, coordinates='rc', inc=1.00)

            if DEBUG:
                plt.subplot(223)
                plt.cla()
                plt.imshow(cell_seg)
                plt.plot((bbox_x0, bbox_xend, bbox_xend, bbox_x0, bbox_x0),
                         (bbox_y0, bbox_y0, bbox_yend, bbox_yend, bbox_y0))

            # crop image and masks according to bounding box
            window_im = cytometer.utils.extract_bbox(im_array, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))
            window_seg_gtruth = cytometer.utils.extract_bbox(cell_seg_gtruth, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))
            window_seg = cytometer.utils.extract_bbox(cell_seg, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))

            if DEBUG:
                plt.clf()
                plt.subplot(221)
                plt.cla()
                plt.imshow(im_array)
                plt.contour(cell_seg_gtruth, linewidths=1, levels=[0.5], colors='green')
                plt.contour(cell_seg, linewidths=1, levels=[0.5], colors='blue')
                plt.plot((bbox_x0, bbox_xend, bbox_xend, bbox_x0, bbox_x0),
                         (bbox_y0, bbox_y0, bbox_yend, bbox_yend, bbox_y0), 'black')

                plt.subplot(222)
                plt.cla()
                plt.imshow(window_im)
                plt.contour(window_seg_gtruth, linewidths=1, levels=[0.5], colors='green')
                plt.contour(window_seg, linewidths=1, levels=[0.5], colors='blue')

            # input to the CNN: multiply histology by +1/-1 segmentation mask
            window_im = \
                cytometer.utils.quality_model_mask(window_seg.astype(np.float32), im=window_im.astype(np.float32),
                                                   quality_model_type='-1_1')[0, :, :, :]

            # output of the CNN: segmentation - ground truth
            window_out = window_seg.astype(np.float32) - window_seg_gtruth.astype(np.float32)

            # output mask for the CNN: loss mask
            # window_mask_loss

            if DEBUG:
                plt.subplot(223)
                plt.cla()
                aux = 0.2989 * window_im[:, :, 0] + 0.5870 * window_im[:, :, 1] + 0.1140 * window_im[:, :, 2]
                plt.imshow(aux)
                plt.title('CNN input: histology * +1/-1 segmentation mask')

                plt.subplot(224)
                plt.cla()
                plt.imshow(window_out)


    # memory-map the precomputed data
    result = np.load(os.path.join(saved_models_dir, original_experiment_id + '_data.npz'), mmap_mode='r')
    window_idx_all = result['window_idx_all']

    # get cell indices for test and training, based on the image indices
    idx_test = np.where([x in idx_test for x in window_idx_all[:, 0]])[0]

    print('## len(idx_test) = ' + str(len(idx_test)))

    # get testing data
    window_im_test = window_im_all[idx_test, :, :, :]
    window_out_test = window_out_all[idx_test, :, :]
    window_mask_loss_test = window_mask_loss_all[idx_test, :]

