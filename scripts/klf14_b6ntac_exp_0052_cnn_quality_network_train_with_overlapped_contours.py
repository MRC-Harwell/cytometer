"""
Training all folds of DenseNet for quality assessement of individual cells based on classification of
thresholded Dice coefficient (Dice >= 0.9). Here the loss is binary focal loss.

The reason is to center the decision boundary on 0.9, to get finer granularity around that threshold.

Mask one-cell histology windows with 0/-1/+1 mask. The mask has a band with a width of 20% the equivalent radius
of the cell (equivalent radius is the radius of a circle with the same area as the cell).

The difference of this one with 0046 is that here we remove segmentations with Dice < 0.5 (automatic segmentations that
have poor ground truth) from the training dataset.

This is part of a series of experiments with different types of masks: 0039, 0040, 0041, 0042, 0045, 0046, 0048.
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
import inspect

# other imports
import glob
import shutil
import datetime
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import cv2

# use CPU for testing on laptop
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

from keras.applications import densenet
# from keras_applications.densenet import DenseNet
from keras.models import Model
from keras.layers import Dense

# for data parallelism in keras models
from keras.utils import multi_gpu_model

import cytometer.model_checkpoint_parallel
import cytometer.utils
import cytometer.data
import tensorflow as tf

# limit GPU memory used
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False

# number of folds for k-fold cross validation
n_folds = 11

# number of epochs for training
epochs = 40

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

# data augmentation factor (e.g. "10" means that we generate 9 augmented images + the original input image)
augment_factor = 10

# generator of random transformations
data_gen_args_float32 = dict(
    rotation_range=90,     # randomly rotate images up to 90 degrees
    fill_mode="constant",  # fill points outside boundaries with zeros
    cval=0,                #
    zoom_range=.1,         # [1-zoom_range, 1+zoom_range]
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True,    # randomly flip images
    dtype=np.float32       # explicit type out
)
datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args_float32)

'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

saved_contour_model_basename = 'klf14_b6ntac_exp_0034_cnn_contour'  # contour

# script name to identify this experiment
experiment_id = inspect.getfile(inspect.currentframe())
if experiment_id == '<input>':
    experiment_id = 'unknownscript'
else:
    experiment_id = os.path.splitext(os.path.basename(experiment_id))[0]

# # list of images, and indices for training vs. testing indices
# contour_model_kfold_filename = os.path.join(saved_models_dir, saved_contour_model_basename + '_info.pickle')
# with open(contour_model_kfold_filename, 'rb') as f:
#     aux = pickle.load(f)
# im_orig_file_list = aux['file_list']
# idx_orig_test_all = aux['idx_test_all']
#
# # correct home directory, if necessary
# im_orig_file_list = cytometer.data.change_home_directory(im_orig_file_list, home_path_from='/users/rittscher/rcasero',
#                                                          home_path_to=home,
#                                                          check_isfile=False)

'''Process the data
'''


# we are interested only in .tif files for which we created hand segmented contours
file_list = glob.glob(os.path.join(training_data_dir, '*.svg'))

for i, file_svg in enumerate(file_list):

    print('file ' + str(i) + '/' + str(len(file_list) - 1))

    # change file extension from .svg to .tif
    file_tif = file_svg.replace('.svg', '.tif')

    # open histology training image
    im = Image.open(file_tif)

    # make array copy
    im_array = np.array(im)

    if DEBUG:
        plt.clf()
        plt.imshow(im)

    # read the ground truth cell contours in the SVG file. This produces a list [contour_0, ..., contour_N-1]
    # where each contour_i = [(X_0, Y_0), ..., (X_P-1, X_P-1)]
    contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False)

    # loop ground truth cell contours
    for j, contour in enumerate(contours):

        # centre of current cell
        xy_c = (np.mean([p[0] for p in contour]), np.mean([p[1] for p in contour]))

        if DEBUG:
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

        # equivalent radius of the ground truth segmentation
        a_gtruth = np.count_nonzero(cell_seg_gtruth)  # segmentation area (pix^2)
        r_gtruth = np.sqrt(a_gtruth / np.pi)  # equivalent circle's radius

        # loop different perturbations in the mask to have a collection of better and worse
        # segmentations
        for inc in [-0.20, -0.15, -0.10, -.07, -0.03, 0.0, 0.03, 0.07, 0.10, 0.15, 0.20]:

            # erode or dilate the ground truth mask to create the segmentation mask
            cell_seg = cytometer.utils.quality_model_mask(cell_seg_gtruth, quality_model_type='0_1_prop_band',
                                                          quality_model_type_param=inc)[0, :, :, 0].astype(np.int8)

            # compute bounding box that contains the mask, and leaves some margin
            bbox_x0, bbox_y0, bbox_xend, bbox_yend = \
                cytometer.utils.bounding_box_with_margin(cell_seg_gtruth + cell_seg, coordinates='xy', inc=0.40)
            bbox_r0, bbox_c0, bbox_rend, bbox_cend = \
                cytometer.utils.bounding_box_with_margin(cell_seg_gtruth + cell_seg, coordinates='rc', inc=0.40)

            if DEBUG:
                plt.subplot(223)
                plt.cla()
                plt.imshow(im_array)
                plt.contour(cell_seg_gtruth, linewidths=1, levels=0.5, colors='green')
                plt.contour(cell_seg, linewidths=1, levels=0.5, colors='blue')
                plt.plot((bbox_x0, bbox_xend, bbox_xend, bbox_x0, bbox_x0),
                         (bbox_y0, bbox_y0, bbox_yend, bbox_yend, bbox_y0))
                plt.xlim(405, 655)
                plt.ylim(285, 35)

            train_im = np.expand_dims(im_array[bbox_r0:bbox_rend, bbox_c0:bbox_cend, :], axis=0)
            train_seg = np.expand_dims(np.expand_dims(cell_seg[bbox_r0:bbox_rend, bbox_c0:bbox_cend], axis=0), axis=3)
            train_seg_gtruth = np.expand_dims(np.expand_dims(cell_seg_gtruth[bbox_r0:bbox_rend, bbox_c0:bbox_cend], axis=0), axis=3)

            # extract training image / segmentation / mask
            train_im, train_seg, _ = \
                cytometer.utils.one_image_per_label(np.expand_dims(im_array, axis=0),
                                                    np.expand_dims(np.expand_dims(cell_seg, axis=0), axis=3),
                                                    training_window_len=bbox_rend - bbox_r0)
            _, train_seg_gtruth, _ = \
                cytometer.utils.one_image_per_label(np.expand_dims(im_array, axis=0),
                                                    np.expand_dims(np.expand_dims(cell_seg_gtruth, axis=0), axis=3),
                                                    training_window_len=bbox_rend - bbox_r0)

            if DEBUG:
                plt.subplot(224)
                plt.cla()
                plt.imshow(train_im[0, :, :, :])
                plt.contour(train_seg_gtruth[0, :, :, 0], linewidths=1, levels=0.5, colors='green')
                plt.contour(train_seg[0, :, :, 0], linewidths=1, levels=0.5, colors='blue')




            # resize the training image to training window size
            assert(train_im.dtype == np.uint8)
            assert(train_seg.dtype == np.uint8)
            assert(train_mask.dtype == np.int32)
            train_im = Image.fromarray(train_im[0, :, :, :])
            train_seg = Image.fromarray(train_seg[0, :, :, 0])
            train_mask = Image.fromarray(train_mask[0, :, :, 0], mode='I')

            train_im = train_im.resize(size=(training_window_len, training_window_len), resample=Image.NEAREST)
            train_seg = train_seg.resize(size=(training_window_len, training_window_len), resample=Image.NEAREST)
            train_mask = train_mask.resize(size=(training_window_len, training_window_len), resample=Image.NEAREST)

            train_im = np.array(train_im)
            train_seg = np.array(train_seg)
            train_mask = np.array(train_mask)

            if DEBUG:
                plt.subplot(224)
                plt.cla()
                plt.imshow(train_im)
                plt.contour(train_mask, linewidths=1, levels=0.5)

            # multiply the image by the mask
            train_im[:, :, 0] * train_im[:, :, 0] * train_mask
            train_im[:, :, 1] * train_im[:, :, 1] * train_mask
            train_im[:, :, 2] * train_im[:, :, 2] * train_mask






'''Save the log of computations
'''

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
