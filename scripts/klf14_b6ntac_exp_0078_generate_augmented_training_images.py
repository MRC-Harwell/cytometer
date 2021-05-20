"""
scripts/klf14_b6ntac_exp_0078_generate_augmented_training_images.py

Script to generate augmented training data for the klf14_b6ntac experiments.

It loads instance segmentation, and computes  distance transformations. It augments
the training dataset with random rotations, flips and scale changes (consistent between corresponding data).

This script creates dmap, im, mask, contour and lab in klf14_b6ntac_training_augmented.
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

# other imports
import pickle
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random
from skimage.transform import warp

# limit number of GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import cytometer.data
import cytometer.utils

# # limit GPU memory used
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# set_session(tf.Session(config=config))


DEBUG = False


'''Load data
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

# list of training image patches where there are hand segmentations
im_file_list = glob.glob(os.path.join(training_dir, '*.svg'))
im_file_list = [x.replace('.svg', '.tif') for x in im_file_list]

# list of corresponding automatic non-overlap segmentations
non_overlap_file_list = []
for file in im_file_list:
    non_overlap_file = file.replace(training_dir, training_non_overlap_data_dir)
    if not os.path.isfile(non_overlap_file):
        raise FileExistsError('File does not exist: ' + non_overlap_file)
    non_overlap_file_list.append(non_overlap_file)
assert(len(im_file_list) == len(non_overlap_file_list))

# load segmentations and compute distance maps
dmap, mask, lab = cytometer.data.load_watershed_seg_and_compute_dmap(non_overlap_file_list)

# load corresponding images and convert to float format in [0, 255]
im = cytometer.data.load_file_list_to_array(im_file_list)
im = im.astype('float32', casting='safe')
im /= 255

# number of training images
n_im = im.shape[0]

# save copy of the labelled image
contour = lab.copy()

# set all inside pixels of segmentation to same value
contour = (contour < 2).astype(np.uint8)

# keep only the contours, not the background
contour = contour * mask

# dilate contours
for i in range(n_im):
    contour[i, :, :, 0] = cv2.dilate(contour[i, :, :, 0], kernel=np.ones(shape=(3, 3)))

'''Copy original data
'''

# copy or save the original data to the augmented directory, so that later we can use a simple flow_from_directory()
# method to read all the augmented data, and we don't need to recompute the distance transformations
for i, base_file in enumerate(im_file_list):

    print('File ' + str(i) + '/' + str(len(im_file_list) - 1))

    if DEBUG:
        plt.clf()
        plt.subplot(221)
        plt.imshow(dmap[i, :, :, 0])
        plt.subplot(222)
        plt.imshow(mask[i, :, :, 0])
        plt.subplot(223)
        plt.imshow(contour[i, :, :, 0])
        plt.subplot(224)
        plt.imshow(lab[i, :, :, 0])

    # create filenames based on the original foo.tif, so that we have
    # * im_seed_nan_foo.tif
    # * dmap_seed_nan_foo.tif
    # * mask_seed_nan_foo.tif
    # * labels_seed_nan_foo.tif
    # where seed_nan refers to the original data without augmentation variations
    base_path, base_name = os.path.split(base_file)
    im_file = os.path.join(training_augmented_dir, 'im_seed_nan_' + base_name)
    dmap_file = os.path.join(training_augmented_dir, 'dmap_seed_nan_' + base_name)
    mask_file = os.path.join(training_augmented_dir, 'mask_seed_nan_' + base_name)
    contour_file = os.path.join(training_augmented_dir, 'contour_seed_nan_' + base_name)
    lab_file = os.path.join(training_augmented_dir, 'lab_seed_nan_' + base_name)

    # check whether output files already exist
    is_im_file = os.path.isfile(im_file)
    is_dmap_file = os.path.isfile(dmap_file)
    is_mask_file = os.path.isfile(mask_file)
    is_contour_file = os.path.isfile(contour_file)
    is_lab_file = os.path.isfile(lab_file)
    if np.all([is_im_file, is_dmap_file, is_mask_file, is_contour_file, is_lab_file]):
        print('Skipping ... output files already exist')
        continue
    else:
        print('Saving')

    # copy the image file
    shutil.copy2(base_file, im_file)

    # save distance transforms (note: we have to save as float mode)
    im_out = Image.fromarray(dmap[i, :, :, 0].astype(np.float32), mode='F')
    im_out.save(dmap_file)

    # save mask (note: we can save as black and white 1 byte)
    im_out = Image.fromarray(mask[i, :, :, 0].astype(np.uint8), mode='L')
    im_out.save(mask_file)

    # set all contour pixels (note: we can save as black and white 1 byte)
    im_out = Image.fromarray(contour[i, :, :, 0].astype(np.uint8), mode='L')
    im_out.save(contour_file)

    # save labels (0-255 levels with 1 byte)
    if np.max(lab[i, :, :, 0]) > 255:
        raise ValueError('There is a label > 255, so it cannot be saved with a uint8 type')
    im_out = Image.fromarray(lab[i, :, :, 0].astype(np.uint8), mode='L')
    im_out.save(lab_file)


'''Data augmentation
'''

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

# augment data, using the same seed so that all corresponding images, dmaps and masks undergo
# the same transformations
# this is a hack so that we have a different seed for each batch of augmented data
for seed in range(augment_factor - 1):

    print('* Augmentation round: ' + str(seed + 1) + '/' + str(augment_factor - 1))

    # generate random transformations. We give this its own loop, so that it's easy to replicate, if necessary
    transform = []
    random.seed(seed)
    for i in range(n_im):
        # generate random transformation
        transform.append(datagen.get_random_transform(img_shape=im.shape[1:3],
                                                      seed=random.randint(0, 2**32)))

    # loop applying transformations to data
    for i in range(n_im):

        print('  ** Image: ' + str(i) + '/' + str(n_im - 1))

        # create filenames based on the original foo.tif, so that we have im_seed_001_foo.tif, dmap_seed_001_foo.tif,
        # mask_seed_001_foo.tif, where seed_001 means data augmented using seed=1
        base_file = im_file_list[i]
        base_path, base_name = os.path.split(base_file)
        transform_file = os.path.join(training_augmented_dir, 'transform_seed_' + str(seed).zfill(3) + '_'
                                      + base_name.replace('.tif', '.pickle'))
        im_file = os.path.join(training_augmented_dir, 'im_seed_' + str(seed).zfill(3) + '_' + base_name)
        dmap_file = os.path.join(training_augmented_dir, 'dmap_seed_' + str(seed).zfill(3) + '_' + base_name)
        mask_file = os.path.join(training_augmented_dir, 'mask_seed_' + str(seed).zfill(3) + '_' + base_name)
        contour_file = os.path.join(training_augmented_dir, 'contour_seed_' + str(seed).zfill(3) + '_' + base_name)
        lab_file = os.path.join(training_augmented_dir, 'lab_seed_' + str(seed).zfill(3) + '_' + base_name)

        # check whether output files already exist
        is_im_file = os.path.isfile(im_file)
        is_dmap_file = os.path.isfile(dmap_file)
        is_mask_file = os.path.isfile(mask_file)
        is_contour_file = os.path.isfile(contour_file)
        is_lab_file = os.path.isfile(lab_file)
        if np.all([is_im_file, is_dmap_file, is_mask_file, is_contour_file, is_lab_file]):
            print('Skipping ... output files already exist')
            continue
        else:
            print('Saving')

        # convert transform from keras to skimage format
        transform_skimage, _ = cytometer.utils.keras2skimage_transform(transform[i], input_shape=im.shape[1:3])

        # apply affine transformation
        im_augmented = warp(im[i, :, :, :], transform_skimage.inverse, order=1, preserve_range=True)
        dmap_augmented = warp(dmap[i, :, :, 0], transform_skimage.inverse, order=1, preserve_range=True)
        mask_augmented = warp(mask[i, :, :, 0], transform_skimage.inverse, order=0, preserve_range=True)
        contour_augmented = warp(contour[i, :, :, 0], transform_skimage.inverse, order=0, preserve_range=True)
        lab_augmented = warp(lab[i, :, :, 0], transform_skimage.inverse, order=0, preserve_range=True)

        # convert to types for display and save to file
        im_augmented = (255 * im_augmented).astype(np.uint8)
        dmap_augmented = dmap_augmented.astype(np.float32)
        mask_augmented = mask_augmented.astype(np.uint8)
        contour_augmented = contour_augmented.astype(np.uint8)
        lab_augmented = lab_augmented.astype(np.uint8)

        if DEBUG:
            plt.clf()
            plt.subplot(321)
            plt.imshow(im[i, :, :, :])
            plt.subplot(322)
            plt.imshow(im_augmented)
            plt.subplot(323)
            plt.imshow(dmap_augmented)
            plt.subplot(324)
            plt.imshow(mask_augmented)
            plt.subplot(325)
            plt.imshow(contour_augmented)
            plt.subplot(326)
            plt.imshow(lab_augmented)

        # save tranformation in keras format
        pickle.dump(transform[i], open(transform_file, 'wb'))

        # save transformed image
        im_out = Image.fromarray(im_augmented, mode='RGB')
        im_out.save(im_file)

        # save distance transforms (note: we have to save as float mode)
        im_out = Image.fromarray(dmap_augmented, mode='F')
        im_out.save(dmap_file)

        # save mask (note: we can save as black and white 1 byte)
        im_out = Image.fromarray(mask_augmented, mode='L')
        im_out.save(mask_file)

        # save contours (note: we can save as black and white 1 byte)
        im_out = Image.fromarray(contour_augmented, mode='L')
        im_out.save(contour_file)

        # save labels (note: we can save as black and white 1 byte)
        im_out = Image.fromarray(lab_augmented, mode='L')
        im_out.save(lab_file)
