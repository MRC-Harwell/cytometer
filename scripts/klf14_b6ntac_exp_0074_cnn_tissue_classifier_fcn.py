"""
Tissue classifier, using sherrah2016 CNN.

CHANGE over 0061: Train by image instead of by object. No scaling. We just map the histo
input to a pixel-wise binary classifier.

Use hand traced areas of white adipocytes and "other" tissues to train classifier to differentiate.

.svg labels:
  * 0: "Cell" = white adipocyte
  * 1: "Other" = other types of tissue
  * 1: "Brown" = brown adipocytes
  * 0: "Background" = flat background

We assign cells to train or test sets grouped by image. This way, we guarantee that at testing time, the
network has not seen neighbour cells to the ones used for training.
"""

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0074_cnn_tissue_classifier_fcn'
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
import inspect

# other imports
import glob
import shutil
import datetime
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import time
import random

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AvgPool2D, Activation

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

# number of epochs for training
epochs = 10

# data augmentation factor (e.g. "10" means that we generate 9 augmented images + the original input image)
augment_factor = 10

# area (pixel**2) of the smallest object we accept as a cell (pi * (16 pixel)**2 = 804.2 pixel**2)
smallest_cell_area = 804

# training window length
training_window_len = 401

# remove from training cells that don't have a good enough overlap with a reference label
smallest_dice = 0.5

# segmentations with Dice >= threshold are accepted
dice_threshold = 0.9

# batch size for training
batch_size = 2


'''CNN Model
'''


def fcn_sherrah2016_classifier(input_shape, for_receptive_field=False):

    cnn_input = Input(shape=input_shape, dtype='float32', name='input_image')

    x = Conv2D(filters=32, kernel_size=(5, 5), strides=1, dilation_rate=1, padding='same')(cnn_input)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(3, 3), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)

    x = Conv2D(filters=int(96/2), kernel_size=(5, 5), strides=1, dilation_rate=2, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(5, 5), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(x)

    x = Conv2D(filters=int(128/2), kernel_size=(3, 3), strides=1, dilation_rate=4, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(9, 9), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(9, 9), strides=1, padding='same')(x)

    x = Conv2D(filters=int(196/2), kernel_size=(3, 3), strides=1, dilation_rate=8, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(17, 17), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(17, 17), strides=1, padding='same')(x)

    x = Conv2D(filters=int(512/2), kernel_size=(3, 3), strides=1, dilation_rate=16, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
    else:
        x = Activation('relu')(x)

    # dimensionality reduction
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
    else:
        x = Activation('relu')(x)

    x = Conv2D(filters=8, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)

    if for_receptive_field:
        x = Activation('linear')(x)
    else:
        x = Activation('relu')(x)

    # x = Conv2D(filters=1, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)
    x = Conv2D(filters=2, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)

    # classification output
    # classification_output = Activation('hard_sigmoid', name='classification_output')(x)
    classification_output = Activation('softmax', name='classification_output')(x)

    return Model(inputs=cnn_input, outputs=[classification_output])


'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

saved_contour_model_basename = 'klf14_b6ntac_exp_0055_cnn_contour'

# load list of images, and indices for training vs. testing indices
contour_model_kfold_filename = os.path.join(saved_models_dir, saved_contour_model_basename + '_kfold_info.pickle')
with open(contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
file_list = aux['file_list']
idx_test_all = aux['idx_test']
idx_train_all = aux['idx_train']

# number of images
n_im = len(file_list)

'''Process the data
'''

# generator of random transformations
data_gen_args_float32 = dict(
    rotation_range=90,     # randomly rotate images up to 90 degrees
    fill_mode="constant",  # fill points outside boundaries with zeros
    cval=0,                #
    zoom_range=0.1,        # [1-zoom_range, 1+zoom_range]
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True,    # randomly flip images
    shear_range=15,        # shear angle in counter-clockwise direction in degrees
    dtype=np.float32       # explicit type out
)
datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args_float32)

# start timer
t0 = time.time()

# init output
im_array_all = []
out_class_all = []
out_mask_all = []
contour_type_all = []
transform_all = []
i_all = []

# correct home directory in file paths
file_list = cytometer.data.change_home_directory(list(file_list), '/users/rittscher/rcasero', home, check_isfile=True)

# loop files with hand traced contours
for i, file_svg in enumerate(file_list):

    print('file ' + str(i) + '/' + str(len(file_list) - 1))

    # change file extension from .svg to .tif
    file_tif = file_svg.replace('.svg', '.tif')

    # open histology training image
    im = Image.open(file_tif)

    # make array copy
    im_array_0 = np.array(im)

    if DEBUG:
        plt.clf()
        plt.imshow(im)

    # read the ground truth cell contours in the SVG file. This produces a list [contour_0, ..., contour_N-1]
    # where each contour_i = [(X_0, Y_0), ..., (X_P-1, X_P-1)]
    cell_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False,
                                                            minimum_npoints=3)
    other_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Other', add_offset_from_filename=False,
                                                             minimum_npoints=3)
    brown_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Brown', add_offset_from_filename=False,
                                                             minimum_npoints=3)
    background_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Background', add_offset_from_filename=False,
                                                                  minimum_npoints=3)
    contours = cell_contours + other_contours + brown_contours + background_contours

    # make a list with the type of cell each contour is classified as
    contour_type = [np.zeros(shape=(len(cell_contours),), dtype=np.uint8),  # 0: white-adipocyte
                    np.ones(shape=(len(other_contours),), dtype=np.uint8),  # 1: other types of tissue
                    np.ones(shape=(len(brown_contours),), dtype=np.uint8),  # 1: brown cells (treated as "other" tissue)
                    np.zeros(shape=(len(background_contours),), dtype=np.uint8)] # 0: background
    contour_type = np.concatenate(contour_type)
    contour_type_all.append(contour_type)

    print('Cells: ' + str(len(cell_contours)))
    print('Other: ' + str(len(other_contours)))
    print('Brown: ' + str(len(brown_contours)))
    print('Background: ' + str(len(background_contours)))

    if (len(contours) == 0):
        print('No contours... skipping')
        continue

    # estimate a large enough size to contain most of the rotations applied with augmentation
    transform_worst = {'theta': 45, 'tx': 0, 'ty': 0, 'shear': 0, 'zx': 1.0, 'zy': 1.0,
                       'flip_horizontal': 0, 'flip_vertical': 0, 'channel_shift_intensity': None, 'brightness': None}
    skimage_transform_worst, output_shape_worst = \
        cytometer.utils.keras2skimage_transform(transform_worst, input_shape=im_array_0.shape[0:2], output_shape='full')

    # generate random transformations. We give this its own loop, so that it's easy to replicate, if necessary
    transform = []
    random.seed(i)
    for aug in range(augment_factor):
        # generate random transformation
        transform.append(datagen.get_random_transform(img_shape=im_array_0.shape,
                                                      seed=random.randint(0, 2 ** 32)))
    # blank out the first transform, so that the original data will be used without transformation
    transform[0] = {'theta': 0, 'tx': 0, 'ty': 0, 'shear': 0, 'zx': 1.0, 'zy': 1.0,
                    'flip_horizontal': 0, 'flip_vertical': 0, 'channel_shift_intensity': None, 'brightness': None}

    for aug in range(augment_factor):

        # convert keras transform to skimage format
        skimage_transform, output_shape = \
            cytometer.utils.keras2skimage_transform(transform[aug], input_shape=im_array_0.shape[0:2],
                                                    output_shape=output_shape_worst)

        # apply augmentation transformation to image
        im_array = cytometer.utils.transform_im(im_array_0, skimage_transform, order=1,
                                                output_shape=output_shape_worst)

        # initialise arrays for training
        out_class = np.zeros(shape=im_array.shape[0:2], dtype=np.uint8)
        out_mask = np.zeros(shape=im_array.shape[0:2], dtype=np.uint8)

        if DEBUG:
            plt.clf()
            plt.subplot(121)
            plt.imshow(im_array_0)
            plt.scatter((im_array_0.shape[1]-1)/2.0, (im_array_0.shape[0]-1)/2.0)
            plt.subplot(122)
            plt.cla()
            plt.imshow(im_array)
            plt.scatter((im_array.shape[1]-1)/2.0, (im_array.shape[0]-1)/2.0)

        # loop ground truth cell contours
        for j, contour in enumerate(contours):

            # apply transformation to contour
            contour_0 = contour.copy()
            contour = cytometer.utils.transform_coords(contour, skimage_transform)

            if DEBUG:
                plt.clf()

                plt.subplot(121)
                plt.imshow(im_array_0)
                plt.plot([p[0] for p in contour_0], [p[1] for p in contour_0])
                xy_c = (np.mean([p[0] for p in contour_0]), np.mean([p[1] for p in contour_0]))
                plt.scatter(xy_c[0], xy_c[1])

                plt.subplot(122)
                plt.imshow(im_array)
                plt.plot([p[0] for p in contour], [p[1] for p in contour])
                xy_c = (np.mean([p[0] for p in contour]), np.mean([p[1] for p in contour]))
                plt.scatter(xy_c[0], xy_c[1])

            # rasterise current ground truth segmentation
            cell_seg_gtruth = Image.new("1", im_array.shape[0:2][::-1], "black")  # I = 32-bit signed integer pixels
            draw = ImageDraw.Draw(cell_seg_gtruth)
            draw.polygon(contour, outline="white", fill="white")
            cell_seg_gtruth = np.array(cell_seg_gtruth, dtype=np.bool)

            if DEBUG:
                plt.subplot(121)
                plt.cla()
                plt.imshow(im_array)
                plt.contour(cell_seg_gtruth.astype(np.uint8))

            # add current object to training output and mask
            out_mask[cell_seg_gtruth] = 1
            out_class[cell_seg_gtruth] = contour_type[j]

        # end for j, contour in enumerate(contours):

        if DEBUG:
            plt.clf()
            plt.subplot(121)
            plt.imshow(im_array)
            plt.contour(out_mask.astype(np.uint8), colors='r')
            plt.title('Mask', fontsize=14)
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(im_array)
            plt.contour(out_class.astype(np.uint8), colors='k')
            plt.title('Class', fontsize=14)
            plt.axis('off')
            plt.tight_layout()

        # add dummy dimensions for keras
        im_array = np.expand_dims(im_array, axis=0)
        out_class = np.expand_dims(out_class, axis=0)
        out_class = np.expand_dims(out_class, axis=3)
        out_mask = np.expand_dims(out_mask, axis=0)

        # convert to float32, if necessary
        im_array = im_array.astype(np.float32)
        out_class = out_class.astype(np.float32)
        out_mask = out_mask.astype(np.float32)

        # scale image intensities from [0, 255] to [0.0, 1.0]
        im_array /= 255

        # append input/output/mask for later use in training
        im_array_all.append(im_array)
        out_class_all.append(out_class)
        out_mask_all.append(out_mask)
        transform_all.append(transform)
        i_all.append(i)

        print('Time so far: ' + str("{:.1f}".format(time.time() - t0)) + ' s')

# collapse lists into arrays
im_array_all = np.concatenate(im_array_all)
out_class_all = np.concatenate(out_class_all)
out_mask_all = np.concatenate(out_mask_all)

'''Convolutional neural network training

    Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
    function
    '''

# list all CPUs and GPUs
device_list = K.get_session().list_devices()

# number of GPUs
# gpu_number = np.count_nonzero(['GPU' in str(x) for x in device_list])
gpu_number = 2

for i_fold in range(len(idx_test_all)):

    print('# Fold ' + str(i_fold) + '/' + str(len(idx_test_all) - 1))

    # test and training image indices
    idx_test = idx_test_all[i_fold]
    idx_train = idx_train_all[i_fold]

    # get cell indices for test and training, based on the image indices
    idx_test = np.where([x in idx_test for x in i_all])[0]
    idx_train = np.where([x in idx_train for x in i_all])[0]

    print('## len(idx_train) = ' + str(len(idx_train)))
    print('## len(idx_test) = ' + str(len(idx_test)))

    # split data into training and testing
    im_array_train = im_array_all[idx_train, :, :, :]
    im_array_test = im_array_all[idx_test, :, :, :]

    out_class_train = out_class_all[idx_train, :, :, :]
    out_class_test = out_class_all[idx_test, :, :, :]

    out_mask_train = out_mask_all[idx_train, :, :]
    out_mask_test = out_mask_all[idx_test, :, :]

    # one-hot encoding of the class
    out_class_train = np.concatenate((1-out_class_train, out_class_train), axis=3)
    out_class_test = np.concatenate((1-out_class_test, out_class_test), axis=3)

    # instantiate model
    with tf.device('/cpu:0'):
        model = fcn_sherrah2016_classifier(input_shape=im_array_train.shape[1:])

    saved_model_filename = os.path.join(saved_models_dir, experiment_id + '_model_fold_' + str(i_fold) + '.h5')

    if gpu_number > 1:  # compile and train model: Multiple GPUs

        # checkpoint to save model after each epoch
        checkpointer = cytometer.model_checkpoint_parallel.ModelCheckpoint(filepath=saved_model_filename,
                                                                           verbose=1, save_best_only=True)
        # compile model
        parallel_model = multi_gpu_model(model, gpus=gpu_number)
        parallel_model.compile(loss={'classification_output': cytometer.utils.binary_focal_loss(alpha=.25, gamma=2)},
                               optimizer='Adadelta',
                               metrics={'classification_output': ['acc']},
                               sample_weight_mode='element')

        # train model
        tic = datetime.datetime.now()

        parallel_model.fit(im_array_train,
                           {'classification_output': out_class_train},
                           sample_weight={'classification_output': out_mask_train},
                           validation_data=(im_array_test,
                                            {'classification_output': out_class_test},
                                            {'classification_output': out_mask_test}),
                           batch_size=batch_size, epochs=epochs, initial_epoch=0,
                           callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))

    else:  # compile and train model: One GPU

        # checkpoint to save model after each epoch
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=saved_model_filename,
                                                       verbose=1, save_best_only=True)

        # compile model
        model.compile(loss={'classification_output': cytometer.utils.binary_focal_loss(alpha=.25, gamma=2)},
                      optimizer='Adadelta',
                      metrics={'classification_output': ['acc']},
                      sample_weight_mode='element')

        # train model
        tic = datetime.datetime.now()
        model.fit(im_array_train,
                  {'classification_output': out_class_train},
                  sample_weight={'classification_output': out_mask_train},
                  validation_data=(im_array_test,
                                   {'classification_output': out_class_test},
                                   {'classification_output': out_mask_test}),
                  batch_size=batch_size, epochs=epochs, initial_epoch=0,
                  callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))


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
