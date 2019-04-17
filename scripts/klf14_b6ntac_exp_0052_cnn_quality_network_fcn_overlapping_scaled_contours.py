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
import time
import random

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

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

# number of folds for k-fold cross validation
n_folds = 10

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


'''CNN Model
'''


def fcn_sherrah2016_regression(input_shape, for_receptive_field=False):

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

    # regression output
    regression_output = Conv2D(filters=1, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same',
                               name='regression_output')(x)

    return Model(inputs=cnn_input, outputs=[regression_output])


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

# start timer
t0 = time.time()

# we are interested only in .tif files for which we created hand segmented contours
file_list = glob.glob(os.path.join(training_data_dir, '*.svg'))

# init output
window_im_all = []
window_out_all = []
window_mask_loss_all = []
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

        # loop different perturbations in the mask to have a collection of better and worse
        # segmentations
        for inc in [-0.20, -0.15, -0.10, -.07, -0.03, 0.0, 0.03, 0.07, 0.10, 0.15, 0.20]:

            # erode or dilate the ground truth mask to create the segmentation mask
            cell_seg = cytometer.utils.quality_model_mask(cell_seg_gtruth, quality_model_type='0_1_prop_band',
                                                          quality_model_type_param=inc)[0, :, :, 0].astype(np.uint8)

            if DEBUG:
                plt.subplot(223)
                plt.imshow(cell_seg)

            # create the loss mask. This will be the largest mask too, so we use it to decide the size of the bounding
            # box to crop

            #   all space covered by either the ground truth or segmentation
            cell_mask_loss = np.logical_or(cell_seg_gtruth, cell_seg).astype(np.uint8)

            #   dilate loss mask so that it also covers part of the background
            cell_mask_loss = cytometer.utils.quality_model_mask(cell_mask_loss, quality_model_type='0_1_prop_band',
                                                                quality_model_type_param=0.30)[0, :, :, 0]

            # compute bounding box that contains the mask, and leaves some margin
            bbox_x0, bbox_y0, bbox_xend, bbox_yend = \
                cytometer.utils.bounding_box_with_margin(cell_mask_loss, coordinates='xy', inc=0.40)
            bbox_r0, bbox_c0, bbox_rend, bbox_cend = \
                cytometer.utils.bounding_box_with_margin(cell_mask_loss, coordinates='rc', inc=0.40)

            if DEBUG:
                plt.subplot(224)
                plt.cla()
                plt.imshow(cell_mask_loss)
                plt.plot((bbox_x0, bbox_xend, bbox_xend, bbox_x0, bbox_x0),
                         (bbox_y0, bbox_y0, bbox_yend, bbox_yend, bbox_y0))

            # crop image and masks according to bounding box
            window_im = cytometer.utils.extract_bbox(im_array, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))
            window_seg_gtruth = cytometer.utils.extract_bbox(cell_seg_gtruth, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))
            window_seg = cytometer.utils.extract_bbox(cell_seg, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))
            window_mask_loss = cytometer.utils.extract_bbox(cell_mask_loss, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))

            if DEBUG:
                plt.clf()
                plt.subplot(221)
                plt.cla()
                plt.imshow(im_array)
                plt.contour(cell_seg_gtruth, linewidths=1, levels=0.5, colors='green')
                plt.contour(cell_seg, linewidths=1, levels=0.5, colors='blue')
                plt.contour(cell_mask_loss, linewidths=1, levels=0.5, colors='red')
                plt.plot((bbox_x0, bbox_xend, bbox_xend, bbox_x0, bbox_x0),
                         (bbox_y0, bbox_y0, bbox_yend, bbox_yend, bbox_y0), 'black')

                plt.subplot(222)
                plt.cla()
                plt.imshow(window_im)
                plt.contour(window_seg_gtruth, linewidths=1, levels=0.5, colors='green')
                plt.contour(window_seg, linewidths=1, levels=0.5, colors='blue')
                plt.contour(window_mask_loss, linewidths=1, levels=0.5, colors='red')

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

            # resize the training image to training window size
            training_size = (training_window_len, training_window_len)
            window_im = cytometer.utils.resize(window_im, size=training_size, resample=Image.LINEAR)
            window_out = cytometer.utils.resize(window_out, size=training_size, resample=Image.NEAREST)
            window_mask_loss = cytometer.utils.resize(window_mask_loss, size=training_size, resample=Image.NEAREST)

            if DEBUG:
                plt.subplot(224)
                plt.cla()
                aux = 0.2989 * window_im[:, :, 0] + 0.5870 * window_im[:, :, 1] + 0.1140 * window_im[:, :, 2]
                plt.imshow(aux)
                plt.contour(window_out, linewidths=1, levels=(-0.5, 0.5), colors='white')
                plt.contour(window_mask_loss, linewidths=1, levels=0.5, colors='red')

            # add dummy dimensions for keras
            window_im = np.expand_dims(window_im, axis=0)

            window_out = np.expand_dims(window_out, axis=0)
            window_out = np.expand_dims(window_out, axis=3)

            window_mask_loss = np.expand_dims(window_mask_loss, axis=0)

            # check sizes and types
            assert(window_im.ndim == 4 and window_im.dtype == np.float32)
            assert(window_out.ndim == 4 and window_out.dtype == np.float32)
            assert(window_mask_loss.ndim == 3 and window_mask_loss.dtype == np.float32)

            # append images to use for training
            window_im_all.append(window_im)
            window_out_all.append(window_out)
            window_mask_loss_all.append(window_mask_loss)

    print('Time so far: ' + str("{:.1f}".format(time.time() - t0)) + ' s')

# collapse lists into arrays
window_im_all = np.concatenate(window_im_all)
window_out_all = np.concatenate(window_out_all)
window_mask_loss_all = np.concatenate(window_mask_loss_all)

'''Split data into training and testing for k-folds
'''

# number of images
n_im = window_im_all.shape[0]

# create k-fold sets to split the data into training vs. testing
kfold_seed = 0
random.seed(kfold_seed)
idx_all = random.sample(range(n_im), n_im)
idx_test_all = np.array_split(idx_all, n_folds)

'''Convolutional neural network training

    Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
    function
    '''

# list all CPUs and GPUs
device_list = K.get_session().list_devices()

# number of GPUs
gpu_number = np.count_nonzero(['GPU' in str(x) for x in device_list])

for k_fold in range(n_folds):

    # test and training indices
    idx_test = idx_test_all[k_fold]
    idx_train = list(set(idx_all) - set(idx_test))

    # split data into training and testing
    window_im_train = window_im_all[idx_train, :, :, :]
    window_im_test = window_im_all[idx_test, :, :, :]

    window_out_train = window_out_all[idx_train, :, :]
    window_out_test = window_out_all[idx_test, :, :]

    window_mask_loss_train = window_mask_loss_all[idx_train, :]
    window_mask_loss_test = window_mask_loss_all[idx_test, :]

    # instantiate model
    with tf.device('/cpu:0'):
        model = fcn_sherrah2016_regression(input_shape=window_im_train.shape[1:])

    saved_model_filename = os.path.join(saved_models_dir, experiment_id + '_model_fold_' + str(k_fold) + '.h5')

    if gpu_number > 1:  # compile and train model: Multiple GPUs

        # checkpoint to save model after each epoch
        checkpointer = cytometer.model_checkpoint_parallel.ModelCheckpoint(filepath=saved_model_filename,
                                                                           verbose=1, save_best_only=True)
        # compile model
        parallel_model = multi_gpu_model(model, gpus=gpu_number)
        parallel_model.compile(loss={'regression_output': 'mse'},
                               optimizer='Adadelta',
                               metrics={'regression_output': ['mse', 'mae']},
                               sample_weight_mode='element')

        # train model
        tic = datetime.datetime.now()
        parallel_model.fit(window_im_train,
                           {'regression_output': window_out_train},
                           sample_weight={'regression_output': window_mask_loss_train},
                           validation_data=(window_im_test,
                                            {'regression_output': window_out_test},
                                            {'regression_output': window_mask_loss_test}),
                           batch_size=10, epochs=epochs, initial_epoch=0,
                           callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))

    else:  # compile and train model: One GPU

        # checkpoint to save model after each epoch
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=saved_model_filename,
                                                       verbose=1, save_best_only=True)

        # compile model
        model.compile(loss={'regression_output': 'mse'},
                      optimizer='Adadelta',
                      metrics={'regression_output': ['mse', 'mae']},
                      sample_weight_mode='element')

        # train model
        tic = datetime.datetime.now()
        model.fit(window_im_train,
                  {'regression_output': window_out_train},
                  sample_weight={'regression_output': window_mask_loss_train},
                  validation_data=(window_im_test,
                                   {'regression_output': window_out_test},
                                   {'regression_output': window_mask_loss_test}),
                  batch_size=10, epochs=epochs, initial_epoch=0,
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
