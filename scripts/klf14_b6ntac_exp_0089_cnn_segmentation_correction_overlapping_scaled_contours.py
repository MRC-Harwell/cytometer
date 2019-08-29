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
experiment_id = 'klf14_b6ntac_exp_0089_cnn_segmentation_correction_overlapping_scaled_contours'
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
import keras_contrib

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AvgPool2D, Activation, BatchNormalization

# for data parallelism in keras models
from keras.utils import multi_gpu_model

import cytometer.model_checkpoint_parallel
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

# whether to process images/segmentations/contours/etc. to create the inputs and outputs to train the neural network.
# If False, then we assume those data have been processed previously, and will be loaded from a file
PREPARE_TRAINING_DATA = False

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


'''CNN Model
'''


def fcn_sherrah2016_regression(input_shape, for_receptive_field=False):

    def activation_if(for_receptive_field, x):
        if for_receptive_field:
            # for estimation of receptive field, we need to use linear operators
            x = Activation('linear')(x)
            x = BatchNormalization(axis=3)(x)
        else:
            # for regular training and inference, we use non-linear operators
            x = Activation('relu')(x)
            x = BatchNormalization(axis=3)(x)
        return x

    def activation_pooling_if(for_receptive_field, pool_size, x):
        if for_receptive_field:
            # for estimation of receptive field, we need to use linear operators
            x = Activation('linear')(x)
            x = AvgPool2D(pool_size=pool_size, strides=1, padding='same')(x)
            x = BatchNormalization(axis=3)(x)
        else:
            # for regular training and inference, we use non-linear operators
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=pool_size, strides=1, padding='same')(x)
            x = BatchNormalization(axis=3)(x)
        return x

    cnn_input = Input(shape=input_shape, dtype='float32', name='input_image')

    x = Conv2D(filters=32, kernel_size=(5, 5), strides=1, dilation_rate=1, padding='same',
               kernel_initializer='he_uniform')(cnn_input)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(3, 3), x=x)

    x = Conv2D(filters=48, kernel_size=(5, 5), strides=1, dilation_rate=2, padding='same',
               kernel_initializer='he_uniform')(x)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(5, 5), x=x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, dilation_rate=4, padding='same',
               kernel_initializer='he_uniform')(x)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(9, 9), x=x)

    x = Conv2D(filters=98, kernel_size=(3, 3), strides=1, dilation_rate=8, padding='same',
               kernel_initializer='he_uniform')(x)
    x = activation_pooling_if(for_receptive_field=for_receptive_field, pool_size=(17, 17), x=x)

    x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, dilation_rate=16, padding='same',
               kernel_initializer='he_uniform')(x)
    x = activation_if(for_receptive_field=for_receptive_field, x=x)

    # dimensionality reduction to 1 feature map
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same',
               kernel_initializer='he_uniform')(x)
    x = activation_if(for_receptive_field=for_receptive_field, x=x)

    x = Conv2D(filters=8, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same',
               kernel_initializer='he_uniform')(x)
    x = activation_if(for_receptive_field=for_receptive_field, x=x)

    regression_output = Conv2D(filters=1, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same',
                               kernel_initializer='he_uniform', name='regression_output')(x)

    return Model(inputs=cnn_input, outputs=[regression_output])


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

'''Prepare the training and test data
'''

if PREPARE_TRAINING_DATA:
    # start timer
    t0 = time.time()

    # init output
    window_im_all = []
    window_out_all = []
    window_mask_loss_all = []
    window_idx_all = []
    for i, file_svg in enumerate(file_svg_list):

        print('file ' + str(i) + '/' + str(len(file_svg_list) - 1))

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

            # loop different perturbations in the mask to have a collection of better and worse
            # segmentations
            for inc in [-0.20, -0.15, -0.10, -.07, -0.03, 0.0, 0.03, 0.07, 0.10, 0.15, 0.20]:

                # erode or dilate the ground truth mask to create the segmentation mask
                cell_seg = cytometer.utils.quality_model_mask(cell_seg_gtruth, quality_model_type='0_1_prop_band',
                                                              quality_model_type_param=inc)[0, :, :, 0].astype(np.uint8)

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

                # create the loss mask

                #   all space covered by either the ground truth or segmentation
                cell_mask_loss = np.logical_or(cell_seg_gtruth, cell_seg).astype(np.uint8)

                #   dilate loss mask so that it also covers part of the background
                cell_mask_loss = cytometer.utils.quality_model_mask(cell_mask_loss, quality_model_type='0_1_prop_band',
                                                                    quality_model_type_param=0.30)[0, :, :, 0]

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
                    plt.contour(cell_seg_gtruth, linewidths=1, levels=[0.5], colors='green')
                    plt.contour(cell_seg, linewidths=1, levels=[0.5], colors='blue')
                    plt.contour(cell_mask_loss, linewidths=1, levels=[0.5], colors='red')
                    plt.plot((bbox_x0, bbox_xend, bbox_xend, bbox_x0, bbox_x0),
                             (bbox_y0, bbox_y0, bbox_yend, bbox_yend, bbox_y0), 'black')

                    plt.subplot(222)
                    plt.cla()
                    plt.imshow(window_im)
                    plt.contour(window_seg_gtruth, linewidths=1, levels=[0.5], colors='green')
                    plt.contour(window_seg, linewidths=1, levels=[0.5], colors='blue')
                    plt.contour(window_mask_loss, linewidths=1, levels=[0.5], colors='red')

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
                    plt.contour(window_mask_loss, linewidths=1, levels=[0.5], colors='red')

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
                window_idx_all.append(np.array([i, j]))

        print('Time so far: ' + str("{:.1f}".format(time.time() - t0)) + ' s')

    # collapse lists into arrays
    window_out_all = np.concatenate(window_out_all)
    window_mask_loss_all = np.concatenate(window_mask_loss_all)
    window_idx_all = np.vstack(window_idx_all)
    window_im_all = np.concatenate(window_im_all)

    # scale intensities to [0.0, 1.0]
    window_im_all /= 255

    # save data to file
    np.savez(os.path.join(saved_models_dir, experiment_id + '_data.npz'),
             window_im_all=window_im_all, window_out_all=window_out_all,
             window_mask_loss_all=window_mask_loss_all, window_idx_all=window_idx_all)

else:  # PREPARE_TRAINING_DATA

    result = np.load(os.path.join(saved_models_dir, experiment_id + '_data.npz'))
    window_im_all = result['window_im_all']
    window_out_all = result['window_out_all']
    window_mask_loss_all = result['window_mask_loss_all']
    window_idx_all = result['window_idx_all']
    del result

'''Convolutional neural network training

    Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
    function
    '''

# list all CPUs and GPUs
device_list = K.get_session().list_devices()

# number of GPUs
gpu_number = np.count_nonzero([':GPU:' in str(x) for x in device_list])

for i_fold in range(n_folds):

    print('# Fold ' + str(i_fold) + '/' + str(n_folds - 1))

    # HACK:
    if i_fold <= 1:
        print('Skipping')
        continue

    # test and training image indices
    idx_test = idx_test_all[i_fold]
    idx_train = idx_train_all[i_fold]

    # memory-map the precomputed data
    result = np.load(os.path.join(saved_models_dir, experiment_id + '_data.npz'), mmap_mode='r')
    window_idx_all = result['window_idx_all']

    # get cell indices for test and training, based on the image indices
    idx_test = np.where([x in idx_test for x in window_idx_all[:, 0]])[0]
    idx_train = np.where([x in idx_train for x in window_idx_all[:, 0]])[0]

    print('## len(idx_train) = ' + str(len(idx_train)))
    print('## len(idx_test) = ' + str(len(idx_test)))

    # split data into training and testing
    window_im_test = window_im_all[idx_test, :, :, :]
    window_im_train = window_im_all[idx_train, :, :, :]

    window_out_test = window_out_all[idx_test, :, :]
    window_out_train = window_out_all[idx_train, :, :]

    window_mask_loss_test = window_mask_loss_all[idx_test, :]
    window_mask_loss_train = window_mask_loss_all[idx_train, :]

    # window_im_test = result['window_im_all'][idx_test, :, :, :]
    # window_im_train = result['window_im_all'][idx_train, :, :, :]
    #
    # window_out_test = result['window_out_all'][idx_test, :, :]
    # window_out_train = result['window_out_all'][idx_train, :, :]
    #
    # window_mask_loss_test = result['window_mask_loss_all'][idx_test, :]
    # window_mask_loss_train = result['window_mask_loss_all'][idx_train, :]

    # instantiate model
    with tf.device('/cpu:0'):
        model = fcn_sherrah2016_regression(input_shape=window_im_train.shape[1:])

    # output filenames
    saved_model_filename = os.path.join(saved_models_dir, experiment_id + '_model_fold_' + str(i_fold) + '.h5')
    saved_logs_dir = os.path.join(saved_models_dir, experiment_id + '_logs_fold_' + str(i_fold))

    # checkpoint to save model after each epoch
    checkpointer = cytometer.model_checkpoint_parallel.ModelCheckpoint(filepath=saved_model_filename,
                                                                       verbose=1, save_best_only=False)
    
    # callback to write a log for TensorBoard
    # Note: run this on the server where the training is happening:
    #       tensorboard --logdir=saved_logs_dir
    tensorboard = keras.callbacks.TensorBoard(log_dir=saved_logs_dir)

    # call for cyclical learning rate
    clr = keras_contrib.callbacks.CyclicLR(
        mode='triangular2',
        base_lr=1e-7,
        max_lr=1e-2,
        step_size=8 * (window_im_train.shape[0] // batch_size))

    # compile model
    parallel_model = multi_gpu_model(model, gpus=gpu_number)
    parallel_model.compile(loss={'regression_output': 'mse'},
                           optimizer='Adadelta',
                           metrics={'regression_output': ['mse', 'mae']},
                           sample_weight_mode='element')

    # train model
    tic = datetime.datetime.now()
    hist = parallel_model.fit(window_im_train,
                              {'regression_output': window_out_train},
                              sample_weight={'regression_output': window_mask_loss_train},
                              validation_data=(window_im_test,
                                               {'regression_output': window_out_test},
                                               {'regression_output': window_mask_loss_test}),
                              batch_size=batch_size, epochs=epochs, initial_epoch=0,
                              callbacks=[checkpointer, clr, tensorboard])
    toc = datetime.datetime.now()
    print('Training duration: ' + str(toc - tic))

    # cast history values to a type that is JSON serializable
    history = hist.history
    for key in history.keys():
        history[key] = list(map(float, history[key]))

    # save training history
    history_filename = os.path.join(saved_models_dir, experiment_id + '_history_fold_' + str(i_fold) + '.json')
    with open(history_filename, 'w') as f:
        json.dump(history, f)

    cytometer.utils.clear_mem()
    del window_im_train
    del window_im_test
    del window_out_train
    del window_out_test
    del window_mask_loss_train
    del window_mask_loss_test
