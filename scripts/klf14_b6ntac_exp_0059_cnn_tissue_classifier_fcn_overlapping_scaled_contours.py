"""
Tissue classifier, using sherrah2016 CNN.

Use hand traced areas of white adipocytes and "other" tissues to train classifier to differentiate.

.svg labels:
  * 0: "Cell" = white adipocyte
  * 1: "Other" = other types of tissue
  * 1: "Brown" = brown adipocytes

We create a bounding box to crop the images, and the scale everything to the same training window size, to remove
differences between cell sizes.

We assign cells to train or test sets grouped by image. This way, we guarantee that at testing time, the
network has not seen neighbour cells to the ones used for training.
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

    # output
    output = Conv2D(filters=2, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same',
                    activation='softmax', name='output')(x)

    return Model(inputs=cnn_input, outputs=[output])


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

# script name to identify this experiment
experiment_id = inspect.getfile(inspect.currentframe())
if experiment_id == '<input>':
    experiment_id = 'unknownscript'
else:
    experiment_id = os.path.splitext(os.path.basename(experiment_id))[0]

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

# start timer
t0 = time.time()

# init output
window_im_all = []
window_seg_gtruth_all = []
window_idx_all = []
window_out_all = []
contour_type_all = []

# loop files with hand traced contours
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
    cell_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False,
                                                            minimum_npoints=3)
    other_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Other', add_offset_from_filename=False,
                                                             minimum_npoints=3)
    brown_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Brown', add_offset_from_filename=False,
                                                             minimum_npoints=3)

    # make a list with the type of cell each contour is classified as
    contour_type = [np.zeros(shape=(len(cell_contours),), dtype=np.uint8),  # 0: white-adipocyte
                    np.ones(shape=(len(other_contours),), dtype=np.uint8),  # 1: other types of tissue
                    np.ones(shape=(len(brown_contours),), dtype=np.uint8)]  # 1: brown cells (treated as "other" tissue)
    contour_type = np.concatenate(contour_type)
    contour_type_all.append(contour_type)

    print('Cells: ' + str(len(cell_contours)))
    print('Other: ' + str(len(other_contours)))
    print('Brown: ' + str(len(brown_contours)))

    # loop ground truth cell contours
    for j, contour in enumerate(cell_contours + other_contours + brown_contours):

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

        # compute bounding box that contains the mask, and leaves some margin
        bbox_x0, bbox_y0, bbox_xend, bbox_yend = \
            cytometer.utils.bounding_box_with_margin(cell_seg_gtruth, coordinates='xy', inc=1.00)
        bbox_r0, bbox_c0, bbox_rend, bbox_cend = \
            cytometer.utils.bounding_box_with_margin(cell_seg_gtruth, coordinates='rc', inc=1.00)

        if DEBUG:
            plt.subplot(222)
            plt.cla()
            plt.imshow(cell_seg_gtruth)
            plt.plot((bbox_x0, bbox_xend, bbox_xend, bbox_x0, bbox_x0),
                     (bbox_y0, bbox_y0, bbox_yend, bbox_yend, bbox_y0))

        # crop image and masks according to bounding box
        window_im = cytometer.utils.extract_bbox(im_array, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))
        window_seg_gtruth = cytometer.utils.extract_bbox(cell_seg_gtruth, (bbox_r0, bbox_c0, bbox_rend, bbox_cend))

        # one-hot encoding of output
        window_out = np.zeros(shape=window_seg_gtruth.shape + (2,), dtype=np.float32)
        window_out[:, :, contour_type[j]] = window_seg_gtruth

        if DEBUG:
            plt.subplot(223)
            plt.cla()
            plt.imshow(window_im)
            plt.contour(window_seg_gtruth, linewidths=1, levels=0.5, colors='green')

        # resize the training image to training window size
        training_size = (training_window_len, training_window_len)
        window_im = cytometer.utils.resize(window_im, size=training_size, resample=Image.LINEAR)
        window_seg_gtruth = cytometer.utils.resize(window_seg_gtruth, size=training_size, resample=Image.NEAREST)
        window_out = cytometer.utils.resize(window_out, size=training_size, resample=Image.NEAREST)

        if DEBUG:
            plt.subplot(224)
            plt.cla()
            plt.imshow(window_im)
            plt.contour(window_seg_gtruth, linewidths=1, levels=0.5, colors='green')

        # add dummy dimensions for keras
        window_im = np.expand_dims(window_im, axis=0)
        window_out = np.expand_dims(window_out, axis=0)
        window_seg_gtruth = np.expand_dims(window_seg_gtruth, axis=0)

        # convert to float32, if necessary
        window_im = window_im.astype(np.float32)
        window_out = window_out.astype(np.float32)
        window_seg_gtruth = window_seg_gtruth.astype(np.float32)

        # scale image intensities from [0, 255] to [0.0, 1.0]
        window_im /= 255

        # check sizes and types
        assert(window_im.ndim == 4 and window_im.dtype == np.float32)
        assert(window_out.ndim == 4 and window_out.dtype == np.float32)
        assert(window_seg_gtruth.ndim == 3 and window_seg_gtruth.dtype == np.float32)

        # append images to use for training
        window_im_all.append(window_im)
        window_out_all.append(window_out)
        window_seg_gtruth_all.append(window_seg_gtruth)
        window_idx_all.append(np.array([i, j]))

    print('Time so far: ' + str("{:.1f}".format(time.time() - t0)) + ' s')

# collapse lists into arrays
window_im_all = np.concatenate(window_im_all)
window_out_all = np.concatenate(window_out_all)
window_seg_gtruth_all = np.concatenate(window_seg_gtruth_all)
window_idx_all = np.vstack(window_idx_all)
contour_type_all = np.concatenate(contour_type_all)

'''Convolutional neural network training

    Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
    function
    '''

# list all CPUs and GPUs
device_list = K.get_session().list_devices()

# number of GPUs
gpu_number = np.count_nonzero(['GPU' in str(x) for x in device_list])

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

    # split data into training and testing
    window_im_train = window_im_all[idx_train, :, :, :]
    window_im_test = window_im_all[idx_test, :, :, :]

    window_out_train = window_out_all[idx_train, :, :, :]
    window_out_test = window_out_all[idx_test, :, :, :]

    window_seg_gtruth_train = window_seg_gtruth_all[idx_train, :, :]
    window_seg_gtruth_test = window_seg_gtruth_all[idx_test, :, :]

    # instantiate model
    with tf.device('/cpu:0'):
        model = fcn_sherrah2016_classifier(input_shape=window_im_train.shape[1:])

    saved_model_filename = os.path.join(saved_models_dir, experiment_id + '_model_fold_' + str(i_fold) + '.h5')

    if gpu_number > 1:  # compile and train model: Multiple GPUs

        # checkpoint to save model after each epoch
        checkpointer = cytometer.model_checkpoint_parallel.ModelCheckpoint(filepath=saved_model_filename,
                                                                           verbose=1, save_best_only=True)
        # compile model
        parallel_model = multi_gpu_model(model, gpus=gpu_number)
        parallel_model.compile(loss={'output': cytometer.utils.focal_loss(alpha=.25, gamma=2)},
                               optimizer='Adadelta',
                               metrics={'output': ['acc']},
                               sample_weight_mode='element')

        # train model
        tic = datetime.datetime.now()

        parallel_model.fit(window_im_train,
                           {'output': window_out_train},
                           sample_weight={'output': window_seg_gtruth_train},
                           validation_data=(window_im_test,
                                            {'output': window_out_test},
                                            {'output': window_seg_gtruth_test}),
                           batch_size=batch_size, epochs=epochs, initial_epoch=0,
                           callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))

    else:  # compile and train model: One GPU

        # checkpoint to save model after each epoch
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=saved_model_filename,
                                                       verbose=1, save_best_only=True)

        # compile model
        model.compile(loss={'output': cytometer.utils.focal_loss(alpha=.25, gamma=2)},
                      optimizer='Adadelta',
                      metrics={'output': ['acc']},
                      sample_weight_mode='element')

        # train model
        tic = datetime.datetime.now()
        model.fit(window_im_train,
                  {'output': window_out_train},
                  sample_weight={'output': window_seg_gtruth_train},
                  validation_data=(window_im_test,
                                   {'output': window_out_test},
                                   {'output': window_seg_gtruth_test}),
                  batch_size=batch_size, epochs=epochs, initial_epoch=0,
                  callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))

        cytometer.utils.clear_mem()
        del window_im_train
        del window_im_test
        del window_out_train
        del window_out_test
        del window_seg_gtruth_train
        del window_seg_gtruth_test

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
