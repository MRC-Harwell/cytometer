"""
Contour segmentation for all folds using focal loss. Use transfer learning from dmap networks.
The CNN has 3 dimensionality reduction layers at the end, instead of 1.

(klf14_b6ntac_exp_0006_cnn_contour only computes fold 0.)
(klf14_b6ntac_exp_0055_cnn_contour.py uses binary crossentropy.)

Training vs testing is done at the histology slide level, not at the window level. This way, we really guarantee that
the network has not been trained with data sampled from the same image as the test data.

Training for the CNN:
* Input: histology
* Output: hand tracked contours, dilated a bit.
* Other: mask for the loss function, to avoid looking outside of where we have contours.
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

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

import cytometer.data
import tensorflow as tf

# limit GPU memory used
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

SAVE_FIGS = False
DEBUG = False

# number of folds for k-fold cross validation
n_folds = 10

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
figures_dir = os.path.join(root_data_dir, 'figures')

# script name that created the folds
folds_basename = 'klf14_b6ntac_exp_0055_cnn_contour'

# script that trained the contour models
contour_model_basename = 'klf14_b6ntac_exp_0068_cnn_contour'

# experiment we are inspecting
experiment_id = 'klf14_b6ntac_exp_0068_cnn_contour'

# load k-folds training and testing data
kfold_info_filename = os.path.join(saved_models_dir, folds_basename + '_kfold_info.pickle')
with open(kfold_info_filename, 'rb') as f:
    kfold_info = pickle.load(f)
file_list = kfold_info['file_list']
idx_test_all = kfold_info['idx_test']
idx_train_all = kfold_info['idx_train']
del kfold_info

# number of images
n_im = len(file_list)

'''Check how the contour CNN responds to histology images
'''

for i_fold in range(n_folds):

    # list of test files in this fold
    file_list_test = np.array(file_list)[idx_test_all[i_fold]]

    for i, file_svg in enumerate(file_list_test):

        print('file ' + str(i) + '/' + str(len(idx_test_all[i_fold]) - 1))

        # change file extension from .svg to .tif
        file_tif = file_svg.replace('.svg', '.tif')

        # open histology training image
        im = Image.open(file_tif)

        # read pixel size information
        xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
        yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

        # make array copy, and cast to expected type and values range
        im_array = np.array(im)
        im_array = im_array.astype(np.float32)
        im_array /= 255.0

        if DEBUG:
            plt.clf()
            plt.subplot(121)
            plt.imshow(im)

        # load contour model
        contour_model_filename = os.path.join(saved_models_dir, contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
        contour_model = keras.models.load_model(contour_model_filename)

        # set input layer to size of images
        contour_model = cytometer.models.change_input_size(contour_model, batch_shape=(None,) + im_array.shape)

        # process histology
        contour = contour_model.predict(np.expand_dims(im_array, axis=0))

        if DEBUG:
            plt.subplot(122)
            plt.cla()
            plt.imshow(contour[0, :, :, 0])

'''Check model weights
'''

for i_fold in range(n_folds):

    print('Fold = ' + str(i_fold) + '/' + str(n_folds - 1))

    # list of test files in this fold
    file_list_test = np.array(file_list)[idx_test_all[i_fold]]

    # load contour model
    contour_model_filename = os.path.join(saved_models_dir,
                                          contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    contour_model = keras.models.load_model(contour_model_filename)

    for lay in [1, 4, 7, 10, 13, 15]:
        layer = contour_model.get_layer(index=lay)
        print('' + str(np.percentile(layer.get_weights()[0], [0, 25, 50, 75, 100])))

    print('')

'''Check outputs from intermediate layers
'''

x = np.expand_dims(im_array, axis=0)
intermediate_layer_model = keras.models.Model(inputs=contour_model.input,
                                              outputs=contour_model.get_layer(index=lay).output)
print(contour_model.get_layer(index=lay).name)
intermediate_output = intermediate_layer_model.predict(x[:, 0:500, 0:500, :])

plt.clf()
plt.imshow(intermediate_output[0, :, :, 0])
# plt.imshow(im_array[0:500, 0:500, :])

lay = 15
w = contour_model.get_layer(index=lay).get_weights()
plt.plot(w[0][0, 0, :, 0])
plt.plot(w[1][0, 0, :, 0])

'''Plot metrics and convergence
'''

log_filename = os.path.join(saved_models_dir, experiment_id + '.log')

if os.path.isfile(log_filename):

    # read Keras output
    df_list = cytometer.data.read_keras_training_output(log_filename)

    # plot metrics with every iteration
    plt.clf()
    for df in df_list:
        plt.subplot(211)
        # loss_plot, = plt.semilogy(df.index, df.loss, label='Loss')
        loss_plot, = plt.plot(df.index, df.loss, label='Loss')
        epoch_ends = np.concatenate((np.where(np.diff(df.epoch))[0], [len(df.epoch)-1, ]))
        # epoch_ends_plot1, = plt.semilogy(epoch_ends, df.loss[epoch_ends], 'ro', label='End of epoch')
        epoch_ends_plot1, = plt.plot(epoch_ends, df.loss[epoch_ends], 'ro', label='End of epoch')
        plt.legend(handles=[loss_plot, epoch_ends_plot1])
        plt.tick_params(labelsize=16)
        plt.subplot(212)
        regr_mae_plot, = plt.plot(df.index, df.acc, label='Acc')
        regr_mae_epoch_ends_plot2, = plt.plot(epoch_ends, df.acc[epoch_ends], 'ro', label='End of epoch')
        plt.legend(handles=[regr_mae_plot, regr_mae_epoch_ends_plot2])
        plt.tick_params(labelsize=16)
        plt.xlabel('Steps', fontsize=16)

if SAVE_FIGS:
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_inspect_exp_0068_training_convergence.png'))
