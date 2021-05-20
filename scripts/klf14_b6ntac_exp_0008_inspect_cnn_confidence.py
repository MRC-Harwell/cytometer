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

import glob
import numpy as np

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# limit GPU memory used
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

# Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
# function
import keras
import keras.backend as K
import cytometer.data
import cytometer.models
import matplotlib.pyplot as plt
from receptivefield.keras import KerasReceptiveField

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = True

'''Load data
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

saved_model_basename = 'klf14_b6ntac_exp_0008_cnn_confidence'  # confidence on cell segmentation

model_name = saved_model_basename + '*.h5'

# load model weights for each fold
model_files = glob.glob(os.path.join(saved_models_dir, model_name))
n_folds = len(model_files)

# load k-fold sets that were used to train the models
saved_model_kfold_filename = os.path.join(saved_models_dir, saved_model_basename + '_info.pickle')
with open(saved_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
im_file_list = aux['file_list']
idx_test_all = aux['idx_test_all']

# correct home directory if we are in a different system than what was used to train the models
im_file_list = cytometer.data.change_home_directory(im_file_list, '/users/rittscher/rcasero', home, check_isfile=True)

'''Load example data to display'''
#
# # load im, seg and mask datasets
# datasets, _, _ = cytometer.data.load_datasets(im_file_list, prefix_from='im', prefix_to=['im', 'seg', 'mask'])
# im = datasets['im']
# seg = datasets['seg']
# mask = datasets['mask']
# del datasets
#
# # number of training images
# n_im = im.shape[0]
#
# if DEBUG:
#     i = 10
#     print('  ** Image: ' + str(i) + '/' + str(n_im - 1))
#     plt.clf()
#     plt.subplot(221)
#     plt.imshow(im[i, :, :, :])
#     plt.title('Histology: ' + str(i))
#     plt.subplot(222)
#     plt.imshow(seg[i, :, :, 0])
#     plt.title('Labels')
#     plt.subplot(223)
#     plt.imshow(mask[i, :, :, 0])
#     plt.title('Mask')

'''Load model and visualise results
'''

# list of model files to inspect
model_files = glob.glob(os.path.join(saved_models_dir, model_name))

fold_i = 0
model_file = model_files[fold_i]

# split the data into training and testing datasets
im_test_file_list, _ = cytometer.data.split_list(im_file_list, idx_test_all[fold_i])

# load im, seg and mask datasets
test_datasets, _, _ = cytometer.data.load_datasets(im_test_file_list, prefix_from='im',
                                                   prefix_to=['im', 'mask'], nblocks=2)
im_test = test_datasets['im']
mask_test = test_datasets['mask']
del test_datasets

# load model
model = keras.models.load_model(model_file)

# set input layer to size of test images
model = cytometer.models.change_input_size(model, batch_shape=(None,) + im_test.shape[1:])

# visualise results
i = 0
# run image through network
confidence_test_pred = model.predict(im_test[i, :, :, :].reshape((1,) + im_test.shape[1:]))

# plot results
plt.clf()
plt.subplot(131)
plt.imshow(im_test[i, :, :, :])
plt.title('histology, i = ' + str(i))
plt.subplot(132)
plt.imshow(mask_test[i, :, :, 0])
plt.title('training confidence')
plt.subplot(133)
plt.imshow(confidence_test_pred[0, :, :, 1])
plt.title('estimated confidence')

# visualise results
i = 18
# run image through network
confidence_test_pred = model.predict(im_test[i, :, :, :].reshape((1,) + im_test.shape[1:]))

# plot results
plt.clf()
plt.subplot(131)
plt.imshow(im_test[i, :, :, :])
plt.title('histology, i = ' + str(i))
plt.subplot(132)
plt.imshow(mask_test[i, :, :, 0])
plt.title('training confidence')
plt.subplot(133)
plt.imshow(confidence_test_pred[0, :, :, 1])
plt.title('estimated confidence')


'''Plot metrics and convergence
'''

fold_i = 0
model_file = model_files[fold_i]

log_filename = os.path.join(saved_models_dir, model_name.replace('*.h5', '.log'))

if os.path.isfile(log_filename):

    # read Keras output
    df_list = cytometer.data.read_keras_training_output(log_filename)

    # plot metrics with every iteration
    plt.clf()
    for df in df_list:
        plt.subplot(311)
        loss_plot, = plt.semilogy(df.index, df.loss, label='loss')
        epoch_ends = np.concatenate((np.where(np.diff(df.epoch))[0], [len(df.epoch)-1, ]))
        epoch_ends_plot1, = plt.semilogy(epoch_ends, df.loss[epoch_ends], 'ro', label='end of epoch')
        plt.legend(handles=[loss_plot, epoch_ends_plot1])
        plt.subplot(312)
        acc_plot, = plt.plot(df.index, df.acc, label='acc')
        epoch_ends = np.concatenate((np.where(np.diff(df.epoch))[0], [len(df.epoch)-1, ]))
        epoch_ends_plot1, = plt.plot(epoch_ends, df.acc[epoch_ends], 'ro', label='end of epoch')
        plt.legend(handles=[acc_plot, epoch_ends_plot1])

