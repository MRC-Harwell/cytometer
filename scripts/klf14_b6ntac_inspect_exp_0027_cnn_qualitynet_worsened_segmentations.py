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
import numpy as np
import matplotlib.pyplot as plt

# use CPU for testing on laptop
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

import cytometer.data
import cytometer.models

# limit GPU memory used
import tensorflow as tf
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
epochs = 20

# area (pixel**2) of the smallest object we accept as a cell (pi * (16 pixel)**2 = 804.2 pixel**2)
smallest_cell_area = 804

# training window length
training_window_len = 401

# threshold for valid/invalid segmentation
valid_threshold = 0.9

'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0027_cnn_qualitynet_worsened_segmentations'

'''Load data
'''

# in this script, we actually only work with fold 0
i_fold = 0

# one-cell windows from the segmentations obtained with our pipeline
# ['train_cell_im', 'train_cell_reflab', 'train_cell_testlab', 'train_cell_dice', 'test_cell_im',
# 'test_cell_reflab', 'test_cell_testlab', 'test_cell_dice']
onecell_filename = os.path.join(training_augmented_dir, 'onecell_kfold_' + str(i_fold).zfill(2) + '_worsen_000.npz')
dataset_orig = np.load(onecell_filename)

# read the data into memory
train_cell_im = dataset_orig['train_cell_im']
train_cell_reflab = dataset_orig['train_cell_reflab']
train_cell_testlab = dataset_orig['train_cell_testlab']
train_cell_dice = dataset_orig['train_cell_dice']
test_cell_im = dataset_orig['test_cell_im']
test_cell_reflab = dataset_orig['test_cell_reflab']
test_cell_testlab = dataset_orig['test_cell_testlab']
test_cell_dice = dataset_orig['test_cell_dice']

# multiply histology by segmentations to have a single input tensor
masked_test_cell_im = test_cell_im * np.repeat(test_cell_testlab.astype(np.float32),
                                               repeats=test_cell_im.shape[3], axis=3)

# save memory
del dataset_orig

'''Load neural network for predictions
'''

model_name = experiment_id + '_model_fold_' + str(i_fold) + '.h5'

saved_model_filename = os.path.join(saved_models_dir, model_name)

# list of model files to inspect
model_files = glob.glob(os.path.join(saved_models_dir, model_name))

model_file = model_files[i_fold]

# load model
model = keras.models.load_model(model_file)

test_cell_preddice = np.zeros(shape=test_cell_dice.shape, dtype=test_cell_dice.dtype)
for i in range(test_cell_im.shape[0]):

    if not i % 20:
        print('Test image: ' + str(i) + '/' + str(test_cell_im.shape[0]-1))

    # predict Dice coefficient for test segmentation
    test_cell_preddice[i] = model.predict(np.expand_dims(masked_test_cell_im[i, :, :, :], axis=0))

# plot scatter plot all predicted vs. ground truth Dice values
if DEBUG:
    plt.clf()
    plt.scatter(test_cell_dice, test_cell_preddice)
    plt.plot([0.2, 1.0], [0.2, 1.0])
    plt.plot([.9, .9], [.1, 1.0], 'r')
    plt.plot([.1, 1.0], [.9, .9], 'r')
    plt.xlabel('Ground truth')
    plt.ylabel('Estimated')
    plt.title('Dice coefficient')

# Pearson coefficient
print('Pearson coeff = ' + "{:.2f}".format(np.corrcoef(np.stack((test_cell_dice, test_cell_preddice)))[0, 1]))

# confusion table: estimated vs ground truth
est0_gt0 = np.count_nonzero(np.logical_and(test_cell_preddice < valid_threshold, test_cell_dice < valid_threshold)) \
    / len(test_cell_preddice)
est0_gt1 = np.count_nonzero(np.logical_and(test_cell_preddice < valid_threshold, test_cell_dice >= valid_threshold)) \
    / len(test_cell_preddice)
est1_gt0 = np.count_nonzero(np.logical_and(test_cell_preddice >= valid_threshold, test_cell_dice < valid_threshold)) \
    / len(test_cell_preddice)
est1_gt1 = np.count_nonzero(np.logical_and(test_cell_preddice >= valid_threshold, test_cell_dice >= valid_threshold)) \
    / len(test_cell_preddice)

print(np.array([["{:.2f}".format(est1_gt0), "{:.2f}".format(est1_gt1)],
                ["{:.2f}".format(est0_gt0), "{:.2f}".format(est0_gt1)]]))


# plot prediction
if DEBUG:
    i = 150
    plt.clf()
    plt.imshow(test_cell_im[i, :, :, :])
    plt.contour(test_cell_testlab[i, :, :, 0], levels=1, colors='green')
    plt.title('Dice: ' + "{:.2f}".format(test_cell_dice[i]) + ' (ground truth)\n'
              + "{:.2f}".format(test_cell_preddice[i]) + ' (estimated)')

    i = 1001
    plt.clf()
    plt.imshow(test_cell_im[i, :, :, :])
    plt.contour(test_cell_testlab[i, :, :, 0], levels=1, colors='green')
    plt.title('Dice: ' + "{:.2f}".format(test_cell_dice[i]) + ' (ground truth)\n'
              + "{:.2f}".format(test_cell_preddice[i]) + ' (estimated)')

'''Look at distribution of Dice coefficients
'''

plt.clf()
plt.hist(np.concatenate((test_cell_dice, train_cell_dice)))
plt.title('Distribution of Dice coeff values')
plt.xlabel('Dice coeff')

'''Plot metrics and convergence
'''

fold_i = 0
model_file = model_files[fold_i]

log_filename = os.path.join(saved_models_dir, experiment_id + '.log')

if os.path.isfile(log_filename):

    # read Keras output
    df_list = cytometer.data.read_keras_training_output(log_filename)

    # plot metrics with every iteration
    plt.clf()
    for df in df_list:
        plt.subplot(211)
        loss_plot, = plt.semilogy(df.index, df.loss, label='loss')
        epoch_ends = np.concatenate((np.where(np.diff(df.epoch))[0], [len(df.epoch)-1, ]))
        epoch_ends_plot1, = plt.semilogy(epoch_ends, df.loss[epoch_ends], 'ro', label='end of epoch')
        plt.legend(handles=[loss_plot, epoch_ends_plot1])
        plt.subplot(212)
        regr_mae_plot, = plt.plot(df.index, df.mean_absolute_error, label='dmap mae')
        regr_mse_plot, = plt.plot(df.index, np.sqrt(df.mean_squared_error), label='sqrt(dmap mse)')
        regr_mae_epoch_ends_plot2, = plt.plot(epoch_ends, df.mean_absolute_error[epoch_ends], 'ro', label='end of epoch')
        regr_mse_epoch_ends_plot2, = plt.plot(epoch_ends, np.sqrt(df.mean_squared_error[epoch_ends]), 'ro', label='end of epoch')
        plt.legend(handles=[regr_mae_plot, regr_mse_plot, regr_mae_epoch_ends_plot2])
