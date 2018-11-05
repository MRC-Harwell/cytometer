"""
Compare the cell area distributions of:

* Hand segmented cells (ground truth)
* Automatically segmented cells that overlap with a ground truth cell and Dice>=0.9
* All automatically segmented cells accepted by quality assessment network
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
import PIL
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import watershed
from skimage.measure import regionprops

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
experiment_id = 'klf14_b6ntac_exp_0031_cell_population_from_segmentation_pipeline'

'''
************************************************************************************************************************
Hand segmented cells (ground truth), no-overlap approximation
************************************************************************************************************************
'''

'''Load data
'''

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(root_data_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# list of all non-overlap original files
im_file_list = glob.glob(os.path.join(training_augmented_dir, 'im_seed_nan_*.tif'))

# read pixel size information
orig_file = os.path.basename(im_file_list[0]).replace('im_seed_nan_', '')
im = PIL.Image.open(os.path.join(training_dir, orig_file))
xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

# load the train and test data: im, seg, dmap and mask data
full_dataset, full_file_list, full_shuffle_idx = \
    cytometer.data.load_datasets(im_file_list, prefix_from='im', prefix_to=['im', 'lab'], nblocks=1)

# remove borders between cells in the lab_train data. For this experiment, we want labels touching each other
for i in range(full_dataset['lab'].shape[0]):
    full_dataset['lab'][i, :, :, 0] = watershed(image=np.zeros(shape=full_dataset['lab'].shape[1:3],
                                                               dtype=full_dataset['lab'].dtype),
                                                markers=full_dataset['lab'][i, :, :, 0],
                                                watershed_line=False)

# relabel background as "0" instead of "1"
full_dataset['lab'][full_dataset['lab'] == 1] = 0

# plot example of data
if DEBUG:
    i = 0
    plt.clf()
    plt.subplot(121)
    plt.imshow(full_dataset['im'][i, :, :, :])
    plt.subplot(122)
    plt.imshow(full_dataset['lab'][i, :, :, 0])

# loop images
df = None
for i in range(full_dataset['lab'].shape[0]):

    # get area of each cell in um^2
    props = regionprops(full_dataset['lab'][i, :, :, 0])
    area = np.array([x['area'] for x in props]) * xres * yres

    # create dataframe with metainformation from mouse
    df_window = cytometer.data.tag_values_with_mouse_info(metainfo, os.path.basename(full_file_list['im'][i]),
                                                          area, values_tag='area', tags_to_keep=['id', 'ko', 'sex'])

    # add a column with the window filename. This is later used in the linear models
    df_window['file'] = os.path.basename(full_file_list['im'][i])

    # create new total dataframe, or concat to existing one
    if df is None:
        df = df_window
    else:
        df = pd.concat([df, df_window], axis=0, ignore_index=True)


# make sure that in the boxplots PAT comes before MAT
df['ko'] = df['ko'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))

# plot boxplots for f/m, PAT/MAT comparison as in Nature Genetics paper
plt.clf()
ax = plt.subplot(121)
df[df['sex'] == 'f'].boxplot(column='area', by='ko', ax=ax, notch=True)
#ax.set_ylim(0, 2e4)
ax.set_title('female', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
ax = plt.subplot(122)
df[df['sex'] == 'm'].boxplot(column='area', by='ko', ax=ax, notch=True)
#ax.set_ylim(0, 2e4)
ax.set_title('male', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)

# TODO Here

'''
************************************************************************************************************************

************************************************************************************************************************
'''

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

# free up memory
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
