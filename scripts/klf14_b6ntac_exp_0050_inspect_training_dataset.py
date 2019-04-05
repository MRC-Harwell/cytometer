"""
Using exp 0049 as starting point, examine the training dataset, to try to figure out why the
pipeline can't get right small cells.
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
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import watershed

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

import cytometer.utils
import cytometer.data
import tensorflow as tf

import PIL

# limit GPU memory used
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False
SAVE_FIGS = True

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
quality_threshold = 0.9

# batch size for training
batch_size = 16

'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')
figures_dir = os.path.join(root_data_dir, 'figures_0050')

saved_contour_model_basename = 'klf14_b6ntac_exp_0034_cnn_contour'  # contour

# script name to identify this experiment
experiment_id = inspect.getfile(inspect.currentframe())
if experiment_id == '<input>':
    experiment_id = 'unknownscript'
else:
    experiment_id = os.path.splitext(os.path.basename(experiment_id))[0]

# list of images, and indices for training vs. testing indices
contour_model_kfold_filename = os.path.join(saved_models_dir, saved_contour_model_basename + '_info.pickle')
with open(contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
im_orig_file_list = aux['file_list']
idx_orig_test_all = aux['idx_test_all']

# correct home directory, if necessary
im_orig_file_list = cytometer.data.change_home_directory(im_orig_file_list, home_path_from='/users/rittscher/rcasero',
                                                         home_path_to=home,
                                                         check_isfile=False)

# read pixel size information
im = PIL.Image.open(im_orig_file_list[0])
xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

'''Loop folds
'''

# loop each fold: we split the data into train vs test, train a model, and compute errors with the
# test data. In each fold, the test data is different
# for i_fold, idx_test in enumerate(idx_test_all):
for i_fold, idx_test in enumerate(idx_orig_test_all):

    print('## Fold ' + str(i_fold) + '/' + str(len(idx_orig_test_all) - 1))

    '''Load data
    '''

    # split the data list into training and testing lists
    im_test_file_list, im_train_file_list = cytometer.data.split_list(im_orig_file_list, idx_test)

    # add the augmented image files
    im_train_file_list = cytometer.data.augment_file_list(im_train_file_list, '_nan_', '_*_')

    # load training dataset
    datasets, _, _ = cytometer.data.load_datasets(im_train_file_list, prefix_from='im',
                                                  prefix_to=['im', 'lab', 'seg', 'mask',
                                                             'predlab_kfold_' + str(i_fold).zfill(2)], nblocks=1)
    train_im = datasets['im']
    train_seg = datasets['seg']
    train_mask = datasets['mask']
    train_reflab = datasets['lab']
    train_predlab = datasets['predlab_kfold_' + str(i_fold).zfill(2)]
    del datasets

    # remove borders between labels
    for i in range(train_reflab.shape[0]):
        train_reflab[i, :, :, 0] = watershed(image=np.zeros(shape=train_reflab[i, :, :, 0].shape, dtype=np.uint8),
                                             markers=train_reflab[i, :, :, 0], watershed_line=False)
    # change the background label from 1 to 0
    train_reflab[train_reflab == 1] = 0

    if DEBUG:
        i = 250
        plt.clf()
        plt.subplot(221)
        plt.imshow(train_im[i, :, :, :])
        plt.subplot(222)
        plt.imshow(train_seg[i, :, :, 0])
        plt.subplot(223)
        plt.imshow(train_reflab[i, :, :, 0])
        plt.subplot(224)
        plt.imshow(train_predlab[i, :, :, 0])

    # create one image per pipeline segmentation with a ground truth segmentation
    train_onecell_im, train_onecell_testlab, train_onecell_index_list, train_onecell_reflab, train_onecell_dice = \
        cytometer.utils.one_image_per_label(train_im, train_predlab,
                                            dataset_lab_ref=train_reflab,
                                            training_window_len=training_window_len,
                                            smallest_cell_area=smallest_cell_area,
                                            clear_border_lab=True,
                                            smallest_dice=smallest_dice,
                                            allow_repeat_ref=True)

    # clear memory
    del train_im
    del train_predlab
    del train_reflab
    del train_seg
    del train_mask

    # compute the masks for the quality network
    quality_mask = cytometer.utils.quality_model_mask(train_onecell_testlab, im=None,
                                                      quality_model_type='-1_1_prop_band')

    for j in range(train_onecell_im.shape[0]):
        plt.clf()
        plt.subplot(121)
        plt.imshow(train_onecell_im[j, :, :, :])
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the left edge are off
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # labels along the bottom edge are off
            labelbottom=False, labelleft=False)
        plt.title('fold = ' + str(i_fold) + ', im = ' + str(train_onecell_index_list[j][0]) +
                  ', cell = ' + str(train_onecell_index_list[j][1]))
        plt.subplot(122)
        plt.imshow(train_onecell_im[j, :, :, :])
        plt.contour(quality_mask[j, :, :, 0], linewidths=1, colors='black')
        plt.contour(train_onecell_reflab[j, :, :, 0], linewidths=1, colors='red')
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the left edge are off
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # labels along the bottom edge are off
            labelbottom=False, labelleft=False)
        area_reflab = np.count_nonzero(train_onecell_reflab[j, :, :, 0]) * xres * yres
        area_testlab = np.count_nonzero(train_onecell_testlab[j, :, :, 0]) * xres * yres
        plt.title(
            'Dice = ' + "{:.2f}".format(train_onecell_dice[j]) + ',\narea$_{ref}$ = ' + "{:.0f}".format(area_reflab)
            + ', area$_{test}$ = ' + "{:.0f}".format(area_testlab))

        if SAVE_FIGS:
            filename = 'fold_' + str(i_fold) + '_im_' + str(train_onecell_index_list[j][0]) + \
                       '_cell_' + str(train_onecell_index_list[j][1]) + '_dice_' + \
                       "{:.2f}".format(train_onecell_dice[j]) + '_arearef_' + \
                       "{:.0f}".format(area_reflab) + '_areatest_' + \
                       "{:.0f}".format(area_testlab) + '.jpg'
            plt.savefig(os.path.join(figures_dir, filename))

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
