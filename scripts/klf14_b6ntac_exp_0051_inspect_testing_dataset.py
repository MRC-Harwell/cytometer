"""
Using exp 0049 as starting point, examine the testing dataset, to try to figure out why the
quality network rejects small cells.
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
import pandas as pd

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
dice_threshold = 0.9

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

saved_contour_model_basename = 'klf14_b6ntac_exp_0034_cnn_contour'
saved_dmap_model_basename = 'klf14_b6ntac_exp_0035_cnn_dmap'
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
# quality network
saved_quality_model_basename = 'klf14_b6ntac_exp_0049_cnn_qualitynet_pm_1_prop_band_focal_loss_dice_geq_0_5'
quality_model_name = saved_quality_model_basename + '*.h5'

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

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(root_data_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

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

    # load training dataset
    datasets, _, _ = cytometer.data.load_datasets(im_test_file_list, prefix_from='im',
                                                  prefix_to=['im', 'lab'], nblocks=1)
    im = datasets['im']
    reflab = datasets['lab']
    del datasets

    # number of images
    n_im = im.shape[0]

    # select the models that correspond to current fold
    contour_model_file = os.path.join(saved_models_dir, saved_contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    dmap_model_file = os.path.join(saved_models_dir, saved_dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    quality_model_file = os.path.join(saved_models_dir, saved_quality_model_basename + '_model_fold_' + str(i_fold) + '.h5')

    # load models
    contour_model = keras.models.load_model(contour_model_file)
    dmap_model = keras.models.load_model(dmap_model_file)
    quality_model = keras.models.load_model(quality_model_file)

    '''Cell segmentation with quality control
    '''

    # segment histology
    labels, labels_info = \
        cytometer.utils.segmentation_pipeline(im,
                                              contour_model, dmap_model, quality_model,
                                              quality_model_type='-1_1_prop_band',
                                              smallest_cell_area=smallest_cell_area)

    for i in range(im.shape[0]):

        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(im[i, :, :, :])
            plt.subplot(222)
            plt.imshow(im[i, :, :, :])
            plt.contour(labels[i, :, :, 0], levels=np.unique(labels[i, :, :, 0]), colors='C0')

        # list of labels that are on the edges
        lab_edge = cytometer.utils.edge_labels(labels[i, :, :, 0])

        # delete edge cell labels from labels_info
        idx_delete = np.where(np.logical_and(labels_info['im'] == i, np.isin(labels_info['label'], lab_edge)))[0]
        labels_info = np.delete(labels_info, idx_delete)

        # delete edge cells from the segmentation
        labels[i, :, :, 0] = np.logical_not(np.isin(labels[i, :, :, 0], lab_edge)) * labels[i, :, :, 0]

        if DEBUG:
            plt.subplot(223)
            plt.imshow(im[i, :, :, :])
            plt.contour(labels[i, :, :, 0], levels=np.unique(labels[i, :, :, 0]), colors='C0')
            plt.contour(reflab[i, :, :, 0], levels=np.unique(labels[i, :, :, 0]), colors='C1')

        # find correspondence between segmented cells and ground truth
        match_info = cytometer.utils.match_overlapping_labels(labels_ref=reflab[i, :, :, 0],
                                                              labels_test=labels[i, :, :, 0],
                                                              allow_repeat_ref=False)

        # delete correspondences with Dice < 0.5, because that's usually the segmentation of one cell
        # barely touch the hand tracing of a different cell
        idx = match_info['dice'] >= smallest_dice
        match_info = match_info[idx]

        # remove background label
        idx = match_info['lab_test'] != 0
        match_info = match_info[idx]

        # create dataframe: one cell per row, tagged with mouse metainformation
        df = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(im_test_file_list[i]),
                                                       values=match_info['lab_test'], values_tag='lab_test',
                                                       tags_to_keep=['id', 'ko', 'sex'])

        # add to dataframe: rest of match columns
        df['lab_ref'] = match_info['lab_ref']
        df['area_test'] = match_info['area_test']
        df['area_ref'] = match_info['area_ref']
        df['dice'] = match_info['dice']

        df['fold'] = i_fold
        df['im'] = i

        # look up table for quality values assigned by the pipeline to automatic segmentations
        quality_lut = np.zeros(shape=(np.max(labels[i, :, :, 0]) + 1, ))
        quality_lut[:] = np.nan
        i_labels_info = labels_info[labels_info['im'] == i]
        quality_lut[i_labels_info['label']] = i_labels_info['quality']

        df['quality'] = quality_lut[df['lab_test']]

        if DEBUG:
            plt.subplot(224)
            plt.scatter(df['dice'], df['quality'])
            plt.xlabel('dice')
            plt.ylabel('quality')

        # concatenate results
        if len(df_gtruth_pipeline) == 0:
            df_gtruth_pipeline = df
        else:
            df_gtruth_pipeline = pd.concat([df_gtruth_pipeline, df])


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
