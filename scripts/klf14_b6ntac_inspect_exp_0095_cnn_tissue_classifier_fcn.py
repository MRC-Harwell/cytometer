'''
Tissue classifier, using sherrah2016 CNN.

Like 0061, but:
  * Train by image instead of by object.
  * No scaling.
  * We just map the histo input to a pixel-wise binary classifier.
Like 0074, but:
  * k-folds = 10 instead of 11.
Like 0088:
  * add new "other" tissue data samples to training dataset.

Use hand traced areas of white adipocytes and "other" tissues to train classifier to differentiate.

.svg labels:
  * 0: "Cell" = white adipocyte
  * 1: "Other" = other types of tissue
  * 1: "Brown" = brown adipocytes
  * 0: "Background" = flat background

We assign cells to train or test sets grouped by image. This way, we guarantee that at testing time, the
network has not seen neighbour cells to the ones used for training.
'''

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_inspect_exp_0095_cnn_tissue_classifier_fcn'
original_experiment_id = 'klf14_b6ntac_exp_0095_cnn_tissue_classifier_fcn'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import json
import pickle
import warnings

# other imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

import cytometer.data
import cytometer.utils
import cytometer.model_checkpoint_parallel

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False

'''Directories and filenames'''

# data paths
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
klf14_training_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training')
klf14_training_non_overlap_data_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training_non_overlap')
klf14_training_augmented_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training_augmented')

saved_models_dir = os.path.join(klf14_root_data_dir, 'saved_models')

# k-folds only for KLF14 data
saved_kfolds_filename = 'klf14_b6ntac_exp_0094_generate_extra_training_images.pickle'


'''Load folds'''

# load list of images, and indices for training vs. testing indices
contour_model_kfold_filename = os.path.join(saved_models_dir, saved_kfolds_filename)
with open(contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
svg_file_list = aux['file_list']
idx_test_all = aux['idx_test']
idx_train_all = aux['idx_train']

# correct home directory
svg_file_list = [x.replace('/users/rittscher/rcasero', home) for x in svg_file_list]
svg_file_list = [x.replace('/home/rcasero', home) for x in svg_file_list]

'''Inspect training convergence'''

if DEBUG:

    plt.clf()
    for i_fold in range(len(idx_test_all)):
        history_filename = os.path.join(saved_models_dir,
                                        original_experiment_id + '_history_fold_' + str(i_fold) + '.json')
        with open(history_filename, 'r') as f:
            history = json.load(f)
        plt.plot(history['acc'], label='fold = ' + str(i_fold))
    plt.tick_params(axis="both", labelsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.legend()
    plt.tight_layout()

    plt.clf()
    for i_fold in range(len(idx_test_all)):
        history_filename = os.path.join(saved_models_dir,
                                        original_experiment_id + '_history_fold_' + str(i_fold) + '.json')
        with open(history_filename, 'r') as f:
            history = json.load(f)
        plt.plot(history['val_acc'], label='fold = ' + str(i_fold))
    plt.tick_params(axis="both", labelsize=14)
    plt.ylabel('Validation accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.legend()
    plt.tight_layout()

    plt.clf()
    for i_fold in range(len(idx_test_all)):
        history_filename = os.path.join(saved_models_dir,
                                        original_experiment_id + '_history_fold_' + str(i_fold) + '.json')
        with open(history_filename, 'r') as f:
            history = json.load(f)
        plt.plot(history['loss'], label='fold = ' + str(i_fold))
    plt.tick_params(axis="both", labelsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.legend()
    plt.tight_layout()

    plt.clf()
    for i_fold in range(len(idx_test_all)):
        history_filename = os.path.join(saved_models_dir,
                                        original_experiment_id + '_history_fold_' + str(i_fold) + '.json')
        with open(history_filename, 'r') as f:
            history = json.load(f)
        plt.plot(history['val_loss'], label='fold = ' + str(i_fold))
    plt.tick_params(axis="both", labelsize=14)
    plt.ylabel('Validation loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.legend()
    plt.tight_layout()


'''TensorBoard logs'''

# list of metrics in the logs (we assume all folds have the same)
size_guidance = {  # limit number of elements that can be loaded so that memory is not overfilled
    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
    event_accumulator.IMAGES: 4,
    event_accumulator.AUDIO: 4,
    event_accumulator.SCALARS: 0,
    event_accumulator.HISTOGRAMS: 1,
}
i_fold = 0
saved_logs_dir = os.path.join(saved_models_dir, original_experiment_id + '_logs_fold_' + str(i_fold))
ea = event_accumulator.EventAccumulator(saved_logs_dir, size_guidance=size_guidance)
ea.Reload()  # loads events from file
tags = ea.Tags()['scalars']

if DEBUG:

    plt.clf()

    for i_fold in range(10):

        saved_logs_dir = os.path.join(saved_models_dir, original_experiment_id + '_logs_fold_' + str(i_fold))
        if not os.path.isdir(saved_logs_dir):
            continue

        # load log data for current fold
        ea = event_accumulator.EventAccumulator(saved_logs_dir, size_guidance=size_guidance)
        ea.Reload()

        # plot curves for each metric
        for i, tag in enumerate(tags):
            df = pd.DataFrame(ea.Scalars(tag))

            # plot
            plt.subplot(2, 2, i + 1)
            plt.plot(df['step'], df['value'], label='fold ' + str(i_fold))

    for i, tag in enumerate(tags):
        plt.subplot(2, 2, i + 1)
        plt.tick_params(axis="both", labelsize=14)
        plt.ylabel(tag, fontsize=14)
        plt.xlabel('Epoch', fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()


'''Inspect model results'''

for i_fold, idx_test in enumerate(idx_test_all):

    print('Fold ' + str(i_fold) + '/' + str(len(idx_test_all)-1))

    '''Load data'''

    # change file extension from .svg to .tif
    tif_file_list = [x.replace('.svg', '.tif') for x in svg_file_list]

    # list of test files for this fold
    im_test_file_list = np.array(tif_file_list)[idx_test]

    # load tissue classifier
    saved_model_filename = os.path.join(saved_models_dir, original_experiment_id + '_model_fold_' + str(i_fold) + '.h5')
    tissue_model = keras.models.load_model(saved_model_filename)
    tissue_model = cytometer.utils.change_input_size(tissue_model, batch_shape=(1, 1001, 1001, 3))

    for i, file_tif in enumerate(tif_file_list):

        # load image
        im = Image.open(file_tif)
        im = np.array(im)
        im = im.astype(np.float32) / 255
        im = np.expand_dims(im, axis=0)

        # classify tissue
        # pred_tissue = tissue_model.predict(im, batch_size=4)
        pred_tissue = tissue_model.predict(im, batch_size=4)

        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(im[0, ...])
            plt.axis('off')
            plt.subplot(222)
            plt.imshow(im[0, ...])
            plt.contourf(pred_tissue[0, :, :, 0] > 0.4, alpha=0.5, levels=[0, 0.5, 1], colors=['C0', 'C1'])
            plt.axis('off')
            plt.subplot(212)
            plt.imshow(pred_tissue[0, :, :, 0])
            plt.axis('off')
            plt.tight_layout()
