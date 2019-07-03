"""
Tissue classifier, using sherrah2016 CNN.

CHANGE over 0061: Train by image instead of by object. No scaling. We just map the histo
input to a pixel-wise binary classifier.

Use hand traced areas of white adipocytes and "other" tissues to train classifier to differentiate.

.svg labels:
  * 0: "Cell" = white adipocyte
  * 1: "Other" = other types of tissue
  * 1: "Brown" = brown adipocytes

We assign cells to train or test sets grouped by image. This way, we guarantee that at testing time, the
network has not seen neighbour cells to the ones used for training.
"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_inspect_exp_0074_cnn_tissue_classifier_fcn'
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

# data augmentation factor (e.g. "10" means that we generate 9 augmented images + the original input image)
augment_factor = 10

# area (pixel**2) of the smallest object we accept as a cell (pi * (16 pixel)**2 = 804.2 pixel**2)
smallest_cell_area = 804

# training window length
training_window_len = 401

# remove from training cells that don't have a good enough overlap with a reference label
smallest_dice = 0.5

# segmentations with Dice >= threshold are accepted
dice_threshold = 0.9

# batch size for training
batch_size = 2


'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

kfold_basename = 'klf14_b6ntac_exp_0055_cnn_contour'
classifier_model_basename = 'klf14_b6ntac_exp_0074_cnn_tissue_classifier_fcn'

# load list of images, and indices for training vs. testing indices
kfold_filename = os.path.join(saved_models_dir, kfold_basename + '_kfold_info.pickle')
with open(kfold_filename, 'rb') as f:
    aux = pickle.load(f)
file_list = aux['file_list']
idx_test_all = aux['idx_test']
idx_train_all = aux['idx_train']

# number of images
n_im = len(file_list)

'''Inspect the data
'''

# start timer
t0 = time.time()

# init output
im_array_all = []
out_class_all = []
out_mask_all = []
contour_type_all = []
i_all = []

# correct home directory in file paths
file_list = cytometer.data.change_home_directory(list(file_list), '/users/rittscher/rcasero', home, check_isfile=True)

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
    contours = cell_contours + other_contours + brown_contours

    # make a list with the type of cell each contour is classified as
    contour_type = [np.zeros(shape=(len(cell_contours),), dtype=np.uint8),  # 0: white-adipocyte
                    np.ones(shape=(len(other_contours),), dtype=np.uint8),  # 1: other types of tissue
                    np.ones(shape=(len(brown_contours),), dtype=np.uint8)]  # 1: brown cells (treated as "other" tissue)
    contour_type = np.concatenate(contour_type)
    contour_type_all.append(contour_type)

    print('Cells: ' + str(len(cell_contours)))
    print('Other: ' + str(len(other_contours)))
    print('Brown: ' + str(len(brown_contours)))

    if (len(contours) == 0):
        print('No contours... skipping')
        continue

    # initialise arrays for training
    out_class = np.zeros(shape=im_array.shape[0:2], dtype=np.uint8)
    out_mask = np.zeros(shape=im_array.shape[0:2], dtype=np.uint8)

    if DEBUG:
        plt.clf()
        plt.imshow(im_array)
        plt.scatter((im_array.shape[1] - 1) / 2.0, (im_array.shape[0] - 1) / 2.0)

    # loop ground truth cell contours
    for j, contour in enumerate(contours):

        if DEBUG:
            plt.clf()
            plt.imshow(im_array)
            plt.plot([p[0] for p in contour], [p[1] for p in contour])
            xy_c = (np.mean([p[0] for p in contour]), np.mean([p[1] for p in contour]))
            plt.scatter(xy_c[0], xy_c[1])

        # rasterise current ground truth segmentation
        cell_seg_gtruth = Image.new("1", im_array.shape[0:2][::-1], "black")  # I = 32-bit signed integer pixels
        draw = ImageDraw.Draw(cell_seg_gtruth)
        draw.polygon(contour, outline="white", fill="white")
        cell_seg_gtruth = np.array(cell_seg_gtruth, dtype=np.bool)

        if DEBUG:
            plt.clf()
            plt.subplot(121)
            plt.imshow(im_array)
            plt.plot([p[0] for p in contour], [p[1] for p in contour])
            xy_c = (np.mean([p[0] for p in contour]), np.mean([p[1] for p in contour]))
            plt.scatter(xy_c[0], xy_c[1])
            plt.subplot(122)
            plt.imshow(im_array)
            plt.contour(cell_seg_gtruth.astype(np.uint8))

        # add current object to training output and mask
        out_mask[cell_seg_gtruth] = 1
        out_class[cell_seg_gtruth] = contour_type[j]

    # end for j, contour in enumerate(contours):

    if DEBUG:
        plt.clf()
        plt.subplot(121)
        plt.imshow(im_array)
        plt.contour(out_mask.astype(np.uint8), colors='r')
        plt.title('Mask', fontsize=14)
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(im_array)
        plt.contour(out_class.astype(np.uint8), colors='k')
        plt.title('Class', fontsize=14)
        plt.axis('off')
        plt.tight_layout()

    # add dummy dimensions for keras
    im_array = np.expand_dims(im_array, axis=0)
    out_class = np.expand_dims(out_class, axis=0)
    out_class = np.expand_dims(out_class, axis=3)
    out_mask = np.expand_dims(out_mask, axis=0)

    # convert to float32, if necessary
    im_array = im_array.astype(np.float32)
    out_class = out_class.astype(np.float32)
    out_mask = out_mask.astype(np.float32)

    # scale image intensities from [0, 255] to [0.0, 1.0]
    im_array /= 255

    # append input/output/mask for later use in training
    im_array_all.append(im_array)
    out_class_all.append(out_class)
    out_mask_all.append(out_mask)
    i_all.append(i)

    print('Time so far: ' + str("{:.1f}".format(time.time() - t0)) + ' s')

# collapse lists into arrays
im_array_all = np.concatenate(im_array_all)
out_class_all = np.concatenate(out_class_all)
out_mask_all = np.concatenate(out_mask_all)

'''Convolutional neural network training

    Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
    function
    '''

# list all CPUs and GPUs
device_list = K.get_session().list_devices()

# number of GPUs
gpu_number = np.count_nonzero(['GPU' in str(x) for x in device_list])

for i_fold in range(len(idx_test_all)):

    print('# Fold ' + str(i_fold) + '/' + str(len(idx_test_all) - 1))

    # test and training image indices. These indices refer to file_list
    idx_test = idx_test_all[i_fold]
    idx_train = idx_train_all[i_fold]

    # map the indices from file_list to im_array_all (there's an image that had no WAT or Other contours and was
    # skipped)
    idx_lut = np.full(shape=(len(file_list), ), fill_value=-1, dtype=idx_test.dtype)
    idx_lut[i_all] = range(len(i_all))
    idx_train = idx_lut[idx_train]
    idx_test = idx_lut[idx_test]

    print('## len(idx_train) = ' + str(len(idx_train)))
    print('## len(idx_test) = ' + str(len(idx_test)))

    # split data into training and testing
    im_array_train = im_array_all[idx_train, :, :, :]
    im_array_test = im_array_all[idx_test, :, :, :]

    out_class_train = out_class_all[idx_train, :, :, :]
    out_class_test = out_class_all[idx_test, :, :, :]

    out_mask_train = out_mask_all[idx_train, :, :]
    out_mask_test = out_mask_all[idx_test, :, :]

    # load classification model
    classifier_model_filename = os.path.join(saved_models_dir, classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    classifier_model = keras.models.load_model(classifier_model_filename)

    # reshape model input
    classifier_model = cytometer.utils.change_input_size(classifier_model, batch_shape=im_array_test.shape)

    # apply classification to test data
    predict_class_test = classifier_model.predict(im_array_test, batch_size=batch_size)

    if DEBUG:
        for i in range(len(idx_test)):

            plt.clf()
            plt.subplot(221)
            plt.imshow(im_array_test[i, :, :, :])
            plt.contour(out_mask_test[i, :, :].astype(np.uint8), colors='r')
            plt.title('i = ' + str(i) + ', Mask', fontsize=14)
            plt.axis('off')
            plt.subplot(222)
            plt.imshow(im_array_test[i, :, :, :])
            plt.imshow(out_class_test[i, :, :, 0].astype(np.uint8), alpha=0.5)
            plt.title('Class', fontsize=14)
            plt.axis('off')
            plt.subplot(212)
            plt.imshow(im_array_test[i, :, :, :])
            plt.imshow(predict_class_test[i, :, :, 0].astype(np.uint8), alpha=0.5)
            plt.title('Predicted class', fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.pause(5)


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
