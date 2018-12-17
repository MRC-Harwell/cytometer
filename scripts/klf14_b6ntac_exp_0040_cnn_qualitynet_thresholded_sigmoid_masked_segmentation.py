"""
Training of all folds of DenseNet for quality assessement of individual cells based on classification of
thresholded Dice coefficient (Dice >= 0.9). Here the loss is binary crossentropy.

The reason is to center the decision boundary on 0.9, to get finer granularity around that threshold.

It adds augmented training data to exp 0037.
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
import re

# other imports
import glob
import shutil
import datetime
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import watershed

# use CPU for testing on laptop
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

from keras.applications import densenet
# from keras_applications.densenet import DenseNet
from keras.models import Model
from keras.layers import Dense, Input

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

# number of folds for k-fold cross validation
n_folds = 11

# number of epochs for training
epochs = 40

# area (pixel**2) of the smallest object we accept as a cell (pi * (16 pixel)**2 = 804.2 pixel**2)
smallest_cell_area = 804

# training window length
training_window_len = 401

# segmetations with Dice >= threshold are accepted
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

'''Loop folds
'''

# loop each fold: we split the data into train vs test, train a model, and compute errors with the
# test data. In each fold, the test data is different
# for i_fold, idx_test in enumerate(idx_test_all):
for i_fold, idx_test in enumerate(idx_orig_test_all):

    print('## Fold ' + str(i_fold) + '/' + str(len(idx_orig_test_all)))

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

    # create one image per cell
    train_onecell_im, train_onecell_testlab, train_onecell_index_list, train_onecell_reflab, train_onecell_dice = \
        cytometer.utils.one_image_per_label(train_im, train_predlab,
                                            dataset_lab_ref=train_reflab,
                                            training_window_len=training_window_len,
                                            smallest_cell_area=smallest_cell_area)

    # # stretch intensity histogram of images
    # train_onecell_im = cytometer.utils.rescale_intensity(train_onecell_im, ignore_value=0.0)

    if DEBUG:
        plt.clf()
        plt.subplot(121)
        i = 1006
        plt.imshow(train_onecell_im[i, :, :, :])
        plt.contour(train_onecell_reflab[i, :, :, 0], levels=1, colors='black')
        plt.contour(train_onecell_testlab[i, :, :, 0], levels=1, colors='green')
        plt.title('Dice = ' + str("{:.2f}".format(train_onecell_dice[i])))
        plt.text(175, 180, '+1', fontsize=14, verticalalignment='top')
        plt.text(100, 75, '0', fontsize=14, verticalalignment='top', color='white')
        plt.subplot(122)
        i = 1705
        plt.imshow(train_onecell_im[i, :, :, :])
        plt.contour(train_onecell_reflab[i, :, :, 0], levels=1, colors='black')
        plt.contour(train_onecell_testlab[i, :, :, 0], levels=1, colors='red')
        plt.title('Dice = ' + str("{:.2f}".format(train_onecell_dice[i])))
        plt.text(175, 180, '+1', fontsize=14, verticalalignment='top')
        plt.text(100, 75, '0', fontsize=14, verticalalignment='top', color='white')

    # histogram of image intensities by channel
    if DEBUG:
        plt.clf()
        plt.subplot(121)
        i = 1006
        plt.hist(train_onecell_im[i, :, :, 0].flatten(), histtype='step', color='red')
        plt.hist(train_onecell_im[i, :, :, 1].flatten(), histtype='step', color='green')
        plt.hist(train_onecell_im[i, :, :, 2].flatten(), histtype='step', color='blue')
        plt.subplot(122)
        i = 1705
        plt.hist(train_onecell_im[i, :, :, 0].flatten(), histtype='step', color='red')
        plt.hist(train_onecell_im[i, :, :, 1].flatten(), histtype='step', color='green')
        plt.hist(train_onecell_im[i, :, :, 2].flatten(), histtype='step', color='blue')

    # load test dataset
    datasets, _, _ = cytometer.data.load_datasets(im_test_file_list, prefix_from='im',
                                                  prefix_to=['im', 'lab', 'seg', 'mask',
                                                             'predlab_kfold_' + str(i_fold).zfill(2)], nblocks=1)
    test_im = datasets['im']
    test_seg = datasets['seg']
    test_mask = datasets['mask']
    test_reflab = datasets['lab']
    test_predlab = datasets['predlab_kfold_' + str(i_fold).zfill(2)]
    del datasets

    # remove borders between labels
    for i in range(test_reflab.shape[0]):
        test_reflab[i, :, :, 0] = watershed(image=np.zeros(shape=test_reflab[i, :, :, 0].shape, dtype=np.uint8),
                                            markers=test_reflab[i, :, :, 0], watershed_line=False)

    # change the background label from 1 to 0
    test_reflab[test_reflab == 1] = 0

    # create one image per cell
    test_onecell_im, test_onecell_testlab, test_onecell_index_list, test_onecell_reflab, test_onecell_dice = \
        cytometer.utils.one_image_per_label(test_im, test_predlab,
                                            dataset_lab_ref=test_reflab,
                                            training_window_len=training_window_len,
                                            smallest_cell_area=smallest_cell_area)

    # # stretch intensity histogram of images
    # test_onecell_im = cytometer.utils.rescale_intensity(test_onecell_im, ignore_value=0.0)

    if DEBUG:
        plt.clf()
        plt.subplot(121)
        i = 50
        plt.imshow(test_onecell_im[i, :, :, :])
        plt.contour(test_onecell_reflab[i, :, :, 0], levels=1, colors='black')
        plt.contour(test_onecell_testlab[i, :, :, 0], levels=1, colors='green')
        plt.title('Dice = ' + str("{:.2f}".format(test_onecell_dice[i])))
        plt.text(175, 180, '+1', fontsize=14, verticalalignment='top')
        plt.text(100, 75, '0', fontsize=14, verticalalignment='top', color='white')
        plt.subplot(122)
        i = 150
        plt.imshow(test_onecell_im[i, :, :, :])
        plt.contour(test_onecell_reflab[i, :, :, 0], levels=1, colors='black')
        plt.contour(test_onecell_testlab[i, :, :, 0], levels=1, colors='red')
        plt.title('Dice = ' + str("{:.2f}".format(test_onecell_dice[i])))
        plt.text(175, 180, '+1', fontsize=14, verticalalignment='top')
        plt.text(100, 75, '0', fontsize=14, verticalalignment='top', color='white')

    # histogram of image intensities by channel
    if DEBUG:
        plt.clf()
        plt.subplot(121)
        i = 50
        plt.hist(test_onecell_im[i, :, :, 0].flatten(), histtype='step', color='red')
        plt.hist(test_onecell_im[i, :, :, 1].flatten(), histtype='step', color='green')
        plt.hist(test_onecell_im[i, :, :, 2].flatten(), histtype='step', color='blue')
        plt.subplot(122)
        i = 150
        plt.hist(test_onecell_im[i, :, :, 0].flatten(), histtype='step', color='red')
        plt.hist(test_onecell_im[i, :, :, 1].flatten(), histtype='step', color='green')
        plt.hist(test_onecell_im[i, :, :, 2].flatten(), histtype='step', color='blue')

    # multiply histology by segmentation 0/+1 mask
    train_onecell_im *= np.repeat(train_onecell_testlab.astype(np.float32), repeats=train_onecell_im.shape[3], axis=3)

    if DEBUG:
        plt.clf()
        plt.subplot(121)
        i = 1006
        plt.imshow(train_onecell_im[i, :, :, :])
        plt.contour(train_onecell_testlab[i, :, :, 0], levels=1, colors='green')
        plt.title('Dice = ' + str("{:.2f}".format(train_onecell_dice[i])))
        plt.text(175, 180, '+1', fontsize=14, verticalalignment='top')
        plt.text(100, 75, '0', fontsize=14, verticalalignment='top', color='white')
        plt.subplot(122)
        i = 1705
        plt.imshow(train_onecell_im[i, :, :, :])
        plt.contour(train_onecell_testlab[i, :, :, 0], levels=1, colors='red')
        plt.title('Dice = ' + str("{:.2f}".format(train_onecell_dice[i])))
        plt.text(175, 180, '+1', fontsize=14, verticalalignment='top')
        plt.text(100, 75, '0', fontsize=14, verticalalignment='top', color='white')

    # multiply histology by segmentations to have a single input tensor
    test_onecell_im *= np.repeat(test_onecell_testlab.astype(np.float32), repeats=test_onecell_im.shape[3], axis=3)

    if DEBUG:
        plt.clf()
        plt.subplot(121)
        i = 50
        plt.imshow(test_onecell_im[i, :, :, :])
        plt.contour(test_onecell_testlab[i, :, :, 0], levels=1, colors='green')
        plt.title('Dice = ' + str("{:.2f}".format(test_onecell_dice[i])))
        plt.text(175, 180, '+1', fontsize=14, verticalalignment='top')
        plt.text(100, 75, '0', fontsize=14, verticalalignment='top', color='white')
        plt.subplot(122)
        i = 150
        plt.imshow(test_onecell_im[i, :, :, :])
        plt.contour(test_onecell_testlab[i, :, :, 0], levels=1, colors='red')
        plt.title('Dice = ' + str("{:.2f}".format(test_onecell_dice[i])))
        plt.text(175, 180, '+1', fontsize=14, verticalalignment='top')
        plt.text(100, 75, '0', fontsize=14, verticalalignment='top', color='white')

    '''Neural network training
    '''

    # list all CPUs and GPUs
    device_list = K.get_session().list_devices()

    # number of GPUs
    gpu_number = np.count_nonzero(['device:GPU' in str(x) for x in device_list])

    # instantiate model
    with tf.device('/cpu:0'):
        # we start with the DenseNet without the final Dense layer, because it has a softmax activation, and we only
        # want to classify 1 class. So then we manually add an extra Dense layer with a sigmoid activation as final
        # output
        #
        # DenseNet121: blocks=[6, 12, 24, 16]
        base_model = densenet.DenseNet121(include_top=False, weights=None,
                                          input_shape=(401, 401, 3), pooling='avg')
        x = Dense(units=1, activation='sigmoid', name='fc1')(base_model.output)
        model = Model(inputs=base_model.input, outputs=x)

    saved_model_filename = os.path.join(saved_models_dir, experiment_id + '_model_fold_' + str(i_fold) + '.h5')

    # the number of training images has to be a multiple of the batch_size. Otherwise, BatchNormalization
    # produces NaNs
    num_train_onecell_im_to_use = int(np.floor(train_onecell_im.shape[0] / batch_size) * batch_size)

    if gpu_number > 1:  # compile and train model: Multiple GPUs
        # checkpoint to save model after each epoch
        checkpointer = cytometer.model_checkpoint_parallel.ModelCheckpoint(filepath=saved_model_filename,
                                                                           verbose=1, save_best_only=True)
        # compile model
        parallel_model = multi_gpu_model(model, gpus=gpu_number)
        parallel_model.compile(loss={'fc1': 'binary_crossentropy'},
                               optimizer='Adadelta',
                               metrics={'fc1': ['acc']})

        # train model
        tic = datetime.datetime.now()
        parallel_model.fit(train_onecell_im[0:num_train_onecell_im_to_use, :, :, :],
                           {'fc1': (train_onecell_dice[0:num_train_onecell_im_to_use] >= quality_threshold).astype(np.float32)},
                           validation_data=(test_onecell_im,
                                            {'fc1': (test_onecell_dice >= quality_threshold).astype(np.float32)}),
                           batch_size=batch_size, epochs=epochs, initial_epoch=0,
                           callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))

    else:  # compile and train model: One GPU

        # checkpoint to save model after each epoch
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=saved_model_filename,
                                                       verbose=1, save_best_only=True)

        # compile model
        model.compile(loss={'fc1':  'binary_crossentropy'},
                      optimizer='Adadelta',
                      metrics={'fc1': ['acc']})

        # train model
        tic = datetime.datetime.now()
        model.fit(train_onecell_im,
                  {'fc1': (train_onecell_dice >= quality_threshold).astype(np.float32)},
                  validation_data=(test_onecell_im,
                                   {'fc1': (test_onecell_dice >= quality_threshold).astype(np.float32)}),
                  batch_size=16, epochs=epochs, initial_epoch=0,
                  callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))

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
