'''
Contour segmentation for all folds using binary crossentropy. Use transfer learning from dmap networks.
The CNN has 3 dimensionality reduction layers at the end, instead of 1.

Exp 0055 was a contour detector trained with binary crossentropy. But it used 1 layer for dimensionality , and no
transfer learning from dmap.
This experiment uses 3 layers for dimensionality reduction.

(klf14_b6ntac_exp_0006_cnn_contour only computes fold 0.)
(klf14_b6ntac_exp_0055_cnn_contour.py uses binary crossentropy.)

Training vs testing is done at the histology slide level, not at the window level. This way, we really guarantee that
the network has not been trained with data sampled from the same image as the test data.

Training for the CNN:
* Input: histology
* Output: hand tracked contours, dilated a bit.
* Other: mask for the loss function, to avoid looking outside of where we have contours.
'''

experiment_id = 'klf14_b6ntac_exp_0069_cnn_contour'

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

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AvgPool2D, Activation

# for data parallelism in keras models
from keras.utils import multi_gpu_model

import cytometer.data
import cytometer.utils
import cytometer.model_checkpoint_parallel
import random
import tensorflow as tf

# limit GPU memory used
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False

# number of blocks to split each image into so that training fits into GPU memory
nblocks = 2

# number of folds for k-fold cross validation
n_folds = 11

# number of epochs for training
epochs = 20

'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

kfold_basename = 'klf14_b6ntac_exp_0055_cnn_contour'
dmap_model_basename = 'klf14_b6ntac_exp_0056_cnn_dmap'

'''CNN Model
'''


def fcn_sherrah2016_classifier(input_shape, for_receptive_field=False):

    input = Input(shape=input_shape, dtype='float32', name='input_image')

    x = Conv2D(filters=32, kernel_size=(5, 5), strides=1, dilation_rate=1, padding='same')(input)
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

    # dimensionality reduction
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
    else:
        x = Activation('relu')(x)

    x = Conv2D(filters=8, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)

    if for_receptive_field:
        x = Activation('linear')(x)
    else:
        x = Activation('relu')(x)

    x = Conv2D(filters=1, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same')(x)

    # classification output
    classification_output = Activation('hard_sigmoid', name='classification_output')(x)

    return Model(inputs=input, outputs=[classification_output])


'''Main code
'''

# load list of images, and indices for training vs. testing indices
kfold_filename = os.path.join(saved_models_dir, kfold_basename + '_kfold_info.pickle')
with open(kfold_filename, 'rb') as f:
    aux = pickle.load(f)
im_svg_file_list = aux['file_list']
idx_orig_test_all = aux['idx_test']

# list of non-overlapped segmentations
im_orig_file_list = []
for i, file in enumerate(im_svg_file_list):
    im_orig_file_list.append(file.replace('.svg', '.tif'))
    im_orig_file_list[i] = os.path.join(training_augmented_dir, 'im_seed_nan_' + os.path.basename(im_orig_file_list[i]))

# loop each fold: we split the data into train vs test, train a model, and compute errors with the
# test data. In each fold, the test data is different
for i_fold, idx_test in enumerate(idx_orig_test_all):

    print('## Fold ' + str(i_fold) + '/' + str(len(idx_orig_test_all) - 1))

    '''Load data
    '''

    # split the data into training and testing datasets
    im_test_file_list, im_train_file_list = cytometer.data.split_list(im_orig_file_list, idx_test)

    # add the augmented image files
    im_train_file_list = cytometer.data.augment_file_list(im_train_file_list, '_nan_', '_*_')
    im_test_file_list = cytometer.data.augment_file_list(im_test_file_list, '_nan_', '_*_')

    # load the train and test data: im, seg, dmap and mask data
    train_dataset, train_file_list, train_shuffle_idx = \
        cytometer.data.load_datasets(im_train_file_list, prefix_from='im', prefix_to=['im', 'seg', 'mask'],
                                     nblocks=nblocks, shuffle_seed=i_fold)
    test_dataset, test_file_list, test_shuffle_idx = \
        cytometer.data.load_datasets(im_test_file_list, prefix_from='im', prefix_to=['im', 'seg', 'mask'],
                                     nblocks=nblocks, shuffle_seed=i_fold)

    # add seg pixels to the mask, because the mask doesn't fully cover the contour
    train_dataset['mask'] = np.logical_or(train_dataset['mask'], train_dataset['seg'])
    test_dataset['mask'] = np.logical_or(test_dataset['mask'], test_dataset['seg'])

    # cast data to float32
    train_dataset['mask'] = train_dataset['mask'].astype(np.float32)
    test_dataset['mask'] = test_dataset['mask'].astype(np.float32)

    train_dataset['seg'] = train_dataset['seg'].astype(np.float32)
    test_dataset['seg'] = test_dataset['seg'].astype(np.float32)

    # remove training data where the mask has very few valid pixels
    train_dataset = cytometer.data.remove_poor_data(train_dataset, prefix='seg', threshold=1900)
    test_dataset = cytometer.data.remove_poor_data(test_dataset, prefix='seg', threshold=1900)

    if DEBUG:
        i = 150
        plt.clf()
        for pi, prefix in enumerate(train_dataset.keys()):
            plt.subplot(1, len(train_dataset.keys()), pi + 1)
            if train_dataset[prefix].shape[-1] < 3:
                plt.imshow(train_dataset[prefix][i, :, :, 0])
            else:
                plt.imshow(train_dataset[prefix][i, :, :, :])
            plt.title('out[' + prefix + ']')

        i = 22
        plt.clf()
        for pi, prefix in enumerate(test_dataset.keys()):
            plt.subplot(1, len(test_dataset.keys()), pi + 1)
            if test_dataset[prefix].shape[-1] < 3:
                plt.imshow(test_dataset[prefix][i, :, :, 0])
            else:
                plt.imshow(test_dataset[prefix][i, :, :, :])
            plt.title('out[' + prefix + ']')

    '''Convolutional neural network training
    
    Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
    function
    '''

    # list all CPUs and GPUs
    device_list = K.get_session().list_devices()

    # number of GPUs
    gpu_number = np.count_nonzero(['GPU' in str(x) for x in device_list])

    # load dmap model that we are going to use as the basis for the contour model
    dmap_model_filename = os.path.join(saved_models_dir, dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    dmap_model = keras.models.load_model(dmap_model_filename)

    # instantiate contour model
    with tf.device('/cpu:0'):
        contour_model = fcn_sherrah2016_classifier(input_shape=train_dataset['im'].shape[1:])

    for lay in [1, 4, 7]:
        # transfer weights from dmap to contour model in the first 3 convolutional layers
        dmap_layer = dmap_model.get_layer(index=lay)
        contour_layer = contour_model.get_layer(index=lay)
        contour_layer.set_weights(dmap_layer.get_weights())

        # fix 3 first convolutional layers so that they don't get trained
        contour_model.get_layer(index=lay).trainable = False

    # delete dmap model
    del dmap_model

    # name of file to save contour model to
    saved_model_filename = os.path.join(saved_models_dir, experiment_id + '_model_fold_' + str(i_fold) + '.h5')

    if gpu_number > 1:  # compile and train model: Multiple GPUs

        # checkpoint to save model after each epoch
        checkpointer = cytometer.model_checkpoint_parallel.ModelCheckpoint(filepath=saved_model_filename,
                                                                           verbose=1, save_best_only=True)
        # compile model
        parallel_model = multi_gpu_model(contour_model, gpus=gpu_number)
        parallel_model.compile(loss={'classification_output': 'binary_crossentropy'},
                               optimizer='Adadelta',
                               metrics={'classification_output': 'accuracy'},
                               sample_weight_mode='element')

        # train model
        tic = datetime.datetime.now()
        parallel_model.fit(train_dataset['im'],
                           {'classification_output': train_dataset['seg']},
                           sample_weight={'classification_output': train_dataset['mask'][..., 0]},
                           validation_data=(test_dataset['im'],
                                            {'classification_output': test_dataset['seg']},
                                            {'classification_output': test_dataset['mask'][..., 0]}),
                           batch_size=10, epochs=epochs, initial_epoch=0,
                           callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))

    else:  # compile and train model: One GPU

        # checkpoint to save model after each epoch
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=saved_model_filename,
                                                       verbose=1, save_best_only=True)

        # compile model
        contour_model.compile(loss={'classification_output': 'binary_crossentropy'},
                              optimizer='Adadelta',
                              metrics={'classification_output': 'accuracy'},
                              sample_weight_mode='element')

        # train model
        tic = datetime.datetime.now()
        contour_model.fit(train_dataset['im'],
                          {'classification_output': train_dataset['seg']},
                          sample_weight={'classification_output': train_dataset['mask'][..., 0]},
                          validation_data=(test_dataset['im'],
                                   {'classification_output': test_dataset['seg']},
                                   {'classification_output': test_dataset['mask'][..., 0]}),
                          batch_size=10, epochs=epochs, initial_epoch=0,
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
