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
import pysto.imgproc as pystoim
import matplotlib.pyplot as plt

# use CPU for testing on laptop
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AvgPool2D, Activation
from keras.layers.normalization import BatchNormalization

import cytometer.data
import cytometer.models as models
import random
import tensorflow as tf

# limit GPU memory used
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

# for data parallelism in keras models
from keras.utils import multi_gpu_model

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False

# number of blocks to split each image into so that training fits into GPU memory
nblocks = 2

# number of folds for k-fold cross validation
n_folds = 11

# number of epochs for training
epochs = 15

# timestamp at the beginning of loading data and processing so that all folds have a common name
timestamp = datetime.datetime.now()

'''Load data
'''

# data paths, using c3h_backup which has 33% of original augmented data
root_data_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
training_data_dir = '/home/gcientanni/OneDrive/c3h_backup/c3h_hfd_training'
training_nooverlap_data_dir = '/home/gcientanni/OneDrive/c3h_backup/c3h_hfd_training_non_overlap'
training_augmented_dir = '/home/gcientanni/OneDrive/c3h_backup/c3h_hfd_training_augmented'
saved_models_dir = '/home/gcientanni/OneDrive/c3h_backup/saved_models'

# timestamp and script name to identify this experiment
experiment_id = inspect.getfile(inspect.currentframe())
if experiment_id == '<input>':
    experiment_id = 'unknownscript'
else:
    experiment_id = os.path.splitext(experiment_id)[0]

'''CNN Model
'''


def fcn_sherrah2016_regression_and_classifier(input_shape, for_receptive_field=False):

    if K.image_data_format() == 'channels_first':
        norm_axis = 1
    elif K.image_data_format() == 'channels_last':
        norm_axis = -1

    input = Input(shape=input_shape, dtype='float32', name='input_image')

    x = Conv2D(filters=32, kernel_size=(5, 5), strides=1, dilation_rate=1, padding='same')(input)
    x = BatchNormalization(axis=norm_axis)(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(3, 3), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)

    x = Conv2D(filters=int(96/2), kernel_size=(5, 5), strides=1, dilation_rate=2, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(5, 5), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(x)

    x = Conv2D(filters=int(128/2), kernel_size=(3, 3), strides=1, dilation_rate=4, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(9, 9), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(9, 9), strides=1, padding='same')(x)

    x = Conv2D(filters=int(196/2), kernel_size=(3, 3), strides=1, dilation_rate=8, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(17, 17), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(17, 17), strides=1, padding='same')(x)

    x = Conv2D(filters=int(512/2), kernel_size=(3, 3), strides=1, dilation_rate=16, padding='same')(x)
    x = BatchNormalization(axis=norm_axis)(x)
    if for_receptive_field:
        x = Activation('linear')(x)
    else:
        x = Activation('relu')(x)

    # regression output
    regression_output = Conv2D(filters=1, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same', name='regression_output')(x)

    # classification output
    x = Conv2D(filters=1, kernel_size=(32, 32), strides=1, dilation_rate=1, padding='same')(regression_output)
    x = BatchNormalization(axis=norm_axis)(x)
    classification_output = Activation('hard_sigmoid', name='classification_output')(x)

    return Model(inputs=input, outputs=[regression_output, classification_output])

'''Prepare folds
'''

# list of original training images, pre-augmentation
im_orig_file_list = glob.glob(os.path.join(training_augmented_dir, 'im_*_nan_*.tif'))

# number of original training images
n_orig_im = len(im_orig_file_list)

# create k-fold sets to split the data into training vs. testing
seed = 0
random.seed(seed)
idx = random.sample(range(n_orig_im), n_orig_im)
idx_test_all = np.array_split(idx, n_folds)

# loop each fold: we split the data into train vs test, train a model, and compute errors with the
# test data. In each fold, the test data is different
for i_fold, idx_test in enumerate(idx_test_all):

    # the training dataset is all images minus the test ones
    idx_train = list(set(range(n_orig_im)) - set(idx_test))

    # list of original training and test files
    im_train_file_list = list(np.array(im_orig_file_list)[idx_train])
    im_test_file_list = list(np.array(im_orig_file_list)[idx_test])

    # add the augmented image files
    im_train_file_list = [os.path.basename(x).replace('_nan_', '_*_') for x in im_train_file_list]
    im_train_file_list = [glob.glob(os.path.join(training_augmented_dir, x)) for x in im_train_file_list]
    im_train_file_list = [item for sublist in im_train_file_list for item in sublist]

    im_test_file_list = [os.path.basename(x).replace('_nan_', '_*_') for x in im_test_file_list]
    im_test_file_list = [glob.glob(os.path.join(training_augmented_dir, x)) for x in im_test_file_list]
    im_test_file_list = [item for sublist in im_test_file_list for item in sublist]

    # list of distance transformation and mask_train files
    dmap_train_file_list = [x.replace('im_', 'dmap_') for x in im_train_file_list]
    mask_train_file_list = [x.replace('im_', 'mask_') for x in im_train_file_list]

    dmap_test_file_list = [x.replace('im_', 'dmap_') for x in im_test_file_list]
    mask_test_file_list = [x.replace('im_', 'mask_') for x in im_test_file_list]

    # load images as numpy arrays ready for training, converts data to type float.32,
    train_dataset, train_file_list, train_shuffle_idx = \
        cytometer.data.load_datasets(im_train_file_list, prefix_from='im', prefix_to=['im', 'dmap', 'mask'],
                                     nblocks=nblocks, shuffle_seed=i_fold)

    test_dataset, test_file_list, test_shuffle_idx = \
        cytometer.data.load_datasets(im_test_file_list, prefix_from='im', prefix_to=['im', 'dmap', 'mask'],
                                     nblocks=nblocks, shuffle_seed=i_fold)

    # remove training data where the mask has very few valid pixels
    train_dataset = cytometer.data.remove_poor_data(train_dataset, prefix='mask', threshold=1000)
    test_dataset = cytometer.data.remove_poor_data(test_dataset, prefix='mask', threshold=1000)

    if DEBUG:
        for i in range(len(im_train_file_list)):
            plt.clf()
            plt.subplot(221)
            plt.imshow(train_file_list[i, :, :, :])
            plt.subplot(222)
            plt.imshow(train_file_list['dmap'][i, :, :, :].reshape(train_file_list['dmap'].shape[1:3]))
            plt.subplot(223)
            plt.imshow(train_file_list['mask'][i, :, :, :].reshape(train_file_list['mask'].shape[1:3]))
            plt.subplot(224)
            a = train_file_list[i, :, :, :]
            b = train_file_list['mask'][i, :, :, :].reshape(train_file_list['mask'].shape[1:3])
            plt.imshow(pystoim.imfuse(b, a))
            plt.show()

        for i in range(len(im_test_file_list)):
            plt.clf()
            plt.subplot(221)
            plt.imshow(test_file_list[i, :, :, :])
            plt.subplot(222)
            plt.imshow(test_file_list['dmap'][i, :, :, :].reshape(test_file_list['dmap'].shape[1:3]))
            plt.subplot(223)
            plt.imshow(test_file_list['mask'][i, :, :, :].reshape(test_file_list['mask'].shape[1:3]))
            plt.subplot(224)
            a = test_file_list[i, :, :, :]
            b = test_file_list['mask'][i, :, :, :].reshape(test_file_list['mask'].shape[1:3])
            plt.imshow(pystoim.imfuse(b, a))
            plt.show()

    # remove a 1-pixel so that images are 1000x1000 and we can split them into 2x2 tiles
    dmap_train = dmap_train[:, 0:-1, 0:-1, :]
    mask_train = mask_train[:, 0:-1, 0:-1, :]
    im_train = im_train[:, 0:-1, 0:-1, :]

    dmap_test = dmap_test[:, 0:-1, 0:-1, :]
    mask_test = mask_test[:, 0:-1, 0:-1, :]
    im_test = im_test[:, 0:-1, 0:-1, :]

    # split images into smaller blocks to avoid GPU memory overflows in training
    _, dmap_train, _ = pystoim.block_split(dmap_train, nblocks=(1, 2, 2, 1))
    _, im_train, _ = pystoim.block_split(im_train, nblocks=(1, 2, 2, 1))
    _, mask_train, _ = pystoim.block_split(mask_train, nblocks=(1, 2, 2, 1))

    _, dmap_test, _ = pystoim.block_split(dmap_test, nblocks=(1, 2, 2, 1))
    _, im_test, _ = pystoim.block_split(im_test, nblocks=(1, 2, 2, 1))
    _, mask_test, _ = pystoim.block_split(mask_test, nblocks=(1, 2, 2, 1))

    dmap_train = np.concatenate(dmap_train, axis=0)
    im_train = np.concatenate(im_train, axis=0)
    mask_train = np.concatenate(mask_train, axis=0)

    dmap_test = np.concatenate(dmap_test, axis=0)
    im_test = np.concatenate(im_test, axis=0)
    mask_test = np.concatenate(mask_test, axis=0)

    # find images that have no valid pixels, to remove them from the dataset
    idx_to_keep = np.sum(np.sum(np.sum(mask_train, axis=3), axis=2), axis=1)
    idx_to_keep = idx_to_keep != 0
    dmap_train = dmap_train[idx_to_keep, :, :, :]
    im_train = im_train[idx_to_keep, :, :, :]
    mask_train = mask_train[idx_to_keep, :, :, :]

    idx_to_keep = np.sum(np.sum(np.sum(mask_test, axis=3), axis=2), axis=1)
    idx_to_keep = idx_to_keep != 0
    dmap_test = dmap_test[idx_to_keep, :, :, :]
    im_test = im_test[idx_to_keep, :, :, :]
    mask_test = mask_test[idx_to_keep, :, :, :]

    # update number of training images with number of tiles
    n_im_train = im_train.shape[0]
    n_im_test = im_test.shape[0]

    if DEBUG:
        for i in range(n_im_train):
            plt.clf()
            plt.subplot(221)
            plt.imshow(im_train[i, :, :, :])
            plt.subplot(222)
            plt.imshow(dmap_train[i, :, :, :].reshape(dmap_train.shape[1:3]))
            plt.subplot(223)
            plt.imshow(mask_train[i, :, :, :].reshape(mask_train.shape[1:3]))
            plt.subplot(224)
            a = im_train[i, :, :, :]
            b = mask_train[i, :, :, :].reshape(mask_train.shape[1:3])
            plt.imshow(pystoim.imfuse(a, b))
            plt.show()

        for i in range(n_im_test):
            plt.clf()
            plt.subplot(221)
            plt.imshow(im_test[i, :, :, :])
            plt.subplot(222)
            plt.imshow(dmap_test[i, :, :, :].reshape(dmap_test.shape[1:3]))
            plt.subplot(223)
            plt.imshow(mask_test[i, :, :, :].reshape(mask_test.shape[1:3]))
            plt.subplot(224)
            a = im_test[i, :, :, :]
            b = mask_test[i, :, :, :].reshape(mask_test.shape[1:3])
            plt.imshow(pystoim.imfuse(a, b))
            plt.show()

    # shuffle data
    np.random.seed(i_fold)

    idx = np.arange(n_im_train)
    np.random.shuffle(idx)
    dmap_train = dmap_train[idx, ...]
    im_train = im_train[idx, ...]
    mask_train = mask_train[idx, ...]

    idx = np.arange(n_im_test)
    np.random.shuffle(idx)
    dmap_test = dmap_test[idx, ...]
    im_test = im_test[idx, ...]
    mask_test = mask_test[idx, ...]

    '''Convolutional neural network training

    Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
    function
    '''

    # list all CPUs and GPUs
    device_list = K.get_session().list_devices()

    # number of GPUs
    gpu_number = np.count_nonzero(['GPU' in str(x) for x in device_list])

    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    if gpu_number > 1:  # compile and train model: Multiple GPUs

        # instantiate model
        with tf.device('/cpu:0'):
            model = models.fcn_sherrah2016_regression(input_shape=im_train.shape[1:])

        # load pre-trained model
        # model = cytometer.models.fcn_sherrah2016_regression(input_shape=im_train.shape[1:])
        # weights_filename = '2018-08-09T18_59_10.294550_fcn_sherrah2016_fold_0.h5'.replace('_0.h5', '_' +
        #                                                                                   str(i_fold) + '.h5')
        # weights_filename = os.path.join(saved_models_dir, weights_filename)
        # model = keras.models.load_model(weights_filename)

        # compile model
        parallel_model = multi_gpu_model(model, gpus=gpu_number)
        parallel_model.compile(loss='mse', optimizer='Adadelta', metrics=['mse', 'mae'], sample_weight_mode='element')

        # train model
        tic = datetime.datetime.now()
        parallel_model.fit(im_train, dmap_train, sample_weight=mask_train,
                           validation_data=(im_test, dmap_test, mask_test),
                           batch_size=4, epochs=epochs, initial_epoch=6)
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))

    else:  # compile and train model: One GPU

        # instantiate model
        with tf.device('/cpu:0'):
            model = models.fcn_sherrah2016_regression(input_shape=im_train.shape[1:])

        # compile model
        model.compile(loss='mse', optimizer='Adadelta', metrics=['mse', 'mae'], sample_weight_mode='element')

        # train model
        tic = datetime.datetime.now()
        model.fit(im_train, dmap_train, sample_weight=mask_train,
                  validation_data=(im_test, dmap_test, mask_test),
                  batch_size=4, epochs=epochs)
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))

    # save result (note, we save the template model, not the multiparallel object)
    saved_model_filename = os.path.join(saved_models_dir, timestamp.isoformat() +
                                        '_fcn_sherrah2016_fold_' + str(i_fold) + '.h5')
    saved_model_filename = saved_model_filename.replace(':', '_')
    model.save(saved_model_filename)

# if we ran the script with nohup in linux, the output is in file nohup.out.
# Save it to saved_models directory (
log_filename = os.path.join(saved_models_dir, timestamp.isoformat() + '_fcn_sherrah2016.h5')
if os.path.isfile('nohup.out'):
    shutil.copy2('nohup.out', log_filename)
