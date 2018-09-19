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
import cytometer.model_checkpoint_parallel
import random
import tensorflow as tf

# limit GPU memory used
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
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

'''Directories and filepaths
'''

# data paths
root_data_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
training_data_dir = os.path.join(home, 'OneDrive/backup/c3h/c3h_hfd_training')
training_nooverlap_data_dir = os.path.join(home, 'OneDrive/backup/c3h/c3h_hfd_training_non_overlap')
training_augmented_dir = os.path.join(home, 'OneDrive/backup/c3h/c3h_hfd_training_augmented')
saved_models_dir = os.path.join(home, 'OneDrive/backup/c3h/saved_models')

# timestamp and script name to identify this experiment
experiment_id = inspect.getfile(inspect.currentframe())
if experiment_id == '<input>':
    experiment_id = 'unknownscript'
else:
    experiment_id = os.path.splitext(os.path.basename(experiment_id))[0]

'''CNN Model
'''

def fcn_sherrah2016_regression(input_shape, for_receptive_field=False):

    input = Input(shape=input_shape, dtype='float32', name='input_image')

    x = Conv2D(filters=32, kernel_size=(5, 5), strides=1, dilation_rate=1, padding='same')(input)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(3, 3), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)

    x = Conv2D(filters=96, kernel_size=(5, 5), strides=1, dilation_rate=2, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(5, 5), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, dilation_rate=4, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(9, 9), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(9, 9), strides=1, padding='same')(x)

    x = Conv2D(filters=196, kernel_size=(3, 3), strides=1, dilation_rate=8, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(17, 17), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(17, 17), strides=1, padding='same')(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), strides=1, dilation_rate=16, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
    else:
        x = Activation('relu')(x)

    # dimensionality reduction layer
    main_output = Conv2D(filters=1, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same',
                         name='main_output')(x)

    return Model(inputs=input, outputs=main_output)

'''Prepare folds
'''

# list of original training images, pre-augmentation
im_orig_file_list = glob.glob(os.path.join(training_augmented_dir, 'im_*_nan_*.tif'))

# number of original training images
n_orig_im = len(im_orig_file_list)

# create k-fold sets to split the data into training vs. testing
kfold_seed = 0
random.seed(kfold_seed)
idx = random.sample(range(n_orig_im), n_orig_im)
idx_test_all = np.array_split(idx, n_folds)

# save the k-fold description for future reference
saved_model_datainfo_filename = os.path.join(saved_models_dir, experiment_id + '_info.pickle')
with open(saved_model_datainfo_filename, 'wb') as f:
    x = {'file_list': im_orig_file_list, 'idx_test_all': idx_test_all, 'kfold_seed': kfold_seed}
    pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)

# loop each fold: we split the data into train vs test, train a model, and compute errors with the
# test data. In each fold, the test data is different
for i_fold, idx_test in enumerate(idx_test_all):

    # split the data into training and testing datasets
    im_test_file_list, im_train_file_list = cytometer.data.split_list(im_orig_file_list, idx_test)

    # add the augmented image files
    im_train_file_list = cytometer.data.augment_file_list(im_train_file_list, '_nan_', '_*_')
    im_test_file_list = cytometer.data.augment_file_list(im_test_file_list, '_nan_', '_*_')


    # reduce dataset size to 15% of original if too large to load on RAM

    # im_train_file_list = np.random.permutation(im_train_file_list)
    # reduction_factor = int(len(im_train_file_list) * 0.85)
    # im_train_file_list = im_train_file_list[reduction_factor:]
    #
    # im_test_file_list = np.random.permutation(im_test_file_list)
    # reduction_factor = int(len(im_test_file_list) * 0.85)
    # im_test_file_list = im_test_file_list[reduction_factor:]

    # test on single batch

    # im_train_file_list = im_train_file_list[int(len(im_train_file_list)-1):]
    # im_test_file_list = im_test_file_list[int(len(im_test_file_list) -1):]


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

    '''Convolutional neural network training

    Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
    function
    '''

    # list all CPUs and GPUs
    device_list = K.get_session().list_devices()

    # number of GPUs
    gpu_number = np.count_nonzero(['GPU' in str(x) for x in device_list])

    # instantiate model
    with tf.device('/cpu:0'):
        model = fcn_sherrah2016_regression(input_shape=train_dataset['im'].shape[1:])

    # checkpoint to save model after each epoch
    saved_model_filename = os.path.join(saved_models_dir, experiment_id + '_model_fold_' + str(i_fold) + '.h5')


    # # checkpoint to save metrics every epoch
    # save_history_filename = os.path.join(saved_models_dir, experiment_id + '_history_fold_' + str(i_fold) + '.csv')
    # csv_logger = CSVLogger(save_history_filename, append=True, separator=',')

    if gpu_number > 1:  # compile and train model: Multiple GPUs

        checkpointer = cytometer.model_checkpoint_parallel.ModelCheckpoint(filepath=saved_model_filename,
                                                                           verbose=1, save_best_only=True)
        # compile model
        parallel_model = multi_gpu_model(model, gpus=gpu_number)
        parallel_model.compile(loss='mse', optimizer='Adadelta', metrics=['mse', 'mae'], sample_weight_mode='element')

        # train model
        tic = datetime.datetime.now()
        parallel_model.fit(train_dataset['im'], train_dataset['dmap'],
                           sample_weight=train_dataset['mask'],
                           validation_data=(test_dataset['im'],
                                            test_dataset['dmap'],
                                            test_dataset['mask']),
                           batch_size=4, epochs=epochs, initial_epoch=0, callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))

    else:  # compile and train model: One GPU

        checkpointer = keras.callbacks.ModelCheckpoint(filepath=saved_model_filename, verbose=1, save_best_only=True)

        # compile model
        model.compile(loss='mse', optimizer='Adadelta', metrics=['mse', 'mae'], sample_weight_mode='element')

        # train model
        tic = datetime.datetime.now()

        model.fit(train_dataset['im'], train_dataset['dmap'],
                           sample_weight=train_dataset['mask'],
                           validation_data=(test_dataset['im'],
                                            test_dataset['dmap'],
                                            test_dataset['mask']),
                           batch_size=4, epochs=epochs, initial_epoch=0, callbacks=[checkpointer], verbose=1)
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))


# if we ran the script with nohup in linux, the output is in file nohup.out.
# Save it to saved_models directory (
log_filename = os.path.join(saved_models_dir, experiment_id + '.log')
nohup_filename = os.path.join(home, 'Software', 'cytometer', 'scripts', 'nohup.out')
if os.path.isfile(nohup_filename):
    shutil.copy2(nohup_filename, log_filename)

