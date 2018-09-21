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
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AvgPool2D, Activation
from keras.layers.normalization import BatchNormalization
# from keras.callbacks import CSVLogger

# for data parallelism in keras models
from keras.utils import multi_gpu_model

import cytometer.data
import cytometer.model_checkpoint_parallel
import random
import tensorflow as tf

# # limit GPU memory used
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1.0
# set_session(tf.Session(config=config))

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False

# number of blocks to split each image into so that training fits into GPU memory
nblocks = 3

# number of folds for k-fold cross validation
n_folds = 11

# number of epochs for training
epochs = 20

# timestamp at the beginning of loading data and processing so that all folds have a common name
timestamp = datetime.datetime.now()

'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

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
    # x = BatchNormalization(axis=norm_axis)(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(3, 3), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)

    x = Conv2D(filters=int(96/2), kernel_size=(5, 5), strides=1, dilation_rate=2, padding='same')(x)
    # x = BatchNormalization(axis=norm_axis)(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(5, 5), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(x)

    x = Conv2D(filters=int(128/2), kernel_size=(3, 3), strides=1, dilation_rate=4, padding='same')(x)
    # x = BatchNormalization(axis=norm_axis)(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(9, 9), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(9, 9), strides=1, padding='same')(x)

    x = Conv2D(filters=int(196/2), kernel_size=(3, 3), strides=1, dilation_rate=8, padding='same')(x)
    # x = BatchNormalization(axis=norm_axis)(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(17, 17), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(17, 17), strides=1, padding='same')(x)

    x = Conv2D(filters=int(512/2), kernel_size=(3, 3), strides=1, dilation_rate=16, padding='same')(x)
    # x = BatchNormalization(axis=norm_axis)(x)
    if for_receptive_field:
        x = Activation('linear')(x)
    else:
        x = Activation('relu')(x)

    # regression output
    regression_output = Conv2D(filters=1, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same', name='regression_output')(x)

    # classification output
    x = Conv2D(filters=2, kernel_size=(32, 32), strides=1, dilation_rate=1, padding='same')(regression_output)
    # x = BatchNormalization(axis=norm_axis)(x)
    classification_output = Activation('hard_sigmoid', name='classification_output')(x)

    return Model(inputs=input, outputs=[regression_output, classification_output])


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
# for i_fold, idx_test in enumerate(idx_test_all):
for i_fold, idx_test in enumerate([idx_test_all[0]]):

    '''Load data
    '''

    # split the data into training and testing datasets
    im_test_file_list, im_train_file_list = cytometer.data.split_list(im_orig_file_list, idx_test)

    # add the augmented image files
    im_train_file_list = cytometer.data.augment_file_list(im_train_file_list, '_nan_', '_*_')
    im_test_file_list = cytometer.data.augment_file_list(im_test_file_list, '_nan_', '_*_')

    # load the train and test data: im, seg, dmap and mask data
    train_dataset, train_file_list, train_shuffle_idx = \
        cytometer.data.load_datasets(im_train_file_list, prefix_from='im', prefix_to=['im', 'seg', 'dmap', 'mask'],
                                     nblocks=nblocks, shuffle_seed=i_fold)
    test_dataset, test_file_list, test_shuffle_idx = \
        cytometer.data.load_datasets(im_test_file_list, prefix_from='im', prefix_to=['im', 'seg', 'dmap', 'mask'],
                                     nblocks=nblocks, shuffle_seed=i_fold)

    # remove training data where the mask has very few valid pixels
    train_dataset = cytometer.data.remove_poor_data(train_dataset, prefix='mask', threshold=1000)
    test_dataset = cytometer.data.remove_poor_data(test_dataset, prefix='mask', threshold=1000)

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

    # instantiate model
    with tf.device('/cpu:0'):
        model = fcn_sherrah2016_regression_and_classifier(input_shape=train_dataset['im'].shape[1:])

    saved_model_filename = os.path.join(saved_models_dir, experiment_id + '_model_fold_' + str(i_fold) + '.h5')

    # # checkpoint to save metrics every epoch
    # save_history_filename = os.path.join(saved_models_dir, experiment_id + '_history_fold_' + str(i_fold) + '.csv')
    # csv_logger = CSVLogger(save_history_filename, append=True, separator=',')

    if gpu_number > 1:  # compile and train model: Multiple GPUs

        # checkpoint to save model after each epoch
        checkpointer = cytometer.model_checkpoint_parallel.ModelCheckpoint(filepath=saved_model_filename,
                                                                           verbose=1, save_best_only=True)
        # compile model
        parallel_model = multi_gpu_model(model, gpus=gpu_number)
        parallel_model.compile(loss={'regression_output': 'mse',
                                     'classification_output': 'binary_crossentropy'},
                               loss_weights={'regression_output': 1.0,
                                             'classification_output': 1000.0},
                               optimizer='Adadelta',
                               metrics={'regression_output': ['mse', 'mae'],
                                        'classification_output': 'accuracy'},
                               sample_weight_mode='element')

        # train model
        tic = datetime.datetime.now()
        parallel_model.fit(train_dataset['im'],
                           {'regression_output': train_dataset['dmap'],
                            'classification_output': train_dataset['seg']},
                           sample_weight={'regression_output': train_dataset['mask'][..., 0],
                                          'classification_output': train_dataset['mask'][..., 0]},
                           validation_data=(test_dataset['im'],
                                            {'regression_output': test_dataset['dmap'],
                                             'classification_output': test_dataset['seg']},
                                            {'regression_output': test_dataset['mask'][..., 0],
                                             'classification_output': test_dataset['mask'][..., 0]}),
                           batch_size=4, epochs=epochs, initial_epoch=0,
                           callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))

        # FOO
        model.save('/users/rittscher/rcasero/Dropbox/klf14/saved_models/foo.h5')

    else:  # compile and train model: One GPU

        # checkpoint to save model after each epoch
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=saved_model_filename,
                                                       verbose=1, save_best_only=True)

        # compile model
        model.compile(loss={'regression_output': 'mse',
                            'classification_output': 'binary_crossentropy'},
                      loss_weights={'regression_output': 1.0,
                                    'classification_output': 1000.0},
                      optimizer='Adadelta',
                      metrics={'regression_output': ['mse', 'mae'],
                               'classification_output': 'accuracy'},
                      sample_weight_mode='element')

        # train model
        tic = datetime.datetime.now()
        model.fit(train_dataset['im'],
                  {'regression_output': train_dataset['dmap'],
                   'classification_output': train_dataset['seg']},
                  sample_weight={'regression_output': train_dataset['mask'],
                                 'classification_output': train_dataset['mask']},
                  validation_data=(test_dataset['im'],
                                   {'regression_output': test_dataset['dmap'],
                                    'classification_output': test_dataset['seg']},
                                   {'regression_output': test_dataset['mask'],
                                    'classification_output': test_dataset['mask']}),
                  batch_size=4, epochs=epochs, initial_epoch=0,
                  callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))

# if we ran the script with nohup in linux, the output is in file nohup.out.
# Save it to saved_models directory (
log_filename = os.path.join(saved_models_dir, experiment_id + '.log')
nohup_filename = os.path.join(home, 'Software', 'cytometer', 'scripts', 'nohup.out')
if os.path.isfile(nohup_filename):
    shutil.copy2(nohup_filename, log_filename)
