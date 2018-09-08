# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import pickle

# other imports
import glob
import shutil
import datetime
import numpy as np
import pysto.imgproc as pystoim
import matplotlib.pyplot as plt

# use CPU for testing on laptop
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K
import cytometer.data
import cytometer.models as models
import random
import tensorflow as tf

# limit GPU memory used
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
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
epochs = 10

# timestamp at the beginning of loading data and processing so that all folds have a common name
timestamp = datetime.datetime.now()

'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Dropbox/klf14')
training_dir = os.path.join(home, 'Dropbox/klf14/klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(home, 'OfflineData/klf14/klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(home, 'Dropbox/klf14/saved_models')

saved_model_basename = os.path.join(saved_models_dir, timestamp.isoformat() + '_fcn_sherrah2016_dmap_contour')
saved_model_basename = saved_model_basename.replace(':', '_')

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

# save the k-fold description for future reference
saved_model_kfold_filename = saved_model_basename + '_kfold.pickle'
with open(saved_model_kfold_filename, 'wb') as f:
    x = {'file_list': im_orig_file_list, 'idx_test_all': idx_test_all, 'seed': seed}
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
    train_out, train_file_list, train_shuffle_idx = \
        cytometer.data.load_training_data(im_train_file_list, prefix_from='im', prefix_to=['im', 'seg', 'dmap', 'mask'],
                                          nblocks=nblocks, shuffle_seed=i_fold)
    test_out, test_file_list, test_shuffle_idx = \
        cytometer.data.load_training_data(im_test_file_list, prefix_from='im', prefix_to=['im', 'seg', 'dmap', 'mask'],
                                          nblocks=nblocks, shuffle_seed=i_fold)

    if DEBUG:
        i = 100
        plt.clf()
        for pi, prefix in enumerate(train_out.keys()):
            plt.subplot(1, len(train_out.keys()), pi + 1)
            if train_out[prefix].shape[-1] == 1:
                plt.imshow(train_out[prefix][i, :, :, 0])
            else:
                plt.imshow(train_out[prefix][i, :, :, :])
            plt.title('out[' + prefix + ']')

        i = 4
        plt.clf()
        for pi, prefix in enumerate(test_out.keys()):
            plt.subplot(1, len(test_out.keys()), pi + 1)
            if test_out[prefix].shape[-1] == 1:
                plt.imshow(test_out[prefix][i, :, :, 0])
            else:
                plt.imshow(test_out[prefix][i, :, :, :])
            plt.title('out[' + prefix + ']')

    '''Convolutional neural network training
    
    Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
    function
    '''

    # filename to save model to
    saved_model_filename = saved_model_basename + '_fold_' + str(i_fold) + '.h5'

    # list all CPUs and GPUs
    device_list = K.get_session().list_devices()

    # number of GPUs
    gpu_number = np.count_nonzero(['GPU' in str(x) for x in device_list])

    if gpu_number > 1:  # compile and train model: Multiple GPUs

        # instantiate model
        with tf.device('/cpu:0'):
            model = models.fcn_sherrah2016_regression_and_classifier(input_shape=train_out['im'].shape[1:])

        # # load pre-trained model
        # # model = cytometer.models.fcn_sherrah2016_regression(input_shape=im_train.shape[1:])
        # weights_filename = '2018-08-09T18_59_10.294550_fcn_sherrah2016_fold_0.h5'.replace('_0.h5', '_' +
        #                                                                                   str(i_fold) + '.h5')
        # weights_filename = os.path.join(saved_models_dir, weights_filename)
        # model = keras.models.load_model(weights_filename)

        # compile model
        parallel_model = multi_gpu_model(model, gpus=gpu_number)
        parallel_model.compile(loss={'regression_output': 'mse',
                                     'classification_output': 'binary_crossentropy'},
                               loss_weights={'regression_output': 1.0,
                                             'classification_output': 100.0},
                               optimizer='Adadelta', metrics=['mse', 'mae'],
                               sample_weight_mode='element')

        # checkpoint to save model after each epoch
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=saved_model_filename,
                                                       verbose=1, save_best_only=True)

        # train model
        tic = datetime.datetime.now()
        parallel_model.fit(train_out['im'], [train_out['dmap'], train_out['seg']],
                           sample_weight=[train_out['mask'], train_out['mask']],
                           validation_data=(test_out['im'],
                                            [test_out['dmap'], test_out['seg']],
                                            [test_out['mask'], test_out['mask']]),
                           batch_size=4, epochs=epochs, initial_epoch=0,
                           callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))

    else:  # compile and train model: One GPU

        # instantiate model
        with tf.device('/cpu:0'):
            model = models.fcn_sherrah2016_regression_and_classifier(input_shape=train_out['im'].shape[1:])

        # compile model
        model.compile(loss={'regression_output': 'mse',
                            'classification_output': 'binary_crossentropy'},
                      loss_weights={'regression_output': 1.0,
                                    'classification_output': 100.0},
                      optimizer='Adadelta', metrics=['mse', 'mae'],
                      sample_weight_mode='element')

        # checkpoint to save model after each epoch
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=saved_model_filename,
                                                       verbose=1, save_best_only=True)

        # train model
        tic = datetime.datetime.now()
        model.fit(train_out['im'], [train_out['dmap'], train_out['seg']],
                  sample_weight=[train_out['mask'], train_out['mask']],
                  validation_data=(test_out['im'],
                                   [test_out['dmap'], test_out['seg']],
                                   [test_out['mask'], test_out['mask']]),
                  batch_size=4, epochs=epochs, initial_epoch=0,
                  callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))

    # save result (note, we save the template model, not the multiparallel object)
    model.save(saved_model_filename)

# if we ran the script with nohup in linux, the output is in file nohup.out.
# Save it to saved_models directory (
log_filename = os.path.join(saved_models_dir, timestamp.isoformat() + '_fcn_sherrah2016_dmap_contour.log')
log_filename = log_filename.replace(':', '_')
nohup_filename = os.path.join(home, 'Software', 'cytometer', 'scripts', 'nohup.out')
if os.path.isfile(nohup_filename):
    shutil.copy2(nohup_filename, log_filename)
