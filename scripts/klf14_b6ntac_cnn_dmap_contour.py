# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])

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

'''Load data
'''

# data paths
root_data_dir = os.path.join(home, 'Dropbox/klf14')
training_dir = os.path.join(home, 'Dropbox/klf14/klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(home, 'OfflineData/klf14/klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(home, 'Dropbox/klf14/saved_models')

# list of original training images, pre-augmentation
im_orig_file_list = glob.glob(os.path.join(training_augmented_dir, 'im_*_nan_*.tif'))

# number of original training images
n_orig_im = len(im_orig_file_list)

# create k-fold splitting of data
seed = 0
random.seed(seed)
idx = random.sample(range(n_orig_im), n_orig_im)
idx_test_all = np.array_split(idx, n_folds)

# loop each fold: we split the data into train vs test, train a model, and compute errors with the
# test data. In each fold, the test data is different
# for i_fold, idx_test in enumerate(idx_test_all):
for i_fold, idx_test in enumerate(idx_test_all[0]):

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
    seg_train_file_list = [x.replace('im_', 'seg_') for x in im_train_file_list]

    dmap_test_file_list = [x.replace('im_', 'dmap_') for x in im_test_file_list]
    mask_test_file_list = [x.replace('im_', 'mask_') for x in im_test_file_list]
    seg_test_file_list = [x.replace('im_', 'seg_') for x in im_test_file_list]

    # number of training images
    n_im_train = len(im_train_file_list)
    n_im_test = len(im_test_file_list)

    # load images
    im_train = cytometer.data.load_im_file_list_to_array(im_train_file_list)
    dmap_train = cytometer.data.load_im_file_list_to_array(dmap_train_file_list)
    mask_train = cytometer.data.load_im_file_list_to_array(mask_train_file_list)
    seg_train = cytometer.data.load_im_file_list_to_array(seg_train_file_list)

    im_test = cytometer.data.load_im_file_list_to_array(im_test_file_list)
    dmap_test = cytometer.data.load_im_file_list_to_array(dmap_test_file_list)
    mask_test = cytometer.data.load_im_file_list_to_array(mask_test_file_list)
    seg_test = cytometer.data.load_im_file_list_to_array(seg_test_file_list)

    # convert uint8 images to float, and rescale RBG values to [0.0, 1.0]
    im_train = im_train.astype(np.float32)
    im_train /= 255
    mask_train = mask_train.astype(np.float32)
    seg_train = seg_train.astype(np.uint8)

    im_test = im_test.astype(np.float32)
    im_test /= 255
    mask_test = mask_test.astype(np.float32)
    seg_test = seg_test.astype(np.uint8)

    if DEBUG:
        for i in range(n_im_train):
            plt.clf()
            plt.subplot(221)
            plt.imshow(im_train[i, :, :, :])
            plt.subplot(222)
            plt.imshow(dmap_train[i, :, :, 0])
            plt.subplot(223)
            plt.imshow(mask_train[i, :, :, 0])
            plt.subplot(224)
            # plt.imshow(seg_train[i, :, :, 0])
            a = im_train[i, :, :, :]
            b = mask_train[i, :, :, 0]
            plt.imshow(pystoim.imfuse(b, a))

        for i in range(n_im_test):
            plt.clf()
            plt.subplot(221)
            plt.imshow(im_test[i, :, :, :])
            plt.subplot(222)
            plt.imshow(dmap_test[i, :, :, 0])
            plt.subplot(223)
            plt.imshow(mask_test[i, :, :, 0])
            plt.subplot(224)
            plt.imshow(seg_test[i, :, :, 0])
            a = im_test[i, :, :, :]
            b = mask_test[i, :, :, 0]
            plt.imshow(pystoim.imfuse(b, a))

    # split image into smaller blocks so that the training fits into GPU memory
    if nblocks > 1:
        dmap_train = cytometer.data.split_images(dmap_train, nblocks=nblocks)
        mask_train = cytometer.data.split_images(mask_train, nblocks=nblocks)
        im_train = cytometer.data.split_images(im_train, nblocks=nblocks)
        seg_train = cytometer.data.split_images(seg_train, nblocks=nblocks)

        dmap_test = cytometer.data.split_images(dmap_test, nblocks=nblocks)
        mask_test = cytometer.data.split_images(mask_test, nblocks=nblocks)
        im_test = cytometer.data.split_images(im_test, nblocks=nblocks)
        seg_test = cytometer.data.split_images(seg_test, nblocks=nblocks)

    # find images that have few valid pixels, to remove them from the dataset
    idx_to_keep = np.sum(np.sum(np.sum(mask_train, axis=3), axis=2), axis=1)
    idx_to_keep = idx_to_keep > 100
    dmap_train = dmap_train[idx_to_keep, :, :, :]
    im_train = im_train[idx_to_keep, :, :, :]
    mask_train = mask_train[idx_to_keep, :, :, :]
    seg_train = seg_train[idx_to_keep, :, :, :]

    idx_to_keep = np.sum(np.sum(np.sum(mask_test, axis=3), axis=2), axis=1)
    idx_to_keep = idx_to_keep > 100
    dmap_test = dmap_test[idx_to_keep, :, :, :]
    im_test = im_test[idx_to_keep, :, :, :]
    mask_test = mask_test[idx_to_keep, :, :, :]
    seg_test = seg_test[idx_to_keep, :, :, :]

    # update number of training images with number of tiles
    n_im_train = im_train.shape[0]
    n_im_test = im_test.shape[0]

    if DEBUG:
        for i in range(n_im_train):
            plt.clf()
            plt.subplot(321)
            plt.imshow(im_train[i, :, :, :])
            plt.subplot(322)
            plt.imshow(dmap_train[i, :, :, 0])
            plt.subplot(323)
            plt.imshow(mask_train[i, :, :, 0])
            plt.subplot(324)
            a = im_train[i, :, :, :]
            b = mask_train[i, :, :, 0]
            plt.imshow(pystoim.imfuse(a, b))
            plt.subplot(325)
            plt.imshow(seg_train[i, :, :, 0])

        for i in range(n_im_test):
            plt.clf()
            plt.subplot(321)
            plt.imshow(im_test[i, :, :, :])
            plt.subplot(322)
            plt.imshow(dmap_test[i, :, :, 0])
            plt.subplot(323)
            plt.imshow(mask_test[i, :, :, 0])
            plt.subplot(324)
            a = im_test[i, :, :, :]
            b = mask_test[i, :, :, 0]
            plt.imshow(pystoim.imfuse(a, b))
            plt.subplot(325)
            plt.imshow(seg_test[i, :, :, 0])

    # shuffle data
    np.random.seed(i_fold)

    idx = np.arange(n_im_train)
    np.random.shuffle(idx)
    dmap_train = dmap_train[idx, ...]
    im_train = im_train[idx, ...]
    mask_train = mask_train[idx, ...]
    seg_train = seg_train[idx, ...]

    idx = np.arange(n_im_test)
    np.random.shuffle(idx)
    dmap_test = dmap_test[idx, ...]
    im_test = im_test[idx, ...]
    mask_test = mask_test[idx, ...]
    seg_test = seg_test[idx, ...]

    '''Convolutional neural network training
    
    Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
    function
    '''

    # filename to save model to
    saved_model_filename = os.path.join(saved_models_dir, timestamp.isoformat() +
                                        '_fcn_sherrah2016_fold_' + str(i_fold) + '.h5')
    saved_model_filename = saved_model_filename.replace(':', '_')

    # list all CPUs and GPUs
    device_list = K.get_session().list_devices()

    # number of GPUs
    gpu_number = np.count_nonzero(['GPU' in str(x) for x in device_list])

    if gpu_number > 1:  # compile and train model: Multiple GPUs

        # instantiate model
        with tf.device('/cpu:0'):
            model = models.fcn_sherrah2016_regression_and_classifier(input_shape=im_train.shape[1:])

        # # load pre-trained model
        # # model = cytometer.models.fcn_sherrah2016(input_shape=im_train.shape[1:])
        # weights_filename = '2018-08-09T18_59_10.294550_fcn_sherrah2016_fold_0.h5'.replace('_0.h5', '_' +
        #                                                                                   str(i_fold) + '.h5')
        # weights_filename = os.path.join(saved_models_dir, weights_filename)
        # model = keras.models.load_model(weights_filename)

        # compile model
        parallel_model = multi_gpu_model(model, gpus=gpu_number)
        parallel_model.compile(loss={'regression_output': 'mse',
                                     'classification_output': 'binary_crossentropy'},
                               loss_weights={'regression_output': 40.0,
                                             'classification_output': 1.0},
                               optimizer='Adadelta', metrics=['mse', 'mae'],
                               sample_weight_mode='element')

        # checkpoint to save model after each epoch
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=saved_model_filename,
                                                       verbose=1, save_best_only=True)

        # train model
        tic = datetime.datetime.now()
        parallel_model.fit(im_train, [dmap_train, seg_train], sample_weight=[mask_train, mask_train],
                           validation_data=(im_test, [dmap_test, seg_test], [mask_test, mask_test]),
                           batch_size=4, epochs=epochs, initial_epoch=0,
                           callbacks=[checkpointer])
        toc = datetime.datetime.now()
        print('Training duration: ' + str(toc - tic))

    else:  # compile and train model: One GPU

        # instantiate model
        with tf.device('/cpu:0'):
            model = models.fcn_sherrah2016(input_shape=im_train.shape[1:])

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
    model.save(saved_model_filename)

# if we ran the script with nohup in linux, the output is in file nohup.out.
# Save it to saved_models directory (
log_filename = os.path.join(saved_models_dir, timestamp.isoformat() + '_fcn_sherrah2016.h5')
if os.path.isfile('nohup.out'):
    shutil.copy2('nohup.out', log_filename)
