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

# number of folds for k-fold cross validation
n_folds = 11

'''Load data
'''

# data paths
root_data_dir = os.path.join(home, 'Dropbox/klf14')
training_dir = os.path.join(home, 'Dropbox/klf14/klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(home, 'OfflineData/klf14/klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(home, 'Dropbox/klf14/saved_models')

# list of original training images, pre-augmentation
orig_im_file_list = glob.glob(os.path.join(training_augmented_dir, 'im_*_nan_*.tif'))

# number of original training images
n_orig_im = len(orig_im_file_list)

# create k-fold splitting of data
seed = 0
random.seed(seed)
idx = random.sample(range(n_orig_im), n_orig_im)
idx_train = np.array_split(idx, n_folds)

# list of training images
im_file_list = glob.glob(os.path.join(training_augmented_dir, 'im_*.tif'))
dmap_file_list = [x.replace('im_', 'dmap_') for x in im_file_list]
mask_file_list = [x.replace('im_', 'mask_') for x in im_file_list]

# number of training images
n_im = len(im_file_list)

# load images
im = cytometer.data.load_im_file_list_to_array(im_file_list)
dmap = cytometer.data.load_im_file_list_to_array(dmap_file_list)
dmap = dmap.reshape(dmap.shape + (1,))
mask = cytometer.data.load_im_file_list_to_array(mask_file_list)
mask = mask.reshape(mask.shape + (1,))

# convert to float
im = im.astype(np.float32)
im /= 255
mask = mask.astype(np.float32)

if DEBUG:
    for i in range(im.shape[0]):
        plt.clf()
        plt.subplot(221)
        plt.imshow(im[i, :, :, :])
        plt.subplot(222)
        plt.imshow(dmap[i, :, :, :].reshape(dmap.shape[1:3]))
        plt.subplot(223)
        plt.imshow(mask[i, :, :, :].reshape(mask.shape[1:3]))
        plt.subplot(224)
        a = im[i, :, :, :]
        b = mask[i, :, :, :].reshape(mask.shape[1:3])
        plt.imshow(pystoim.imfuse(b, a))
        plt.show()

# # remove a 1-pixel thick border so that images are 999x999 and we can split them into 3x3 tiles
# dmap = dmap[:, 1:-1, 1:-1, :]
# mask = mask[:, 1:-1, 1:-1, :]
# seg = seg[:, 1:-1, 1:-1, :]
# im = im[:, 1:-1, 1:-1, :]

# remove a 1-pixel so that images are 1000x1000 and we can split them into 2x2 tiles
dmap = dmap[:, 0:-1, 0:-1, :]
mask = mask[:, 0:-1, 0:-1, :]
im = im[:, 0:-1, 0:-1, :]

# split images into smaller blocks to avoid GPU memory overflows in training
_, dmap, _ = pystoim.block_split(dmap, nblocks=(1, 2, 2, 1))
_, im, _ = pystoim.block_split(im, nblocks=(1, 2, 2, 1))
_, mask, _ = pystoim.block_split(mask, nblocks=(1, 2, 2, 1))

dmap = np.concatenate(dmap, axis=0)
im = np.concatenate(im, axis=0)
mask = np.concatenate(mask, axis=0)

# find images that have no valid pixels, to remove them from the dataset
idx_to_keep = np.sum(np.sum(np.sum(mask, axis=3), axis=2), axis=1)
idx_to_keep = idx_to_keep != 0

dmap = dmap[idx_to_keep, :, :, :]
im = im[idx_to_keep, :, :, :]
mask = mask[idx_to_keep, :, :, :]

if DEBUG:
    for i in range(dmap.shape[0]):
        plt.clf()
        plt.subplot(221)
        plt.imshow(im[i, :, :, :])
        plt.subplot(222)
        plt.imshow(dmap[i, :, :, :].reshape(dmap.shape[1:3]))
        plt.subplot(223)
        plt.imshow(mask[i, :, :, :].reshape(mask.shape[1:3]))
        plt.subplot(224)
        a = im[i, :, :, :]
        b = mask[i, :, :, :].reshape(mask.shape[1:3])
        plt.imshow(pystoim.imfuse(a, b))
        plt.show()

# shuffle data
np.random.seed(0)
idx = np.arange(im.shape[0])
np.random.shuffle(idx)
dmap = dmap[idx, ...]
im = im[idx, ...]
mask = mask[idx, ...]

'''Convolutional neural network training

Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
function
'''

# list all CPUs and GPUs
device_list = K.get_session().list_devices()

# number of GPUs
gpu_number = np.count_nonzero(['GPU' in str(x) for x in device_list])

if gpu_number > 1:  # compile and train model: Multiple GPUs

    # instantiate model
    with tf.device('/cpu:0'):
        model = models.fcn_sherrah2016(input_shape=im.shape[1:])

    # compile model
    parallel_model = multi_gpu_model(model, gpus=gpu_number)
    parallel_model.compile(loss='mse', optimizer='Adadelta', metrics=['mse', 'mae'], sample_weight_mode='element')

    # train model
    tic = datetime.datetime.now()
    parallel_model.fit(im, dmap, batch_size=4, epochs=10, validation_split=.1, sample_weight=mask)
    toc = datetime.datetime.now()
    print('Training duration: ' + str(toc - tic))

else:  # compile and train model: One GPU

    # instantiate model
    with tf.device('/cpu:0'):
        model = models.fcn_sherrah2016(input_shape=im.shape[1:])

    # compile model
    model.compile(loss='mse', optimizer='Adadelta', metrics=['mse', 'mae'], sample_weight_mode='element')

    # train model
    tic = datetime.datetime.now()
    model.fit(im, dmap, batch_size=1, epochs=5, validation_split=.1, sample_weight=mask)
    toc = datetime.datetime.now()
    print('Training duration: ' + str(toc - tic))

# save result (note, we save the template model, not the multiparallel object)
saved_model_filename = os.path.join(saved_models_dir, datetime.datetime.utcnow().isoformat() + '_fcn_sherrah2016.h5')
saved_model_filename = saved_model_filename.replace(':', '_')
model.save(saved_model_filename)
