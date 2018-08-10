# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])

import gc
import glob
import numpy as np
import pysto.imgproc as pystoim

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# limit GPU memory used
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

# Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
# function
import keras.models
import cytometer.data
import cytometer.models as models
import matplotlib.pyplot as plt
from receptivefield.keras import KerasReceptiveField


DEBUG = True

'''Load data
'''

# data paths
root_data_dir = os.path.join(home, 'Dropbox/klf14')
training_dir = os.path.join(home, 'Dropbox/klf14/klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(home, 'OfflineData/klf14/klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(home, 'Dropbox/klf14/saved_models')

# model_name = '2018-07-27T10_00_57.382521_fcn_sherrah2016.h5'
# model_name = '2018-07-28T00_40_03.181899_fcn_sherrah2016.h5'
# model_name = '2018-07-31T12_48_02.977755_fcn_sherrah2016.h5'
# model_name = '2018-08-06T18_02_55.864612_fcn_sherrah2016.h5'
model_name = '2018-08-09T18_59_10.294550_fcn_sherrah2016*.h5'

# list of training images
im_file_list = glob.glob(os.path.join(training_augmented_dir, 'im_*.tif'))
dmap_file_list = [x.replace('im_', 'dmap_') for x in im_file_list]
mask_file_list = [x.replace('im_', 'mask_') for x in im_file_list]

# number of training images
n_im = len(im_file_list)

# load images
im = cytometer.data.load_im_file_list_to_array(im_file_list)
dmap = cytometer.data.load_im_file_list_to_array(dmap_file_list)
mask = cytometer.data.load_im_file_list_to_array(mask_file_list)

# convert to float
im = im.astype(np.float32)
im /= 255
mask = mask.astype(np.float32)

if DEBUG:
    for i in range(n_im):
        print('  ** Image: ' + str(i) + '/' + str(n_im - 1))
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

# remove a 1-pixel so that images are 1000x1000 and we can split them into 2x2 tiles
dmap = dmap[:, 0:-1, 0:-1, :]
mask = mask[:, 0:-1, 0:-1, :]
im = im[:, 0:-1, 0:-1, :]

# split images into smaller blocks to avoid GPU memory overflows
_, im, _ = pystoim.block_split(im, nblocks=(1, 2, 2, 1))
im = np.concatenate(im, axis=0)
gc.collect()
_, dmap, _ = pystoim.block_split(dmap, nblocks=(1, 2, 2, 1))
dmap = np.concatenate(dmap, axis=0)
gc.collect()
_, mask, _ = pystoim.block_split(mask, nblocks=(1, 2, 2, 1))
mask = np.concatenate(mask, axis=0)
gc.collect()

if DEBUG:
    for i in range(n_im):
        print('  ** Image: ' + str(i) + '/' + str(n_im - 1))
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

'''Receptive field

'''

# list of model files to inspect
model_files = glob.glob(os.path.join(saved_models_dir, model_name))

receptive_field_size = []
for model_file in model_files:

    print(model_file)

    # estimate receptive field of the model
    def model_build_func(input_shape):
        model = models.fcn_sherrah2016(input_shape=input_shape, for_receptive_field=True)
        model.load_weights(model_file)
        return model


    rf = KerasReceptiveField(model_build_func, init_weights=False)

    rf_params = rf.compute(
        input_shape=(500, 500, 3),
        input_layer='input_image',
        output_layers=['main_output'])
    print(rf_params)

    receptive_field_size. append(rf._rf_params[0].size)

for i, model_file in enumerate(model_files):
    print(model_file)
    print('Receptive field size: ' + str(receptive_field_size[i]))

'''Load model and visualise results
'''

for model_file in model_files:

    # load model
    model = keras.models.load_model(model_file)

    # visualise results
    if DEBUG:
        for i in range(im.shape[0]):

            # run image through network
            dmap_pred = model.predict(im[i, :, :, :].reshape((1,) + im.shape[1:]))

            plt.clf()
            plt.subplot(221)
            plt.imshow(im[i, :, :, :])
            plt.title('histology')
            plt.subplot(222)
            plt.imshow(dmap[i, :, :, :].reshape(dmap.shape[1:3]))
            plt.title('ground truth dmap')
            plt.subplot(223)
            plt.imshow(dmap_pred.reshape(dmap_pred.shape[1:3]))
            plt.title('estimated dmap')
            plt.subplot(224)
            a = dmap[i, :, :, :].reshape(dmap.shape[1:3])
            b = dmap_pred.reshape(dmap.shape[1:3])
            plt.imshow(b - a)
            plt.colorbar()
            plt.title('error (est - gt)')
            plt.show()


'''Plot metrics and convergence
'''

log_filename = os.path.join(saved_models_dir, model_name.replace('*.h5', '.log'))

if os.path.isfile(log_filename):
    # read Keras output
    df_list = cytometer.data.read_keras_training_output(log_filename)

    # plot metrics
    plt.clf()
    plt.subplot(311)
    loss_plot, = plt.plot(df.index, df.loss, label='loss')
    #epoch_starts = np.where(np.diff(np.concatenate(([0, ], df.epoch))))[0]
    epoch_ends = np.where(np.diff(df.epoch))[0]
    epoch_starts_plot1, = plt.plot(epoch_ends, df.loss[epoch_ends], 'ro-', label='epoch ends')
    plt.legend(handles=[loss_plot, epoch_starts_plot1])
    plt.subplot(312)
    mae_plot, = plt.plot(df.index, df.mean_absolute_error, label='mae')
    epoch_ends_plot2, = plt.plot(epoch_ends, df.mean_absolute_error[epoch_ends], 'ro-', label='epoch ends')
    plt.legend(handles=[mae_plot, epoch_ends_plot2])
    plt.subplot(313)
    mse_plot, = plt.plot(df.index, df.mean_squared_error, label='mse')
    epoch_ends_plot2, = plt.plot(epoch_ends, df.mean_squared_error[epoch_ends], 'ro-', label='epoch ends')
    plt.legend(handles=[mse_plot, epoch_ends_plot2])
