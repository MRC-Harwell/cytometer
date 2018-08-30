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
import random

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


DEBUG = False

'''Load data
'''

# data paths
root_data_dir = os.path.join(home, 'Dropbox/klf14')
training_dir = os.path.join(home, 'Dropbox/klf14/klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(home, 'OfflineData/klf14/klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(home, 'Dropbox/klf14/saved_models')

# model_name = '2018-08-09T18_59_10.294550_fcn_sherrah2016*.h5'  # dmap regression trained with 6 epochs
# model_name = '2018-08-11T23_10_03.296260_fcn_sherrah2016*.h5'  # dmap regression trained with 15 epochs
# model_name = '2018-08-20T12_15_24.854266_fcn_sherrah2016*.h5'  # First working network with dmap regression + contour classification
model_name = '2018-08-20T16_47_04.292112_fcn_sherrah2016_contour_fold_0.h5'

# list of training images
im_file_list = glob.glob(os.path.join(training_augmented_dir, 'im_*_nan_*.tif'))
seg_file_list = [x.replace('im_', 'seg_') for x in im_file_list]
mask_file_list = [x.replace('im_', 'mask_') for x in im_file_list]

# number of training images
n_im = len(im_file_list)

# load k-folds of the model
model_files = glob.glob(os.path.join(saved_models_dir, model_name))
n_folds = len(model_files)

# create k-fold sets to split the data into training vs. testing
seed = 0
random.seed(seed)
idx = random.sample(range(n_im), n_im)
idx_test_all = np.array_split(idx, n_folds)

# load images
im = cytometer.data.load_im_file_list_to_array(im_file_list)
seg = cytometer.data.load_im_file_list_to_array(seg_file_list)
mask = cytometer.data.load_im_file_list_to_array(mask_file_list)

# convert to float
im = im.astype(np.float32)
im /= 255
mask = mask.astype(np.float32)
seg = seg.astype(np.float32)

if DEBUG:
    for i in range(n_im):
        print('  ** Image: ' + str(i) + '/' + str(n_im - 1))
        plt.clf()
        plt.subplot(221)
        plt.imshow(im[i, :, :, :])
        plt.title('Histology: ' + str(i))
        plt.subplot(222)
        plt.imshow(seg[i, :, :, 0] * mask[i, :, :, 0])
        plt.title('Contour')
        plt.subplot(223)
        plt.imshow(mask[i, :, :, 0])
        plt.subplot(224)
        a = im[i, :, :, :]
        b = mask[i, :, :, 0]
        plt.imshow(pystoim.imfuse(a, b))

'''Receptive field

'''

# list of model files to inspect
model_files = glob.glob(os.path.join(saved_models_dir, model_name))

receptive_field_size = []
for model_file in model_files:

    print(model_file)

    # estimate receptive field of the model
    def model_build_func(input_shape):
        model = models.fcn_sherrah2016_contour(input_shape=input_shape, for_receptive_field=True)
        model.load_weights(model_file)
        return model


    rf = KerasReceptiveField(model_build_func, init_weights=False)

    rf_params = rf.compute(
        input_shape=(500, 500, 3),
        input_layer='input_image',
        output_layers=['classification_output'])
    print(rf_params)

    receptive_field_size.append(rf._rf_params[0].size)

for i, model_file in enumerate(model_files):
    print(model_file)
    print('Receptive field size: ' + str(receptive_field_size[i]))

'''Load model and visualise results
'''

for fold_i, model_file in enumerate(model_files):

    # select test data (data not used for training)
    idx_test = idx_test_all[fold_i]
    im_test = im[idx_test, ...]
    seg_test = seg[idx_test, ...]
    mask_test = mask[idx_test, ...]

    # split data into blocks
    seg_test = seg_test[:, 0:-1, 0:-1, :]
    mask_test = mask_test[:, 0:-1, 0:-1, :]
    im_test = im_test[:, 0:-1, 0:-1, :]

    _, seg_test, _ = pystoim.block_split(seg_test, nblocks=(1, 2, 2, 1))
    _, im_test, _ = pystoim.block_split(im_test, nblocks=(1, 2, 2, 1))
    _, mask_test, _ = pystoim.block_split(mask_test, nblocks=(1, 2, 2, 1))

    seg_test = np.concatenate(seg_test, axis=0)
    im_test = np.concatenate(im_test, axis=0)
    mask_test = np.concatenate(mask_test, axis=0)

    # load model
    model = cytometer.models.fcn_sherrah2016_contour(input_shape=im_test.shape[1:])
    model.load_weights(model_file)

    # visualise results
    if DEBUG:
        for i in range(im_test.shape[0]):

            # run image through network
            seg_test_pred = model.predict(im_test[i, :, :, :].reshape((1,) + im_test.shape[1:]))

            plt.clf()
            plt.subplot(221)
            plt.imshow(im_test[i, :, :, :])
            plt.title('histology, i = ' + str(i))
            plt.subplot(222)
            plt.imshow(seg_test[i, :, :, 0] * mask_test[i, :, :, 0])
            plt.title('ground truth contours')
            plt.subplot(223)
            plt.imshow(seg_test_pred[0, :, :, 0])
            plt.title('estimated contours')

            input("Press Enter to continue...")


'''Plot metrics and convergence
'''

log_filename = os.path.join(saved_models_dir, model_name.replace('*.h5', '.log'))

if os.path.isfile(log_filename):

    # read Keras output
    df_list = cytometer.data.read_keras_training_output(log_filename)

    # plot metrics with every iteration
    plt.clf()
    for df in df_list:
        plt.subplot(311)
        loss_plot, = plt.semilogy(df.index, df.loss, label='loss')
        epoch_ends = np.concatenate((np.where(np.diff(df.epoch))[0], [len(df.epoch)-1, ]))
        epoch_ends_plot1, = plt.semilogy(epoch_ends, df.loss[epoch_ends], 'ro', label='end of epoch')
        plt.legend(handles=[loss_plot, epoch_ends_plot1])
        plt.subplot(312)
        regr_mae_plot, = plt.plot(df.index, df.regression_output_mean_absolute_error, label='regr mae')
        clas_mae_plot, = plt.plot(df.index, df.classification_output_mean_absolute_error, label='clas mae')
        regr_epoch_ends_plot2, = plt.plot(epoch_ends, df.regression_output_mean_absolute_error[epoch_ends], 'ro', label='end of epoch')
        clas_epoch_ends_plot2, = plt.plot(epoch_ends, df.classification_output_mean_absolute_error[epoch_ends], 'ro', label='end of epoch')
        plt.legend(handles=[regr_mae_plot, clas_mae_plot, epoch_ends_plot2])
        plt.subplot(313)
        regr_mse_plot, = plt.semilogy(df.index, df.regression_output_mean_squared_error, label='regr mse')
        clas_mse_plot, = plt.semilogy(df.index, df.classification_output_mean_squared_error, label='clas mse')
        regr_epoch_ends_plot2, = plt.semilogy(epoch_ends, df.regression_output_mean_squared_error[epoch_ends], 'ro', label='end of epoch')
        clas_epoch_ends_plot2, = plt.semilogy(epoch_ends, df.classification_output_mean_squared_error[epoch_ends], 'ro', label='end of epoch')
        plt.legend(handles=[regr_mse_plot, clas_mse_plot, regr_epoch_ends_plot2])


    # plot metrics at end of each epoch
    plt.clf()
    for df in df_list:
        plt.subplot(311)
        epoch_ends = np.concatenate((np.where(np.diff(df.epoch))[0], [len(df.epoch)-1, ]))
        epoch_ends_plot1, = plt.semilogy(epoch_ends, df.loss[epoch_ends], '-', label='loss')
        plt.subplot(312)
        epoch_ends_plot2, = plt.plot(epoch_ends, df.mean_absolute_error[epoch_ends], '-', label='mae')
        plt.subplot(313)
        epoch_ends_plot2, = plt.semilogy(epoch_ends, df.mean_squared_error[epoch_ends], '-', label='mse')

    # concatenate metrics from all folds
    epoch_ends_all = np.empty((0,))
    loss_all = np.empty((0,))
    mae_all = np.empty((0,))
    mse_all = np.empty((0,))
    for df in df_list:
        epoch_ends = np.concatenate((np.where(np.diff(df.epoch))[0], [len(df.epoch) - 1, ]))
        epoch_ends_all = np.concatenate((epoch_ends_all, epoch_ends))
        loss_all = np.concatenate((loss_all, df.loss[epoch_ends]))
        mae_all = np.concatenate((mae_all, df.mean_absolute_error[epoch_ends]))
        mse_all = np.concatenate((mse_all, df.mean_squared_error[epoch_ends]))
        print(str(len(df.mean_squared_error[epoch_ends])))

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

    # Instanciate a Gaussian Process model
    kernel = Matern(nu=1.5)
    gp_loss = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1,
                                       normalize_y=True)
    gp_mae = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1,
                                      normalize_y=True)
    gp_mse = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=10,
                                      normalize_y=True)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp_loss.fit(epoch_ends_all.reshape(-1, 1), loss_all.reshape(-1, 1))
    gp_mae.fit(epoch_ends_all.reshape(-1, 1), mae_all.reshape(-1, 1))
    gp_mse.fit(epoch_ends_all.reshape(-1, 1), mse_all.reshape(-1, 1))

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    xx = np.array(range(460, int(np.max(epoch_ends_all)))).reshape(-1, 1)
    loss_pred, loss_sigma = gp_loss.predict(xx, return_std=True)
    mae_pred, mae_sigma = gp_mae.predict(xx, return_std=True)
    mse_pred, mse_sigma = gp_mse.predict(xx, return_std=True)

    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    plt.clf()
    plt.subplot(311)
    plt.plot(xx, loss_pred, 'k', label='loss')
    plt.plot(epoch_ends_all, loss_all, 'k.', markersize=10, label=u'Observations')
    plt.ylabel('loss')
    plt.subplot(312)
    plt.plot(xx, mae_pred, 'k', label='mae')
    plt.plot(epoch_ends_all, mae_all, 'k.', markersize=10, label=u'Observations')
    plt.ylabel('mae')
    plt.subplot(313)
    plt.plot(xx, mse_pred, 'k', label='mse')
    plt.plot(epoch_ends_all, mse_all, 'k.', markersize=10, label=u'Observations')
    plt.ylabel('mse')
    plt.xlabel('iteration')

