# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import pickle

import glob
import numpy as np

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

# limit GPU memory used
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

# Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
# function
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AvgPool2D, Activation
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import cytometer.data
import cytometer.models as models
from cytometer.utils import principal_curvatures_range_image
import matplotlib.pyplot as plt
from receptivefield.keras import KerasReceptiveField

import cv2
from skimage.morphology import skeletonize

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = True

'''Load data
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

saved_model_basename = 'klf14_b6ntac_exp_0000_cnn_dmap_contour'  # dmap + contour, classification loss weight 1000, hard_sigmoid for classification

model_name = saved_model_basename + '*.h5'

# load model weights for each fold
model_files = glob.glob(os.path.join(saved_models_dir, model_name))
n_folds = len(model_files)

# load k-fold sets that were used to train the models
saved_model_kfold_filename = os.path.join(saved_models_dir, saved_model_basename + '_info.pickle')
with open(saved_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
im_file_list = aux['file_list']
idx_test_all = aux['idx_test_all']

# correct home directory if we are in a different system than what was used to train the models
im_file_list = cytometer.data.change_home_directory(im_file_list, '/users/rittscher/rcasero', home, check_isfile=True)

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


'''Load example data to display'''

# load im, seg and mask datasets
datasets, _, _ = cytometer.data.load_datasets(im_file_list, prefix_from='im', prefix_to=['im', 'seg', 'mask', 'dmap'])
im = datasets['im']
seg = datasets['seg']
mask = datasets['mask']
dmap = datasets['dmap']
del datasets

# number of training images
n_im = im.shape[0]

if DEBUG:
    i = 10
    print('  ** Image: ' + str(i) + '/' + str(n_im - 1))
    plt.clf()
    plt.subplot(221)
    plt.imshow(im[i, :, :, :])
    plt.title('Histology: ' + str(i))
    plt.subplot(222)
    plt.imshow(seg[i, :, :, 0])
    plt.title('Labels')
    plt.subplot(223)
    plt.imshow(dmap[i, :, :, 0])
    plt.title('Distance transformation')
    plt.subplot(224)
    plt.imshow(mask[i, :, :, 0])
    plt.title('Mask')

'''Receptive field

'''

# list of model files to inspect
model_files = glob.glob(os.path.join(saved_models_dir, model_name))

receptive_field_size = []
for model_file in model_files:

    print(model_file)

    # estimate receptive field of the model
    def model_build_func(input_shape):
        model = fcn_sherrah2016_regression_and_classifier(input_shape=input_shape, for_receptive_field=True)
        model.load_weights(model_file)
        return model


    rf = KerasReceptiveField(model_build_func, init_weights=False)

    rf_params = rf.compute(
        input_shape=(500, 500, 3),
        input_layer='input_image',
        output_layers=['regression_output', 'classification_output'])
    print(rf_params)

    receptive_field_size.append(rf._rf_params[0].size)

for i, model_file in enumerate(model_files):
    print(model_file)
    print('Receptive field size: ' + str(receptive_field_size[i]))

'''Load model and visualise results
'''

fold_i = 0
model_file = model_files[fold_i]

# split the data into training and testing datasets
im_test_file_list, _ = cytometer.data.split_list(im_file_list, idx_test_all[fold_i])

# load im, seg and mask datasets
test_datasets, _, _ = cytometer.data.load_datasets(im_test_file_list, prefix_from='im',
                                                   prefix_to=['im', 'seg', 'mask', 'dmap'], nblocks=2)
im_test = test_datasets['im']
seg_test = test_datasets['seg']
mask_test = test_datasets['mask']
dmap_test = test_datasets['dmap']
del test_datasets

# load model
model = fcn_sherrah2016_regression_and_classifier(input_shape=im_test.shape[1:])
model.load_weights(model_file)

# visualise results
i = 0
# run image through network
dmap_test_pred, contour_test_pred = model.predict(im_test[i, :, :, :].reshape((1,) + im_test.shape[1:]))

# compute mean curvature from dmap
_, mean_curvature, _, _ = principal_curvatures_range_image(dmap_test_pred[0, :, :, 0], sigma=10)

# plot results
plt.clf()
plt.subplot(321)
plt.imshow(im_test[i, :, :, :])
plt.title('histology, i = ' + str(i))
plt.subplot(322)
plt.imshow(dmap_test[i, :, :, 0])
plt.title('ground truth dmap')
plt.subplot(323)
plt.imshow(contour_test_pred[0, :, :, 0])
plt.title('predicted contours')
plt.subplot(324)
plt.imshow(dmap_test_pred[0, :, :, 0])
plt.title('predicted dmap')
plt.subplot(325)
plt.imshow(mean_curvature)
plt.title('mean curvature of dmap')
plt.subplot(326)
plt.imshow(mean_curvature * contour_test_pred[0, :, :, 0])
plt.title('predicted contours * mean curvature')

# visualise results
i = 18
# run image through network
dmap_test_pred, contour_test_pred = model.predict(im_test[i, :, :, :].reshape((1,) + im_test.shape[1:]))

# compute mean curvature from dmap
_, mean_curvature, _, _ = principal_curvatures_range_image(dmap_test_pred[0, :, :, 0], sigma=10)

# plot results
plt.clf()
plt.subplot(321)
plt.imshow(im_test[i, :, :, :])
plt.title('histology, i = ' + str(i))
plt.subplot(322)
plt.imshow(dmap_test[i, :, :, 0])
plt.title('ground truth dmap')
plt.subplot(323)
plt.imshow(contour_test_pred[0, :, :, 0])
plt.title('predicted contours')
plt.subplot(324)
plt.imshow(dmap_test_pred[0, :, :, 0])
plt.title('predicted dmap')
plt.subplot(325)
plt.imshow(mean_curvature)
plt.title('mean curvature of dmap')
plt.subplot(326)
plt.imshow(mean_curvature * contour_test_pred[0, :, :, 0])
plt.title('predicted contours * mean curvature')


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
        regr_mae_plot, = plt.plot(df.index, df.regression_output_mean_absolute_error, label='dmap mae')
        regr_mse_plot, = plt.plot(df.index, np.sqrt(df.regression_output_mean_squared_error), label='sqrt(dmap mse)')
        regr_mae_epoch_ends_plot2, = plt.plot(epoch_ends, df.regression_output_mean_absolute_error[epoch_ends], 'ro', label='end of epoch')
        regr_mse_epoch_ends_plot2, = plt.plot(epoch_ends, np.sqrt(df.regression_output_mean_squared_error[epoch_ends]), 'ro', label='end of epoch')
        plt.legend(handles=[regr_mae_plot, regr_mse_plot, regr_mae_epoch_ends_plot2])
        plt.subplot(313)
        clas_mae_plot, = plt.plot(df.index, df.classification_output_mean_absolute_error, label='cont mae')
        clas_mse_plot, = plt.plot(df.index, np.sqrt(df.classification_output_mean_squared_error), label='sqrt(cont mse)')
        clas_mae_epoch_ends_plot2, = plt.plot(epoch_ends, df.classification_output_mean_absolute_error[epoch_ends], 'ro', label='end of epoch')
        clas_mse_epoch_ends_plot2, = plt.plot(epoch_ends, np.sqrt(df.classification_output_mean_squared_error[epoch_ends]), 'ro', label='end of epoch')
        plt.legend(handles=[clas_mae_plot, clas_mse_plot, clas_mae_epoch_ends_plot2])

