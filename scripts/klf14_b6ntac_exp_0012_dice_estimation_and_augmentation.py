'''
Train a CNN to estimate the quality of the segmentation produced by my algorithm
(CNN dmap estimation + contour detection + watershed).

The quality measure is the Dice coefficient associated to each segmented cell.
'''

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
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'

# limit GPU memory used
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

# Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
# function
import keras
import keras.backend as K
import cytometer.data
import cytometer.models
from cytometer.utils import principal_curvatures_range_image
import pysto.imgproc as pystoim
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import watershed
from mahotas.labeled import borders

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False

'''Load model
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

saved_contour_model_basename = 'klf14_b6ntac_exp_0006_cnn_contour'  # contour
saved_dmap_model_basename = 'klf14_b6ntac_exp_0007_cnn_dmap'  # dmap

contour_model_name = saved_contour_model_basename + '*.h5'
dmap_model_name = saved_dmap_model_basename + '*.h5'

# filenames of models of each k-fold
contour_model_files = glob.glob(os.path.join(saved_models_dir, contour_model_name))
dmap_model_files = glob.glob(os.path.join(saved_models_dir, dmap_model_name))
contour_n_folds = len(contour_model_files)
dmap_n_folds = len(dmap_model_files)

# load k-fold sets that were used to train the models (we assume they are the same for contours and dmaps)
saved_contour_model_kfold_filename = os.path.join(saved_models_dir, saved_contour_model_basename + '_info.pickle')
with open(saved_contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
im_file_list = aux['file_list']
idx_test_all = aux['idx_test_all']

# correct home directory if we are in a different system than what was used to train the models
im_file_list = cytometer.data.change_home_directory(im_file_list, '/users/rittscher/rcasero', home, check_isfile=True)

'''Data augmentation parameters
'''

# data augmentation factor (e.g. "10" means that we generate 9 augmented images + the original input image)
augment_factor = 10

# we create two instances with the same arguments
data_gen_args = dict(
    rotation_range=90,     # randomly rotate images up to 90 degrees
    fill_mode="constant",  # fill points outside boundaries with zeros
    cval=0,                #
    zoom_range=.1,         # [1-zoom_range, 1+zoom_range]
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True     # randomly flip images
)

dice_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

'''Load data 
'''

fold_i = 0

# We want the same datasets that were used for training, because we want to generate training Dice coefficient images
# for the quality estimation. If we were to compute Dice coeffs on the test data, and then train a CNN with it, we
# wouldn't have any data left over to validate the Dice estimator
_, im_train_file_list = cytometer.data.split_list(im_file_list, idx_test_all[fold_i])

# load datasets.
train_datasets, _, _ = cytometer.data.load_datasets(im_train_file_list, prefix_from='im',
                                                    prefix_to=['im', 'lab', 'seg', 'mask'], nblocks=1)
im_train = train_datasets['im']
seg_train = train_datasets['seg']
mask_train = train_datasets['mask']
lab_train = train_datasets['lab']
del train_datasets

# number of images
n_im = im_train.shape[0]

# list of trained model
contour_model_files = glob.glob(os.path.join(saved_models_dir, contour_model_name))
dmap_model_files = glob.glob(os.path.join(saved_models_dir, dmap_model_name))

# select one of the models
contour_model_file = contour_model_files[fold_i]
dmap_model_file = dmap_model_files[fold_i]

# load model
contour_model = keras.models.load_model(contour_model_file)
dmap_model = keras.models.load_model(dmap_model_file)

# set input layer to size of images
contour_model = cytometer.models.change_input_size(contour_model, batch_shape=(None,) + im_train.shape[1:])
dmap_model = cytometer.models.change_input_size(dmap_model, batch_shape=(None,) + im_train.shape[1:])

border_train = lab_train.copy()
for i in range(lab_train.shape[0]):

    # remove borders between cells in the lab_train data. For this experiment, we want labels touching each other
    lab_train[i, :, :, 0] = watershed(image=np.zeros(shape=lab_train[i, :, :, 0].shape, dtype=np.uint8),
                                      markers=lab_train[i, :, :, 0], watershed_line=False)

    # extract the borders of all labels
    border_train[i, :, :, 0] = borders(lab_train[i, :, :, 0])

# change the background label from 1 to 0
lab_train[lab_train == 1] = 0

'''Segmentation and Dice coefficients
'''

# loop images to compute
labels_test_qual = np.zeros(shape=im_train.shape[:-1] + (1,), dtype=np.float32)
for i in range(im_train.shape[0]):

    # run histology image through network
    contour_train_pred = contour_model.predict(im_train[i, :, :, :].reshape((1,) + im_train.shape[1:]))
    dmap_train_pred = dmap_model.predict(im_train[i, :, :, :].reshape((1,) + im_train.shape[1:]))

    # cell segmentation
    labels, labels_borders = cytometer.utils.segment_dmap_contour(dmap_train_pred[0, :, :, 0],
                                                                  contour=contour_train_pred[0, :, :, 1],
                                                                  border_dilation=0)

    # plot results of cell segmentation
    if DEBUG:

        plt.clf()
        plt.subplot(231)
        plt.imshow(im_train[i, :, :, :])
        plt.title('histology, i = ' + str(i))
        plt.subplot(232)
        plt.imshow(contour_train_pred[0, :, :, 1])
        plt.title('predicted contours')
        plt.subplot(233)
        plt.imshow(dmap_train_pred[0, :, :, 0])
        plt.title('predicted dmap')
        plt.subplot(234)
        plt.imshow(labels)
        plt.title('labels')
        plt.subplot(235)
        plt.imshow(labels_borders)
        plt.title('borders on histology')
        plt.subplot(236)
        plt.imshow(seg_train[i, :, :, 1])
        plt.title('ground truth borders')

    # compute quality measure of estimated labels
    qual = cytometer.utils.segmentation_quality(labels_test=labels,
                                                labels_ref=lab_train[i, :, :, 0])

    # colour the estimated labels with their quality
    lut = np.zeros(shape=(np.max(qual['lab_test']) + 1,), dtype=qual['dice'].dtype)
    lut.fill(np.nan)
    lut[qual['lab_test']] = qual['dice']
    labels_test_qual[i, :, :, 0] = lut[labels]

    # plot validation of cell segmentation
    if DEBUG:
        plt.clf()
        plt.subplot(221)
        plt.imshow(im_train[i, :, :, :])
        plt.title('histology, i = ' + str(i))
        plt.subplot(222)
        plt.imshow(border_train[i, :, :, 0])
        plt.title('ground truth labels')
        plt.subplot(223)
        aux = np.zeros(shape=labels_borders.shape + (3,), dtype=np.float32)
        aux[:, :, 0] = border_train[i, :, :, 0]
        aux[:, :, 1] = labels_borders
        aux[:, :, 2] = border_train[i, :, :, 0]
        plt.imshow(aux)
        plt.title('estimated (green) vs. ground truth (purple)')
        plt.subplot(224)
        aux = np.zeros(shape=labels_borders.shape + (3,), dtype=np.float32)
        aux[:, :, 0] = labels_test_qual[i, :, :, 0]
        aux[:, :, 1] = labels_test_qual[i, :, :, 0]
        aux[:, :, 2] = labels_test_qual[i, :, :, 0]
        aux_r = aux[:, :, 0]
        aux_r[border_train[i, :, :, 0] == 1.0] = 1.0
        aux[:, :, 0] = aux_r
        aux[:, :, 2] = aux_r
        aux_g = aux[:, :, 1]
        aux_g[labels_borders == 1.0] = 1.0
        aux[:, :, 1] = aux_g
        plt.imshow(aux, cmap='Greys_r')
        plt.title('Dice coeff')

    # filenames for the Dice coefficient files
    base_file = im_file_list[i]
    base_path, base_name = os.path.split(base_file)
    dice_file = os.path.join(training_augmented_dir, base_file.replace('im_', 'dice_'))

    # save the Dice coefficient labels
    im_out = labels_test_qual[i, :, :, 0]
    im_out = Image.fromarray(im_out, mode='F')
    im_out.save(dice_file)


# augment Dice images
for seed in range(augment_factor - 1):
    print('* Augmentation round: ' + str(seed + 1) + '/' + str(augment_factor - 1))

    # random rotation, flip and scaling of the image
    labels_test_qual_augmented = dice_datagen.flow(labels_test_qual, seed=seed, shuffle=False, batch_size=n_im).next()

    if DEBUG:
        i = 0

        # file name of the histology corresponding to the random transformation of the Dice coefficient image
        base_file = im_file_list[i]
        base_path, base_name = os.path.split(base_file)
        im_file = os.path.join(training_augmented_dir, base_file.replace('seed_nan', 'seed_' + str(seed).zfill(3)))

        # load histology
        aux_dataset, _, _ = cytometer.data.load_datasets([im_file], prefix_from='im', prefix_to=['im'], nblocks=1)

        # compare randomly transformed histology to corresponding Dice coefficient
        plt.clf()
        plt.subplot(311)
        plt.imshow(aux_dataset['im'][0, :, :, :])
        plt.subplot(312)
        plt.imshow(labels_test_qual_augmented[i, :, :, 0], cmap='Greys_r')
        plt.subplot(313)
        aux = pystoim.imfuse(aux_dataset['im'][0, :, :, :], labels_test_qual_augmented[i, :, :, 0])
        plt.imshow(aux, cmap='Greys_r')

