'''
Take the models for fold 0, and segment all 55 images (train+test). Compute Dice coeffs for each segmented cell.
Augment Dice images to prepare training of classifier.

This script creates files dice_kfold_00_seed_XXX_*.tif in klf14_b6ntac_training_augmented.
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
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

# limit GPU memory used
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
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
from skimage.transform import warp


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
saved_dmap_model_basename = 'klf14_b6ntac_exp_0015_cnn_dmap'  # dmap

contour_model_name = saved_contour_model_basename + '*.h5'
dmap_model_name = saved_dmap_model_basename + '*.h5'

# filenames of models of each k-fold
contour_model_files = glob.glob(os.path.join(saved_models_dir, contour_model_name))
dmap_model_files = glob.glob(os.path.join(saved_models_dir, dmap_model_name))
contour_n_folds = len(contour_model_files)
dmap_n_folds = len(dmap_model_files)

# list of segmented files
seg_file_list = glob.glob(os.path.join(training_non_overlap_data_dir, '*.tif'))

# list of images
im_file_list = [seg_file.replace(training_non_overlap_data_dir, training_dir) for seg_file in seg_file_list]

# replace path to the augmented directory, and add prefix 'im_seed_nan_' to filenames
for i in range(len(im_file_list)):
    base_path, base_name = os.path.split(im_file_list[i])
    im_file_list[i] = os.path.join(training_augmented_dir, 'im_seed_nan_' + base_name)

'''Data augmentation parameters
'''

# data augmentation factor (e.g. "10" means that we generate 9 augmented images + the original input image)
augment_factor = 10

'''Load data 
'''

fold_i = 0

# load datasets.
# we need to load all images, whether for training or testing, because otherwise
# ImageDataGenerator is not going to produce the same transformations in
# klf14_b6ntac_exp_0078_generate_augmented_training_images.py
datasets, _, _ = cytometer.data.load_datasets(im_file_list, prefix_from='im',
                                              prefix_to=['im', 'lab', 'seg', 'mask'], nblocks=1)
im = datasets['im']
seg = datasets['seg']
mask = datasets['mask']
reflab = datasets['lab']
del datasets

# number of images
n_im = im.shape[0]

# list of trained model
contour_model_files = glob.glob(os.path.join(saved_models_dir, contour_model_name))
dmap_model_files = glob.glob(os.path.join(saved_models_dir, dmap_model_name))

# select the model that corresponds to current fold
contour_model_file = contour_model_files[fold_i]
dmap_model_file = dmap_model_files[fold_i]

# load model
contour_model = keras.models.load_model(contour_model_file)
dmap_model = keras.models.load_model(dmap_model_file)

# set input layer to size of images
contour_model = cytometer.models.change_input_size(contour_model, batch_shape=(None,) + im.shape[1:])
dmap_model = cytometer.models.change_input_size(dmap_model, batch_shape=(None,) + im.shape[1:])


border = reflab.copy()
for i in range(reflab.shape[0]):

    # remove borders between cells in the lab_train data. For this experiment, we want labels touching each other
    reflab[i, :, :, 0] = watershed(image=np.zeros(shape=reflab[i, :, :, 0].shape, dtype=np.uint8),
                                   markers=reflab[i, :, :, 0], watershed_line=False)

    # extract the borders of all labels
    border[i, :, :, 0] = borders(reflab[i, :, :, 0])

# change the background label from 1 to 0
reflab[reflab == 1] = 0

'''Cell segmentation, match estimated segmentations to ground truth segmentations and Dice coefficients
'''

# loop images to compute
labels = np.zeros(shape=im.shape[:-1] + (1,), dtype=np.int32)
labels_borders = np.zeros(shape=im.shape[:-1] + (1,), dtype=np.uint8)
labels_qual = np.zeros(shape=im.shape[:-1] + (1,), dtype=np.float32)
for i in range(n_im):

    print('Image ' + str(i) + '/' + str(n_im-1))

    # run histology image through network
    contour_pred = contour_model.predict(im[i, :, :, :].reshape((1,) + im.shape[1:]))
    dmap_pred = dmap_model.predict(im[i, :, :, :].reshape((1,) + im.shape[1:]))

    # cell segmentation
    labels[i, :, :, 0], labels_borders[i, :, :, 0] \
        = cytometer.utils.segment_dmap_contour(dmap_pred[0, :, :, 0],
                                               contour=contour_pred[0, :, :, 0],
                                               border_dilation=0)

    # plot results of cell segmentation
    if DEBUG:
        plt.clf()
        plt.subplot(231)
        plt.imshow(im[i, :, :, :])
        plt.title('histology, i = ' + str(i))
        plt.subplot(232)
        plt.imshow(contour_pred[0, :, :, 0])
        plt.title('predicted contours')
        plt.subplot(233)
        plt.imshow(dmap_pred[0, :, :, 0])
        plt.title('predicted dmap')
        plt.subplot(234)
        plt.imshow(labels[i, :, :, 0])
        plt.title('labels')
        plt.subplot(235)
        plt.imshow(labels_borders[i, :, :, 0])
        plt.title('label borders')
        plt.subplot(236)
        plt.imshow(seg[i, :, :, 0])
        plt.title('ground truth borders')

    # compute quality measure of estimated labels
    qual = cytometer.utils.match_overlapping_labels(labels_test=labels[i, :, :, 0],
                                                    labels_ref=reflab[i, :, :, 0])

    # colour the estimated labels with their quality
    lut = np.zeros(shape=(np.max(qual['lab_test']) + 1,), dtype=qual['dice'].dtype)
    lut.fill(np.nan)
    lut[qual['lab_test']] = qual['dice']
    labels_qual[i, :, :, 0] = lut[labels[i, :, :, 0]]

    # plot validation of cell segmentation
    if DEBUG:
        plt.clf()
        plt.subplot(221)
        plt.imshow(im[i, :, :, :])
        plt.title('histology, i = ' + str(i))
        plt.subplot(222)
        plt.imshow(border[i, :, :, 0])
        plt.title('ground truth labels')
        plt.subplot(223)
        aux = np.zeros(shape=labels_borders[i, :, :, 0].shape + (3,), dtype=np.float32)
        aux[:, :, 0] = border[i, :, :, 0]
        aux[:, :, 1] = labels_borders[i, :, :, 0]
        aux[:, :, 2] = border[i, :, :, 0]
        plt.imshow(aux)
        plt.title('estimated (green) vs. ground truth (purple)')
        plt.subplot(224)
        aux = np.zeros(shape=labels_borders[i, :, :, 0].shape + (3,), dtype=np.float32)
        aux[:, :, 0] = labels_qual[i, :, :, 0]
        aux[:, :, 1] = labels_qual[i, :, :, 0]
        aux[:, :, 2] = labels_qual[i, :, :, 0]
        aux_r = aux[:, :, 0]
        aux_r[border[i, :, :, 0] == 1.0] = 1.0
        aux[:, :, 0] = aux_r
        aux[:, :, 2] = aux_r
        aux_g = aux[:, :, 1]
        aux_g[labels_borders[i, :, :, 0] == 1.0] = 1.0
        aux[:, :, 1] = aux_g
        plt.imshow(aux, cmap='Greys_r')
        plt.title('Dice coeff')

    # filenames for output files
    base_file = im_file_list[i]
    base_path, base_name = os.path.split(base_file)
    predlab_file = os.path.join(training_augmented_dir,
                                base_file.replace('im_', 'predlab_kfold_' + str(fold_i).zfill(2) + '_'))
    predseg_file = os.path.join(training_augmented_dir,
                                base_file.replace('im_', 'predseg_kfold_' + str(fold_i).zfill(2) + '_'))
    preddice_file = os.path.join(training_augmented_dir,
                                 base_file.replace('im_', 'preddice_kfold_' + str(fold_i).zfill(2) + '_'))

    # save the predicted labels (one per cell)
    im_out = Image.fromarray(labels[i, :, :, 0], mode='I')  # int32
    im_out.save(predlab_file)

    # save the predicted contours
    im_out = Image.fromarray(labels_borders[i, :, :, 0], mode='L')  # uint8
    im_out.save(predseg_file)

    # save the Dice coefficient labels
    im_out = Image.fromarray(labels_qual[i, :, :, 0], mode='F')  # float32
    im_out.save(preddice_file)


# augment Dice images
for seed in range(augment_factor - 1):

    print('* Augmentation round: ' + str(seed + 1) + '/' + str(augment_factor - 1))

    for i in range(n_im):

        print('** Image ' + str(i) + '/' + str(n_im - 1))

        # file name of the histology corresponding to the random transformation of the Dice coefficient image
        base_file = im_file_list[i]
        base_path, base_name = os.path.split(base_file)
        transform_file = os.path.join(base_path, base_name.replace('im_', 'transform_')
                                      .replace('.tif', '.pickle').replace('seed_nan', 'seed_' + str(seed).zfill(3)))
        transform = pickle.load(open(transform_file, 'br'))

        # convert transform to skimage format
        transform_skimage = cytometer.utils.keras2skimage_transform(keras_transform=transform,
                                                                    input_shape=labels[1:3],
                                                                    output_shape=im.shape[1:3])

        # apply transform to reference images
        labels_augmented = warp(labels[i, :, :, 0], transform_skimage.inverse, order=0, preserve_range=True)
        labels_borders_augmented = warp(labels_borders[i, :, :, 0], transform_skimage.inverse, order=0, preserve_range=True)
        labels_qual_augmented = warp(labels_qual[i, :, :, 0], transform_skimage.inverse, order=1, preserve_range=True)

        # apply flips
        if transform['flip_horizontal'] == 1:
            labels_augmented = labels_augmented[:, ::-1]
            labels_borders_augmented = labels_borders_augmented[:, ::-1]
            labels_qual_augmented = labels_qual_augmented[:, ::-1]
        if transform['flip_vertical'] == 1:
            labels_augmented = labels_augmented[::-1, :]
            labels_borders_augmented = labels_borders_augmented[::-1, :]
            labels_qual_augmented = labels_qual_augmented[::-1, :]

        # convert to types for display and save to file
        labels_augmented = labels_augmented.astype(np.int32)
        labels_borders_augmented = labels_borders_augmented.astype(np.uint8)
        labels_qual_augmented = labels_qual_augmented.astype(np.float32)

        if DEBUG:

            # file name of the histology corresponding to the random transformation of the Dice coefficient image
            im_file_augmented = os.path.join(training_augmented_dir, base_file.replace('seed_nan', 'seed_' + str(seed).zfill(3)))

            # load histology
            aux_dataset, _, _ = cytometer.data.load_datasets([im_file_augmented], prefix_from='im', prefix_to=['im'], nblocks=1)

            # compare randomly transformed histology to corresponding Dice coefficient
            plt.clf()
            plt.subplot(321)
            plt.imshow(aux_dataset['im'][0, :, :, :])
            plt.subplot(322)
            plt.imshow(labels_qual_augmented, cmap='Greys_r')
            plt.subplot(323)
            aux = pystoim.imfuse(aux_dataset['im'][0, :, :, :], labels_qual_augmented)
            plt.imshow(aux, cmap='Greys_r')
            plt.subplot(324)
            plt.imshow(labels_augmented)
            plt.subplot(325)
            plt.imshow(labels_borders_augmented)

        # filenames for the Dice coefficient augmented files
        predlab_file = os.path.join(training_augmented_dir,
                                    base_name.replace('im_seed_nan_',
                                                      'predlab_kfold_' + str(fold_i).zfill(2) + '_seed_' + str(seed).zfill(3) + '_'))
        predseg_file = os.path.join(training_augmented_dir,
                                    base_name.replace('im_seed_nan_',
                                                      'predseg_kfold_' + str(fold_i).zfill(2) + '_seed_' + str(seed).zfill(3) + '_'))
        preddice_file = os.path.join(training_augmented_dir,
                                     base_name.replace('im_seed_nan_',
                                                       'preddice_kfold_' + str(fold_i).zfill(2) + '_seed_' + str(seed).zfill(3) + '_'))

        # save the predicted labels (one per cell)
        im_out = Image.fromarray(labels_augmented.astype(np.int32), mode='I')  # int32
        im_out.save(predlab_file)

        # save the predicted contours
        im_out = Image.fromarray(labels_borders_augmented.astype(np.uint8), mode='L')  # uint8
        im_out.save(predseg_file)

        # save the Dice coefficient labels
        im_out = Image.fromarray(labels_qual_augmented.astype(np.float32), mode='F')  # float32
        im_out.save(preddice_file)
