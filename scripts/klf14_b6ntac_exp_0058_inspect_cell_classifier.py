"""
Load testing data from each fold according to exp 0053, apply segmentation and quality network.
"""

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
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from mahotas.features import haralick

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

import cytometer.utils
import cytometer.data
import tensorflow as tf
from sklearn import manifold

# limit GPU memory used
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False

# number of folds for k-fold cross validation
n_folds = 10

# number of epochs for training
epochs = 25

# area (pixel**2) of the smallest object we accept as a cell (pi * (16 pixel)**2 = 804.2 pixel**2)
smallest_cell_area = 804

# training window length
training_window_len = 401

# remove from training cells that don't have a good enough overlap with a reference label
smallest_dice = 0.5

# segmentations with Dice >= threshold are accepted
dice_threshold = 0.9

# batch size for training
batch_size = 16


'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0054_inspect_testing_dataset'

# load k-folds training and testing data
kfold_info_filename = os.path.join(saved_models_dir, 'klf14_b6ntac_exp_0053_cnn_quality_network_fcn_overlapping_scaled_contours_kfold_info.pickle')
with open(kfold_info_filename, 'rb') as f:
    kfold_info = pickle.load(f)
file_list = kfold_info['file_list']
idx_test_all = kfold_info['idx_test']
idx_train_all = kfold_info['idx_train']
del kfold_info

# model names
contour_model_basename = 'klf14_b6ntac_exp_0055_cnn_contour_model'
dmap_model_basename = 'klf14_b6ntac_exp_0056_cnn_dmap_model'
quality_model_basename = 'klf14_b6ntac_exp_0053_cnn_quality_network_fcn_overlapping_scaled_contours_model'
classifier_model_basename = 'klf14_b6ntac_exp_0057_cnn_tissue_classifier_fcn_overlapping_scaled_contours_model'

# number of images
n_im = len(file_list)


'''Load the texture of each cell in each image
'''

# init output
contour_type_all = []
cell_features_all = []
window_idx_all = []
cell_seg_gtruth_all = []
cell_im_all = []

# loop files with hand traced contours
for i, file_svg in enumerate(file_list):

    print('file ' + str(i) + '/' + str(len(file_list) - 1))

    # change file extension from .svg to .tif
    file_tif = file_svg.replace('.svg', '.tif')

    # open histology training image
    im = Image.open(file_tif)

    # make array copy
    im_array = np.array(im)

    if DEBUG:
        plt.clf()
        plt.imshow(im)

    # read the ground truth cell contours in the SVG file. This produces a list [contour_0, ..., contour_N-1]
    # where each contour_i = [(X_0, Y_0), ..., (X_P-1, X_P-1)]
    cell_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False,
                                                            minimum_npoints=3)
    other_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Other', add_offset_from_filename=False,
                                                             minimum_npoints=3)
    damaged_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Damaged', add_offset_from_filename=False,
                                                               minimum_npoints=3)

    # make a list with the type of cell each contour is classified as
    contour_type = [np.zeros(shape=(len(cell_contours),), dtype=np.uint8),
                    np.ones(shape=(len(other_contours),), dtype=np.uint8),
                    2 * np.ones(shape=(len(damaged_contours),), dtype=np.uint8)]
    contour_type = np.concatenate(contour_type)
    contour_type_all.append(contour_type)

    print('Cells: ' + str(len(cell_contours)))
    print('Other: ' + str(len(other_contours)))
    print('Damaged: ' + str(len(damaged_contours)))
    print('')

    # loop ground truth cell contours
    for j, contour in enumerate(cell_contours + other_contours + damaged_contours):

        if DEBUG:
            # centre of current cell
            xy_c = (np.mean([p[0] for p in contour]), np.mean([p[1] for p in contour]))

            plt.clf()
            plt.subplot(221)
            plt.imshow(im)
            plt.plot([p[0] for p in contour], [p[1] for p in contour])
            plt.scatter(xy_c[0], xy_c[1])

        # rasterise current ground truth segmentation
        cell_seg_gtruth = Image.new("1", im.size, "black")  # I = 32-bit signed integer pixels
        draw = ImageDraw.Draw(cell_seg_gtruth)
        draw.polygon(contour, outline="white", fill="white")
        cell_seg_gtruth = np.array(cell_seg_gtruth, dtype=np.uint8)

        # mask histology with segmentation mask
        cell_masked_im = cytometer.utils.quality_model_mask(cell_seg_gtruth, im=im_array, quality_model_type='0_1')
        cell_masked_im = cell_masked_im[0, :, :, :]

        if DEBUG:
            plt.subplot(222)
            plt.cla()
            plt.imshow(cell_masked_im)

        # compute texture vectors per channel
        cell_features = (haralick(cell_masked_im[:, :, 0], ignore_zeros=True),
                         haralick(cell_masked_im[:, :, 1], ignore_zeros=True),
                         haralick(cell_masked_im[:, :, 2], ignore_zeros=True))
        cell_features = np.vstack(cell_features)
        cell_features = cell_features.flatten()

        # append results to total vectors
        cell_features_all.append(cell_features)
        window_idx_all.append(np.array([i, j]))
        cell_seg_gtruth_all.append(np.expand_dims(cell_seg_gtruth, axis=0))
        im_array_all.append(np.expand_dims(im_array, axis=0))

# collapse lists into arrays
contour_type_all = np.concatenate(contour_type_all)
cell_features_all = np.vstack(cell_features_all)
window_idx_all = np.vstack(window_idx_all)
cell_seg_gtruth_all = np.concatenate(cell_seg_gtruth_all)
im_array_all = np.concatenate(im_array_all)


'''t-SNE embedding
'''

for i_fold in range(0, n_folds):

    print('# Fold ' + str(i_fold) + '/' + str(n_folds - 1))

    # test and training image indices
    idx_test = idx_test_all[i_fold]
    idx_train = idx_train_all[i_fold]

    # get cell indices for test and training, based on the image indices
    idx_test = np.where([x in idx_test for x in window_idx_all[:, 0]])[0]
    idx_train = np.where([x in idx_train for x in window_idx_all[:, 0]])[0]

    print('## len(idx_train) = ' + str(len(idx_train)))
    print('## len(idx_test) = ' + str(len(idx_test)))

    # embedding
    tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    cell_features_embedding = tsne.fit_transform(cell_features_all)

    if DEBUG:
        color = np.array(['C0', 'C1', 'C2'])
        plt.clf()
        plt.scatter(cell_features_embedding[:, 0], cell_features_embedding[:, 1], c=color[contour_type_all],
                    cmap=plt.cm.Spectral, s=2)

        plt.show()

'''Validate classifier network
'''

for i_fold in range(0, n_folds):

    print('# Fold ' + str(i_fold) + '/' + str(n_folds - 1))

    # test and training image indices
    idx_test = idx_test_all[i_fold]
    idx_train = idx_train_all[i_fold]

    # get cell indices for test and training, based on the image indices
    idx_test = np.where([x in idx_test for x in window_idx_all[:, 0]])[0]
    idx_train = np.where([x in idx_train for x in window_idx_all[:, 0]])[0]

    print('## len(idx_train) = ' + str(len(idx_train)))
    print('## len(idx_test) = ' + str(len(idx_test)))

    # extract test arrays
    contour_type_test = contour_type_all[idx_test]
    window_idx_test = window_idx_all[idx_test, :]
    cell_seg_gtruth_test = cell_seg_gtruth_all[idx_test, :, :]
    im_array_test = im_array_all[idx_test, :, :, :]

    # prepare arrays for CNN
    im_array_test = im_array_test.astype(np.float32)
    im_array_test /= 255

    # load classifier network
    classifier_model_filename = os.path.join(saved_models_dir,
                                             classifier_model_basename + '_fold_' + str(i_fold) + '.h5')
    classifier_model = keras.models.load_model(classifier_model_filename)

    # apply classification network to cell histology
    cell_ = classifier_model.predict(im_array_test[j, :, :, :], batch_size=batch_size)

    for j in range(len(idx_test)):

