'''
Combine the contours estimated:
* directly with the classification CNN
* computing normal curvature on dmap estimated with the regression CNN

Extract cells using watershed.
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
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
import matplotlib.pyplot as plt

from skimage import measure
from skimage.morphology import watershed
from mahotas.labeled import borders
import cv2

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

saved_contour_model_basename = 'klf14_b6ntac_exp_0006_cnn_contour'  # contour
saved_dmap_model_basename = 'klf14_b6ntac_exp_0007_cnn_dmap'  # dmap

contour_model_name = saved_contour_model_basename + '*.h5'
dmap_model_name = saved_dmap_model_basename + '*.h5'

# load model weights for each fold
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

'''Load model and visualise results
'''

fold_i = 0

# split the data into training and testing datasets
im_test_file_list, _ = cytometer.data.split_list(im_file_list, idx_test_all[fold_i])

# load im, seg and mask datasets
test_datasets, _, _ = cytometer.data.load_datasets(im_test_file_list, prefix_from='im',
                                                   prefix_to=['im', 'seg', 'mask'], nblocks=2)
im_test = test_datasets['im']
seg_test = test_datasets['seg']
mask_test = test_datasets['mask']
del test_datasets

# list of model files to inspect
contour_model_files = glob.glob(os.path.join(saved_models_dir, contour_model_name))
dmap_model_files = glob.glob(os.path.join(saved_models_dir, dmap_model_name))

contour_model_file = contour_model_files[fold_i]
dmap_model_file = dmap_model_files[fold_i]

# load models
contour_model = keras.models.load_model(contour_model_file)
dmap_model = keras.models.load_model(dmap_model_file)

# set input layer to size of test images
contour_model = cytometer.models.change_input_size(contour_model, batch_shape=(None,) + im_test.shape[1:])
dmap_model = cytometer.models.change_input_size(dmap_model, batch_shape=(None,) + im_test.shape[1:])

# visualise results
i = 0
# i = 18
# run image through network
contour_test_pred = contour_model.predict(im_test[i, :, :, :].reshape((1,) + im_test.shape[1:]))
dmap_test_pred = dmap_model.predict(im_test[i, :, :, :].reshape((1,) + im_test.shape[1:]))

# compute mean curvature from dmap
_, mean_curvature, _, _ = principal_curvatures_range_image(dmap_test_pred[0, :, :, 0], sigma=10)

# multiply mean curvature by estimated contours
contour_weighted = contour_test_pred[0, :, :, 1] * mean_curvature

# rough segmentation of inner areas
labels = (contour_weighted <= 0).astype('uint8')

# label areas with a different label per connected area
labels = measure.label(labels)

# remove very small labels (noise)
labels_prop = measure.regionprops(labels)
for j in range(1, np.max(labels)):
    # label of region under consideration is not the same as index j
    lab = labels_prop[j]['label']
    if labels_prop[j]['area'] < 50:
        labels[labels == lab] = 0

# extend labels using watershed
# labels_ext = watershed(-dmap_test_pred[0, :, :, 0], labels)
labels_ext = watershed(mean_curvature, labels)

# extract borders of watershed regions for plots
labels_borders = borders(labels_ext)

# dilate borders for easier visualization
kernel = np.ones((3, 3), np.uint8)
labels_borders = cv2.dilate(labels_borders.astype(np.uint8), kernel=kernel) > 0

# add borders as coloured curves
im_test_r = im_test[i, :, :, 0].copy()
im_test_g = im_test[i, :, :, 1].copy()
im_test_b = im_test[i, :, :, 2].copy()
im_test_r[labels_borders] = 0.0
im_test_g[labels_borders] = 1.0
im_test_b[labels_borders] = 0.0
im_borders = np.concatenate((np.expand_dims(im_test_r, axis=2),
                             np.expand_dims(im_test_g, axis=2),
                             np.expand_dims(im_test_b, axis=2)), axis=2)

# plot results
plt.clf()
plt.subplot(331)
plt.imshow(im_test[i, :, :, :])
plt.title('histology, i = ' + str(i))
plt.subplot(332)
plt.imshow(contour_test_pred[0, :, :, 1])
plt.title('predicted contours')
plt.subplot(333)
plt.imshow(dmap_test_pred[0, :, :, 0])
plt.title('predicted dmap')
plt.subplot(334)
plt.imshow(mean_curvature)
plt.title('dmap\'s mean curvature')
plt.subplot(335)
plt.imshow(contour_weighted)
plt.title('contour * curvature')
plt.subplot(336)
plt.imshow(labels)
plt.title('labels')
plt.subplot(337)
plt.imshow(labels_ext)
plt.title('watershed on labels')
plt.subplot(338)
plt.imshow(im_borders)
plt.title('watershed on labels')
