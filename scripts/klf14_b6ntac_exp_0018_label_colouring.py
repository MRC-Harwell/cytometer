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
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# limit GPU memory used
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = .9
set_session(tf.Session(config=config))

# Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
# function
import keras
import keras.backend as K
import cytometer.data
import cytometer.models
import matplotlib.pyplot as plt
import cv2
from skimage.future.graph import rag_mean_color
from skimage.measure import regionprops
import networkx as nx
import pysto.imgproc as pystoim

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

dice_saved_model_basename = 'klf14_b6ntac_exp_0013_cnn_dice_coeff_with_weights'  # Dice coefficient regression model

dice_model_name = dice_saved_model_basename + '*.h5'

# load model weights for each fold
dice_model_files = glob.glob(os.path.join(saved_models_dir, dice_model_name))
n_folds = len(dice_model_files)

# load k-fold sets that were used to train the models
saved_model_kfold_filename = os.path.join(saved_models_dir, 'klf14_b6ntac_exp_0015_cnn_dmap_info.pickle')
with open(saved_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
im_file_list = aux['file_list']
idx_test_all = aux['idx_test_all']

# correct home directory if we are in a different system than what was used to train the models
im_file_list = cytometer.data.change_home_directory(im_file_list, '/users/rittscher/rcasero', home, check_isfile=True)

'''Load model and visualise results
'''

# list of model files to inspect
dice_model_files = glob.glob(os.path.join(saved_models_dir, dice_model_name))

fold_i = 0
dice_model_file = dice_model_files[fold_i]

# split the data into training and testing datasets
im_test_file_list, _ = cytometer.data.split_list(im_file_list, idx_test_all[fold_i])

# load datasets
test_datasets, _, _ = cytometer.data.load_datasets(im_test_file_list, prefix_from='im',
                                                   prefix_to=['im', 'seg', 'lab',
                                                              'predseg_kfold_' + str(fold_i).zfill(2),
                                                              'predlab_kfold_' + str(fold_i).zfill(2),
                                                              'preddice_kfold_' + str(fold_i).zfill(2)],
                                                   nblocks=2)
im_test = test_datasets['im']
seg_test = test_datasets['seg']
lab_test = test_datasets['lab']
predseg_test = test_datasets['predseg_kfold_00']
predlab_test = test_datasets['predlab_kfold_00']
preddice_test = test_datasets['preddice_kfold_00']
del test_datasets

# load model
dice_model = keras.models.load_model(dice_model_file)

# set input layer to size of test images
dice_model = cytometer.models.change_input_size(dice_model, batch_shape=(None,) + im_test.shape[1:])

# visualise results
i = 0

# run image through network
preddice_test_pred = dice_model.predict(im_test[i, :, :, :].reshape((1,) + im_test.shape[1:]))

"""Split segmentation into multiple masked segmentations where the network can only see one cell at a time
"""

# simplify notation
labels = predlab_test[i, :, :, 0]

# compute the center of mass for each labelled region. E.g. centroids_xy[53, :] is the centroid for label 53.
labels_prop = regionprops(labels, coordinates='rc')
centroids_xy = {}
for lp in labels_prop:
    centroids_xy[lp['label']] = lp['centroid'][::-1]

# compute Region Adjacency Graph (RAG) for labels. Note that we don't care about the mean colour difference
# between regions. We only care about whether labels are adjacent to others or not
rag = rag_mean_color(image=labels, labels=labels)

# colour the nodes in the graph, so that the effective receptive field can only see one cell at a time

# plot results
plt.clf()
plt.subplot(321)
plt.imshow(im_test[i, :, :, :])
plt.title('histology, i = ' + str(i))
plt.subplot(323)
aux = cv2.dilate(predseg_test[i, :, :, 0], kernel=np.ones(shape=(3, 3)))  # dilate for better visualisation
plt.imshow(aux)
plt.title('predicted contours')
plt.subplot(324)
plt.imshow(predlab_test[i, :, :, 0])
plt.title('predicted labels')
plt.subplot(325)
plt.imshow(predlab_test[i, :, :, 0])
nx.draw(rag, pos=centroids_xy, node_size=30)
plt.title('cell adjacency graph')

