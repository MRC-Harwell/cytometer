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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
import cytometer.utils
import matplotlib.pyplot as plt
import cv2
import pysto.imgproc as pystoim
from skimage.future.graph import rag_mean_color
from skimage.measure import regionprops
import networkx as nx

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

# compute Region Adjacency Graph (RAG) for labels
rag = rag_mean_color(image=predlab_test[i, :, :, 0], labels=predlab_test[i, :, :, 0])
labels_prop = regionprops(predlab_test[i, :, :, 0], coordinates='rc')
centroids_rc = {}
for lp in labels_prop:
    centroids_rc[lp['label']] = lp['centroid']
centroids_xy = centroids_rc.copy()
for n in centroids_rc.keys():
    centroids_xy[n] = centroids_rc[n][::-1]

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

labels = predlab_test[i, :, :, 0]
receptive_field = (162, 162)

# colour labels
colours, coloured_labels = cytometer.utils.colour_labels_with_receptive_field(labels, receptive_field)

plt.clf()
plt.subplot(221)
plt.imshow(labels)
plt.title('labels')
plt.subplot(222)
plt.imshow(labels)
nx.draw(rag, pos=centroids_xy, node_size=30)
plt.title('label adjacency graph')
plt.subplot(223)
plt.imshow(coloured_labels, cmap='tab10')
c = centroids_rc[38]
plt.plot(c[1], c[0], 'ok')
if receptive_field[0] % 2:
    receptive_field_half = ((receptive_field[0] - 1) / 2,)
else:
    receptive_field_half = (receptive_field[0] / 2,)
if receptive_field[1] % 2:
    receptive_field_half += ((receptive_field[1] - 1) / 2,)
else:
    receptive_field_half += (receptive_field[1] / 2,)
rmin = int(max(0.0, np.round(c[0] - receptive_field_half[0])))
rmax = int(min(labels.shape[0] - 1.0, np.round(c[0] + receptive_field_half[0])))
cmin = int(max(0.0, np.round(c[1] - receptive_field_half[1])))
cmax = int(min(labels.shape[1] - 1.0, np.round(c[1] + receptive_field_half[1])))
plt.plot([cmin, cmax, cmax, cmin, cmin], [rmin, rmin, rmax, rmax, rmin], 'k')
plt.title('coloured labels')
