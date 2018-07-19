import os

# environment variables
os.environ['KERAS_BACKEND'] = 'tensorflow'

# remove warning "Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Just disables the warning, doesn't enable AVX/FMA

# different versions of conda keep the path in different variables
if 'CONDA_ENV_PATH' in os.environ:
    conda_env_path = os.environ['CONDA_ENV_PATH']
elif 'CONDA_PREFIX' in os.environ:
    conda_env_path = os.environ['CONDA_PREFIX']
else:
    conda_env_path = '.'

os.environ['PYTHONPATH'] = os.path.join(os.environ['HOME'], 'Software', 'cytometer', 'cytometer') \
                           + ':' + os.environ['PYTHONPATH']


import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import keras.backend as K
import tensorflow as tf
import keras.preprocessing.image
K.set_image_dim_ordering('tf')
print(K.image_data_format())

import cytometer.models as models

# keras model
from keras.models import Sequential
from keras.layers import Activation, Conv2D, MaxPooling2D
from cytometer.layers import DilatedMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

DEBUG = False



# data directories
root_data_dir = '/home/rcasero/Dropbox/klf14'
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')

''' create training dataset objects
========================================================================================================================
'''

# list of segmented files
seg_file_list = glob.glob(os.path.join(training_non_overlap_data_dir, '*.tif'))

# loop segmented files
mask = np.zeros(shape=(len(seg_file_list), 1001, 1001), dtype='float32')
seg = np.zeros(shape=(len(seg_file_list), 1001, 1001), dtype='uint8')
im = np.zeros(shape=(len(seg_file_list), 1001, 1001, 3), dtype='uint8')
for i, seg_file in enumerate(seg_file_list):

    # load segmentation
    seg_aux = np.array(Image.open(seg_file))

    # we are not going to use for training the pixels that have no cell or contour label. They could be background, but
    # also a broken cell, another type of cell, etc
    mask[i, :, :] = seg_aux != 1

    # set pixels that were background in the watershed algorithm to background here too
    seg_aux[seg_aux == 1] = 0

    # copy data to whole array
    seg[i, :, :] = seg_aux

    # plot image
    if DEBUG:
        plt.clf()
        plt.subplot(221)
        plt.imshow(seg[i, :, :], cmap="gray")
        plt.title('Cell labels')
        plt.subplot(222)
        plt.imshow(mask[i, :, :], cmap="gray")
        plt.title('Training weight')

    # compute distance map to the cell contours
    dmap = ndimage.distance_transform_edt(seg_aux)

    # plot distance map
    if DEBUG:
        plt.subplot(223)
        plt.imshow(dmap)
        plt.title('Distance map')

    # load corresponding original image
    im_file = seg_file.replace(training_non_overlap_data_dir, training_data_dir)
    im[i, :, :, :] = np.array(Image.open(im_file))

    # plot original image
    if DEBUG:
        plt.subplot(224)
        plt.imshow(im[i, :, :, :])
        plt.title('Histology')


# we have to add a dummy dimension to comply with Keras expected format
mask = mask.reshape((mask.shape + (1,)))
seg = seg.reshape((seg.shape + (1,)))
