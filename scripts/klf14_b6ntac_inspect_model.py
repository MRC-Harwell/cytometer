# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])

import glob
import numpy as np
import pysto.imgproc as pystoim
import tensorflow as tf
# Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
# function
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

# list of training images
im_file_list = glob.glob(os.path.join(training_augmented_dir, 'im_*.tif'))
dmap_file_list = [x.replace('im_', 'dmap_') for x in im_file_list]
mask_file_list = [x.replace('im_', 'mask_') for x in im_file_list]

# number of training images
n_im = len(im_file_list)

# load images
im = cytometer.data.load_im_file_list_to_array(im_file_list)
dmap = cytometer.data.load_im_file_list_to_array(dmap_file_list)
dmap = dmap.reshape(dmap.shape + (1,))
mask = cytometer.data.load_im_file_list_to_array(mask_file_list)
mask = mask.reshape(mask.shape + (1,))

# convert to float
im = im.astype(np.float32)
im /= 255
mask = mask.astype(np.float32)

'''Receptive field

'''

model_file = os.path.join(saved_models_dir, '2018-07-27T10_00_57.382521_fcn_sherrah2016.h5')

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

print('Receptive field size: ' + str(rf._rf_params[0].size))

'''Load model and visualise results
'''

# # load model
# model = keras.models.load_model(model_file)

# load model
model = models.fcn_sherrah2016(input_shape=im.shape[1:])

# load model weights
model.load_weights(model_file)

# visualise results
if DEBUG:
    for i in range(im.shape[0]):

        # run image through network
        dmap_pred = model.predict(im[i, :, :, :].reshape((1,) + im.shape[1:]))

        plt.clf()
        plt.subplot(221)
        plt.imshow(im[i, :, :, :])
        plt.subplot(222)
        plt.imshow(dmap[i, :, :, :].reshape(dmap.shape[1:3]))
        plt.subplot(223)
        plt.imshow(dmap_pred.reshape(dmap_pred.shape[1:3]))
        plt.subplot(224)
        a = dmap[i, :, :, :].reshape(dmap.shape[1:3])
        b = dmap_pred.reshape(dmap.shape[1:3])
        imax = np.max((np.max(a), np.max(b)))
        a /= imax
        b /= imax
        plt.imshow(pystoim.imfuse(a, b))
        plt.show()

# extract and visualise weights

layer_1 = model.layers[1]
weights_1 = layer_1.get_weights()[0]

if DEBUG:

    plt.clf()
    norm_max = np.max(np.linalg.norm(weights_1, axis=2))
    for i in range(32):
        plt.subplot(8, 4, i + 1)
        plt.imshow(np.linalg.norm(weights_1[:, :, :, i], axis=2) / norm_max)

        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)

