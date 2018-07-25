# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer'),
                 os.path.join(home, 'Software/cytometer')])

import glob
import numpy as np
import pysto.imgproc as pystoim
import tensorflow as tf
import cytometer.data
import cytometer.models as models
import matplotlib.pyplot as plt

DEBUG = True

'''Load data
'''

# data paths
root_data_dir = os.path.join(home, 'Dropbox/klf14')
training_dir = os.path.join(home, 'Dropbox/klf14/klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
saved_models_dir = os.path.join(home, 'Dropbox/klf14/saved_models')

# list of segmented files
seg_file_list = glob.glob(os.path.join(training_non_overlap_data_dir, '*.tif'))

# list of corresponding image patches
im_file_list = [seg_file.replace(training_non_overlap_data_dir, training_dir) for seg_file in seg_file_list]

# load segmentations and compute distance maps
dmap, mask, seg = cytometer.data.load_watershed_seg_and_compute_dmap(seg_file_list)

# load corresponding images
im = cytometer.data.load_im_file_list_to_array(im_file_list)


'''CNN

Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
function
'''

# load model
with tf.device('/cpu:0'):
    model = models.fcn_sherrah2016_modified(input_shape=im.shape[1:])

# load model weights
model.load_weights(os.path.join(saved_models_dir, '2018-07-24T17_56_50.661912_fcn_sherrah2016_modified.h5'))


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
