DEBUG = False

import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


# environment variables
os.environ['KERAS_BACKEND'] = 'tensorflow'

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

# data directories
root_data_dir = '/home/rcasero/Dropbox/klf14'
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')

# list of segmented files
seg_file_list = glob.glob(os.path.join(training_non_overlap_data_dir, '*.tif'))

# loop segmented files
for seg_file in seg_file_list:

    # load segmentations
    seg = Image.open(seg_file)
    seg = np.array(seg)

    # set pixels that were background in the watershed algorithm to background here too
    seg[seg == 1] = 0

    # plot image
    if DEBUG:
        plt.clf()
        plt.subplot(221)
        plt.imshow(seg, cmap="gray")

    # compute distance map to the cell contours
    dmap = ndimage.distance_transform_edt(seg)

    # plot distance map
    if DEBUG:
        plt.subplot(222)
        plt.imshow(dmap)

    # load corresponding original image
    im_file = seg_file.replace(training_non_overlap_data_dir, training_data_dir)
    im = Image.open(im_file)

    # plot original image
    if DEBUG:
        plt.subplot(223)
        plt.imshow(im)
