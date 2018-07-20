import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


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
dmap = np.zeros(shape=(len(seg_file_list), 1001, 1001), dtype='float32')
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
        plt.imshow(seg_aux, cmap="gray")
        plt.title('Cell labels')
        plt.subplot(222)
        plt.imshow(mask[i, :, :], cmap="gray")
        plt.title('Training weight')

    # compute distance map to the cell contours
    dmap[i, :, :] = ndimage.distance_transform_edt(seg_aux)

    # plot distance map
    if DEBUG:
        plt.subplot(223)
        plt.imshow(dmap[i, :, :])
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
dmap = dmap.reshape((dmap.shape + (1,)))
