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
import random
import cv2
from scipy.ndimage.morphology import binary_dilation

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# limit GPU memory used
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

# Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
# function
import keras.backend as K
import cytometer.data
import cytometer.models as models
import matplotlib.pyplot as plt

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False

'''Load data
'''

# data paths
root_data_dir = os.path.join(home, 'Dropbox/klf14')
training_dir = os.path.join(home, 'Dropbox/klf14/klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(home, 'OfflineData/klf14/klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(home, 'Dropbox/klf14/saved_models')

# model_name = '2018-07-27T10_00_57.382521_fcn_sherrah2016.h5'
# model_name = '2018-07-28T00_40_03.181899_fcn_sherrah2016.h5'
# model_name = '2018-07-31T12_48_02.977755_fcn_sherrah2016.h5'
# model_name = '2018-08-06T18_02_55.864612_fcn_sherrah2016.h5'
# model_name = '2018-08-09T18_59_10.294550_fcn_sherrah2016*.h5'
model_name = '2018-08-11T23_10_03.296260_fcn_sherrah2016*.h5'

# list of training images
im_file_list = glob.glob(os.path.join(training_augmented_dir, 'im_*_nan_*.tif'))
dmap_file_list = [x.replace('im_', 'dmap_') for x in im_file_list]
mask_file_list = [x.replace('im_', 'mask_') for x in im_file_list]

# number of training images
n_im = len(im_file_list)

# load k-folds of the model
model_files = np.sort(glob.glob(os.path.join(saved_models_dir, model_name)))
n_folds = len(model_files)

# create k-fold splitting of data
seed = 0
random.seed(seed)
idx = random.sample(range(n_im), n_im)
idx_test_all = np.array_split(idx, n_folds)

# load images
im = cytometer.data.load_im_file_list_to_array(im_file_list)
dmap = cytometer.data.load_im_file_list_to_array(dmap_file_list)
mask = cytometer.data.load_im_file_list_to_array(mask_file_list)

# convert to float
im = im.astype(np.float32)
im /= 255
mask = mask.astype(np.float32)

if DEBUG:
    for i in range(n_im):
        print('  ** Image: ' + str(i) + '/' + str(n_im - 1))
        plt.clf()
        plt.subplot(221)
        plt.imshow(im[i, :, :, :])
        plt.subplot(222)
        plt.imshow(dmap[i, :, :, 0])
        plt.subplot(223)
        plt.imshow(mask[i, :, :, 0])
        plt.subplot(224)
        a = im[i, :, :, :]
        b = mask[i, :, :, 0]
        plt.imshow(pystoim.imfuse(a, b))
        plt.show()

'''Load model and estimate dmap
'''

def inverse_dmap(dmap):

    inv_dmap = np.zeros(shape=dmap.shape, dtype=np.float32)

    if DEBUG:
        plt.clf()
        plt.subplot(221)
        plt.imshow(dmap)

    # loop pixels in dmap
    for ij in range(dmap.size):

        # row, column indices
        i, j = np.unravel_index(ij, dmap.shape)

        # skip pixels where the distance estimate is saturated
        if dmap[i, j] >= 72:
            continue

        # circle radius is the distance value in the dmap
        radius = dmap[i, j]

        # circle's centre point is the middle of the pixel (Note: i,j -> y,x)
        center = (j, i)

        aux = np.zeros(shape=dmap.shape, dtype=np.float32)
        inv_dmap += cv2.circle(aux, center=center, radius=radius, color=1.0)

        if DEBUG:
            plt.subplot(222)
            plt.imshow(aux)

    # plot result
    if DEBUG:
        plt.subplot(222)
        plt.imshow(inv_dmap)

        foo = plt.hist(inv_dmap.flatten())
        d = foo[0]
        bins = (foo[1][1:] + foo[1][:-1])/2
        plt.cla()
        plt.semilogy(bins, d)

    # threshold the inverse dmap
    inv_max = np.max(inv_dmap)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    foo = inv_dmap / np.max(inv_dmap)
    # foo = foo.astype(np.uint8)
    # dst = cv2.adaptiveThreshold(foo, maxValue=1, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
    #                             thresholdType=cv2.THRESH_BINARY,
    #                             blockSize=67, C=0)
    plt.subplot(224)
    plt.cla()
    plt.imshow(foo > .2)



# debug
fold_i = 0
model_file = model_files[fold_i]

for fold_i, model_file in enumerate(model_files):

    # select test data (data not used for training)
    idx_test = idx_test_all[fold_i]
    im_test = im[idx_test, ...]
    dmap_test = dmap[idx_test, ...]
    mask_test = mask[idx_test, ...]

    # split data into blocks
    dmap_test = dmap_test[:, 0:-1, 0:-1, :]
    mask_test = mask_test[:, 0:-1, 0:-1, :]
    im_test = im_test[:, 0:-1, 0:-1, :]

    _, dmap_test, _ = pystoim.block_split(dmap_test, nblocks=(1, 2, 2, 1))
    _, im_test, _ = pystoim.block_split(im_test, nblocks=(1, 2, 2, 1))
    _, mask_test, _ = pystoim.block_split(mask_test, nblocks=(1, 2, 2, 1))

    dmap_test = np.concatenate(dmap_test, axis=0)
    im_test = np.concatenate(im_test, axis=0)
    mask_test = np.concatenate(mask_test, axis=0)

    # load model
    model = cytometer.models.fcn_sherrah2016_regression(input_shape=im_test.shape[1:])
    model.load_weights(model_file)


    # visualise results
    if DEBUG:

        # debug
        i = 10

        for i in range(im_test.shape[0]):

            # run image through network
            dmap_test_pred = model.predict(im_test[i, :, :, :].reshape((1,) + im_test.shape[1:]))
            dmap_test_pred = dmap_test_pred.reshape(dmap_test_pred.shape[1:3])

            if DEBUG:
                plt.clf()
                plt.subplot(221)
                plt.imshow(im_test[i, :, :, :])
                plt.title('histology, i = ' + str(i))
                plt.subplot(222)
                plt.imshow(dmap_test[i, :, :, :].reshape(dmap_test.shape[1:3]))
                plt.title('ground truth dmap')
                plt.subplot(223)
                plt.imshow(dmap_test_pred)
                plt.title('estimated dmap')
                plt.subplot(224)
                a = dmap_test[i, :, :, :].reshape(dmap_test.shape[1:3])
                b = dmap_test_pred
                c = mask_test[i, :, :, 0]
                plt.imshow(np.abs((b - a)) * c)
                plt.colorbar()
                plt.title('error |est - gt| * mask')

                input("Press Enter to continue...")

            mask = np.zeros(shape=)


            from skimage.segmentation import morphological_geodesic_active_contour
            contours = morphological_geodesic_active_contour(dmap_test_pred[0, :, :, 0],
                                                             iterations=10000,
                                                             init_level_set='checkerboard',
                                                             smoothing=4,
                                                             balloon=0.0)

            contours = cv2.Laplacian(dmap_test_pred[0, :, :, 0], cv2.CV_32F)

            import CGAL
            from CGAL.CGAL_Point_set_processing_3 import jet_estimate_normals

            plt.clf()
            plt.subplot(211)
            plt.imshow(dmap_test_pred.reshape(dmap_test_pred.shape[1:3]))
            plt.subplot(212)
            plt.imshow(contours)


            if DEBUG:
                plt.clf()
                plt.subplot(221)
                plt.imshow(im_test[i, :, :, :])
                plt.title('histology, i = ' + str(i))
                plt.subplot(222)
                plt.imshow(dmap_test[i, :, :, :].reshape(dmap_test.shape[1:3]))
                plt.title('ground truth dmap')
                plt.subplot(223)
                plt.imshow(dmap_test_pred.reshape(dmap_test_pred.shape[1:3]))
                plt.title('estimated dmap')
                plt.subplot(224)
                plt.imshow(contours)
                plt.show()

                input("Press Enter to continue...")
