"""
scripts/klf14_b6ntac_generate_augmented_training_images.py

Script to generate augmented training data for the klf14_b6ntac experiments.

This script loads histology windows, masks, distance transformations and labels, and generates
random rotations and scale changes (consistent between corresponding data), and saves them to
file for later use in training.
"""

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])

# other imports
import glob
import shutil
import numpy as np
import pysto.imgproc as pystoim
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# use CPU for testing on laptop
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

# set display for the server
#os.environ['DISPLAY'] = 'localhost:11.0'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import cytometer.data

# limit GPU memory used
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))


DEBUG = False


'''Load data
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(home, 'Data/cytometer_data/klf14/klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(home, 'Data/cytometer_data/klf14/klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(home, 'Data/cytometer_data/klf14/saved_models')

# list of segmented files
seg_file_list = glob.glob(os.path.join(training_non_overlap_data_dir, '*.tif'))

# list of corresponding image patches
im_file_list = [seg_file.replace(training_non_overlap_data_dir, training_dir) for seg_file in seg_file_list]

# load segmentations and compute distance maps
dmap, mask, seg = cytometer.data.load_watershed_seg_and_compute_dmap(seg_file_list)

# load corresponding images and convert to float format
im = cytometer.data.load_file_list_to_array(im_file_list)
im = im.astype('float32', casting='safe')
im /= 255

# number of training images
n_im = im.shape[0]

# set all inside pixels of segmentation to same value
seg = (seg < 2).astype(np.uint8)

# dilate contours
for i in range(n_im):
    seg[i, :, :, 0] = cv2.dilate(seg[i, :, :, 0], kernel=np.ones(shape=(3, 3)))

'''Data augmentation
'''

# copy or save the original data to the augmented directory, so that later we can use a simple flow_from_directory()
# method to read all the augmented data, and we don't need to recompute the distance transformations
for i, base_file in enumerate(im_file_list):

    if DEBUG:
        plt.clf()
        plt.subplot(221)
        plt.imshow(dmap[i, :, :, 0])
        plt.subplot(222)
        plt.imshow(mask[i, :, :, 0])
        plt.subplot(223)
        plt.imshow(seg[i, :, :, 0] * mask[i, :, :, 0])

    # create filenames based on the original foo.tif, so that we have
    # * im_seed_nan_foo.tif
    # * dmap_seed_nan_foo.tif
    # * mask_seed_nan_foo.tif
    # * labels_seed_nan_foo.tif
    # where seed_nan refers to the original data without augmentation variations
    base_path, base_name = os.path.split(base_file)
    im_file = os.path.join(training_augmented_dir, 'im_seed_nan_' + base_name)
    dmap_file = os.path.join(training_augmented_dir, 'dmap_seed_nan_' + base_name)
    mask_file = os.path.join(training_augmented_dir, 'mask_seed_nan_' + base_name)
    seg_file = os.path.join(training_augmented_dir, 'seg_seed_nan_' + base_name)

    # copy the image file and create files for the dmap and mask
    shutil.copy2(base_file, im_file)

    # save distance transforms (note: we have to save as float mode)
    im_out = dmap[i, :, :, 0]
    im_out = Image.fromarray(im_out, mode='F')
    im_out.save(dmap_file)

    # save mask (note: we can save as black and white 1 byte)
    im_out = mask[i, :, :, 0]
    im_out = im_out.astype(np.uint8)
    im_out = Image.fromarray(im_out, mode='L')
    im_out.save(mask_file)

    # set all contour pixels
    im_out = seg[i, :, :, 0] * mask[i, :, :, 0]
    im_out = Image.fromarray(im_out.astype(np.uint32), mode='I')
    im_out.save(seg_file)


# data augmentation factor (e.g. "10" means that we generate 9 augmented images + the original input image)
augment_factor = 10

# we create two instances with the same arguments
data_gen_args = dict(
    rotation_range=90,     # randomly rotate images up to 90 degrees
    fill_mode="constant",  # fill points outside boundaries with zeros
    cval=0,                #
    zoom_range=.1,         # [1-zoom_range, 1+zoom_range]
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True     # randomly flip images
)

dmap_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
im_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
mask_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
seg_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

# augment data, using the same seed so that all corresponding images, dmaps and masks undergo
# the same transformations
# this is a hack so that we have a different seed for each batch of augmented data
for seed in range(augment_factor - 1):

    print('* Augmentation round: ' + str(seed + 1) + '/' + str(augment_factor - 1))

    dmap_augmented = dmap_datagen.flow(dmap, seed=seed, shuffle=False, batch_size=n_im).next()
    im_augmented = im_datagen.flow(im, seed=seed, shuffle=False, batch_size=n_im).next()
    mask_augmented = np.round(mask_datagen.flow(mask, seed=seed, shuffle=False, batch_size=n_im).next())
    seg_augmented = np.round(seg_datagen.flow(seg, seed=seed, shuffle=False, batch_size=n_im).next())
    seg_augmented = seg_augmented.astype(np.uint8)

    for i in range(n_im):

        print('  ** Image: ' + str(i) + '/' + str(n_im - 1))

        # the generator creates an artifact in the masks (the rotated bounding box, for some reason, presents
        # intermitent points, instead of being all zeros). Here we compute connected components and remove the really
        # small ones as noise
        mask_aux = mask_augmented[i, :, :, :].reshape(mask_augmented.shape[1:3]).astype(np.uint8)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_aux)
        lblareas = stats[:, cv2.CC_STAT_AREA]

        # labels of small components, that we assume are edge artifacts
        labels_artifact = np.where(lblareas < 50)[0]
        labels_artifact = list(labels_artifact)

        # clear the artifact objects from the image
        mask_aux[np.isin(labels, labels_artifact)] = 0

        # transfer result to mask array
        mask_augmented[i, :, :, :] = mask_aux.reshape(mask_aux.shape + (1,))

        # we don't need to clear the border artifact from seg, because the mask will make it ignored anyway

        # keep only contours in the segmentation array
        seg_augmented[i, :, :, 0] *= mask_augmented[i, :, :, 0].astype('uint8')

        if DEBUG:
            plt.clf()
            plt.subplot(321)
            plt.imshow(im_augmented[i, :, :, :])
            plt.subplot(322)
            plt.imshow(dmap_augmented[i, :, :, 0])
            plt.subplot(323)
            plt.imshow(mask_augmented[i, :, :, 0])
            plt.subplot(324)
            a = im_augmented[i, :, :, :]
            b = mask_augmented[i, :, :, 0]
            plt.imshow(pystoim.imfuse(a, b))
            plt.subplot(325)
            plt.imshow(seg_augmented[i, :, :, 0])
            plt.subplot(326)
            plt.imshow(pystoim.imfuse(mask_augmented[i, :, :, 0],
                                      seg_augmented[i, :, :, 0]))

        # create filenames based on the original foo.tif, so that we have im_seed_001_foo.tif, dmap_seed_001_foo.tif,
        # mask_seed_001_foo.tif, where seed_001 means data augmented using seed=1
        base_file = im_file_list[i]
        base_path, base_name = os.path.split(base_file)
        im_file = os.path.join(training_augmented_dir, 'im_seed_' + str(seed).zfill(3) + '_' + base_name)
        dmap_file = os.path.join(training_augmented_dir, 'dmap_seed_' + str(seed).zfill(3) + '_' + base_name)
        mask_file = os.path.join(training_augmented_dir, 'mask_seed_' + str(seed).zfill(3) + '_' + base_name)
        seg_file = os.path.join(training_augmented_dir, 'seg_seed_' + str(seed).zfill(3) + '_' + base_name)

        # save transformed image
        im_out = im_augmented[i, :, :, :]
        im_out *= 255
        im_out = im_out.astype(np.uint8)
        im_out = Image.fromarray(im_out, mode='RGB')
        im_out.save(im_file)

        # save distance transforms (note: we have to save as float mode)
        im_out = dmap_augmented[i, :, :, 0]
        im_out = Image.fromarray(im_out, mode='F')
        im_out.save(dmap_file)

        # save mask (note: we can save as black and white 1 byte)
        im_out = mask_augmented[i, :, :, 0]
        im_out = im_out.astype(np.uint8)
        im_out = Image.fromarray(im_out, mode='L')
        im_out.save(mask_file)

        # save labels (note: we can save as uint32)
        im_out = seg_augmented[i, :, :, 0]
        im_out = im_out.astype(np.uint32)
        im_out = Image.fromarray(im_out, mode='I')
        im_out.save(seg_file)
