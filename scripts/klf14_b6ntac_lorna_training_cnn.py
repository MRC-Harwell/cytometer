#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:40:00 2018

@author: Ramon Casero <rcasero@gmail.com>
"""

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

DEBUG = False

import os

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

# imports
import glob

import keras.backend as K
import tensorflow as tf
import keras.preprocessing.image
K.set_image_dim_ordering('tf')
print(K.image_data_format())

import numpy as np
from PIL import Image
import openslide
import matplotlib.pyplot as plt
import cytometer.models as models
import pysto.imgproc as pystoim

# configure Keras, to avoid using file ~/.keras/keras.json
K.set_floatx('float32')
K.set_epsilon(1e-07)
# fix "RuntimeError: Invalid DISPLAY variable" in cluster runs
# import matplotlib
# matplotlib.use('agg')

# limit the amount of GPU memory that Keras can allocate
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=config)
K.set_session(session)

# DEBUG: used while developing the software, not for production
# import importlib
# importlib.reload(models)

"""
Data
"""

# create (5, 250, 250, 3) containing Lorna's 5 RGB training images, decimated from
# size (500, 500) to (256, 256)
def load_list_of_files(file_list, antialias_flag):
    nfiles = len(file_list)
    file_list.sort()
    im = Image.open(file_list[0])
    im = im.resize((256, 256), antialias_flag)
    im = np.array(im)
    data = np.zeros((nfiles,) + im.shape, dtype=im.dtype)
    data[0, ] = im
    for i, filename in enumerate(file_list[1:]):
        im = Image.open(filename)
        im = im.resize((256, 256), antialias_flag)
        data[i+1, ] = im
    return data


# load Lorna's hand segmented data
data_dir = os.path.join('/home/rcasero/Software/cytometer/data', 'adipocyte_500x500_patches')
data_im = load_list_of_files(glob.glob(os.path.join(data_dir, '*_rgb.tif')), Image.ANTIALIAS)
data_seg = load_list_of_files(glob.glob(os.path.join(data_dir, '*_seg.tif')), Image.NEAREST)

# display the training data
if DEBUG:
    plt.ion()
    for i in range(data_im.shape[0]):
        plt.subplot(1, 2, 1)
        plt.imshow(data_im[i, ])
        plt.subplot(1, 2, 2)
        plt.imshow(data_seg[i, ])
        plt.draw_all()
        print('i = ' + str(i))
        plt.pause(0.1)
        input('Press key to continue')

# convert hand segmentation from uint8 to categorical binary data
data_seg_cat = keras.utils.to_categorical(data_seg)

# split image data to avoid GPU memory errors
data_im_split = np.zeros((20, 125, 125, 3), dtype=data_im.dtype)
data_seg_cat_split = np.zeros((20, 125, 125, 4), dtype=data_seg.dtype)
j = 0
for i in range(data_im.shape[0]):
    for row_start, row_end in zip([0, 125], [125, 250]):
        for col_start, col_end in zip([0, 125], [125, 250]):
            data_im_split[j, :, :, :] = data_im[i, row_start:row_end, col_start:col_end, :]
            data_seg_cat_split[j, :, :, :] = data_seg_cat[i, row_start:row_end, col_start:col_end, :]
            j += 1

# display the training data
if DEBUG:
    plt.ion()
    for i in range(data_im_split.shape[0]):
        plt.subplot(1, 2, 1)
        plt.imshow(data_im_split[i, ])
        plt.subplot(1, 2, 2)
        plt.imshow(data_seg_cat_split[i, :, :, 0])
        plt.draw_all()
        print('i = ' + str(i))
        plt.pause(0.1)
        input('Press key to continue')



"""
Keras model
"""

# parameters
batch_size = 10
n_epoch = 20

# rate scheduler from DeepCell
def rate_scheduler(lr = .001, decay = 0.95):
    def output_fn(epoch):
        epoch = np.int(epoch)
        new_lr = lr * (decay ** epoch)
        return new_lr
    return output_fn


model = models.basic_9c3mp()
optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# DEBUG: model visualisation
from keras.utils import plot_model
plot_model(model, to_file='/tmp/model.png', show_shapes=True)

# data augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=90,     # randomly rotate images up to 90 degrees
    fill_mode="reflect",   # how to fill points outside boundaries
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)    # randomly flip images

if DEBUG:
    # save the images generated by image augmentation
    train_generator = train_datagen.flow(data_im_split, data_seg_cat_split, batch_size=batch_size,
                                         save_to_dir='/tmp/preview', save_prefix='foo', save_format='jpeg')
else:
    train_generator = train_datagen.flow(data_im_split, data_seg_cat_split, batch_size=batch_size)

## https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
## https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/

# set seed of random number generator so that we can reproduce results
seed = 0
np.random.seed(seed)

# fit the model on the batches generated by datagen.flow()
loss_history = model.fit_generator(train_generator,
                                   steps_per_epoch=1,
                                   epochs=n_epoch)

# save trained model
model.save('./foo.h5')

# plot loss history
plt.ion()
plt.plot(loss_history.epoch, loss_history.history['acc'])
plt.xlabel('epoch')
plt.ylabel('acc')
plt.draw()
plt.pause(0.01)

##################################################################################################

# full image from Roger's data
#klf14_data_dir = "/home/rcasero/data/roger_data"
#klf14_data_file = os.path.join(klf14_data_dir, "KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38.ndpi")

klf14_data_dir = "/home/rcasero/Dropbox/klf14"
klf14_data_file = os.path.join(klf14_data_dir, "KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38.ndpi")

data_test = openslide.OpenSlide(klf14_data_file)
#downsample_level = data_test.get_best_level_for_downsample(4)
downsample_level = 0

# crop a relatively small patch of the whole image
im_crop = data_test.read_region((10000, 20000), downsample_level, (2000, 2000))

plt.ion()
plt.clf()
plt.imshow(im_crop)
plt.draw()
plt.show()
plt.pause(0.1)

# remove alpha channel
im_crop = np.array(im_crop)
im_crop = im_crop[:, :, 0:3]

# split image into overlapping blocks
im_crop_slices, im_crop_blocks, xout = pystoim.block_split(im_crop, (10, 10, 1),
                                                           pad_width=((100, 100), (100, 100), (0, 0)),
                                                           mode='reflect', reflect_type='even')

# init memory and correct slices for the predictor output (input images are RGB, and predictions have 4 channels)
im_crop_blocks_predicted = im_crop_blocks.copy()
im_crop_slices_predicted = im_crop_slices.copy()
for i, sl in enumerate(im_crop_slices_predicted):
    sl[2] = slice(0, 4, 1)
    im_crop_slices_predicted[i] = sl

# apply trained model to the cropped image
for i, block in enumerate(im_crop_blocks):
    if DEBUG:
        print("Processing block " + str(i) + "/" + str(len(im_crop_blocks)))
    aux = model.predict(block.reshape((1,) + block.shape))
    im_crop_blocks_predicted[i] = aux.reshape(aux.shape[1:])

# reassemble the blocks
im_crop_predicted, foo = pystoim.block_stack(im_crop_blocks_predicted, im_crop_slices_predicted,
                                             pad_width=((100, 100), (100, 100), (0, 0)))

# assign a label to each pixel according to the label with the largest softmax value
im_crop_predicted = np.argmax(im_crop_predicted, axis=2)

# plot segmentation results
plt.clf()
plt.subplot(1, 2, 1)
plt.imshow(im_crop)
plt.subplot(1, 2, 2)
plt.imshow(im_crop_predicted)
plt.show()
plt.pause(.1)

