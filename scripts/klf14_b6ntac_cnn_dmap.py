import sys
sys.path.extend(['/users/rittscher/rcasero/Software/cytometer', '/users/rittscher/rcasero/Software/cytometer'])
import os
import glob
import datetime
import numpy as np
import pysto.imgproc as pystoim
import cytometer.data

# use CPU for testing on laptop
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
import cytometer.models as models
import matplotlib.pyplot as plt

# limit GPU memory used
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

# allow data parallelism
from keras.utils import multi_gpu_model

# cross-platform home directory
from pathlib import Path

DEBUG = False

home = str(Path.home())
root_data_dir = os.path.join(home, 'Dropbox/klf14')
training_dir = os.path.join(home, 'Dropbox/klf14/klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
saved_models_dir = os.path.join(home, 'Dropbox/klf14/saved_models')

'''Load data
'''

# list of segmented files
seg_file_list = glob.glob(os.path.join(training_non_overlap_data_dir, '*.tif'))

# list of corresponding image patches
im_file_list = [seg_file.replace(training_non_overlap_data_dir, training_dir) for seg_file in seg_file_list]

# load segmentations and compute distance maps
dmap, mask, seg = cytometer.data.load_watershed_seg_and_compute_dmap(seg_file_list)

# load corresponding images
im = cytometer.data.load_im_file_list_to_array(im_file_list)

# # remove a 1-pixel thick border so that images are 999x999 and we can split them into 3x3 tiles
# dmap = dmap[:, 1:-1, 1:-1, :]
# mask = mask[:, 1:-1, 1:-1, :]
# seg = seg[:, 1:-1, 1:-1, :]
# im = im[:, 1:-1, 1:-1, :]

# remove a 1-pixel so that images are 1000x1000 and we can split them into 2x2 tiles
dmap = dmap[:, 0:-1, 0:-1, :]
mask = mask[:, 0:-1, 0:-1, :]
seg = seg[:, 0:-1, 0:-1, :]
im = im[:, 0:-1, 0:-1, :]

# split images into smaller blocks to avoid GPU memory overflows in training
dmap_slices, dmap_blocks, _ = pystoim.block_split(dmap, nblocks=(1, 2, 2, 1))
im_slices, im_blocks, _ = pystoim.block_split(im, nblocks=(1, 2, 2, 1))
mask_slices, mask_blocks, _ = pystoim.block_split(mask, nblocks=(1, 2, 2, 1))

dmap_split = np.concatenate(dmap_blocks, axis=0)
im_split = np.concatenate(im_blocks, axis=0)
mask_split = np.concatenate(mask_blocks, axis=0)

# find images that have no valid pixels, to remove them from the dataset
idx_to_keep = np.sum(np.sum(np.sum(mask_split, axis=3), axis=2), axis=1)
idx_to_keep = idx_to_keep != 0

dmap_split = dmap_split[idx_to_keep, :, :, :]
im_split = im_split[idx_to_keep, :, :, :]
mask_split = mask_split[idx_to_keep, :, :, :]


'''CNN

Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
function
'''

# declare network model
with tf.device('/cpu:0'):
    model = models.fcn_sherrah2016_modified(input_shape=im_split.shape[1:])
#model.load_weights(os.path.join(saved_models_dir, 'foo.h5'))

# list all CPUs and GPUs
device_list = K.get_session().list_devices()

# number of GPUs
gpu_number = np.count_nonzero(['GPU' in str(x) for x in device_list])

if gpu_number > 1:  # compile and train model: Multiple GPUs

    # compile model
    parallel_model = multi_gpu_model(model, gpus=gpu_number)
    parallel_model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'], sample_weight_mode='element')

    # train model
    tic = datetime.datetime.now()
    parallel_model.fit(im_split, dmap_split, batch_size=1, epochs=10, validation_split=.1,
                       sample_weight=mask_split)
    toc = datetime.datetime.now()
    print('Training duration: ' + str(toc - tic))

else:  # compile and train model: One GPU

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # train model
    tic = datetime.datetime.now()
    model.fit(im_split, mask_split, batch_size=1, epochs=10, validation_split=.1)
    toc = datetime.datetime.now()
    print('Training duration: ' + str(toc - tic))

# save result (note, we save the template model, not the multiparallel object)
saved_model_filename = os.path.join(saved_models_dir, datetime.datetime.utcnow().isoformat() + '_fcn_sherrah2016_modified.h5')
saved_model_filename = saved_model_filename.replace(':', '_')
model.save(saved_model_filename)

# # visualise results
# if DEBUG:
#     for i in range(im_split.shape[0]):
#
#         # run image through network
#         dmap_pred = model.predict(im_split[i, :, :, :].reshape((1,) + im_split.shape[1:]))
#
#         plt.clf()
#         plt.subplot(221)
#         plt.imshow(im_split[i, :, :, :])
#         plt.subplot(222)
#         plt.imshow(dmap_split[i, :, :, :].reshape(dmap_split.shape[1:3]))
#         plt.subplot(223)
#         plt.imshow(dmap_pred.reshape(dmap_pred.shape[1:3]))
#         plt.subplot(224)
#         a = dmap_split[i, :, :, :].reshape(dmap_split.shape[1:3])
#         b = dmap_pred.reshape(dmap_split.shape[1:3])
#         imax = np.max((np.max(a), np.max(b)))
#         a /= imax
#         b /= imax
#         plt.imshow(pystoim.imfuse(a, b))


'''==================================================================================================================
OLD CODE
=====================================================================================================================
'''

# ## TODO: Once we have things working with plain np.ndarrays, we'll look into data augmentation, flows, etc. (below)
#
# # data augmentation
# train_datagen = keras.preprocessing.image.ImageDataGenerator(
#     rotation_range=90,     # randomly rotate images up to 90 degrees
#     fill_mode="reflect",   # how to fill points outside boundaries
#     horizontal_flip=True,  # randomly flip images
#     vertical_flip=True)    # randomly flip images
#
# # we create two instances with the same arguments
# data_gen_args = dict(featurewise_center=True,
#                      featurewise_std_normalization=True,
#                      rotation_range=90.,
#                      horizontal_flip=True,
#                      vertical_flip=True,
#                      width_shift_range=0.1,
#                      height_shift_range=0.1,
#                      zoom_range=0.2)
# image_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
# mask_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
#
# # Provide the same seed and keyword arguments to the fit and flow methods
# seed = 1
# image_datagen.fit(im, augment=True, seed=seed)
# mask_datagen.fit(mask, augment=True, seed=seed)
#
# if DEBUG:
#     # save the images generated by image augmentation
#     image_generator = image_datagen.flow(im, y=seg, seed=seed, batch_size=batch_size,
#                                          save_to_dir='/tmp/preview', save_prefix='im_', save_format='jpeg')
#     mask_generator = mask_datagen.flow(mask, y=None, seed=seed, batch_size=batch_size,
#                                        save_to_dir='/tmp/preview', save_prefix='mask_', save_format='jpeg')
# else:
#     image_generator = image_datagen.flow(im, y=seg, seed=seed, batch_size=batch_size)
#     mask_generator = mask_datagen.flow(mask, y=None, seed=seed, batch_size=batch_size)
#
#
# # fits the model on batches with real-time data augmentation:
# model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
#                     steps_per_epoch=len(x_train) / 32, epochs=epochs)
#
# # combine generators into one which yields image and masks
# train_generator = zip(image_generator, mask_generator)
#
# # train model
# model.fit_generator(
#     train_generator,
#     steps_per_epoch=10,
#     epochs=n_epoch)
#
# val = np.array([[1, 2], [3, 4]])
# kvar = K.variable(value=val, dtype='float64', name='example_var')
# K.eval(kvar * kvar)
#
#
#
# """
# image_generator = image_datagen.flow_from_directory(
#     'data/images',
#     class_mode=None,
#     seed=seed)
#
# mask_generator = mask_datagen.flow_from_directory(
#     'data/masks',
#     class_mode=None,
#     seed=seed)
#
# # combine generators into one which yields image and masks
# train_generator = zip(image_generator, mask_generator)
#
# model.fit_generator(
#     train_generator,
#     steps_per_epoch=2000,
#     epochs=50)
# """
#
# ## https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# ## https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
#
# # set seed of random number generator so that we can reproduce results
# seed = 0
# np.random.seed(seed)
#
# # fit the model on the batches generated by datagen.flow()
# loss_history = model.fit_generator(train_generator,
#                                    steps_per_epoch=1,
#                                    epochs=n_epoch)
