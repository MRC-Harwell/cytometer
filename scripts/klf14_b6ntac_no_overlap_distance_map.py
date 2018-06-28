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

''' Keras convolutional neural network
========================================================================================================================
'''

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

# TODO: crop images
im = im[:, 0:256, 0:256, :]
seg = seg[:, 0:256, 0:256, :]
mask = mask[:, 0:256, 0:256, :]

if K.image_data_format() == 'channels_first':
    #    default_input_shape = (3, None, None)
    default_input_shape = (im.shape[3], im.shape[1], im.shape[2])
elif K.image_data_format() == 'channels_last':
    #    default_input_shape = (None, None, 3)
    default_input_shape = im.shape[1:4]


# Based on DeepCell's sparse_feature_net_61x61, but here we use no dilation
def regression_9c3mp(input_shape=default_input_shape, reg=0.001, init='he_normal'):

    if K.image_data_format() == 'channels_first':
        norm_axis = 1
    elif K.image_data_format() == 'channels_last':
        norm_axis = 3

    model = Sequential()

    model.add(Conv2D(input_shape=input_shape,
                     filters=32, kernel_size=(3, 3), strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=norm_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=norm_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='same'))

    # model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1,
    #                  kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    # model.add(BatchNormalization(axis=norm_axis))
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1,
    #                  kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    # model.add(BatchNormalization(axis=norm_axis))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(4, 4), strides=1, padding='same'))
    #
    # model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1,
    #                  kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    # model.add(BatchNormalization(axis=norm_axis))
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1,
    #                  kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    # model.add(BatchNormalization(axis=norm_axis))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(8, 8), strides=1, padding='same'))

    model.add(Conv2D(filters=10, kernel_size=(4, 4), strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=norm_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=10, kernel_size=(1, 1), strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=norm_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=1, kernel_size=(1, 1), strides=1,
                     kernel_initializer=init, padding='same', kernel_regularizer=l2(reg)))

    return model


model = regression_9c3mp()
optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])

#run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
#model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'], options=run_opts)


## DEBUG: model visualisation
#from keras.utils import plot_model
#plot_model(model, to_file='/tmp/model.png', show_shapes=True)

model.fit(im, seg, sample_weight=mask, batch_size=32, epochs=3)


## TODO: Once we have things working with plain np.ndarrays, we'll look into data augmentation, flows, etc. (below)

# data augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=90,     # randomly rotate images up to 90 degrees
    fill_mode="reflect",   # how to fill points outside boundaries
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)    # randomly flip images

# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     horizontal_flip=True,
                     vertical_flip=True,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
mask_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(im, augment=True, seed=seed)
mask_datagen.fit(mask, augment=True, seed=seed)

if DEBUG:
    # save the images generated by image augmentation
    image_generator = image_datagen.flow(im, y=seg, seed=seed, batch_size=batch_size,
                                         save_to_dir='/tmp/preview', save_prefix='im_', save_format='jpeg')
    mask_generator = mask_datagen.flow(mask, y=None, seed=seed, batch_size=batch_size,
                                       save_to_dir='/tmp/preview', save_prefix='mask_', save_format='jpeg')
else:
    image_generator = image_datagen.flow(im, y=seg, seed=seed, batch_size=batch_size)
    mask_generator = mask_datagen.flow(mask, y=None, seed=seed, batch_size=batch_size)


# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

# train model
model.fit_generator(
    train_generator,
    steps_per_epoch=10,
    epochs=n_epoch)

val = np.array([[1, 2], [3, 4]])
kvar = K.variable(value=val, dtype='float64', name='example_var')
K.eval(kvar * kvar)



"""
image_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
"""

## https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
## https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/

# set seed of random number generator so that we can reproduce results
seed = 0
np.random.seed(seed)

# fit the model on the batches generated by datagen.flow()
loss_history = model.fit_generator(train_generator,
                                   steps_per_epoch=1,
                                   epochs=n_epoch)
