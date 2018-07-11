import os
import glob
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import pysto.imgproc as pystoim

# use CPU for testing on laptop
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
from keras.models import Model
from keras.layers import Activation, Conv2D, Input, MaxPooling2D
from keras.layers.normalization import BatchNormalization

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))



DEBUG = False

root_data_dir = '/home/rcasero/Dropbox/klf14'
training_dir = '/home/rcasero/Dropbox/klf14/klf14_b6ntac_training'
training_nooverlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')

'''Load data
'''

K.set_image_data_format('channels_last')
norm_axis = 3

labels_file_list = glob.glob(os.path.join(training_nooverlap_data_dir, '*.tif'))

# allocate arrays for data
im = np.zeros((len(labels_file_list), 1001, 1001, 3), dtype='uint8')
labels = np.zeros((len(labels_file_list), 1001, 1001, 1), dtype='uint8')
mask = np.zeros((len(labels_file_list), 1001, 1001, 1), dtype='uint8')

# labels
LABEL_BACKGROUND = 1

# loop loading the segmentation labels, and convert to a mask of cells vs. no cells
for i, labels_file in enumerate(labels_file_list):

    # corresponding image file
    im_file = labels_file.replace(training_nooverlap_data_dir, training_dir)

    # load image
    im[i, :, :, :] = tifffile.imread(im_file)

    # load labels
    labels[i, :, :, 0] = tifffile.imread(labels_file)[0, :, :]

    # convert to mask of cells vs. no cells
    mask[i, :, :, :] = labels[i, :, :, :] != LABEL_BACKGROUND

    # plot labels
    if DEBUG:
        plt.clf()
        plt.subplot(221)
        plt.imshow(im[i, :, :, :])
        plt.subplot(222)
        plt.imshow(labels[i, :, :, 0])
        plt.subplot(223)
        plt.imshow(mask[i, :, :, 0])


'''CNN
'''

# declare network model
input = Input(shape=(1001, 1001, 3), dtype='float32')
x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(input)
x = BatchNormalization(axis=norm_axis)(x)
x = Activation('relu')(x)

x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x)
x = BatchNormalization(axis=norm_axis)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(x)

x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x)
x = BatchNormalization(axis=norm_axis)(x)
x = Activation('relu')(x)

x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x)
x = BatchNormalization(axis=norm_axis)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(4, 4), strides=1, padding='same')(x)

x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x)
x = BatchNormalization(axis=norm_axis)(x)
x = Activation('relu')(x)

x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x)
x = BatchNormalization(axis=norm_axis)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(8, 8), strides=1, padding='same')(x)

x = Conv2D(filters=200, kernel_size=(4, 4), strides=1, padding='same')(x)
x = BatchNormalization(axis=norm_axis)(x)
x = Activation('relu')(x)

x = Conv2D(filters=200, kernel_size=(1, 1), strides=1, padding='same')(x)
x = BatchNormalization(axis=norm_axis)(x)
x = Activation('relu')(x)

x = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same')(x)
x = Activation('softmax')(x)

main_output = Activation('softmax', name='main_output')(x)
model = Model(inputs=input, outputs=main_output)

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
model.fit(im, mask, batch_size=6, epochs=50, validation_split=.1)

model.save('/home/rcasero/Downloads/foo.model')

# visualise results
if DEBUG:
    for i in range(len(labels_file_list)):

        # run image through network
        mask_pred = model.predict(im[i, :, :, :].reshape((1,) + im.shape[1:]))

        plt.clf()
        plt.subplot(221)
        plt.imshow(im[i, :, :, :].transpose((1, 2, 0)))
        plt.subplot(222)
        plt.imshow(mask[i, :, :, :].reshape((1001, 1001)))
        plt.subplot(223)
        plt.imshow(mask_pred.reshape((1001, 1001)) * 255)
        plt.subplot(224)
        a = mask[i, :, :, :].reshape((1001, 1001))
        b = mask_pred.reshape((1001, 1001)) * 255
        plt.imshow(pystoim.imfuse(a, b))

plt.clf()
plt.hist(b.flatten())
