#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 18:32:40 2017

@author: rcasero
"""

# check python interpreter version
import sys
if sys.version_info < (3,0,0):
    raise ImportError('Python 3 required')

# configure Keras, to avoid using file ~/.keras/keras.json
import os
import keras
from importlib import reload
os.environ['KERAS_BACKEND'] = 'tensorflow'
reload(keras.backend)
keras.backend.set_image_dim_ordering('tf')

# load module dependencies
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import keras.backend as K

# constants
data_dir = "/home/rcasero/Software/cytometer/data"
im_file = "KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38_lowres.tif"

# load image
# pillow:  4.1.1
im = Image.open(os.path.join(data_dir, im_file))
imp = np.array(im)

plt.imshow(im)

import importlib
importlib.reload(deepcell_models)
import deepcell_models






import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plot 4 images as gray scale: http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
elif K.image_data_format() == 'channels_last':
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
else:
    raise AssertionError("Unknown data axes ordering in Keras")

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
print('Test score:', history.history['val_loss'][0])
print('Test accuracy:', history.history['val_acc'])

# if model has been precomputed, this is the command to load it
#model = load_model('mnist_cnn_model.h5')

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

# manually save trained model so that we can later quickly load it
# model.save('mnist_cnn_model.h5')

# select one of the test images
idx = 9
X = X_test[idx]
X = X.reshape(1, 28, 28, 1)

# show image
plt.imshow(X.reshape(28, 28), cmap=plt.get_cmap('gray'))

# weights of the 1st convo layer
weights = model.layers[0].get_weights()[0]

# plot the 32 weights
for I in range(0, 32):
    plt.subplot(4, 8, I+1)
    plt.imshow(weights[:,:,0,I], cmap=plt.get_cmap('gray'), interpolation='none')

# show the plot
plt.show()

# output of the 1st convo layer = model.layers[1].output (https://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer)
get_1st_convo_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[1].output])
layer_output = get_1st_convo_layer_output([X, 1])[0]

# plot the 32 features
for I in range(0, 32):
    plt.subplot(4, 8, I+1)
    plt.imshow(layer_output[0, :, :, I], cmap=plt.get_cmap('gray'))

# show the plot
plt.show()

# output of the 1st max-pooling layer = model.layers[1].output (https://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer)
get_1st_maxpool_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[5].output])
layer_output = get_1st_maxpool_layer_output([X, 1])[0]

# plot the 32 features
for I in range(0, 32):
    plt.subplot(4, 8, I+1)
    plt.imshow(layer_output[0, :, :, I], cmap=plt.get_cmap('gray'))
