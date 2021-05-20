"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K
import numpy as np

from keras.models import Model, Sequential
from keras.layers import Activation, Conv2D, Input
from keras.layers.normalization import BatchNormalization

from keras.utils import multi_gpu_model

from keras.metrics import mae


# remove warning "Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Just disables the warning, doesn't enable AVX/FMA

# image_data_format = 'channels_first'
image_data_format = 'channels_last'

K.set_image_data_format(image_data_format)

'''Error when the score_array doesn't have dmin-1 compared to y_test
'''

im = np.zeros((10, 5, 5, 3))
out1 = np.ones((10, 5, 5, 1))
out2 = np.ones((10, 2, 2, 2))

input_tensor = Input(shape=im.shape[1:])
main_output = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same',
                     name='main_output')(input_tensor)
aux_output = Conv2D(filters=2, kernel_size=(1, 1), strides=3, padding='same',
                    name='aux_output')(input_tensor)
model = Model(inputs=input_tensor, outputs=[main_output, aux_output])


def dummy_loss(y_true, y_pred):
    return K.mean(mae(y_true, y_pred), axis=-1)


model.compile(loss=dummy_loss, optimizer='sgd',
              sample_weight_mode=['element', 'element'])

model.fit(im, [out1, out2],
          sample_weight=[out1[..., 0, 0], out2[..., 0, 0]],
          batch_size=3, epochs=2)  # fails here

##################################################################################

# simulate input images
if image_data_format == 'channels_first':
    im = np.zeros(shape=(10, 3, 64, 64), dtype='uint8')

    # simulate network output
    out = 2 * np.ones(shape=(10, 1, 64, 64), dtype='float32')
    aux_out = 5 * np.ones(shape=(10, 1, 22, 22), dtype='float32')
    # simulate training weights for network output
    weight = np.ones(shape=(10, 64, 64), dtype='float32')
    aux_weight = np.ones(shape=(10, 22, 22), dtype='float32')

    # simulate validation data
    im_validation = 3 * np.ones(shape=(5, 3, 64, 64), dtype='uint8')
    out_validation = 4 * np.ones(shape=(5, 1, 64, 64), dtype='float32')

elif image_data_format == 'channels_last':
    im = np.zeros(shape=(10, 64, 64, 3), dtype='uint8')

    # simulate network output
    out = 2 * np.ones(shape=(10, 64, 64, 1), dtype='float32')
    aux_out = 5 * np.ones(shape=(10, 22, 22, 1), dtype='float32')
    # simulate training weights for network output
    # weight = np.ones(shape=(10, 64, 64, 1), dtype='float32')
    weight = np.ones(shape=(10, 64, 64), dtype='float32')
    aux_weight = np.ones(shape=(10, 22, 22), dtype='float32')

    # simulate validation data
    im_validation = 3 * np.ones(shape=(5, 64, 64, 3), dtype='uint8')
    out_validation = 4 * np.ones(shape=(5, 64, 64, 1), dtype='float32')

else:
    raise ValueError('Unrecognised position for channels')

validation_data = (im_validation, out_validation)

# optimizer
optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

'''simple CNN with one output
'''

# create network model
model = Sequential()
model.add(Conv2D(input_shape=im.shape[1:],
                 filters=32, kernel_size=(3, 3), strides=1, padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same'))

'''string format (sample_weights_mode='element')
'''

# compile model
model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'], sample_weight_mode='element')

# train with validation_data
model.fit(im, out, sample_weight=weight, validation_data=validation_data, batch_size=3, epochs=3)

'''dictionary format (sample_weight_mode={'conv2d_2': 'element'})
'''

model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'], sample_weight_mode={'conv2d_2': 'element'})

# train with validation_data
model.fit(im, out, sample_weight=weight, validation_data=validation_data, batch_size=3, epochs=3)


'''parallel training in multiple GPUs
   string format (sample_weights_mode='element')
'''

# list all CPUs and GPUs
device_list = K.get_session().list_devices()

# number of GPUs
gpu_number = np.count_nonzero(['GPU' in str(x) for x in device_list])

if gpu_number > 1:  # compile and train model: Multiple GPUs

    # compile model
    parallel_model = multi_gpu_model(model, gpus=gpu_number)
    parallel_model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'], sample_weight_mode='element')

    # train model
    parallel_model.fit(im, out, sample_weight=weight, validation_data=validation_data, batch_size=gpu_number, epochs=3)

else:
    raise Warning('Cannot test parallel GPU processing, because I could not find 2 or more GPUs')

'''Multi-output CNN
'''

# create network model
input = Input(shape=im.shape[1:], dtype='float32')
x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same')(input)
x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)

main_output = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same', name='main_output')(x)
aux_output = Conv2D(filters=1, kernel_size=(1, 1), strides=3, padding='same', name='aux_output')(x)

model = Model(inputs=input, outputs=[main_output, aux_output])

'''list format (sample_weight_mode=['element', 'element'])
'''

model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'],
              sample_weight_mode=['element', 'element'])

model.fit(im, [out, aux_out], sample_weight=[weight, aux_weight],
          batch_size=3, epochs=3)

'''dictionary format (sample_weight_mode={'conv2d_2': 'element'})
'''

model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'],
              sample_weight_mode={'main_output': 'element', 'aux_output': 'element'})

model.fit(im, {'main_output': out, 'aux_output': aux_out},
          sample_weight={'main_output': weight, 'aux_output': aux_weight},
          batch_size=3, epochs=3)

'''Multi-output CNN with outputs of different number of features
'''

# create network model
input = Input(shape=im.shape[1:], dtype='float32')
x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same')(input)
x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)

main_output = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same', name='main_output')(x)
aux_output = Conv2D(filters=2, kernel_size=(1, 1), strides=3, padding='same', name='aux_output')(x)

model = Model(inputs=input, outputs=[main_output, aux_output])

'''list format (sample_weight_mode=['element', 'element'])
'''

model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'],
              sample_weight_mode=['element', 'element'])

model.fit(im, [out, np.repeat(aux_out, repeats=2, axis=3)],
          sample_weight=[weight, aux_weight],
          batch_size=3, epochs=3)

'''Error when the score_array doesn't have dmin-1 compared to y_test
'''

im = np.zeros((10, 5, 5, 3))
out1 = np.ones((10, 5, 5, 1))
out2 = np.ones((10, 2, 2, 2))

input_tensor = Input(shape=im.shape[1:])
main_output = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same',
                     name='main_output')(input_tensor)
aux_output = Conv2D(filters=2, kernel_size=(1, 1), strides=3, padding='same',
                    name='aux_output')(input_tensor)
model = Model(inputs=input_tensor, outputs=[main_output, aux_output])


def dummy_loss(y_true, y_pred):
    return K.mean(mae(y_true, y_pred), axis=-1)


model.compile(loss=dummy_loss, optimizer='sgd',
              sample_weight_mode=['element', 'element'])

model.fit(im, [out1, out2],
          sample_weight=[out1[..., 0, 0], out2[..., 0, 0]],
          batch_size=3, epochs=2)  # fails here
