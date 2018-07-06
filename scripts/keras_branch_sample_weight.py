import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K
import numpy as np

from keras.models import Sequential
from keras.layers import Activation, Conv2D
from keras.layers.normalization import BatchNormalization

# remove warning "Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Just disables the warning, doesn't enable AVX/FMA

K.set_image_data_format('channels_first')

# simulate input images
im = np.zeros(shape=(10, 3, 64, 64), dtype='uint8')

# simulate network output
out = 2 * np.ones(shape=(10, 1, 64, 64), dtype='float32')

# simulate training weights for network output
weight = np.ones(shape=(10, 1, 64, 64), dtype='float32')

# simulate validation data
im_validation = 3 * np.ones(shape=(5, 3, 64, 64), dtype='uint8')
out_validation = 4 * np.ones(shape=(5, 1, 64, 64), dtype='float32')
validation_data = (im_validation, out_validation)

# create network model
model = Sequential()
model.add(Conv2D(input_shape=(3, 64, 64),
                 filters=32, kernel_size=(3, 3), strides=1, padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same'))

optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

'''string format for sample_weights_mode
'''

model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'], sample_weight_mode='element')

# train with validation_split
##model.fit(im, out, sample_weight=weight, validation_split=0.1, batch_size=3, epochs=3)

# train with validation_data
model.fit(im, out, sample_weight=weight, validation_data=validation_data, batch_size=3, epochs=3)

'''dictionary format for sample_weights_mode
'''

model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'], sample_weight_mode={'conv2d_2': 'element'})

# train with validation_data
model.fit(im, out, sample_weight=weight, validation_data=validation_data, batch_size=3, epochs=3)
