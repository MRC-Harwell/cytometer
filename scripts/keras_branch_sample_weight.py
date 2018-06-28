import os
import keras
import keras.backend as K
import numpy as np

from keras.models import Sequential
from keras.layers import Activation, Conv2D
from keras.layers.normalization import BatchNormalization

# environment variables
os.environ['KERAS_BACKEND'] = 'tensorflow'

# remove warning "Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Just disables the warning, doesn't enable AVX/FMA

# simulate input images
im = np.zeros(shape=(10, 128, 128, 3), dtype='uint8')

# simulate network output
out = np.zeros(shape=(10, 128, 128, 1), dtype='float32')

# simulate training weights for network output
weight = np.zeros(shape=(10, 128, 128, 1), dtype='float32')

# create network model
model = Sequential()
model.add(Conv2D(input_shape=(128, 128, 3),
                 filters=32, kernel_size=(3, 3), strides=1, padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same'))

optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])

model.fit(im, out, sample_weight=weight, batch_size=3, epochs=3)
