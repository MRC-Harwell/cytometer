"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
import cytometer.models as models
from receptivefield.keras import KerasReceptiveField
import matplotlib.pyplot as plt
import tifffile
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
import numpy as np

training_dir = '/home/rcasero/Dropbox/klf14/klf14_b6ntac_training'
saved_models = '/home/rcasero/Dropbox/klf14/saved_models'

'''DeepCell-based network
'''

nblocks_vec = list(range(2, 6))
kernel_len_vec = list(range(2, 5))
dilation_rate_vec = list(range(1, 3))
rf_size = []
output = {'nblocks': [], 'kernel_len': [], 'dilation_rate': [], 'rf_size': []}
for nblocks in nblocks_vec:
    for kernel_len in kernel_len_vec:
        for dilation_rate in dilation_rate_vec:

            # estimate receptive field of the model
            def model_build_func(input_shape):
                return models.fcn_conv_bnorm_maxpool_regression(input_shape=input_shape, for_receptive_field=True,
                                                                nblocks=nblocks, kernel_len=kernel_len,
                                                                dilation_rate=dilation_rate)

            rf = KerasReceptiveField(model_build_func, init_weights=True)

            try:
                rf_params = rf.compute(
                    input_shape=(1000, 1000, 1),
                    input_layer='input_image',
                    output_layer='main_output'
                )
                output['nblocks'].append(nblocks)
                output['kernel_len'].append(kernel_len)
                output['dilation_rate'].append(dilation_rate)
                output['rf_size'].append(rf_params[2].w)

            except ResourceExhaustedError:  # if we run out of memory
                output['nblocks'].append(nblocks)
                output['kernel_len'].append(kernel_len)
                output['dilation_rate'].append(dilation_rate)
                output['rf_size'].append(float('nan'))


# convert to arrays, so that we can index with a vector
output['nblocks'] = np.array(output['nblocks'])
output['kernel_len'] = np.array(output['kernel_len'])
output['dilation_rate'] = np.array(output['dilation_rate'])
output['rf_size'] = np.array(output['rf_size'])

# plot results
plt.clf()
for kernel_len in kernel_len_vec:
    for dilation_rate in dilation_rate_vec[:-1]:
        idx = np.logical_and(output['kernel_len'] == kernel_len, output['dilation_rate'] == dilation_rate)
        p = plt.plot(output['nblocks'][idx], output['rf_size'][idx])
        ymin = output['rf_size'][idx][0]
        xmin = 2.85
        plt.text(xmin, ymin, 'K=' + str(kernel_len) + ', D=' + str(dilation_rate), color=p[0].get_color())
        plt.plot(output['nblocks'][idx], output['rf_size'][idx], 'o', color=p[0].get_color())
        plt.xlabel('convolution blocks')
        plt.ylabel('receptive field size')
        plt.xlim((2.75, 5.15))



idx = np.where(np.array(output['rf_size']) == 400)[0]

np.array(idx)

np.array(output['nblocks'])[idx]
np.array(output['kernel_len'])[idx]
np.array(output['dilation_rate'])[idx]

nblocks = 4
kernel_len = 4
dilation_rate = 2

# load typical histology window
im_file = os.path.join(training_dir, 'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_031860_col_033476.tif')
im = tifffile.imread(im_file)

# plot receptive field on histology image
rf.plot_rf_grid(custom_image=im)
plt.show()

'''
Sherrah 2016 fully convolutional network'''

# estimate receptive field of the model
def model_build_func(input_shape):
    return models.fcn_sherrah2016_regression(input_shape=input_shape, for_receptive_field=True)


rf = KerasReceptiveField(model_build_func, init_weights=True)

rf_params = rf.compute(
    input_shape=(600, 600, 1),
    input_layer='input_image',
    output_layer='main_output')

print('Receptive field size: ' + str(rf_params[2].w))

# load typical histology window
im_file = os.path.join(training_dir, 'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_031860_col_033476.tif')
im = tifffile.imread(im_file)

# plot receptive field on histology image
rf.plot_rf_grid(custom_image=im)
plt.show()

# show model
model = models.fcn_sherrah2016_regression(input_shape=(600, 600, 3))
model.summary()

'''
Extended Sherrah 2016 fully convolutional network'''

# estimate receptive field of the model
def model_build_func(input_shape):
    return models.fcn_sherrah2016_modified(input_shape=input_shape, for_receptive_field=True)


rf = KerasReceptiveField(model_build_func, init_weights=True)

rf_params = rf.compute(
    input_shape=(333, 333, 3),
    input_layer='input_image',
    output_layers=['main_output'])

print('Receptive field size: ' + str(rf._rf_params[0].size))

# load typical histology window
im_file = os.path.join(training_dir, 'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_031860_col_033476.tif')
im = tifffile.imread(im_file)

# plot receptive field on histology image
rf.plot_rf_grids(custom_image=im, figsize=(6, 6))
plt.show()

# show model
model = models.fcn_sherrah2016_regression(input_shape=(600, 600, 3))
model.summary()

'''
Extended Sherrah 2016 fully convolutional network with trained weights'''

# estimate receptive field of the model
def model_build_func(input_shape):
    model = models.fcn_sherrah2016_modified(input_shape=input_shape, for_receptive_field=True)
    model.load_weights(os.path.join(saved_models, '2018-07-22T18_37_14.641079_fcn_sherrah2016_modified.h5'))
    return model


rf = KerasReceptiveField(model_build_func, init_weights=False)

rf_params = rf.compute(
    input_shape=(333, 333, 3),
    input_layer='input_image',
    output_layers=['main_output'])
print(rf_params)

print('Receptive field size: ' + str(rf._rf_params[0].size))

# load typical histology window
im_file = os.path.join(training_dir, 'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_031860_col_033476.tif')
im = tifffile.imread(im_file)

# plot receptive field on histology image
rf.plot_rf_grid(custom_image=im)
plt.show()

# show model
model = models.fcn_sherrah2016_regression(input_shape=(600, 600, 3))
model.summary()

