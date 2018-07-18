import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
import cytometer.models as models
from receptivefield.keras import KerasReceptiveField
import matplotlib.pyplot as plt
import tifffile
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
import numpy as np

training_dir = '/home/rcasero/Dropbox/klf14/klf14_b6ntac_training'


nblocks_vec = list(range(3, 6))
kernel_len_vec = list(range(2, 5))
dilation_rate_vec = list(range(1, 3))
rf_size = []
output = dict('nblocks': [], 'kernel_len': [], 'dilation_rate': [], 'rf_size': [])
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
                    input_shape=(800, 800, 1),
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
    for dilation_rate in dilation_rate_vec:
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

