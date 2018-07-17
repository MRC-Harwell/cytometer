import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
import cytometer.models as models
from receptivefield.keras import KerasReceptiveField
import matplotlib.pyplot as plt
import tifffile
from tensorflow.python.framework.errors_impl import ResourceExhaustedError

training_dir = '/home/rcasero/Dropbox/klf14/klf14_b6ntac_training'


nblocks_vec = list(range(3, 6))
kernel_len_vec = list(range(2, 5))
dilation_rate_vec = list(range(1, 4))
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
                    input_shape=(400, 400, 1),
                    input_layer='input_image',
                    output_layer='main_output'
                )

                rf_size.append(rf_params[2].w)
            except ResourceExhaustedError:  # if we run out of memory
                rf_size.append(float('nan'))



output = {'nblocks': [], 'kernel_len': [], 'dilation_rate': [], 'rf_size': []}
i=0
for nblocks in nblocks_vec:
    for kernel_len in kernel_len_vec:
        for dilation_rate in dilation_rate_vec:
            output['nblocks'].append(nblocks)
            output['kernel_len'].append(kernel_len)
            output['dilation_rate'].append(dilation_rate)
            output['rf_size'].append(rf_size[i])
            i = i + 1


import numpy as np
np.where(np.array(output['rf_size']) == 400)





# load typical histology window
im_file = os.path.join(training_dir, 'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_031860_col_033476.tif')
im = tifffile.imread(im_file)

# plot receptive field on histology image
rf.plot_rf_grid(custom_image=im)
plt.show()

