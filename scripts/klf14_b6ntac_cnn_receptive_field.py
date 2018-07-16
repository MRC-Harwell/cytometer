import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
import cytometer.models as models
from receptivefield.keras import KerasReceptiveField
import matplotlib.pyplot as plt




# estimate receptive field of the model
def model_build_func(input_shape):
    return models.fcn_9_conv_8_bnorm_3_maxpool_binary_classifier(input_shape=input_shape,
                                                                 for_receptive_field=True,
                                                                 dilation_rate=2)

rf = KerasReceptiveField(model_build_func, init_weights=True)

rf_params = rf.compute(
    input_shape=(200, 200, 3),
    input_layer='input_image',
    output_layer='main_output'
)
print(rf_params)

plt.clf()
plt.imshow(im_split[i, :, :, :])
rf.plot_rf_grid(custom_image=im_split[i, :, :, :], figsize=(6, 6))
plt.show()

rf.plot_rf_grid(get_default_image(shape, name='doge'))
