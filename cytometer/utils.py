import math
import cytometer.models as models

def propagate_receptive_field(layer, receptiveField):
    n_in = receptiveField[0]
    j_in = receptiveField[1]
    r_in = receptiveField[2]
    start_in = receptiveField[3]
    if 'conv2d' in layer.name:
        k = layer.kernel.shape[0].value  # kernel size (first dimension)
    elif 'max_pooling2d' in layer.name:
        k = layer.pool_size[0]           # kernel size (first dimension)
    s = layer.strides[0]                 # stride (first dimension)
    if layer.padding == 'same':
        p = math.floor((k - 1) / 2.0)    # unilateral padding (first dimension)
    elif layer.padding == 'valid':
        p = 0                            # zero padding

    n_out = math.floor((n_in - k + 2 * p) / s) + 1
    actualP = (n_out - 1) * s - n_in + k
    pR = math.ceil(actualP / 2)
    pL = math.floor(actualP / 2)

    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in
    start_out = start_in + ((k - 1) / 2 - pL) * j_in
    return n_out, j_out, r_out, start_out


if __name__ == '__main__':

    # first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and start_0 = 0.5
    imsize = 1000
    j_0 = 1
    r_0 = 1
    start_0 = 0.5
    receptiveField = [imsize, j_0, r_0, start_0]
    print('input_image' + ': ' + 'n = ' + str(receptiveField[0]) + ', j = ' + str(receptiveField[1])
          + ', r = ' + str(receptiveField[2]) + ', start = ' + str(receptiveField[3]))

    # loop layers
    for layer in model.layers:

        if 'conv2d' in layer.name or 'max_pooling2d' in layer.name:

            receptiveField = propagate_receptive_field(layer, receptiveField)
            print(layer.name + ': ' + 'n = ' + str(receptiveField[0]) + ', j = ' + str(receptiveField[1])
                  + ', r = ' + str(receptiveField[2]) + ', start = ' + str(receptiveField[3]) )

        else:

            print(layer.name + ': N/A')

