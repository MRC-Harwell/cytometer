"""
cytometer/data.py

Functions to load, save and pre-process data related to the cytometer project.
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import pandas as pd
import ast
from mahotas import bwperim

DEBUG = False


def load_im_file_list_to_array(file_list):
    """
    Loads a list of images, all with the same size, into a numpy array (file, row, col, channel)
    :param file_list:
    :return:
    """
    if not isinstance(file_list, list):
        raise ValueError('file_list must be a list')
    if len(file_list) == 0:
        return np.empty((1, 0))

    # load first image to get the image size
    im0 = np.array(Image.open(file_list[0]))

    # allocate memory for the output
    im_out = np.zeros(shape=(len(file_list),) + im0.shape, dtype=im0.dtype)

    # read files and copy them to output array
    for i, file in enumerate(file_list):
        im_out[i, ...] = np.array(Image.open(file))

        if DEBUG:
            plt.clf()
            plt.imshow(im_out[i, ...])

    # one channel data gets loaded as im_out.shape=(n, rows, cols), but keras requires (n, rows, cols, 1)
    if im_out.ndim == 3:
        im_out = im_out.reshape(im_out.shape + (1,))

    return im_out


def load_watershed_seg_and_compute_dmap(seg_file_list, background_label=1):
    """
    Loads a list of segmentation files and computes distance maps to the objects boundaries/background.

    The segmentation file is assumed to have one integer label (2, 3, 4, ...) per object. The background has label 1.
    The boundaries between objects have label 0 (if boundaries exist).

    Those boundaries==0 are important for this function, because distance maps are computed as the distance of any
    pixel to the closest pixel=0.

    As an example, the watershed method will produce boundaries==0 when expanding segmentation labels.
    :param seg_file_list: list of paths to the files with segmentations, in any format that PIL can load
    :param background_label: label that will be considered to correspond to the background and not an object
    :return: (dmap, mask, seg)
        dmap: np.array with one distance map per segmentation file. It provides the Euclidean distance of each pixel
              to the closest background/boundary pixel.
        mask: np.array with one segmentation mask per file. Background pixels = 0. Foreground/boundary pixels = 1.
        seg: np.array with one segmentation per file. The segmentation labels in the input files.
    """

    if not isinstance(seg_file_list, list):
        raise ValueError('seg_file_list must be a list')
    if len(seg_file_list) == 0:
        return np.empty((1, 0)), np.empty((1, 0)), np.empty((1, 0))

    # get size of first segmentation. All segmentations must have the same size
    seg0 = Image.open(seg_file_list[0])
    seg0_dtype = np.array(seg0).dtype

    # allocate memory for outputs
    seg = np.zeros(shape=(len(seg_file_list), seg0.height, seg0.width), dtype=seg0_dtype)
    mask = np.zeros(shape=(len(seg_file_list), seg0.height, seg0.width), dtype='float32')  # these will be used as weights
    dmap = np.zeros(shape=(len(seg_file_list), seg0.height, seg0.width), dtype='float32')

    # loop segmented files
    for i, seg_file in enumerate(seg_file_list):

        # load segmentation
        seg_aux = np.array(Image.open(seg_file))

        # watershed only labels contours between pairs of cells, not where the cell touches the background.
        # Thus, we compute the missing part of the contours between cells and background
        im_background = seg_aux == 1  # background points in the labels
        im_background = bwperim(im_background, n=8, mode='ignore')  # perimeter of background areas
        im_background[0:2, :] = 0  # remove perimeter artifact on the borders of the image
        im_background[-2:, :] = 0
        im_background[:, 0:2] = 0
        im_background[:, -2:] = 0

        seg_aux[im_background.astype(np.bool)] = 0  # add background contours to segmentation

        # copy segmentation slice to output
        seg[i, :, :] = seg_aux

        # the mask is 0 where we have background pixels, and 1 everywhere else (foreground)
        mask[i, :, :] = seg_aux != background_label

        # add the background pixels to the boundaries
        seg_aux[seg_aux == background_label] = 0

        # plot image
        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(seg_aux, cmap="gray")
            plt.title('Cell labels')
            plt.subplot(222)
            plt.imshow(mask[i, :, :], cmap="gray")
            plt.title('Training weight')

        # compute distance map from very pixel to the closest boundary
        dmap[i, :, :] = ndimage.distance_transform_edt(seg_aux)

        # plot distance map
        if DEBUG:
            plt.subplot(223)
            plt.imshow(dmap[i, :, :])
            plt.title('Distance map')

    # add a dummy channel dimension to comply with Keras format
    mask = mask.reshape((mask.shape + (1,)))
    seg = seg.reshape((seg.shape + (1,)))
    dmap = dmap.reshape((dmap.shape + (1,)))

    return dmap, mask, seg


def read_keras_training_output(filename):
    """
    Read a text file with the keras output of one or multiple trainings. The output from
    each training is expected to look like this:

    <FILE>
    Epoch 1/10

       1/1846 [..............................] - ETA: 1:33:38 - loss: 1611.5088 - mean_squared_error: 933.6124 - mean_absolute_error: 17.6664
       2/1846 [..............................] - ETA: 53:17 - loss: 1128.0714 - mean_squared_error: 593.7890 - mean_absolute_error: 13.8204
    ...
       1846/1846 [==============================] - 734s 398ms/step - loss: 273.1249 - mean_squared_error: 385.5759 - mean_absolute_error: 14.7488 - val_loss: 544.2305 - val_mean_squared_error: 285.2719 - val_mean_absolute_error: 11.1605
    Epoch 2/10

       1/1846 [..............................] - ETA: 11:22 - loss: 638.7009 - mean_squared_error: 241.0196 - mean_absolute_error: 10.6583
    ...
       1846/1846 [==============================] - 734s 398ms/step - loss: 273.1249 - mean_squared_error: 385.5759 - mean_absolute_error: 14.7488 - val_loss: 544.2305 - val_mean_squared_error: 285.2719 - val_mean_absolute_error: 11.1605
    Epoch 2/10

       1/1846 [..............................] - ETA: 11:22 - loss: 638.7009 - mean_squared_error: 241.0196 - mean_absolute_error: 10.6583
    </FILE>

    Any lines until the first "Epoch" are ignored. Then, the rest of the training output is put into a pandas.DataFrame
    like this:

               epoch   ETA       loss  mean_absolute_error  mean_squared_error
    0              1  5618  1611.5088              17.6664            933.6124
    1              1  3197  1128.0714              13.8204            593.7890
    ...
    18448         10     0    88.1862              13.6856            453.2228
    18449         10     0    88.2152              13.6862            453.2333

    [18450 rows x 5 columns]

    :param filename: string with the path and filename of a text file with the training output from keras
    :return: list of pandas.DataFrame
    """

    # ignore all lines until we get to the first "Epoch"
    df_all = []
    file = open(filename, 'r')
    for line in file:
        if line[0:5] == 'Epoch':
            data = []
            epoch = 1
            break

    # loop until the end of the file
    while True:

        if ('Epoch 1/' in line) or not line:  # start of new training or end of file

            if len(data) > 0:
                # convert list of dictionaries to dataframe
                df = pd.DataFrame(data)

                # reorder columns so that epochs go first
                df = df[(df.columns[df.columns == 'epoch']).append(df.columns[df.columns != 'epoch'])]

                # add dataframe to output list of dataframes
                df_all.append(df)

                if not line:
                    # if we are at the end of the file, we stop reading
                    break
                else:
                    # reset the variables for the next training
                    data = []
                    epoch = 1

        elif 'Epoch' in line:  # new epoch of current training

            epoch += 1

        elif 'ETA:' in line:

            # remove whitespaces: '1/1846[..............................]-ETA:1:33:38-loss:1611.5088-mean_squared_error:933.6124-mean_absolute_error:17.6664'
            line = line.strip()
            line = line.replace(' ', '')

            # split line into elements: ['1/1846[..............................]', 'ETA:1:33:38', 'loss:1611.5088', 'mean_squared_error:933.6124', 'mean_absolute_error:17.6664']
            line = line.split('-')

            # remove first element: ['ETA:1:33:38', 'loss:1611.5088', 'mean_squared_error:933.6124', 'mean_absolute_error:17.6664']
            line = line[1:]

            # add "" around the first :, and at the beginning and end of the element
            # ['"ETA":"1:33:38"', '"loss":"1611.5088"', '"mean_squared_error":"933.6124"', '"mean_absolute_error":"17.6664"']
            line = [x.replace(':', '":"', 1) for x in line]
            line = ['"' + x + '"' for x in line]

            # add epoch to collated elements and surround by {}
            # '{"ETA":"1:33:38", "loss":"1611.5088", "mean_squared_error":"933.6124", "mean_absolute_error":"17.6664"}'
            line = '{"epoch":' + str(epoch) + ', ' + ', '.join(line) + '}'

            # convert string to dict
            # {'ETA': '1:33:38', 'loss': '1611.5088', 'mean_squared_error': '933.6124', 'mean_absolute_error': '17.6664'}
            line = ast.literal_eval(line)

            # convert string values to numeric values
            for key in line.keys():
                if key == 'ETA':
                    eta_str = line['ETA'].split(':')
                    if len(eta_str) == 1:  # ETA: 45s
                        eta = int(eta_str[-1].replace('s', ''))
                    else:
                        eta = int(eta_str[-1])
                    if len(eta_str) > 1:  # ETA: 1:08
                        eta += 60 * int(eta_str[-2])
                    if len(eta_str) > 2:  # ETA: 1:1:08
                        eta += 3600 * int(eta_str[-3])
                    if len(eta_str) > 3:
                        raise ValueError('ETA format not implemented')
                    line['ETA'] = eta
                elif key == 'epoch':
                    pass
                else:
                    line[key] = float(line[key])

            # add the dictionary to the output list
            data.append(line)

        # read the next line
        line = file.readline()

    # we have finished reading the log file

    # close the file
    file.close()

    return df_all
