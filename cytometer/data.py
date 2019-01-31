"""
cytometer/data.py

Functions to load, save and pre-process data related to the cytometer project.
"""

import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import pandas as pd
import ast
from mahotas import bwperim
import pysto.imgproc as pystoim
import re
import six
from svgpathtools import svg2paths


DEBUG = False


def split_images(x, nblocks):
    """
    Splits the rows and columns of a data array with shape (n, rows, cols, channels) into blocks.

    If necessary, the array is trimmed off so that all blocks have the same size.

    :param x: numpy.ndarray (images, rows, cols, channels).
    :param nblocks: scalar with the number of blocks to split the rows and columns into.
    :return: numpy.ndarray with x split into blocks.
    """

    # compute how many whole blocks fit in the data, and what length of the image they cover
    _, nrows, ncols, _ = x.shape
    nrows = int(np.floor(nrows / nblocks) * nblocks)
    ncols = int(np.floor(ncols / nblocks) * nblocks)

    # remove the extra bit of the images so that we can split them into equal blocks
    x = x[:, 0:nrows, 0:ncols, :]

    # split images into smaller blocks
    _, x, _ = pystoim.block_split(x, nblocks=(1, nblocks, nblocks, 1), by_reference=True)
    x = np.concatenate(x, axis=0)

    return x


def split_list(x, idx):
    """
    Split a list into two sublists.
    :param x: list to be split.
    :param idx: list of indices. Repeated indices are ignored.
    :return: x[idx], x[~idx]. Here ~idx stands for the indices not in idx.
    """

    # ignore repeated indices
    idx = set(idx)

    # indices for the second sublist
    idx_2 = list(set(range(len(x))) - idx)

    idx = list(idx)

    return list(np.array(x)[idx]), list(np.array(x)[idx_2])


def change_home_directory(file_list, home_path_from, home_path_to, check_isfile=False):
    """
    Change the home directory in a list of file paths and names. Optionally, it checks that the new paths exist in the
    system.

    For example,

    change_home_directory(file_list, '/home/foo', '/users/project/jsmith')

    converts the list

    file_list =
    ['/home/foo/data/file1.txt',
     '/home/foo/results/file2.txt']

    to

    ['/users/project/jsmith/data/file1.txt',
     '/users/project/jsmith/results/file2.txt']


    :param file_list: list of strings with file paths and names, all with the same path directory.
    :param home_path_from: string at the beginning of each path that will be replaced.
    :param home_path_to: string with the new home directory.
    :param check_isfile: (def False) check whether files with the new home directory exist.
    :return:
    """
    if not isinstance(file_list, list):
        file_list = [file_list]
    p = re.compile('^' + home_path_from + '[' + os.path.sep + ']')
    for i, file in enumerate(file_list):
        file_without_home = p.sub('', file)
        file_list[i] = os.path.join(home_path_to, file_without_home)
        if check_isfile and not os.path.isfile(file_list[i]):
            raise FileExistsError(os.path.isfile(file_list[i]))
    return file_list


def augment_file_list(file_list, tag_from, tag_to):
    """
    Replace a tag in the filenames of a list, with another tag, and search for files with the new names.
    This is useful to add filenames of augmented data. For example, if you start with

        im_file_1_nan.tif
        im_file_2_nan.tif
        im_file_3_nan.tif

    and run augment_file_list(file_list, '_nan_', '_*_'), the new file list will contain filenames
    of exisiting files that correspond to

        im_file_1_*.tif
        im_file_2_*.tif
        im_file_3_*.tif

    :param file_list: list of filenames.
    :param tag_from: The tag that will be replaced for the file search. The tag will only be replaced
    in file names, not in file paths.
    :param tag_to: The tag that will replace tag_from in the file search.
    :return: a new file list that includes the files that match the filenames with the new tags.
    """

    file_list_out = []
    for file in file_list:

        # split the filename into the path and the file name itself
        file_path, file_basename = os.path.split(file)

        # replace the tag by the other tag, e.g. '_*_'
        file_basename = file_basename.replace(tag_from, tag_to)

        # search for files with the new tag
        file_list_out += glob.glob(os.path.join(file_path, file_basename))

    return file_list_out


def load_file_list_to_array(file_list):
    """
    Loads a list of images, all with the same size, into a numpy array (file, row, col, channel).

    :param file_list: list of strings with filenames.
    :return: numpy.ndarray.
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


def load_datasets(file_list, prefix_from='im', prefix_to=[], nblocks=1, shuffle_seed=None):
    """
    Loads image files and prepare them for training or testing, returning numpy.ndarrays.
    Image files can be of any type loadable by the PIL module, but they must have the same size.

    Multiple sets of corresponding images can be loaded using prefix_to. For instance,

    im_file_1.tif          seg_file_1.tif          mask_file_1.tif
    im_file_2.tif          seg_file_2.tif          mask_file_2.tif
    im_file_3.tif          seg_file_3.tif          mask_file_3.tif

    will return

    outs['im']   --> numpy.ndarray (3, rows, cols, channels)
    outs['seg']  --> numpy.ndarray (3, rows, cols, channels)
    outs['mask'] --> numpy.ndarray (3, rows, cols, channels)

    This function also provides the following optional functionality:
        * images can be split into equally-sized blocks (this is useful when full images are too
        large for training).
        * images can be shuffled randomly.

    :param file_list: list of paths and filenames (for one of the datasets).
    :param prefix_from: (def 'im') string with the prefix (e.g. 'im') in file_list that when changed gives the other
    datasets. Note that the prefix refers to the name of the file, not its path.
    :param prefix_to: (def []) list of strings with the prefixes of the datasets that will be loaded. E.g.
    ['im', 'mask'] will load the datasets from the filenames that start with 'im' and 'mask'.
    :param nblocks: (def 1) number of equi-sized blocks to split the images into. Note that the edges of the images may
    need to be trimmed so that splitting creates blocks with the same size.
    :param shuffle_seed: (def None) If provided, images are shuffled after splitting.
    :return: out, out_file_list, shuffle_idx:
       * out: dictionary where out[prefix] contains a numpy.ndarray with the data corresponding to the "prefix" dataset.
       * out_file_list: list of the filenames for the out[prefix] dataset.
       * shuffle_idx: list of indices used to shuffle the images after splitting.
    """

    if not isinstance(prefix_to, list):
        raise TypeError('data_prefixes must be a list of strings')

    # using prefixes, get list of filenames for the different datasets
    out_file_list = {}
    for prefix in prefix_to:
        out_file_list[prefix] = []
        for x in file_list:
            x_path, x_basename = os.path.split(x)
            if re.match('^' + prefix_from, x_basename) is None:
                raise ValueError('Prefix not found in filename: ' + x)
            prefix_file = os.path.join(x_path,
                                       re.sub('^' + prefix_from, prefix, x_basename, count=1))
            if not os.path.isfile(prefix_file):
                raise FileExistsError(prefix_file)
            out_file_list[prefix].append(prefix_file)

    # load all datasets
    out = {}
    shuffle_idx = {}
    for prefix in prefix_to:
        # load dataset
        out[prefix] = load_file_list_to_array(out_file_list[prefix])

        # data type conversions
        if prefix == 'im' and out[prefix].dtype == 'uint8':
            out[prefix] = out[prefix].astype(np.float32)
            out[prefix] /= 255
        elif prefix in {'mask', 'dmap'}:
            out[prefix] = out[prefix].astype(np.float32)
        elif prefix == 'seg':
            out[prefix] = out[prefix].astype(np.uint8)

        # split image into smaller blocks, if requested
        if nblocks > 1:
            out[prefix] = split_images(out[prefix], nblocks=nblocks)

        # create shuffling indices if required
        if prefix == prefix_to[0] and shuffle_seed is not None:
            n = out[prefix].shape[0]  # number of images
            shuffle_idx = np.arange(n)
            np.random.seed(shuffle_seed)
            np.random.shuffle(shuffle_idx)

        # shuffle data
        if shuffle_seed is not None:
            out[prefix] = out[prefix][shuffle_idx, ...]

    if DEBUG:
        i = 5
        plt.clf()
        for pi, prefix in enumerate(prefix_to):
            plt.subplot(1, len(prefix_to), pi+1)
            if out[prefix].shape[-1] == 1:
                plt.imshow(out[prefix][i, :, :, 0])
            else:
                plt.imshow(out[prefix][i, :, :, :])
            plt.title('out[' + prefix + ']')

    return out, out_file_list, shuffle_idx


def remove_poor_data(datasets, prefix='mask', threshold=1000):
    """
    Find images where the mask has very few pixels, and remove them from the datasets. Training
    with them can decrease the performance of the model.
    :param datasets: dictionary with numpy.ndarray datasets loaded with load_datasets().
    :param prefix: (def 'mask') string. The number of pixels will be assessed in datasets[prefix].
    :param threshold: (def 1000) integer. Masks
    :return:
    """

    # indices of masks with a large enough number of pixels==1
    idx = np.count_nonzero(datasets[prefix], axis=(1, 2, 3)) > threshold
    # remove corresponding images from all datasets
    for prefix in datasets.keys():
        datasets[prefix] = datasets[prefix][idx, ...]

    return datasets


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

        # compute distance map from every pixel to the closest boundary
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


def read_paths_from_svg_file(file, tag='Cell'):
    """
    Read a SVG file produced by Gimp that contains paths (contours), and return a list of paths, where each path
    is a list of (X,Y) point coordinates.

    Only paths that have a label that starts with the chosen tag are read. This allows having different types of
    objects in the SVG file (e.g. cells, edge cells, background, etc), but only read one type of objects.

    :param file: path and name of SVG file.
    :param tag: (def 'Cell'). Only paths with a label that starts with this tag will be read.
    :return: [ path0, path1, ...] = [ [(X0,Y0), (X1,Y1), ...], ...]
    """

    # extract contour as a list of (X,Y) coordinates
    def extract_contour(path):

        contour = []
        for pt in path:

            # (X, Y) for each point
            contour.append((np.real(pt.start), np.imag(pt.start)))

            if DEBUG:
                plt.plot(*zip(*contour))

        return contour

    # extract all paths from the SVG file
    paths, attributes = svg2paths(file)

    # loop paths
    paths_out = []
    for path, attribute in zip(paths, attributes):

        # skip if the countour is not a cell (we also skip edge cells, as they are incomplete, and thus their area
        # is misleading)
        if not attribute['id'].startswith(tag):
            continue

        # extract contour polygon from the path object, and compute area
        contour = extract_contour(path)
        paths_out.append(contour)

    return paths_out


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

            # split line into elements: ['1/1846[..............................]', 'ETA:1:33:38', 'loss:1611.5088', 'mean_squared_error:933.6124', 'mean_absolute_error:17.6664']
            line = line.split(' - ')

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


def tag_values_with_mouse_info(metainfo, s, values, values_tag='values', tags_to_keep=None):
    """
    If you have a vector with cell areas, values = [4.0 , 7.0 , 9.0 , 7.3], then

    tag_values_with_mouse_info('meta.csv',
                               s='KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_031860_col_033476.svg',
                               values=[4. , 7. , 9. , 7.3],
                               values_tag='area',
                               tags_to_keep=['id', 'ko', 'sex'])

    will read the table of metainformation for all mice from file 'meta.csv', it finds which mouse we are dealing with
    according to s='KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_031860_col_033476.svg' (in this case,
    mouse '37.2g'), and then it creates a dataframe with one row per value, and the mouse's metainformation repeated
    in each row,

          id   ko sex  area
    0  37.4a  PAT   m   4.0
    1  37.4a  PAT   m   7.0
    2  37.4a  PAT   m   9.0
    3  37.4a  PAT   m   7.3

    :param metainfo: List of OrderedDict from read_mouse_csv_info(), or string with CSV filename that contains the
    metainformation.
    :param s: String, typically a filename, that identifies an image, containing the mouse id, e.g.
    'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_031860_col_033476.svg'
    :param values: List or vector of values. Each value creates a row in the output dataframe.
    :param values_tag: Name of the column with the values in the output dataframe.
    :param tags_to_keep: (def None = keep all the tags). If not all the metainformation columns are needed, a list of
    tags can be passed here, e.g. tags_to_keep = ['id', 'ko', 'sex'].
    :return: panda.DataFrame with one row per value.
    """

    # if metainfo is provided as CSV filename, read it to memory as a DataFrame
    if isinstance(metainfo, six.string_types):
        metainfo = pd.read_csv(metainfo)

    # list of mouse IDs
    mouse_ids = metainfo['id']

    # indices of IDs that can be found in the filename
    id_in_sid = np.where([x in s for x in mouse_ids])[0]

    # if more than one ID can be found in the filename, that means that the filename is ambiguous
    if len(id_in_sid) > 1:
        raise ValueError('s is ambiguous, and more than one ID can be found in it: ' + s)
    elif len(id_in_sid) == 0:  # no ID found in the filename
        raise ValueError('No ID can be found in s: ' + s)

    # row with all the metainfo for the selected mouse
    metainfo_row = metainfo.loc[id_in_sid, :]

    # keep only some of the metainformation tags
    if tags_to_keep:
        metainfo_row = metainfo_row[tags_to_keep]

    # repeat the row once per element in values
    df = pd.concat([metainfo_row] * len(values), ignore_index=True)

    # add the values as a new column
    df[values_tag] = values

    return df
