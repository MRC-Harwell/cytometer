"""
cytometer/data.py

Functions to load, save and pre-process data related to the cytometer project.
"""

import os
import glob
import warnings
import pickle
import ujson
import time
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
from scipy import ndimage
import scipy.stats
import pandas as pd
import ast
from mahotas import bwperim
import pysto.imgproc as pystoim
import re
import six
from svgpathtools import svg2paths
import random
import colorsys
import scipy
from shapely.geometry import Polygon

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


def split_file_list_kfolds(file_list, n_folds, ignore_str='', fold_seed=0, save_filename=None):
    """
    Split file list into k-folds for training and testing a neural network.

    If there are N files, each fold gets N/k files for testing, and N*(k-1)/k files for training. If N
    is not divisible by k, the split follows the rules of numpy.array_split().

    :param file_list: List of filenames.
    :param n_folds: Number of folds to split the list into.
    :param ignore_str: (def '') String. This substring will be ignored when making a list of files to be shuffled.
    The string format is the one you'd use in re.sub(ignore_str, '', file_list[i]).
    This is useful if you have several files from the same histology slice, but you want to make sure that
    the network tests run with data from histology slices that haven't been seen at training. For example,

    histo_01_win_01.ndpi
    histo_01_win_02.ndpi
    histo_02_win_01.ndpi
    histo_02_win_02.ndpi

    Using ignore_str='_win_.*' will reduce the file list to

    histo_01
    histo_02

    Thus, both files from histo_01 will be assigned together either to the train or test sets.

    :param fold_seed: Scalar. Seed for the pseudo-random number generator to shuffle the file indices.
    :param save_filename: (def None). If provided, save results to pickle file (*.pickle).
        * 'file_list'
        * 'idx_train'
        * 'idx_test'
        * 'fold_seed'
    :return:
        * idx_train: list of 1D arrays. idx_train[i] are the train indices for training files in the ith-fold.
        * idx_test: list of 1D arrays. idx_train[i] are the train indices for test files in the ith-fold.
    """

    # starting file list
    #
    # im_1_row_4.ndpi
    # im_2_row_8.ndpi
    # im_1_row_2.ndpi
    # im_2_row_3.ndpi
    # im_1_row_1.ndpi

    # create second file list removing the ignore_from* substring
    #
    # im_1
    # im_2
    # im_1
    # im_2
    # im_1
    file_list_reduced = [re.sub(ignore_str, '', x) for x in file_list]

    # create third file list, without duplicates
    #
    # im_1
    # im_2
    file_list_unique = np.unique(file_list_reduced)

    # look up table (LUT) to map back to the full list
    #
    # [[0, 2, 4], [1, 3]]
    file_list_lut = []
    for i, f in enumerate(file_list_unique):
        file_list_lut.append(np.array([j for j, f_all in enumerate(file_list_reduced) if f == f_all]))
    file_list_lut = np.array(file_list_lut)

    # number of reduced files
    n_file = len(file_list_unique)

    # split data into training and testing for k-folds.
    # Here the indices refer to file_list_unique
    #
    # idx_train_all = [[0]]  # im_1
    # idx_test_all = [[1]]   # im_2
    random.seed(fold_seed)
    idx_unique = random.sample(range(n_file), n_file)
    idx_test_all = np.array_split(idx_unique, n_folds)
    idx_train_all = []
    for k_fold in range(n_folds):
        # test and training image indices
        idx_test = idx_test_all[k_fold]
        idx_train = list(set(idx_unique) - set(idx_test))
        random.shuffle(idx_train)
        idx_train_all.append(np.array(idx_train))

    # map reduced indices to full list
    # Here we change the indices to refer to file_list
    #
    # idx_train_all = [[0, 2, 4]]  # im_1_*
    # idx_test_all = [[1, 3]]   # im_2_*
    for i, idx_train in enumerate(idx_train_all):  # loop folds
        # file_list indices
        idx = np.concatenate(file_list_lut[idx_train])
        idx_train_all[i] = idx

    for i, idx_test in enumerate(idx_test_all):  # loop folds
        # file_list indices
        idx = np.concatenate(file_list_lut[idx_test])
        idx_test_all[i] = idx

    # check that there's no overlap between training and test datasets, and that each fold contains
    # all the images
    for i, idx_train in enumerate(idx_train_all):
        assert(set(idx_train).intersection(idx_test_all[i]) == set())
        assert (len(np.concatenate((idx_test_all[i], idx_train))) == len(file_list))

    if save_filename is not None:
        with open(save_filename, 'wb') as f:
            x = {'file_list': file_list, 'idx_train': idx_train_all, 'idx_test': idx_test_all, 'fold_seed': fold_seed}
            pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)

    return idx_train_all, idx_test_all


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
            raise FileExistsError(file_list[i])
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


def read_paths_from_svg_file(file, tag='Cell', add_offset_from_filename=False, minimum_npoints=3):
    """
    Read a SVG file produced by Gimp that contains paths (contours), and return a list of paths, where each path
    is a list of (X,Y) point coordinates.

    Only paths that have a label that starts with the chosen tag are read. This allows having different types of
    objects in the SVG file (e.g. cells, edge cells, background, etc), but only read one type of objects.

    :param file: path and name of SVG file.
    :param tag: (def 'Cell'). Only paths with a label that starts with this tag will be read. The case (upper/lowercase)
    is ignored.
    :param add_offset_from_filename: (def False)
    :param minimum_npoints: (def 3) Contours with less than this number of points will be ignored.
    :return: [ path0, path1, ...] = [ [(X0,Y0), (X1,Y1), ...], ...]
    """

    # convert tag to lowercase, to avoid differentiating between "Cell" and "cell"
    tag = tag.lower()

    # extract contour as a list of (X,Y) coordinates
    def extract_contour(path, x_offset=0, y_offset=0):

        contour = []
        for pt in path:

            # (X, Y) for each point
            contour.append((np.real(pt.start) + x_offset, np.imag(pt.start) + y_offset))

            if DEBUG:
                plt.plot(*zip(*contour))

        return contour

    # extract all paths from the SVG file
    paths, attributes = svg2paths(file)

    # add offset to point coordinates from the file name
    if add_offset_from_filename:
        file_basename = os.path.basename(file)  # remove path from filename
        file_basename, _ = os.path.splitext(file_basename)  # remove extension
        file_basename = file_basename.split('_')
        row_i = file_basename.index('row')
        y_offset = float(file_basename[row_i + 1])
        col_i = file_basename.index('col')
        x_offset = float(file_basename[col_i + 1])
    else:
        x_offset = 0
        y_offset = 0

    # loop paths
    paths_out = []
    for path, attribute in zip(paths, attributes):

        # convert the name of the object to lowercase, to avoid differentiating between "Cell" and "cell"
        attribute_id = attribute['id'].lower()

        # skip if the contour's name doesn't start with the required tag, e.g. 'Cell'
        if not attribute_id.startswith(tag):
            continue

        # extract contour polygon from the path object
        contour = extract_contour(path, x_offset=x_offset, y_offset=y_offset)

        # if the contour has enough points, append to the output
        if len(contour) >= minimum_npoints:
            paths_out.append(contour)

    return paths_out


def area2quantile(areas, quantiles=np.linspace(0.0, 1.0, 101)):
    """
    Return function to map from cell areas to quantiles.

    :param areas: Vector with random sample that is representative of area values in the population. The probability
    distribution and quantiles are computed from this random sample.
    :param quantiles: (def np.linspace(0.0, 1.0, 101)) Quantiles values in [0.0, 1.0] at which the function will be
    linearly interpolated.
    :return:
    * f: scipy.interpolate.interpolate.interp1d interpolation function that maps areas values to [0.0, 1.0]. Area values
    outside the range are mapped to 0.0 (smaller) or 1.0 (larger).
    """

    areas_by_quantiles = scipy.stats.mstats.hdquantiles(areas, prob=quantiles)
    f_area2quantile = scipy.interpolate.interp1d(areas_by_quantiles.data, quantiles, bounds_error=False,
                                               fill_value=(0.0, 1.0))

    if DEBUG:
        plt.clf()
        fig = plt.hist(areas, bins=50, density=True, histtype='step')
        for x in areas_by_quantiles.data:
            plt.plot([x, x], [0, fig[0].max()], 'k')

    return f_area2quantile


def aida_colourmap():
    """
    Create a colourmap that replicates in plt.imshow() the colours that we obtain in AIDA. This colormap is called
    'quantiles_aida'.

    This colourmap is meant to map area quantiles [0.0, 1.0] to a pastel yellow-green-purple colour scale.

    This function can be combined with area2quantile() to map areas to colours:

        import cytometer

        # variables already assigned
        manual_areas # vector with a representative random sample of cell areas. The distribution will be computed from these values
        area_grid  # np.float32 2D array with area values interpolated onto a pixel grid
        area_mask # bool array of the same size that says where there's tissue and where not

        # compute function to map between cell areas and [0.0, 1.0]
        f_area2quantile = cytometer.data.area2quantile(manual_areas)

        # convert area values to quantiles
        quantiles_grid = f_area2quantile(quantiles_grid)

        # make background white in the plot
        quantiles_grid[~areas_mask] = np.nan

        # load AIDA's colourmap
        cm = cytometer.data.aida_colourmap()

        # plot
        plt.imshow(quantiles_grid, vmin=0.0, vmax=1.0, cmap=cm)

    :return:
    * cm: matplotlib.colors.ListedColormap with 101 colours.
    """

    # range of colours in HSL format
    hue = np.linspace(0, 315 / 360, 101)
    lightness = 0.69
    saturation = 0.44
    alpha = 1

    # the colourmap only changes the hue linearly, while keeping the saturation and lightness constant
    cm = [colorsys.hls_to_rgb(h=h, l=lightness, s=saturation) + (alpha,) for h in hue]

    return ListedColormap(cm, name='quantiles_aida')


def aida_contour_items(contours, f_area2quantile, cm='quantiles_aida', xres=1.0, yres=1.0, cell_prob=None):
    """
    Create list of contour items for AIDA.

    This function computes the area of each contour, it's quantile, and maps it to a colour.

    :param contours: List [contour_0, contour_1...], where contour_i is an (Ni, 2)-np.array with Ni 2D points.
    :param f_area2quantile: Function to map areas to quantiles. Compute with cytometer.data.area2quantile(manual_areas).
    :param cm: (def 'quantiles_aida'). String or matplotlib.colors.ListedColormap to map [0.0, 1.0] to RGB colours. If
    cm is a string, currently only 'quantiles_aida' is accepted.
    :param xres: (def 1.0) Pixel size in the x-coordinate.
    :param yres: (def 1.0) Pixel size in the y-coordinate.
    :param cell_prob: (def None) Vector with one value per contour. This value is interpreted as the probability of the
    object being a cell.
    :return: List of dictionaries, each one with the structure of a contour object.
    """


    def aida_contour_item(contour, rgb_colour, cell_prob=None):
        """
        Create an object that describes a closed contour in AIDA. The user provides the coordinates of the contour
        points and the colour for the contour.

        :param contour: np.array or list of points of a contour: [[x0, y0], [x1, y1], ...]
        :param rgb_colour: (r, g, b) or (r, g, b, alpha). RGB colour for this contour.
        :param cell_prob: (def None) Scalar in [0.0, 1.0] with the probability of the contour being a cell.
        :return: item: dictionary with the structure of the contour object.
        """

        if type(contour) != 'numpy.ndarray':
            contour = list(contour)

        # convert RGB to HSL
        hls_colour = colorsys.rgb_to_hls(rgb_colour[0], rgb_colour[1], rgb_colour[2])

        # create the item object
        item = {
            'class': '',
            'type': 'path',
            'color': {
                'fill': {
                    'hue': hls_colour[0] * 360,
                    'saturation': hls_colour[2],
                    'lightness': hls_colour[1],
                    'alpha': 0.7
                },
                'stroke': {
                    'hue': hls_colour[0] * 360,
                    'saturation': hls_colour[2],
                    'lightness': hls_colour[1],
                    'alpha': 1.0
                }
            },
            'segments': contour,
            'closed': True
        }

        if cell_prob is not None:
            item['cell_prob'] = cell_prob

        return item

    ## main function

    # load colourmap
    if cm == 'quantiles_aida':
        cm = aida_colourmap()
    else:
        raise ValueError('Only quantiles_aida colourmap is implemented')

    # compute area of each contour
    areas = [Polygon(c).area * xres * yres for c in contours]  # (um^2)

    # convert to quantiles
    q = f_area2quantile(areas)

    # create the list of contour objects (each item is a contour object)
    if cell_prob is None:
        items = [aida_contour_item(contour=contours[i], rgb_colour=cm(q[i]))
                 for i in range(len(contours))]
    else:
        items = [aida_contour_item(contour=contours[i], rgb_colour=cm(q[i]), cell_prob=cell_prob[i])
                 for i in range(len(contours))]

    return items


def aida_rectangle_items(rectangles):
    """
    Create list of rectangle items for AIDA.

    :param rectangles: List [rectangle_0, rectangle_1...], where rectangle_i is a tuple (x0, y0, width, height).
    :return: List of dictionaries, each one with the structure of a rectangle object.
    """


    def aida_rectangle_item(rectangle):
        """
        Create an object that describes a rectangle in AIDA.

        :param rectangle: (x0, y0, width, height).
        :return: item: dictionary with the structure of the rectangle object.
        """

        # extract rectangle parameters
        (x0, y0, width, height) = rectangle

        # convert RGB to HSL
        hls_colour_black = colorsys.rgb_to_hls(0, 0, 0)
        hls_colour_white = colorsys.rgb_to_hls(1, 1, 1)

        item = {
            'type': 'rectangle',
            'class': '',
            'color': {
                'fill': {
                    'hue': hls_colour_black[0] * 360,
                    'saturation': hls_colour_black[2],
                    'lightness': hls_colour_black[1],
                    'alpha': 0.5},
                'stroke': {
                    'hue': hls_colour_white[0] * 360,
                    'saturation': hls_colour_white[2],
                    'lightness': hls_colour_white[1],
                    'alpha': 1}},
            'x': x0,
            'y': y0,
            'width': width,
            'height': height,
            'locked': False
        }

        return item

    ## main function

    if type(rectangles) != list:
        raise TypeError('rectangles must be a list, even if it only has one item')

    # create the list of rectangle objects (each item is a rectangle object)
    items = [aida_rectangle_item(rectangle=rectangles[i]) for i in range(len(rectangles))]

    return items


def aida_write_new_items(filename, items, mode='append_to_last_layer', indent=0, ensure_ascii=False,
                         double_precision=10, number_of_attempts=1):
    """
    Create a new or update existing AIDA annotations file, adding new items.
    :param filename: String with path to .json annotations file.
    :param items: List of items, obtained e.g. with aida_contour_items() or aida_rectangle_items().
    :param mode:
        - 'append_to_last_layer': (def) Append items to the last layer with the same item type.
        - 'append_new_layer': Create new layer for items.
        - 'w': Overwrite existing file, or create a new one with only these items.
    :param indent: (def 0) Number of indentation spaces for pretty print format. By default, 0, everything is saved to
    one line without spaces.
    :param ensure_ascii: (def False) Limits output to ASCII and escapes all extended characters above 127. Default is
    true. If your end format supports UTF-8 setting this option to false is highly recommended to save space.
    :param double_precision: (def 10) Controls how many decimals to encode for double or decimal values.
    :param number_of_attempts: (def 1) Number of times we try to write the file if the server returns
    ConnectionResetError.
    :return:
    * None
    """

    if type(items) != list:
        raise SyntaxError('items must be a list, but is type: ' + type(items))

    # get item type
    item_type = items[0]['type']

    # layer name that corresponds to this item type
    if item_type == 'path':
        layer_name = 'White adipocyte'
    elif item_type == 'rectangle':
        layer_name = 'Blocks'
    else:
        raise NotImplementedError('item_type not implemented: ' + item_type)

    # if an annotations file already exists with the filename path, read it so that we can add to it
    if mode != 'w' and os.path.isfile(filename):

        # load existing data
        with open(filename) as fp:
            annotations = ujson.load(fp)

        # check that this file has the expected format
        if 'name' not in annotations:
            raise KeyError('Annotations have no \'name\' key')
        if 'layers' not in annotations:
            raise KeyError('Annotations have no \'layers\' key')

    else:  # if previous file doesn't exist, create an annotations file from scratch

        # top-most segmentation object. It contains only one layer
        annotations = {'name': 'DeepCytometer annotations', 'layers': []}

    if DEBUG:
        print('Loaded file: ' + filename)
        print('Annotations name: ' + annotations['name'])
        print('Number of layers: ' + str(len(annotations['layers'])))
        for j in range(len(annotations['layers'])):
            print('    Layer ' + str(j) + ': ' + annotations['layers'][j]['name'] + '. Number of '
                  + annotations['layers'][j]['items'][0]['type'] + ' items: '
                  + str(len(annotations['layers'][j]['items'])))

    # find the last layer for the current item type, if there's any
    i_layer = layer_number = []  # in case there isn't any layer of this type
    for l in range(len(annotations['layers'])):
        if layer_name in annotations['layers'][l]['name']:
            # index of layer where we are going to append items
            i_layer = l
            # number that we are going to add to the layer name. Note that this, in general, will be different from
            # the layer's index
            layer_number = int(annotations['layers'][l]['name'].replace(layer_name, ''))

    # if no layer exists, we'll need to add a new layer
    if i_layer == []:
        mode = 'append_new_layer'
        layer_number = -1

    # if we want to append items to a new layer
    if mode == 'append_new_layer':

        # new layer object
        layer = {'name': layer_name + ' ' + str(layer_number + 1), 'opacity': 1.0, 'items': items}

        # append layer to current annotations
        annotations['layers'] += [layer, ]

    elif mode == 'append_to_last_layer':

        # append items to currently selected layer
        annotations['layers'][i_layer]['items'] += items

    else:
        raise NotImplementedError('mode not implemented: ' + mode)

    # if we are trying to save to a network filesystem, the server hosting the filesystem may return a
    # ConnectionResetError. Because this would corrupt the output file, we let the user try a couple of times, with
    # with a little wait between tries
    for attempt in range(number_of_attempts):
        try:
            # write annotations to file
            with open(filename, 'w') as fp:
                ujson.dump(annotations, fp, indent=indent, ensure_ascii=ensure_ascii, double_precision=double_precision)
            break
        except ConnectionResetError:
            print('# ======> ConnectionResetError. Attempt ' + str(attempt) + '/' + str(number_of_attempts) + ' ...')
            time.sleep(5)  # secs
        except BrokenPipeError:
            print('# ======> BrokenPipeError. Attempt ' + str(attempt) + '/' + str(number_of_attempts) + ' ...')
            time.sleep(5)  # secs
        except:
            raise


def aida_get_contours(annotations, layer_name='.*', return_props=False):
    """
    Concatenate items as contours in an AIDA annotations file or dict. Only 'path' and 'rectangle' types implemented.

    :param annotations: filename or dict with AIDA annotations.
    :param layer_name: (def '.*', which matches any name). Regular expression (see help for re module) that will be used
    as the pattern to march against the layer names. For example, layer_name='White adipocyte.*' will match any layer
    name like 'White adipocyte 0', 'White adipocyte 1', etc.
    :param return_props: (def False). Return extra output variable with properties of contours.
    :return:
    * items: list of concatenated contours from selected layers. [contour_0, contour_1, ...] where
    contour_i = [[x0, y0], [x1, y1], ...].
    * props: dictionary of contour properties. Currently, only implemented output is
        props['cell_prob']
    """

    # if filename provided, load annotations
    if isinstance(annotations, six.string_types):

        # parse the json file
        with open(annotations) as fp:
            annotations = ujson.load(fp)

    if type(annotations) != dict:
        raise TypeError('annotations must be type dict')

    items = []
    if return_props:
        cell_prob = []
    for l in range(len(annotations['layers'])):

        # check whether the regular expression matches the layer name
        if re.match(layer_name, annotations['layers'][l]['name']) is not None:

            for i in range(len(annotations['layers'][l]['items'])):

                if annotations['layers'][l]['items'][i]['type'] == 'path':

                    # add items to list of output items
                    items += [annotations['layers'][l]['items'][i]['segments'],]
                    if return_props:
                        cell_prob += [annotations['layers'][l]['items'][i]['cell_prob'],]

                elif annotations['layers'][l]['items'][i]['type'] == 'rectangle':

                    # extract rectangle first and last corners
                    x0 = annotations['layers'][l]['items'][i]['x']
                    y0 = annotations['layers'][l]['items'][i]['y']
                    xend = x0 + annotations['layers'][l]['items'][i]['width'] - 1
                    yend = y0 + annotations['layers'][l]['items'][i]['height'] - 1

                    # 4 corners of the rectangle, with first repeated for closed contour
                    item = [[x0, y0], [xend, y0], [xend, yend], [x0, yend], [x0, y0]]

                    # add items to list of output items
                    items += [item,]

                else:

                    warnings.warn('Unknown item type found: layer ' + str(l) + ', item ' + str(i) + ': '
                                  + annotations['layers'][l]['items'][i]['type'], SyntaxWarning)

    if return_props:
        return items, {'cell_prob':cell_prob}
    else:
        return items


def read_keras_training_output(filename, every_step=True):
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
    :param every_step: (bool def True) The log file contains one line per training step, as opposed to reading only one
    summary line per epoch
    :return: list of pandas.DataFrame
    """

    def process_metrics(line, epoch):

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

        return line
    # end of def process_metrics():


    # ignore all lines until we get to the first "Epoch"
    df_all = []
    file = open(filename, 'r')
    for line in file:
        if line[0:5] == 'Epoch':
            data = []
            epoch = 1
            break

    # if the user wants to read only the summary line of the training output instead of each step
    if every_step:

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

                # convert line into dictionary with different metric values
                line = process_metrics(line, epoch)

                # add the dictionary to the output list
                data.append(line)

            # read the next line
            line = file.readline()

        # we have finished reading the log file

    else:  # read only one summary line by epoch

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

            elif 'Epoch' in line and not ':' in line:  # new epoch of current training

                epoch += 1

            # we want this line
            #     1748/1748 [==============================] - 188s 108ms/step - loss: 0.3056 - acc: 0.9531 - val_loss: 0.3092 - val_acc: 0.9405
            # we don't want this line
            #     '1740/1748 [============================>.] - ETA: 0s - loss: 0.3058 - acc: 0.95312019-06-14 13:08:41.183265: W tensorflow/stream_executor/cuda/cuda_dnn.cc:3472] \n'
            # we don't want these lines
            #     2019-06-14 13:08:41.183372: W tensorflow/stream_executor/cuda/cuda_dnn.cc:3217]
            elif 'loss' in line and '[=====' in line and '=====]' in line:

                # remove start of line to keep only
                #    loss: 0.3056 - acc: 0.9531 - val_loss: 0.3092 - val_acc: 0.9405
                pattern = re.compile('loss:.*')
                line = pattern.search(line)
                line = line[0]

                # add dummy start at beginning of the line that will be removed by process_metrics()
                line = 'foo: foo - ' + line

                # convert line into dictionary with different metric values
                line = process_metrics(line, epoch)

                # add the dictionary to the output list
                data.append(line)

            # read the next line
            line = file.readline()

        # we have finished reading the log file

    # end "if from_summary_line"

    # close the file
    file.close()

    return df_all


def tag_values_with_mouse_info(metainfo, s, values=None, values_tag='values', tags_to_keep=None, id_tag='id'):
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
    :param id_tag: (def 'id') String with the name of the column that contains the animal ID. This ID is also in the
    filename, and that's how the filename gets matched to a row in the metainfo file.
    :return: panda.DataFrame with one row per value.
    """

    # if metainfo is provided as CSV filename, read it to memory as a DataFrame
    if isinstance(metainfo, six.string_types):
        metainfo = pd.read_csv(metainfo)

    # list of mouse IDs
    mouse_ids = metainfo[id_tag]

    # indices of IDs that can be found in the filename
    id_in_sid = np.where([x in s for x in mouse_ids])[0]

    # if more than one ID can be found in the filename, that means that the filename is ambiguous
    if len(id_in_sid) > 1:
        raise ValueError('s is ambiguous, and more than one ID can be found in it: ' + s)
    elif len(id_in_sid) == 0:  # no ID found in the filename
        warnings.warn('Either s has no valid ID, or metainfo is missing a row for that ID: ' + s)
        metainfo_row = pd.DataFrame(columns=metainfo.columns)
    else:
        # row with all the metainfo for the selected mouse
        metainfo_row = metainfo.loc[id_in_sid, :]

    # keep only some of the metainformation tags
    if tags_to_keep:
        metainfo_row = metainfo_row[tags_to_keep]

    if values is None or len(values) == 0:
        return metainfo_row
    else:
        # repeat the row once per element in values
        df = pd.concat([metainfo_row] * len(values), ignore_index=True)
        df[values_tag] = values
        return df


## Deprecated functions


def write_path_to_aida_json_file(fp, x, hue=170, pretty_print=False):
    """
    DEPRECATED: Use aida_write_new_items() instead.
    Write single contour to a JSON file in AIDA's annotation format.

    (This function only writes the XML code only for the contour, not a full JSON file).

    :param fp: file pointer to text file that is open for writing/appending.
    :param x: numpy.ndarray with two columns, for (x,y)-coordinates of the points of a polygonal contour.
    :param hue: (def 170, which is a greenish blue) The hue of the color as a value in degrees between 0 and 360.
    :param pretty_print: (def False) Boolean. Whether to save the file with '\n' and whitespaces for pretty formatting.
    :return: None.
    """

    warnings.warn('Use aida_write_new_items() instead', DeprecationWarning)

    if pretty_print:
        fp.write('        {\n')
        fp.write('          "class": "",\n')
        fp.write('          "type": "path",\n')
        fp.write('          "color": {\n')
        fp.write('            "fill": {\n')
        fp.write('              "hue": ' + str(hue) + ',\n')
        fp.write('              "saturation": 0.44,\n')
        fp.write('              "lightness": 0.69,\n')
        fp.write('              "alpha": 0.5\n')
        fp.write('            },\n')
        fp.write('            "stroke": {\n')
        fp.write('              "hue": ' + str(hue) + ',\n')
        fp.write('              "saturation": 0.44,\n')
        fp.write('              "lightness": 0.69,\n')
        fp.write('              "alpha": 1.0\n')
        fp.write('            }\n')
        fp.write('          },\n')
        fp.write('          "segments": [\n')
        for i, pt in enumerate(x):
            fp.write('            [' + '{0:.12f}'.format(pt[0]) + ', ' + '{0:.12f}'.format(pt[1]))
            if i == len(x) - 1:
                fp.write(']\n')  # last point in the contour
            else:
                fp.write('],\n')  # any other point in the contour
        fp.write('          ],\n')
        fp.write('          "closed": true\n')
        fp.write('        }')
    else:
        fp.write('{')
        fp.write('"class": "",')
        fp.write('"type": "path",')
        fp.write('"color": {')
        fp.write('"fill": {')
        fp.write('"hue": ' + str(hue) + ',')
        fp.write('"saturation": 0.44,')
        fp.write('"lightness": 0.69,')
        fp.write('"alpha": 0.5')
        fp.write('},')
        fp.write('"stroke": {')
        fp.write('"hue": ' + str(hue) + ',')
        fp.write('"saturation": 0.44,')
        fp.write('"lightness": 0.69,')
        fp.write('"alpha": 1.0')
        fp.write('}')
        fp.write('},')
        fp.write('"segments": [')
        for i, pt in enumerate(x):
            fp.write('[' + '{0:.12f}'.format(pt[0]) + ',' + '{0:.12f}'.format(pt[1]))
            if i == len(x) - 1:
                fp.write(']')  # last point in the contour
            else:
                fp.write('],')  # any other point in the contour
        fp.write('],')
        fp.write('"closed": true')
        fp.write('}')

    return


def write_paths_to_aida_json_file(fp, xs, hue=170, pretty_print=False):
    """
    DEPRECATED: Use aida_write_new_items() instead.
    Write a list/tuple of contours to a JSON file in AIDA's annotation format.

    :param fp: file pointer to text file that is open for writing/appending.
    :param xs: List of contours. Each contour is a list of points given as [x,y] or (x,y).
    :param hue: (def 170, a greenish blue). Integer or list of integers with hue value (between 0 and 360 degrees), one
    per contour in xs. If hue is an integer, the same hue is applied to each contour.
    :param pretty_print: (def False) Boolean. Whether to save the file with '\n' and whitespaces for pretty formatting.
    :return: None.
    """

    warnings.warn('Use aida_write_new_items() instead', DeprecationWarning)

    if np.isscalar(hue):
        hue = [hue] * len(xs)

    # write file header
    if pretty_print:
        fp.write('{\n')
        fp.write('  "name": "Cytometer segmentation",\n')
        fp.write('  "layers": [\n')
        fp.write('    {\n')
        fp.write('      "name": "Cell layer",\n')
        fp.write('      "opacity": 1,\n')
        fp.write('      "items": [\n')
        for i, x in enumerate(xs):
            write_path_to_aida_json_file(fp, x, hue=hue[i], pretty_print=pretty_print)
            if i == len(xs) - 1:
                fp.write('\n')  # last path
            else:
                fp.write(',\n')  # any other path
        fp.write('      ]\n')
        fp.write('    }\n')
        fp.write('  ]\n')
        fp.write('}\n')
    else:
        fp.write('{')
        fp.write('"name": "Cytometer segmentation",')
        fp.write('"layers": [')
        fp.write('{')
        fp.write('"name": "Cell layer",')
        fp.write('"opacity": 1,')
        fp.write('"items": [')
        for i, x in enumerate(xs):
            write_path_to_aida_json_file(fp, x, hue=hue[i], pretty_print=pretty_print)
            if i == len(xs) - 1:
                fp.write('')  # last path
            else:
                fp.write(',')  # any other path
        fp.write(']')
        fp.write('}')
        fp.write(']')
        fp.write('}')

    return


def append_paths_to_aida_json_file(file, xs, hue=170, pretty_print=False):
    """
    DEPRECATED: Use write_aida_annotations() instead.
    Add contours to AIDA's annotation file in JSON format.

    :param file: Path and filename of .json file with AIDA annotations.
    :param xs: List of contours. Each contour is a list of points given as [x,y] or (x,y).
    :param hue: (def 170, a greenish blue). Integer or list of integers with hue value (between 0 and 360 degrees), one
    per contour in xs. If hue is an integer, the same hue is applied to each contour.
    :param pretty_print: (def False) Boolean. Whether to save the file with '\n' and whitespaces for pretty formatting.
    :return: None.
    """

    warnings.warn('Use aida_write_new_items() instead', DeprecationWarning)

    def seek_character(fp, target):
        """
        Read file backwards until finding target character.
        :param fp: File pointer.
        :param target: Character that we are looking for.
        :return:
        c: Found character.
        """

        while fp.tell() > 0:
            c = fp.read(1)
            if c == target:
                break
            else:
                fp.seek(fp.tell() - 2)
        if fp.tell() == 0:
            raise IOError('Beginning of file reached before finding "}"')
        return c

    if len(xs) == 0:
        return

    if np.isscalar(hue):
        hue = [hue] * len(xs)

    # truncate the tail of the annotations file:
    #
    # '          "closed": true'
    # '        }'  ---------> end of last contour
    # '      ]'    ---------> first line of tail (truncate this line and the rest)
    # '    }'
    # '  ]'
    # '}'
    #
    # so that we can append a new contour like
    #
    # '          "closed": true'
    # '        },' ---------> end of last contour
    # '        {'  ---------> beginning of new contour
    # '          "class": "",'
    # '          "type": "path",'
    # '          ...'
    # '          "closed": true'
    # '        }'  ---------> end of last contour
    # '      ]'    ---------> first line of tail
    # '    }'
    # '  ]'
    # '}'
    if pretty_print:
        tail = '      ]\n' + \
               '    }\n' + \
               '  ]\n' + \
               '}\n'
    else:
        tail = ']' + \
               '}' + \
               ']' + \
               '}'
    fp = open(file, 'r')
    if not fp.seekable():
        raise IOError('File does not support random access: ' + file)

    # go to end of the file
    pos = fp.seek(0, os.SEEK_END)

    # find tail characters reading backwards from the end
    c = seek_character(fp, '}')
    c = seek_character(fp, ']')
    c = seek_character(fp, '}')
    c = seek_character(fp, ']')
    c = seek_character(fp, '}')
    pos = fp.tell()

    # reopen file to append contours
    fp.close()
    fp = open(file, 'a')
    fp.seek(pos)

    # truncate the tail
    fp.truncate()

    # add ',' to '        }', so that we can add a new contour
    if pretty_print:
        fp.write(',\n')
    else:
        fp.write(',')

    # append new contours to JSON file
    for i, x in enumerate(xs):
        write_path_to_aida_json_file(fp, x, hue=hue[i], pretty_print=pretty_print)
        if i == len(xs) - 1:
            if pretty_print:
                fp.write('\n')  # last path
            else:
                fp.write('')  # last path
        else:
            if pretty_print:
                fp.write(',\n')  # any other path
            else:
                fp.write(',')  # any other path

    # append tail
    fp.write(tail)

    fp.close()

    return
