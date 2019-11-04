"""
Read full .ndpi slides, extract 1001x1001 windows manually selected to increase the dataset randomly created with 0074.

The windows are saved with row_R_col_C, where R, C are the row, col centroid of the image. You can get the offset of the
image from the centroid as offset = centroid - box_half_size = centroid - 500.
"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0094_generate_extra_training_images'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import openslide
import numpy as np
from statistics import mode
import matplotlib.pyplot as plt
import cv2
from random import randint, seed
import tifffile
import glob
from cytometer.utils import rough_foreground_mask

DEBUG = False

ndpi_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
training_dir = os.path.join(home, 'Data/cytometer_data/klf14/klf14_b6ntac_training')
seg_dir = os.path.join(home, 'Data/cytometer_data/klf14/klf14_b6ntac_seg')
downsample_factor = 8.0

# dimensions of the box that we use to crop the full histology image
box_size = 1001
box_half_size = int((box_size - 1) / 2)
n_samples = 5

# explicit list of files, to avoid differences if the files in the directory change
ndpi_files_list = [
    'KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04.ndpi',
    'KLF14-B6NTAC-MAT-18.2d  60-16 C1 - 2016-02-03 13.13.57.ndpi',
    'KLF14-B6NTAC-MAT-18.2b  58-16 C1 - 2016-02-03 11.10.52.ndpi',
    'KLF14-B6NTAC-MAT-17.2c  66-16 C1 - 2016-02-04 11.46.39.ndpi',
    'KLF14-B6NTAC-MAT-17.2f  68-16 C1 - 2016-02-04 15.05.54.ndpi',
    'KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38.ndpi',
    'KLF14-B6NTAC-MAT-18.1a  50-16 C1 - 2016-02-02 09.12.41.ndpi',
    'KLF14-B6NTAC-MAT-18.2g  63-16 C1 - 2016-02-03 16.58.52.ndpi',
    'KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46.ndpi',
    'KLF14-B6NTAC-37.1d PAT 109-16 C1 - 2016-02-15 15.19.08.ndpi',
    'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33.ndpi',
    'KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38.ndpi',
    'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53.ndpi',
    'KLF14-B6NTAC-37.1c PAT 108-16 C1 - 2016-02-15 14.49.45.ndpi',
    'KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52.ndpi',
    'KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32.ndpi',
    'KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11.ndpi',
    'KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06.ndpi',
    'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52.ndpi',
    'KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00.ndpi'
]
ndpi_files_list = [os.path.join(ndpi_dir, x) for x in ndpi_files_list]

# Note: if you want to read the full list of KLF14*.ndpi
# ndpi_files_list = glob.glob(os.path.join(ndpi_dir, 'KLF14*.ndpi'))

# level that we downsample the image to
level = 4

for i_file, ndpi_file in enumerate(ndpi_files_list):

    print('File ' + str(i_file) + '/' + str(len(ndpi_files_list)) + ': ' + ndpi_file)

    # load file
    im = openslide.OpenSlide(os.path.join(ndpi_dir, ndpi_file))

    # box for left half of image
    box_corner_col = 0
    box_corner_row = 0
    location = (box_corner_col, box_corner_row)
    size = (int(im.level_dimensions[level][0]/2), int(im.level_dimensions[level][1]))

    # extract tile from full resolution image, downsampled so that we can look for regions of interest
    tile_lo = im.read_region(location=location, level=level, size=size)
    tile_lo = np.array(tile_lo)
    tile_lo = tile_lo[:, :, 0:3]

    if DEBUG:
        plt.clf()
        plt.imshow(tile_lo)

for i_file, ndpi_file in enumerate(ndpi_files_list):

    print('File ' + str(i_file) + '/' + str(len(ndpi_files_list)) + ': ' + ndpi_file)

    # load file
    im = openslide.OpenSlide(os.path.join(ndpi_dir, ndpi_file))

    # locations of cropping windows in the coordinates of the downsampled image
    if i_file == 0:
        location_list = ((1059, 1052), (682, 1174), (918, 1533))
    elif i_file == 1:
        location_list = ((883, 866), (864, 1278), (665, 456))
    elif i_file == 2:
        location_list = ((672, 397), (885, 877), (870, 1303))
    elif i_file == 3:
        location_list = ((549, 615), (1094, 1112), (546, 615))
    elif i_file == 4:
        location_list = ((230, 500), (488, 1435), (1225, 1198))
    elif i_file == 5:
        location_list = ((874, 730), (776, 991), (1947, 1170))
    elif i_file == 6:
        location_list = ((786, 874), (995, 950), (1193, 903), (1080, 224))
    elif i_file == 7:
        location_list = ((686, 986), (846, 633), (1677, 999))
    elif i_file == 8:
        location_list = ((553, 415), (341, 274), (164, 872))
    elif i_file == 9:
        location_list = ((563, 827), (116, 628), (1030, 1230))
    elif i_file == 10:
        location_list = ((632, 1508), (1311, 925), (1210, 334))
    elif i_file == 11:
        location_list = ((1627, 2587), (1532, 1804), (1411, 923), (1430, 850))
    elif i_file == 12:
        location_list = ((1133, 1388), (1113, 430), (372, 1187))
    elif i_file == 13:
        location_list = ((330, 744), (264, 350), (556, 451))
    elif i_file == 14:
        location_list = ((591, 1262), (927, 1170), (1140, 1173))
    elif i_file == 15:
        location_list = ((441, 759), (652, 339), (1097, 266))
    elif i_file == 16:
        location_list = ((467, 763), (1197, 855), (1806, 1253))
    elif i_file == 17:
        location_list = ((475, 895), (680, 1334), (1346, 950))
    elif i_file == 18:
        location_list = ((744, 1483), (1216, 992), (1854, 581))
    elif i_file == 19:
        location_list = ((970, 1852), (432, 657), (1015, 803))

    for j in len(location_list):

        location = location_list[j]

        location = np.array(location) * im.level_downsamples[level]

        # extract tile at full resolution
        tile = im.read_region(location=location.astype(np.int), level=0, size=(box_size, box_size))
        tile = np.array(tile)
        tile = tile[:, :, 0:3]

        if DEBUG:
            plt.clf()
            plt.imshow(tile)


########################## BELOW: OLD CODE  ######################################################

    # level for a x8 downsample factor
    level_8 = im.get_best_level_for_downsample(downsample_factor)

    assert (im.level_downsamples[level_8] == downsample_factor)

    if OLD_CODE:  # this is the code that was used for the KLF14 experiments

        # Note: Now we have function cytometer.utils.rough_foreground_mask() to do the following, but the function

        # get downsampled image
        im_8 = im.read_region(location=(0, 0), level=level_8, size=im.level_dimensions[level_8])
        im_8 = np.array(im_8)
        im_8 = im_8[:, :, 0:3]

        if DEBUG:
            plt.clf()
            plt.imshow(im_8)
            plt.pause(.1)

        # reshape image to matrix with one column per colour channel
        im_8_mat = im_8.copy()
        im_8_mat = im_8_mat.reshape((im_8_mat.shape[0] * im_8_mat.shape[1], im_8_mat.shape[2]))

        # background colour
        background_colour = []
        for i in range(3):
            background_colour += [mode(im_8_mat[:, i]), ]
        background_colour_std = np.std(im_8_mat, axis=0)

        # threshold segmentation
        seg = np.ones(im_8.shape[0:2], dtype=bool)
        for i in range(3):
            seg = np.logical_and(seg, im_8[:, :, i] < background_colour[i] - background_colour_std[i])
        seg = seg.astype(dtype=np.uint8)
        seg[seg == 1] = 255

        # dilate the segmentation to fill gaps within tissue
        kernel = np.ones((25, 25), np.uint8)
        seg = cv2.dilate(seg, kernel, iterations=1)
        seg = cv2.erode(seg, kernel, iterations=1)

        # find connected components
        labels = seg.copy()
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(seg)
        lblareas = stats[:, cv2.CC_STAT_AREA]

        # labels of large components, that we assume correspond to tissue areas
        labels_large = np.where(lblareas > 1e5)[0]
        labels_large = list(labels_large)

        # label=0 is the background, so we remove it
        labels_large.remove(0)

        # only set pixels that belong to the large components
        seg = np.zeros(im_8.shape[0:2], dtype=np.uint8)
        for i in labels_large:
            seg[labels == i] = 255

    else:  # not OLD_CODE

        seg, im_8 = rough_foreground_mask(ndpi_file, downsample_factor=8.0, dilation_size=25,
                                          component_size_threshold=1e5, hole_size_treshold=0,
                                          return_im=True)
        seg *= 255

    # save segmentation as a tiff file (with ZLIB compression)
    outfilename = os.path.basename(ndpi_file)
    outfilename = os.path.splitext(outfilename)[0] + '_seg'
    outfilename = os.path.join(seg_dir, outfilename + '.tif')
    tifffile.imsave(outfilename, seg,
                    compress=9,
                    resolution=(int(im.properties["tiff.XResolution"]) / downsample_factor,
                                int(im.properties["tiff.YResolution"]) / downsample_factor,
                                im.properties["tiff.ResolutionUnit"].upper()))

    # plot the segmentation
    if DEBUG:
        plt.clf()
        plt.imshow(seg)
        plt.pause(.1)

   
    else:  # not OLD_CODE

        np.random.seed(i_file)
        sample_centroid = []
        while len(sample_centroid) < n_samples:
            row = randint(0, seg.shape[0] - 1)
            col = randint(0, seg.shape[1] - 1)
            # if the centroid is a pixel that belongs to tissue...

            if seg[row, col] != 0:
                # ... add it to the list of random samples
                sample_centroid.append((row, col))

    # compute the centroid in high-res
    sample_centroid_upsampled = []
    for row, col in sample_centroid:
        sample_centroid_upsampled.append(
            (int(row * downsample_factor + np.round((downsample_factor - 1) / 2)),
             int(col * downsample_factor + np.round((downsample_factor - 1) / 2)))
        )

    # create the training dataset by sampling the full resolution image with boxes around the centroids
    for j in range(len(sample_centroid_upsampled)):

        # high resolution centroid
        if OLD_CODE:
            # Note: this is a bug. The centroid used should have been the high resolution one
            row, col = sample_centroid[j]
        else:
            row, col = sample_centroid_upsampled[j]

        # compute top-left corner of the box from the centroid
        box_corner_row = row - box_half_size
        box_corner_col = col - box_half_size

        # extract tile from full resolution image
        tile = im.read_region(location=(box_corner_col, box_corner_row), level=0, size=(box_size, box_size))
        tile = np.array(tile)
        tile = tile[:, :, 0:3]

        # output filename
        outfilename = os.path.basename(ndpi_file)
        if OLD_CODE:
            outfilename = os.path.splitext(outfilename)[0] + '_row_' + str(row).zfill(6) \
                          + '_col_' + str(col).zfill(6)
        else:
            outfilename = os.path.splitext(outfilename)[0] + '_row_' + str(box_corner_row).zfill(6) \
                          + '_col_' + str(box_corner_col).zfill(6)
        outfilename = os.path.join(training_dir, outfilename + '.tif')

        # save tile as a tiff file with ZLIB compression (LZMA or ZSTD can't be opened by QuPath)
        print('Saving ' + outfilename)
        tifffile.imsave(outfilename, tile,
                        compress=9,
                        resolution=(int(im.properties["tiff.XResolution"]),
                                    int(im.properties["tiff.YResolution"]),
                                    im.properties["tiff.ResolutionUnit"].upper()))

        # plot tile
        if DEBUG:
            plt.clf()
            plt.subplot(121)
            plt.imshow(tile)
            plt.subplot(122)
            foo = tifffile.imread(outfilename)
            plt.imshow(foo)
            plt.pause(.1)

