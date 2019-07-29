"""
Read full .ndpi slides, rough segmentation of tissue areas, random selection of centroids, extract
1001x1001 windows around centroids.

The windows are saved with row_R_col_C, where R, C are the row, col centroid of the image. You can get the offset of the
image from the centroid as offset = centroid - box_half_size = centroid - 500.

We include two version of the code:
    * OLD_CODE=True: This is how things were done for the KLF14 training dataset. There's a bug because instead of using
      the high resolution centroid we used the low resolution centroid. This still created a valid training dataset,
      though.
    * OLD_CODE=False: This is how we would do things in future.
"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0076_generate_training_images'

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

# use the old version of the code that was used for the KLF14 experiments. The new function rough_foreground_mask()
# produces a similar result, but because there are a few different pixels in "seg", the randomly selected training
# windows would be different.
OLD_CODE = True

box_size = 1001
box_half_size = int((box_size - 1) / 2)
n_samples = 5

# explicit list of files, to avoid differences if the files in the directory change (note that the order of the
# files is important, because it's used for the random seed when selecting training windows)
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

for i_file, ndpi_file in enumerate(ndpi_files_list):

    print('File ' + str(i_file) + '/' + str(len(ndpi_files_list)) + ': ' + ndpi_file)

    if OLD_CODE:  # this is the code that was used for the KLF14 experiments

        # load file
        im = openslide.OpenSlide(os.path.join(ndpi_dir, ndpi_file))

        # level for a x8 downsample factor
        level_8 = im.get_best_level_for_downsample(downsample_factor)

        assert(im.level_downsamples[level_8] == downsample_factor)

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

    # these are the low-res centroids within the segmentation masks that were randomly selected in the old script
    if OLD_CODE:
        if os.path.basename(ndpi_file) == 'KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04.ndpi':
            sample_centroid = [(10716, 8924), (19676, 18716), (20860, 64580),
                               (22276, 15492), (26108, 68956)]
        elif os.path.basename(ndpi_file) == 'KLF14-B6NTAC-MAT-18.2d  60-16 C1 - 2016-02-03 13.13.57.ndpi':
            sample_centroid = [(6788, 5516), (13860, 10964), (13980, 4956),
                               (18380, 63068), (24076, 20404)]
        elif os.path.basename(ndpi_file) == 'KLF14-B6NTAC-MAT-18.2b  58-16 C1 - 2016-02-03 11.10.52.ndpi':
            sample_centroid = [(4852, 58628), (6124, 58212), (6268, 13820),
                               (13820, 57052), (14220, 60412)]
        elif os.path.basename(ndpi_file) == 'KLF14-B6NTAC-MAT-17.2c  66-16 C1 - 2016-02-04 11.46.39.ndpi':
            sample_centroid = [(10372, 21228), (22956, 57156), (25644, 26132),
                               (30820, 22204), (33004, 61228)]
        elif os.path.basename(ndpi_file) == 'KLF14-B6NTAC-MAT-17.2f  68-16 C1 - 2016-02-04 15.05.54.ndpi':
            sample_centroid = [(7500, 50372), (11452, 16252), (13348, 19316),
                               (14332, 19564), (26132, 12148)]
        elif os.path.basename(ndpi_file) == 'KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38.ndpi':
            sample_centroid = [(7676, 28900), (9588, 28676), (12476, 31852),
                               (13804, 8004), (21796, 55852)]
        elif os.path.basename(ndpi_file) == 'KLF14-B6NTAC-MAT-18.1a  50-16 C1 - 2016-02-02 09.12.41.ndpi':
            sample_centroid = [(4324, 55084), (6124, 59516), (12908, 10212),
                               (24276, 18652), (28156, 18596)]
        elif os.path.basename(ndpi_file) == 'KLF14-B6NTAC-MAT-18.2g  63-16 C1 - 2016-02-03 16.58.52.ndpi':
            sample_centroid = []
        elif os.path.basename(ndpi_file) == 'KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46.ndpi':
            sample_centroid = [(4124, 12524), (8532, 9804), (17044, 31228),
                               (18852, 29164), (21804, 35412)]
        elif os.path.basename(ndpi_file) == 'KLF14-B6NTAC-37.1d PAT 109-16 C1 - 2016-02-15 15.19.08.ndpi':
            sample_centroid = [(11684, 51748), (12172, 49588), (12700, 52388),
                               (16068, 7276), (20380, 19420)]
        elif os.path.basename(ndpi_file) == 'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33.ndpi':
            sample_centroid = [(9236, 18316), (16540, 18252), (16756, 63692),
                               (24836, 55124), (29564, 19260)]
        elif os.path.basename(ndpi_file) == 'KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38.ndpi':
            sample_centroid = [(5348, 19844), (6548, 15236), (6652, 61724),
                               (6900, 71980), (10356, 31316), (10732, 16692),
                               (12828, 18388), (14980, 27052), (18780, 69468),
                               (27388, 18468)]
        elif os.path.basename(ndpi_file) == 'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53.ndpi':
            sample_centroid = [(8212, 15364), (11004, 5988), (15004, 10364),
                               (19556, 57972), (21812, 22916)]
        elif os.path.basename(ndpi_file) == 'KLF14-B6NTAC-37.1c PAT 108-16 C1 - 2016-02-15 14.49.45.ndpi':
            sample_centroid = [(1292, 4348), (7372, 8556), (12588, 53084),
                               (12732, 39812), (16612, 37372)]
        elif os.path.basename(ndpi_file) == 'KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52.ndpi':
            sample_centroid = [(5428, 58372), (12404, 54316), (13604, 24644),
                               (14628, 69148), (19340, 17348)]
        elif os.path.basename(ndpi_file) == 'KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32.ndpi':
            sample_centroid = [(6412, 12484), (13012, 19820), (31172, 25996),
                               (34628, 40116), (35948, 41492)]
        elif os.path.basename(ndpi_file) == 'KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11.ndpi':
            sample_centroid = [(10084, 58476), (16260, 58300), (19220, 61724),
                               (21012, 57844), (23236, 11084)]
        elif os.path.basename(ndpi_file) == 'KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06.ndpi':
            sample_centroid = [(9644, 61660), (11852, 71620), (13300, 55476),
                               (16332, 65356), (17204, 19444)]
        elif os.path.basename(ndpi_file) == 'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52.ndpi':
            sample_centroid = [(6124, 82236), (7436, 19092), (16556, 10292),
                               (23100, 9220), (31860, 33476)]
        elif os.path.basename(ndpi_file) == 'KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00.ndpi':
            sample_centroid = [(2772, 5332), (7124, 13028), (16812, 17484),
                               (19228, 15060), (29684, 15172)]
        else:
            raise ValueError('Unknown NDPI file:' + ndpi_file)

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

        # plot tile
        if DEBUG:
            plt.clf()
            plt.subplot(121)
            plt.imshow(tile)
            plt.subplot(122)
            foo = tifffile.imread(outfilename)
            plt.imshow(foo)
            plt.pause(.1)

        # save tile as a tiff file with ZLIB compression (LZMA or ZSTD can't be opened by QuPath)
        tifffile.imsave(outfilename, tile,
                        compress=9,
                        resolution=(int(im.properties["tiff.XResolution"]),
                                    int(im.properties["tiff.YResolution"]),
                                    im.properties["tiff.ResolutionUnit"].upper()))
