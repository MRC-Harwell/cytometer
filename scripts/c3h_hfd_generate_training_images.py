import os
import openslide
import numpy as np
from statistics import mode
import matplotlib.pyplot as plt
import cv2
from random import randint, seed
import tifffile
import glob


DEBUG = False

data_dir = '/home/gcientanni/scan_srv2_cox/Maz Yon'
training_dir = '/home/gcientanni/Dropbox/c3h/test_2_training'
seg_dir = '/home/gcientanni/Dropbox/c3h/test_2_seg'
downsample_factor = 8.0

box_size = 1001
box_half_size = int((box_size - 1) / 2)
n_samples = 5

files_list = glob.glob(os.path.join(data_dir, 'C3H*.ndpi'))

for file_i, file in enumerate(files_list):


    print('File ' + str(file_i) + '/' + str(len(files_list)) + ': ' + file)

    # load file
    im = openslide.OpenSlide(os.path.join(data_dir, file))

    # level for a x8 downsample factor
    level_8 = im.get_best_level_for_downsample(downsample_factor)

    assert(im.level_downsamples[level_8] == downsample_factor)

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
    labels_large = np.where(lblareas > 5e5)[0]
    labels_large = list(labels_large)

    # label=0 is the background, so we remove it
    labels_large.remove(0)

    # only set pixels that belong to the large components
    seg = np.zeros(im_8.shape[0:2], dtype=np.uint8)
    for i in labels_large:
        seg[labels == i] = 255

    # save segmentation as a tiff file (with ZLIB compression)
    outfilename = os.path.basename(file)
    outfilename = os.path.splitext(outfilename)[0] + '_seg'
    outfilename = os.path.join(seg_dir, outfilename + '.tif')
    tifffile.imsave(outfilename, seg,
                    compress=9,
                    resolution=(int(im.properties["tiff.XResolution"]) / downsample_factor,
                                int(im.properties["tiff.YResolution"]) / downsample_factor,
                                im.properties["tiff.ResolutionUnit"].upper()))

    # plot the segmentation
    if DEBUG:
        plt.figure()
        plt.clf()
        plt.imshow(seg)
        plt.pause(.1)

    # pick random centroids that belong to one of the set pixels
    sample_centroid = []
    sample_centroid_upsampled = []
    seed(file_i)
    while len(sample_centroid) < n_samples:

        row = randint(0, seg.shape[0]-1)
        col = randint(0, seg.shape[1]-1)

        # if the centroid is a pixel that belongs to tissue...
        if seg[row, col] != 0:
            # ... add it to the list of random samples
            sample_centroid.append((row, col))
            # ... recompute the centroid's coordinates in the full-resolution image (approximately)
            sample_centroid_upsampled.append((int(row * downsample_factor + np.round((downsample_factor - 1) / 2)),
                                              int(col * downsample_factor + np.round((downsample_factor - 1) / 2))))

    # create the training dataset by sampling the full resolution image with boxes around the centroids
    for row, col in sample_centroid_upsampled:

        # compute from the centroid the top-left corner of the box
        box_corner_row = row - box_half_size
        box_corner_col = col - box_half_size
        tile = im.read_region(location=(box_corner_col, box_corner_row), level=0, size=(box_size, box_size))
        tile = np.array(tile)
        tile = tile[:, :, 0:3]

        # plot tile
        if DEBUG:
            plt.clf()
            plt.imshow(tile)
            plt.pause(.1)

        # save tile as a tiff file with ZLIB compression (LZMA or ZSTD can't be opened by QuPath)
        outfilename = os.path.basename(file)
        outfilename = os.path.splitext(outfilename)[0] + '_row_'+ str(row).zfill(6) + '_col_' + str(col).zfill(6)
        outfilename = os.path.join(training_dir, outfilename + '.tif')
        tifffile.imsave(outfilename, tile,
                        compress=9,
                        resolution=(int(im.properties["tiff.XResolution"]),
                                    int(im.properties["tiff.YResolution"]),
                                    im.properties["tiff.ResolutionUnit"].upper()))