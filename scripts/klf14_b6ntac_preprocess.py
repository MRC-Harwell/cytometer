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

data_dir = '/home/rcasero/data/roger_data'
training_dir = '/home/rcasero/Software/cytometer/data/klf14_b6ntac_training'
seg_dir = '/home/rcasero/Software/cytometer/data/klf14_b6ntac_seg'
downsample_factor = 4.0
sample_size = 301
sample_half_size = int((sample_size - 1) / 2)
n_samples = 5

files_list = glob.glob(os.path.join(data_dir, '*.ndpi'))

seed(0)
for file_i, file in enumerate(files_list):

    print('File ' + str(file_i) + '/' + str(len(files_list)) + ': ' + file)

    # load file
    im = openslide.OpenSlide(os.path.join(data_dir, file))

    # level for a x4 downsample factor
    level_4 = im.get_best_level_for_downsample(downsample_factor)

    assert(im.level_downsamples[level_4] == downsample_factor)

    # get downsampled image
    im_4 = im.read_region(location=(0, 0), level=level_4, size=im.level_dimensions[level_4])
    im_4 = np.array(im_4)
    im_4 = im_4[:, :, 0:3]

    if DEBUG:
        plt.imshow(im_4)
        plt.pause(.1)

    # reshape image to matrix with one column per colour channel
    im_4_mat = im_4.copy()
    im_4_mat = im_4_mat.reshape((im_4_mat.shape[0] * im_4_mat.shape[1], im_4_mat.shape[2]))

    # background colour
    background_colour = []
    for i in range(3):
        background_colour += [mode(im_4_mat[:, i]), ]
    background_colour_std = np.std(im_4_mat, axis=0)

    # threshold segmentation
    seg = np.ones(im_4.shape[0:2], dtype=bool)
    for i in range(3):
        seg = np.logical_and(seg, im_4[:, :, i] < background_colour[i] - background_colour_std[i])
    seg = seg.astype(dtype=np.uint8)
    seg[seg == 1] = 255

    # dilate the segmentation to fill gaps within tissue
    kernel = np.ones((50, 50), np.uint8)
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
    seg = np.zeros(im_4.shape[0:2], dtype=np.uint8)
    for i in labels_large:
        seg[labels == i] = 255

    # save segmentation as a tiff file
    outfilename = os.path.basename(file)
    outfilename = os.path.splitext(outfilename)[0] + '_seg'
    outfilename = os.path.join(seg_dir, outfilename + '.tif')
    tifffile.imsave(outfilename, seg,
                    compress='LZMA',
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
    while len(sample_centroid) < n_samples:
        row = randint(sample_half_size+1, seg.shape[0])
        col = randint(sample_half_size+1, seg.shape[1])
        # if the centroid is a pixel that belongs to tissue...
        if seg[row, col] != 0:
            # ... add it to the list of random samples
            sample_centroid.append((row, col))

    # create the training dataset by sampling the images with boxes around the centroids
    for row, col in sample_centroid:
        # extract the sample of the image
        tile = im_4[row-sample_half_size:row+sample_half_size+1, col-sample_half_size:col+sample_half_size+1]

        # save tile as a tiff file
        outfilename = os.path.basename(file)
        outfilename = os.path.splitext(outfilename)[0] + '_row_'+ str(row).zfill(6) + '_col_' + str(col).zfill(6)
        outfilename = os.path.join(training_dir, outfilename + '.tif')
        tifffile.imsave(outfilename, tile,
                        compress='LZMA',
                        resolution=(int(im.properties["tiff.XResolution"]) / downsample_factor,
                                    int(im.properties["tiff.YResolution"]) / downsample_factor,
                                    im.properties["tiff.ResolutionUnit"].upper()))
