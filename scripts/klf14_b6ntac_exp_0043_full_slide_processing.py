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

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

data_dir = os.path.join(home, 'Data/cytometer_data/klf14/Maz Yon')
training_dir = os.path.join(home, 'Data/cytometer_data/klf14/klf14_b6ntac_training')
seg_dir = os.path.join(home, 'Data/cytometer_data/klf14/klf14_b6ntac_seg')
downsample_factor = 8.0

box_size = 1001
box_half_size = int((box_size - 1) / 2)
n_samples = 5

files_list = glob.glob(os.path.join(data_dir, '*.ndpi'))

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
    labels_large = np.where(lblareas > 1e5)[0]
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
