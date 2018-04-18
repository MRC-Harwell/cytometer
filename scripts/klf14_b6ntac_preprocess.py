import os
import openslide
import numpy as np
from statistics import mode
import matplotlib.pyplot as plt
import cv2
import PIL.Image
from random import randint


data_dir = '/home/rcasero/data/roger_data'
sample_size = 301
sample_half_size = int((sample_size - 1) / 2)
n_samples = 10

for file in os.listdir(data_dir):

    # load file
    im = openslide.OpenSlide(os.path.join(data_dir, file))

    # level for a x4 downsample factor
    level_4 = im.get_best_level_for_downsample(4)

    assert(im.level_downsamples[level_4] == 4.0)

    # get downsampled image
    im_4 = im.read_region(location=(0, 0), level=level_4, size=im.level_dimensions[level_4])
    im_4 = np.array(im_4)
    im_4 = im_4[:, :, 0:3]

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

    # dilate the segmentation
    kernel = np.ones((20, 20), np.uint8)
    seg = cv2.dilate(seg, kernel, iterations=1)

    # find connected components
    labels = seg.copy()
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(seg)
    lblareas = stats[:, cv2.CC_STAT_AREA]

    # labels of large components, that we assume correspond to tissue areas (label=0 is the background)
    labels_large = np.where(lblareas > 1e5)[0]
    labels_large = list(labels_large)
    labels_large.remove(0)

    # only set pixels that belong to the large components
    seg = np.zeros(im_4.shape[0:2], dtype=np.uint8)
    for i in labels_large:
        seg[labels == i] = 255

    PIL.Image.fromarray(seg).save('/tmp/foo2.tif')

    # pick random centroids that belong to one of the set pixels
    sample_centroid = []
    while (len(sample_centroid) < n_samples):
        r = randint(sample_half_size+1, seg.shape[0])

    plt.figure()
    plt.clf()
    plt.imshow(seg)
    plt.pause(.1)

    plt.figure()
    plt.imshow(output)
    plt.pause(.1)
