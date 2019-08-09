"""
Rasterise hand traced contours as polygons, and split overlapping areas between contributing cells using watershed.
This script creates the data in klf14_b6ntac_training_non_overlap.
"""

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os

import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from PIL.TiffTags import TAGS
import numpy as np
import cytometer.data
from cv2 import watershed
import mahotas
import tifffile

DEBUG = False

root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_nooverlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')

# we are interested only in .tif files for which we created hand segmented contours
file_list = glob.glob(os.path.join(training_data_dir, '*.svg'))

for n, file_svg in enumerate(file_list):

    print('file ' + str(n) + '/' + str(len(file_list)-1))

    # change file extension from .svg to .tif
    file_tif = file_svg.replace('.svg', '.tif')

    # name of output file
    file_tif_out = file_tif.replace(training_data_dir, training_nooverlap_data_dir)

    # skip if file has been processed already
    if os.path.isfile(file_tif_out):
        print('Skipping... file already processed')
        continue
    else:
        print('Processing file')

    # load image
    im = Image.open(file_tif)

    if DEBUG:
        plt.clf()
        plt.subplot(221)
        plt.imshow(im)

    # pixels for each cell will be assigned a different label
    labels = np.zeros(im.size[::-1], dtype=np.int32)

    # here we count how many cells each pixel belongs to
    cell_count = np.zeros(im.size[::-1], dtype=np.int32)

    # extract contours
    polygon = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell')

    # loop cells
    for i, pg in enumerate(polygon):

        # create empty arrays with the same size as image
        cell_mask = Image.new("1", im.size, "black")  # I = 32-bit signed integer pixels
        draw = ImageDraw.Draw(cell_mask)

        # rasterize current cell
        draw.polygon(pg, outline="white", fill="white")

        # assign a label to the pixels in this cell
        labels[cell_mask] = i + 1

        # increase the counter for each pixel that belongs to this cell
        cell_count += cell_mask

        # wipe out the cell mask because otherwise the next polygon will be added to it
        cell_mask.close()

    if DEBUG:
        plt.subplot(222)
        plt.imshow(labels)
        plt.subplot(223)
        plt.imshow(cell_count)

    ## find compromise for overlapping areas

    if len(polygon) > 0:

        # label non-cell areas
        background_label = i + 2
        labels[labels == 0] = background_label

        # label overlapping areas
        labels[cell_count > 1] = 0

        # apply watershed algorithm to fill in overlap areas. The "image" entry is an array of zeros, so that the
        # boundaries extend uniformly without paying attention to the original histology image (otherwise, the
        # boundaries will be crooked)
        labels = watershed(np.zeros(im.size[::-1] + (3,), dtype=np.uint8), np.array(labels))

        # set background pixels back to zero
        labels[labels == background_label] = 0

        # compute borders between labels, because in some cases, adjacent cells have intermittent overlaps
        # that produce interrupted 0 boundaries
        borders = mahotas.labeled.borders(labels, mode='ignore')

        # add borders to the labels
        labels[borders] = 0

    if DEBUG:
        plt.subplot(224)
        plt.imshow(labels)

    ## save segmentation results

    # TIFF tag codes
    xresolution_tag = 282
    yresolution_tag = 283
    resolution_unit = 296

    assert(TAGS[xresolution_tag] == 'XResolution')
    assert(TAGS[yresolution_tag] == 'YResolution')
    assert(TAGS[resolution_unit] == 'ResolutionUnit')

    # make label values positive
    labels += 1

    tifffile.imsave(file_tif_out, np.reshape(labels.astype(np.uint8), newshape=(1,)+labels.shape),
                    compress=9,
                    resolution=(im.tag[xresolution_tag][0][0],
                                im.tag[yresolution_tag][0][0],
                                im.tag[resolution_unit][0]))

