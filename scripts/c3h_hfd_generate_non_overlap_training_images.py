# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os

import glob
import matplotlib.pyplot as plt
import numpy as np
from svgpathtools import svg2paths
from PIL import Image
import PIL.ImageDraw
from PIL.TiffTags import TAGS
import tifffile
from cv2 import watershed


DEBUG = True

root_data_dir = os.path.join(home, 'Dropbox/klf14')
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_nooverlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')

# extract contour as a list of (X,Y) coordinates
def extract_contour(path, x_res=1.0, y_res=1.0):

    contour = []
    for pt in path:

        # (X, Y) for each point
        contour.append((np.real(pt.start) * x_res, np.imag(pt.start) * y_res))

        if DEBUG:
            plt.plot(*zip(*contour))

    return contour


# extract contours that correspond to non-edge cells in SVG file as list of polygons with (X,Y)-vertices
def extract_cell_contours_from_file(file, x_res=1.0, y_res=1.0):

    # extract all paths from the SVG file
    paths, attributes = svg2paths(file)

    # loop paths
    paths_out = []
    for path, attribute in zip(paths, attributes):

        # skip if the countour is not a cell (we also skip edge cells, as they are incomplete, and thus their area
        # is misleading)
        if not attribute['id'].startswith('Cell'):
            continue

        # extract contour polygon from the path object, and compute area
        contour = extract_contour(path, x_res=x_res, y_res=y_res)
        paths_out.append(contour)

    return paths_out


# we are interested only in .tif files for which we created hand segmented contours
file_list = glob.glob(os.path.join(training_data_dir, '*.svg'))

for n, file_svg in enumerate(file_list):

    print('file ' + str(n) + '/' + str(len(file_list)-1))

    # change file extension from .svg to .tif
    file_tif = file_svg.replace('.svg', '.tif')

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
    polygon = extract_cell_contours_from_file(file_svg)

    # if there are no cells, we skip to the next file
    if len(polygon) == 0:
        continue

    # loop cells
    for i, pg in enumerate(polygon):

        # create empty arrays with the same size as image
        cell_mask = PIL.Image.new("1", im.size, "black")  # I = 32-bit signed integer pixels
        draw = PIL.ImageDraw.Draw(cell_mask)

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

    # label non-cell areas
    background_label = i + 2
    labels[labels == 0] = background_label

    # label overlapping areas
    labels[cell_count > 1] = 0

    # apply watershed algorithm to fill in overlap areas. The "image" entry is an array of zeros, so that the boundaries
    # extend uniformly without paying attention to the original histology image (otherwise, the boundaries will be
    # crooked)
    labels = watershed(np.zeros(im.size[::-1] + (3,), dtype=np.uint8), np.array(labels))

    # set background pixels back to zero
    labels[labels == background_label] = 0

    if DEBUG:
        plt.subplot(224)
        plt.imshow(labels)

    ## save segmentation results
    file_tif_out = file_tif.replace(training_data_dir, training_nooverlap_data_dir)

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

