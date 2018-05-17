import os
import glob
import numpy as np
from svgpathtools import svg2paths
import matplotlib.pyplot as plt
import openslide


image_data_dir = '/home/rcasero/data/roger_data'
training_data_dir = '/home/rcasero/Software/cytometer/data/klf14_b6ntac_training'

mat_file_list = glob.glob(os.path.join(training_data_dir, '*-MAT-*.svg'))
pat_file_list = glob.glob(os.path.join(training_data_dir, '*-PAT-*.svg'))


# Area of Polygon using Shoelace formula
# http://en.wikipedia.org/wiki/Shoelace_formula
# FB - 20120218
# corners must be ordered in clockwise or counter-clockwise direction
def polygon_area(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

# extract contour as a list of (X,Y) coordinates
def extract_contour(path, x_res=1.0, y_res=1.0):

    contour = []
    for pt in path:

        # (X, Y) for each point
        contour.append((np.real(pt.start) * x_res, np.imag(pt.start) * y_res))

        plt.plot(*zip(*contour))

    return contour

# extract the polygon from the path object, and compute polygon area
def extract_contour_and_compute_area(file, x_res=1.0, y_res=1.0):

    # extract all paths from the SVG file
    paths, attributes = svg2paths(file)

    # loop paths
    areas = []
    for path, attribute in zip(paths, attributes):

        # skip if the countour is not a cell (we also skip edge cells, as they are incomplete, and thus their area
        # is misleading)
        if not attribute['id'].startswith('Cell'):
            continue

        # extract contour polygon from the path object, and compute area
        contour = extract_contour(path, x_res=x_res, y_res=y_res)
        areas.append(polygon_area(contour))

    return np.array(areas)

##################################################################################################################
# main programme
##################################################################################################################

for file in mat_file_list:

    ## get pixel size in original image, so that we can compute areas in um^2, instead of pixel^2

    # file = '/path/to/file/KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_row_015004_col_010364.svg'
    # original_file = 'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_row_015004_col_010364.svg'
    original_file = os.path.basename(file)

    # original_file = 'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53'
    original_file = original_file.rsplit(sep='_row_')[0]

    # original_file = '/home/rcasero/data/roger_data/KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53.ndpi'
    original_file = os.path.join(image_data_dir, original_file + '.ndpi')

    # open original image
    im = openslide.OpenSlide(original_file)

    # compute pixel size in the original image, where patches were taken from
    if im.properties['tiff.ResolutionUnit'].lower() = 'centimeter':
        x_res = 1e-2 / float(im.properties['tiff.XResolution']) # meters / pixel
        y_res = 1e-2 / float(im.properties['tiff.YResolution']) # meters / pixel
    else:
        raise ValueError('Only centimeter units implemented')

    areas = extract_contour_and_compute_area(file, x_res=x_res, y_res=y_res)