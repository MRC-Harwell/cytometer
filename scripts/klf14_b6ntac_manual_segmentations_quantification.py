import os
import glob
import numpy as np
from svgpathtools import svg2paths
import matplotlib.pyplot as plt
import openslide
import csv
from enum import Enum


DEBUG = False

image_data_dir = '/home/rcasero/data/roger_data'
root_data_dir = '/home/rcasero/Dropbox/klf14'
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')


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

        if DEBUG:
            plt.plot(*zip(*contour))

    return contour

# extract the polygon from the path object, and compute polygon area
def extract_cell_contour_and_compute_area(file, x_res=1.0, y_res=1.0):

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

## get pixel size in original images, so that we can compute areas in um^2, instead of pixel^2
## Warning: We are assuming that all images have the same resolution, so we only do this once


# # file = '/path/to/file/KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_row_015004_col_010364.svg'
# # original_file = 'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_row_015004_col_010364.svg'
# original_file = os.path.basename(file)
#
# # original_file = 'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53'
# original_file = original_file.rsplit(sep='_row_')[0]
#
# # original_file = '/home/rcasero/data/roger_data/KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53.ndpi'
# original_file = os.path.join(image_data_dir, original_file + '.ndpi')

original_file = os.path.join(root_data_dir, 'KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38.ndpi')

# open original image
im = openslide.OpenSlide(original_file)

# compute pixel size in the original image, where patches were taken from
if im.properties['tiff.ResolutionUnit'].lower() == 'centimeter':
    x_res = 1e-2 / float(im.properties['tiff.XResolution']) # meters / pixel
    y_res = 1e-2 / float(im.properties['tiff.YResolution']) # meters / pixel
else:
    raise ValueError('Only centimeter units implemented')

## read CSV file with female/male labels for mice

with open(os.path.join(root_data_dir, 'klf14_b6ntac_sex_info.csv'), 'r') as f:
    reader = csv.DictReader(f, skipinitialspace=True)
    klf14_info = []
    for row in reader:
        klf14_info.append(row)
f.close()

## read all contour files, and categorise them into MAT/PAT and f/m

# list of mouse IDs
klf14_ids = [x['id'] for x in klf14_info]

cell_areas = {'f': {'MAT': np.empty(0), 'PAT': np.empty(0)},
              'm': {'MAT': np.empty(0), 'PAT': np.empty(0)}}

file_list = glob.glob(os.path.join(training_data_dir, '*.svg'))
for file in file_list:

    # get mouse ID from the file name
    mouse_id = None
    for x in klf14_ids:
        if x in os.path.basename(file):
            mouse_id = x
            break
    if mouse_id is None:
        raise ValueError('Filename does not seem to correspond to any known mouse ID: ' + file)

    # index of mouse ID
    idx = klf14_ids.index(mouse_id)

    # sex and KO-side for this mouse
    mouse_sex = klf14_info[idx]['sex']
    mouse_ko  = klf14_info[idx]['ko']

    # append values of cell areas to corresponding categories
    cell_areas[mouse_sex][mouse_ko] = np.append(cell_areas[mouse_sex][mouse_ko],
                                                extract_cell_contour_and_compute_area(file, x_res=x_res, y_res=y_res))

# compute mean and std of cell size in um^2
F = 0
M = 1
MAT = 0
PAT = 1

cell_areas_mean = np.zeros((2, 2), dtype=np.float32)
cell_areas_std = np.zeros((2, 2), dtype=np.float32)

cell_areas_mean[F, MAT] = np.mean(cell_areas['f']['MAT']) * 1e12
cell_areas_mean[F, PAT] = np.mean(cell_areas['f']['PAT']) * 1e12
cell_areas_mean[M, MAT] = np.mean(cell_areas['m']['MAT']) * 1e12
cell_areas_mean[M, PAT] = np.mean(cell_areas['m']['PAT']) * 1e12

cell_areas_std[F, MAT] = np.std(cell_areas['f']['MAT']) * 1e12
cell_areas_std[F, PAT] = np.std(cell_areas['f']['PAT']) * 1e12
cell_areas_std[M, MAT] = np.std(cell_areas['m']['MAT']) * 1e12
cell_areas_std[M, PAT] = np.std(cell_areas['m']['PAT']) * 1e12

# plot results
width = 0.35
ind = np.arange(2)
fig, ax = plt.subplots()
rects_mat = ax.bar(ind, cell_areas_mean[:, MAT], width=width, color='r',
                 yerr=cell_areas_std[:, MAT])
rects_pat = ax.bar(ind + width, cell_areas_mean[:, PAT], width=width, color='g',
                 yerr=cell_areas_mean[:, PAT])
ax.set_xlabel('Sex')
ax.set_ylabel('Area (um^2)')
ax.set_title('Cell area by sex and KO side')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('f', 'm'))
ax.legend((rects_mat[0], rects_pat[0]), ('MAT', 'PAT'), loc='upper left')
