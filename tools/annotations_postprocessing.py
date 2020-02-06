"""
This script aggregates all the cells in each '*_exp_0097_corrected.json' file and saves them to a
'*_exp_0097_refined.json', then creates a soft links to it that will be read by AIDA so that we can refine the
segmentation by hand.
"""

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import glob
import shutil
import cytometer.data
import pickle
import itertools
from shapely.geometry import Polygon
import PIL
import openslide
import numpy as np

# directories
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
annotations_dir = os.path.join(home, 'Software/AIDA/dist/data/annotations')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')
ndpi_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')

# k-folds file
saved_extra_kfolds_filename = 'klf14_b6ntac_exp_0094_generate_extra_training_images.pickle'

# list of annotations
annotation_files_list = os.path.join(annotations_dir, '*_exp_0097_corrected.json')
annotation_files_list = glob.glob(annotation_files_list)

# variables
overwrite_existing_refined_files = False

########################################################################################################################
## Colourmap for AIDA
########################################################################################################################

# load list of images, and indices for training vs. testing indices
saved_kfolds_filename = os.path.join(saved_models_dir, saved_extra_kfolds_filename)
with open(saved_kfolds_filename, 'rb') as f:
    aux = pickle.load(f)
file_svg_list = aux['file_list']
idx_test_all = aux['idx_test']
idx_train_all = aux['idx_train']

# correct home directory
file_svg_list = [x.replace('/users/rittscher/rcasero', home) for x in file_svg_list]
file_svg_list = [x.replace('/home/rcasero', home) for x in file_svg_list]

# loop files with hand traced contours
manual_areas_all = []
for i, file_svg in enumerate(file_svg_list):

    print('file ' + str(i) + '/' + str(len(file_svg_list) - 1) + ': ' + os.path.basename(file_svg))

    # change file extension from .svg to .tif
    file_tif = file_svg.replace('.svg', '.tif')

    # open histology training image
    im = PIL.Image.open(file_tif)

    # read pixel size information
    xres = 0.0254 / im.info['dpi'][0]  # m
    yres = 0.0254 / im.info['dpi'][1]  # m

    # read the ground truth cell contours in the SVG file. This produces a list [contour_0, ..., contour_N-1]
    # where each contour_i = [(X_0, Y_0), ..., (X_P-1, X_P-1)]
    contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False,
                                                       minimum_npoints=3)

    # compute cell area
    manual_areas_all.append([Polygon(c).area * xres * yres for c in contours])  # (um^2)

manual_areas_all = list(itertools.chain.from_iterable(manual_areas_all))

# compute function to map between cell areas and [0.0, 1.0], that we can use to sample the colourmap uniformly according
# to area quantiles
f_area2quantile = cytometer.data.area2quantile(manual_areas_all)

########################################################################################################################
## Process files linking to overlap segmentations
########################################################################################################################

# for annotation_file in annotation_files_list:
#
#     print('File: ' + os.path.basename(annotation_file))
#
#     # name of the soft link that will be read by AIDA
#     main_json_file = annotation_file.replace('_exp_0097_corrected.json', '.json')
#
#     # check that the target file is a softlink and not a file
#     if os.path.isfile(main_json_file) and not os.path.islink(main_json_file):
#         raise FileExistsError('The main .json file is a file and not a soft link')
#
#     # delete existing soft link
#     if os.path.islink(main_json_file):
#         os.remove(main_json_file)
#
#     # link to annotations file
#     os.symlink(os.path.basename(annotation_file), main_json_file)

########################################################################################################################
## Process files for segmentation refinement
########################################################################################################################

for annotation_file in annotation_files_list:

    print('File: ' + os.path.basename(annotation_file))

    # name of the soft link that will be read by AIDA
    main_json_file = annotation_file.replace('_exp_0097_corrected.json', '.json')

    # name of the file were we are going to have the corrected cells in one layer
    corrected_monolayer_file_left = \
        annotation_file.replace('_exp_0097_corrected.json', '_exp_0097_corrected_monolayer_left.json')
    corrected_monolayer_file_right = \
        annotation_file.replace('_exp_0097_corrected.json', '_exp_0097_corrected_monolayer_right.json')

    # name of the file were we are going to make the manual refinement
    refined_file_left = annotation_file.replace('_exp_0097_corrected.json', '_exp_0097_refined_left.json')
    refined_file_right = annotation_file.replace('_exp_0097_corrected.json', '_exp_0097_refined_right.json')

    # name of the original .ndpi file
    ndpi_file = os.path.basename(annotation_file).replace('_exp_0097_corrected.json', '.ndpi')
    ndpi_file = os.path.join(ndpi_dir, ndpi_file)

    im = openslide.OpenSlide(ndpi_file)
    xres = 1e-2 / float(im.properties['tiff.XResolution'])
    yres = 1e-2 / float(im.properties['tiff.YResolution'])

    # to save time, we don't overwrite the monolayer file if it exists already
    if not os.path.isfile(corrected_monolayer_file_left) or not os.path.isfile(corrected_monolayer_file_right):

        # load contours from annotation file
        cells = cytometer.data.aida_get_contours(annotation_file, layer_name='White adipocyte.*')

        # bounding box for all cells
        cells_array = np.vstack(cells)
        box_min = np.min(cells_array, axis=0)
        box_max = np.max(cells_array, axis=0)
        box_mid = (box_min + box_max) / 2.0
        del cells_array

        # splits cells into cells to the left and to the right of the middle of the bounding box
        cells_left = []
        cells_right = []
        for contour in cells:
            # the contour is assigned to the left side if all its cells are within the left side
            contour_on_left = np.all(np.array(contour)[:, 0] < box_mid[0])
            if contour_on_left:
                cells_left.append(contour)
            else:
                cells_right.append(contour)

        # create AIDA items to contain contours
        items_left = cytometer.data.aida_contour_items(cells_left, f_area2quantile, cm='quantiles_aida', xres=xres, yres=yres)
        items_right = cytometer.data.aida_contour_items(cells_right, f_area2quantile, cm='quantiles_aida', xres=xres, yres=yres)

        # write contours to single layer AIDA file (one to visualise, one to correct manually)
        cytometer.data.aida_write_new_items(corrected_monolayer_file_left, items_left, mode='w', indent=0)
        cytometer.data.aida_write_new_items(corrected_monolayer_file_right, items_right, mode='w', indent=0)

    # check that refinement file doesn't exist already, because we don't want to overwrite
    # a segmentation that we have refined by hand
    if overwrite_existing_refined_files or not os.path.isfile(refined_file_left):
        shutil.copy2(corrected_monolayer_file_left, refined_file_left)
    if overwrite_existing_refined_files or not os.path.isfile(refined_file_right):
        shutil.copy2(corrected_monolayer_file_right, refined_file_right)

########################################################################################################################
## Choose an annotations file for each image
########################################################################################################################

for annotation_file in annotation_files_list:

    print('File: ' + os.path.basename(annotation_file))

    # name of the soft link that will be read by AIDA
    main_json_file = annotation_file.replace('_exp_0097_corrected.json', '.json')

    # name of the file were we are going to have the corrected cells in one layer
    corrected_monolayer_file_left = \
        annotation_file.replace('_exp_0097_corrected.json', '_exp_0097_corrected_monolayer_left.json')
    corrected_monolayer_file_right = \
        annotation_file.replace('_exp_0097_corrected.json', '_exp_0097_corrected_monolayer_right.json')

    # name of the file were we are going to make the manual refinement
    refined_file_left = annotation_file.replace('_exp_0097_corrected.json', '_exp_0097_refined_left.json')
    refined_file_right = annotation_file.replace('_exp_0097_corrected.json', '_exp_0097_refined_right.json')

    # create the softlink
    if os.path.isfile(main_json_file) and not os.path.islink(main_json_file):
        raise FileExistsError('The main .json file is a file and not a soft link')

    # if the main soft link is already point at a file, delete it so that we can link again
    if os.path.islink(main_json_file):
        os.remove(main_json_file)

    # link to copied file
    os.symlink(refined_file_left, main_json_file)
