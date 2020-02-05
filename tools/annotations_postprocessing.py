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
import cytometer.data
import pickle
import itertools
from shapely.geometry import Polygon
import PIL
import openslide

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
delete_existing_refined_files = False

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

    # name of the file were we are going to make the manual refinement
    refined_file = annotation_file.replace('_exp_0097_corrected.json', '_exp_0097_refined.json')

    # name of the original .ndpi file
    ndpi_file = os.path.basename(annotation_file).replace('_exp_0097_corrected.json', '.ndpi')
    ndpi_file = os.path.join(ndpi_dir, ndpi_file)

    im = openslide.OpenSlide(ndpi_file)
    xres = 1e-2 / float(im.properties['tiff.XResolution'])
    yres = 1e-2 / float(im.properties['tiff.YResolution'])

    # delete existing refined file if we want to wipe out previous refined files
    if delete_existing_refined_files and os.path.isfile(refined_file):
        os.remove(refined_file)

    # check that refinement file doesn't exist already, because we don't want to overwrite
    # a segmentation that we have refined by hand
    if not os.path.isfile(refined_file):

        # load contours from annotation file
        cells = cytometer.data.aida_get_contours(annotation_file, layer_name='White adipocyte.*')

        # create AIDA items to contain contours
        items = cytometer.data.aida_contour_items(cells, f_area2quantile, cm='quantiles_aida', xres=xres, yres=yres)

        # write contours to single layer AIDA file
        cytometer.data.aida_write_new_items(refined_file, items, mode='w', indent=0)

    # create the softlink
    if os.path.isfile(main_json_file) and not os.path.islink(main_json_file):
        raise FileExistsError('The main .json file is a file and not a soft link')

    # if the main soft link is already point at a file, delete it so that we can link again
    if os.path.islink(main_json_file):
        os.remove(main_json_file)

    # link to copied file
    os.symlink(refined_file, main_json_file)
