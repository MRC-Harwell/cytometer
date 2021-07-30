"""
This script aggregates all the cells in each '*_exp_0001_[auto | corrected].json' file and saves them to a
'*_exp_0001_[auto | corrected]_aggregate.json', then creates a soft links to it that will be read by AIDA.
"""

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])

import glob
import cytometer.data
import openslide
import numpy as np
import shapely

histology_dir = os.path.join(home, 'jesse_mousedata/GTEx')
area2quantile_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v8')
annotations_dir = os.path.join(home, 'Data/cytometer_data/aida_data_GTEx/annotations')

# file with area->quantile map precomputed from all automatically segmented slides in klf14_b6ntac_exp_0098_full_slide_size_analysis_v7.py
filename_area2quantile = os.path.join(area2quantile_dir, 'klf14_b6ntac_exp_0098_filename_area2quantile.npz')

# suffixes of annotation filenames
auto_filename_suffix = '_exp_0001_auto.json'
corrected_filename_suffix = '_exp_0001_corrected.json'

# list of annotations
auto_annotation_files_list = os.path.join(annotations_dir, '*' + auto_filename_suffix)
auto_annotation_files_list = glob.glob(auto_annotation_files_list)
corrected_annotation_files_list = os.path.join(annotations_dir, '*' + corrected_filename_suffix)
corrected_annotation_files_list = glob.glob(corrected_annotation_files_list)

# filtering parameters
min_area = 200  # um^2
max_area = 160000  # um^2
cell_prob_thr = 0.5  # threshold for objects to be accepted as cells
max_inv_compactness = 2.0  # objects less compact than this are rejected (= more compact^-1)

########################################################################################################################
## Colourmap for AIDA
########################################################################################################################

if os.path.isfile(filename_area2quantile):
    with np.load(filename_area2quantile, allow_pickle=True) as aux:
        f_area2quantile_f = aux['f_area2quantile_f'].item()
        f_area2quantile_m = aux['f_area2quantile_m'].item()
else:
    raise FileNotFoundError('Cannot find file with area->quantile map precomputed from all automatically segmented' +
                            ' slides in klf14_b6ntac_exp_0098_full_slide_size_analysis_v7.py')

# load AIDA's colourmap
cm = cytometer.data.aida_colourmap()

########################################################################################################################
## Process files for segmentation refinement
########################################################################################################################

def process_annotations(annotation_files_list, overwrite_aggregated_annotation_file=False, create_symlink=False):
    """
    Helper function to process a list of JSON files with annotations.
    :param annotation_files_list: list of JSON filenames containing annotations.
    :return:
    """

    for annotation_file in annotation_files_list:

        print('File: ' + os.path.basename(annotation_file))

        # name of the file that we are going to save the aggregated annotations to
        aggregated_annotation_file = annotation_file.replace('.json', '_aggregated.json')

        # name of the original histo file
        histo_file = os.path.basename(annotation_file).replace(auto_filename_suffix, '.svs')
        histo_file = histo_file.replace(corrected_filename_suffix, '.svs')
        histo_file = os.path.join(histology_dir, histo_file)

        im = openslide.OpenSlide(histo_file)
        xres = float(im.properties['aperio.MPP'])  # um/pixel
        yres = float(im.properties['aperio.MPP'])  # um/pixel

        # aggregate cells from all blocks and write/overwrite a file with them
        if not os.path.isfile(aggregated_annotation_file) or overwrite_aggregated_annotation_file:

            # load contours from annotation file
            cells, props = cytometer.data.aida_get_contours(annotation_file, layer_name='White adipocyte.*', return_props=True)

            # compute cell measures
            areas = []
            inv_compactnesses = []
            for cell in cells:
                poly_cell = shapely.geometry.Polygon(cell)
                area = poly_cell.area
                if area > 0:
                    inv_compactness = poly_cell.length ** 2 / (4 * np.pi * area)
                else:
                    inv_compactness = np.nan
                areas.append(area)
                inv_compactnesses.append(inv_compactness)

            # prepare for removal objects that are too large or too small
            idx = (np.array(areas) >= min_area) * (np.array(areas) <= max_area)

            # prepare for removal objects that are not compact enough
            idx *= np.array(inv_compactnesses) <= max_inv_compactness

            # prepare for removal objects unlikely to be cells
            idx *= np.array(props['cell_prob']) >= cell_prob_thr

            # execute the removal of objects
            cells = list(np.array(cells, dtype='object')[idx])
            props['cell_prob'] = list(np.array(props['cell_prob'], dtype='object')[idx])
            # areas = list(np.array(areas)[idx])

            # create AIDA items to contain contours
            items = cytometer.data.aida_contour_items(cells, f_area2quantile_m, cm='quantiles_aida',
                                                      xres=xres, yres=yres)

            # write contours to single layer AIDA file (one to visualise, one to correct manually)
            cytometer.data.aida_write_new_items(aggregated_annotation_file, items, mode='w', indent=0)

        if create_symlink:

            # name expected by AIDA for annotations
            symlink_name = os.path.basename(histo_file).replace('.svs', '.json')
            symlink_name = os.path.join(annotations_dir, symlink_name)

            # create symlink to the aggregated annotation file from the name expected by AIDA
            if os.path.isfile(symlink_name):
                if os.path.islink(symlink_name):
                    # delete existing symlink
                    os.remove(symlink_name)
                else:
                    raise FileExistsError('File found with the name of the symlink we are trying to create')
            else:
                os.symlink(os.path.basename(aggregated_annotation_file), symlink_name)

# create aggreagated annotation files for auto segmentations, and link to them
#process_annotations(auto_annotation_files_list, overwrite_aggregated_annotation_file=True, create_symlink=True)
process_annotations(corrected_annotation_files_list, overwrite_aggregated_annotation_file=False, create_symlink=True)
