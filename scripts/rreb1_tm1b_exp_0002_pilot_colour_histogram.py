"""
Compare the median colour histogram from the KLF14 training data to the new RREB1-TM1B data's colour. We found that the
new data has a different tint.

This is where we create the file with the median, Q1 and Q3 curves for the colour histograms of the KLF14 training data
that we use to correct the histology colour before passing it to DeepCytometer.
"""

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

# load all the training images

# script name to identify this experiment
experiment_id = 'rreb1_tm1b_exp_0002_colour_histogram'
print('Experiment ID: ' + experiment_id)

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])

# other imports
import openslide
import PIL
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import time

# # limit number of GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import cytometer.utils
import cytometer.data

DEBUG = False

# rough_foreground_mask() parameters
downsample_factor = 8.0

# full resolution image window and network expected receptive field parameters
fullres_box_size = np.array([2751, 2751])
receptive_field = np.array([131, 131])

'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v8')
annotations_dir = os.path.join(home, 'Data/cytometer_data/aida_data_Rreb1_tm1b/annotations')
rreb1_data_dir = os.path.join(home, 'scan_srv2_cox/Liz Bentley/Grace/RREB1 Feb19')

# saved_kfolds_filename = 'klf14_b6ntac_exp_0079_generate_kfolds.pickle'
saved_extra_kfolds_filename = 'klf14_b6ntac_exp_0094_generate_extra_training_images.pickle'

# list of new data NDPI files to process
ndpi_files_list = [
    'RREB1-TM1B-B6N-IC-1.1a 1132-18 G1 - 2019-02-20 09.56.50.ndpi'
]

# load colour histograms
klf14_training_colour_histogram_file = os.path.join(saved_models_dir, 'klf14_training_colour_histogram.npz')
with np.load(klf14_training_colour_histogram_file) as aux:
    xbins_edge = aux['xbins_edge']
    xbins = aux['xbins']
    hist_r_q1 = aux['hist_r_q1']
    hist_r_q2 = aux['hist_r_q2']
    hist_r_q3 = aux['hist_r_q3']
    hist_g_q1 = aux['hist_g_q1']
    hist_g_q2 = aux['hist_g_q2']
    hist_g_q3 = aux['hist_g_q3']
    hist_b_q1 = aux['hist_b_q1']
    hist_b_q2 = aux['hist_b_q2']
    hist_b_q3 = aux['hist_b_q3']
    mode_r = aux['mode_r']
    mode_g = aux['mode_g']
    mode_b = aux['mode_b']

# load the NDPI file
i_file = 0
ndpi_file = ndpi_files_list[i_file]

print('File ' + str(i_file) + '/' + str(len(ndpi_files_list) - 1) + ': ' + ndpi_file)

# make full path to ndpi file
ndpi_file = os.path.join(rreb1_data_dir, ndpi_file)

# name of file to save rough mask, current mask, and time steps
rough_mask_file = os.path.basename(ndpi_file)
rough_mask_file = rough_mask_file.replace('.ndpi', '_rough_mask.npz')
rough_mask_file = os.path.join(annotations_dir, rough_mask_file)

# open full resolution histology slide
im = openslide.OpenSlide(ndpi_file)

# pixel size
assert(im.properties['tiff.ResolutionUnit'] == 'centimeter')
xres = 1e-2 / float(im.properties['tiff.XResolution'])
yres = 1e-2 / float(im.properties['tiff.YResolution'])

# load coarse tissue mask
with np.load(rough_mask_file) as aux:
    lores_istissue0 = aux['lores_istissue0']
    im_downsampled = aux['im_downsampled']
    step = aux['step']
    perc_completed_all = list(aux['perc_completed_all'])
    time_step_all = list(aux['time_step_all'])

time_prev = time.time()

# get indices for the next histology window to process
(first_row, last_row, first_col, last_col), \
(lores_first_row, lores_last_row, lores_first_col, lores_last_col) = \
    cytometer.utils.get_next_roi_to_process(lores_istissue0, downsample_factor=downsample_factor,
                                            max_window_size=fullres_box_size,
                                            border=np.round((receptive_field - 1) / 2))

# load window from full resolution slide
tile = im.read_region(location=(first_col, first_row), level=0,
                      size=(last_col - first_col, last_row - first_row))
tile = np.array(tile)
tile = tile[:, :, 0:3]

# interpolate coarse tissue segmentation to full resolution
istissue_tile = lores_istissue0[lores_first_row:lores_last_row, lores_first_col:lores_last_col]
istissue_tile = cytometer.utils.resize(istissue_tile, size=(last_col - first_col, last_row - first_row),
                                       resample=PIL.Image.NEAREST)

# histograms for each channel
hist_r, _ = np.histogram(tile[:, :, 0].flatten(), bins=xbins_edge, density=True)
hist_g, _ = np.histogram(tile[:, :, 1].flatten(), bins=xbins_edge, density=True)
hist_b, _ = np.histogram(tile[:, :, 2].flatten(), bins=xbins_edge, density=True)

# plot new data histograms
plt.subplot(131)
plt.plot(xbins, np.sqrt(hist_r), 'k', label='New data before')
plt.legend()

plt.subplot(132)
plt.plot(xbins, np.sqrt(hist_g), 'k', label='New data before')
plt.legend()

plt.subplot(133)
plt.plot(xbins, np.sqrt(hist_b), 'k', label='New data before')
plt.legend()

# colour channel modes for the RREB1 tile
mode_r_rrbe1 = scipy.stats.mode(tile[:, :, 0], axis=None).mode[0]
mode_g_rrbe1 = scipy.stats.mode(tile[:, :, 1], axis=None).mode[0]
mode_b_rrbe1 = scipy.stats.mode(tile[:, :, 2], axis=None).mode[0]

# recompute the colour histograms after correcting the colour tint
hist_r, _ = np.histogram(tile[:, :, 0].flatten().astype(np.float32) - mode_r_rrbe1 + mode_r, bins=xbins_edge, density=True)
hist_g, _ = np.histogram(tile[:, :, 1].flatten().astype(np.float32) - mode_g_rrbe1 + mode_g, bins=xbins_edge, density=True)
hist_b, _ = np.histogram(tile[:, :, 2].flatten().astype(np.float32) - mode_b_rrbe1 + mode_b, bins=xbins_edge, density=True)

# plot corrected histograms
plt.subplot(131)
plt.plot(xbins, np.sqrt(hist_r), 'k--', label='New data after')
plt.legend()

plt.subplot(132)
plt.plot(xbins, np.sqrt(hist_g), 'k--', label='New data after')
plt.legend()

plt.subplot(133)
plt.plot(xbins, np.sqrt(hist_b), 'k--', label='New data after')
plt.legend()
