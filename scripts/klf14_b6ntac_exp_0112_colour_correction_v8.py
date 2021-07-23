"""
This is where we create the file with the ECDFs of the Klf14 training data
that we use to correct the histology colour before passing it to DeepCytometer.

Differences with klf14_b6ntac_exp_0107_colour_correction_v8.py:
    * We only use the training images with mostly white adipocytes, not the extended dataset.
    * We use HD quantiles to estimate the ECDF, instead of histograms, to avoid having bins with 0 values that translate
      to horizontal segments in the ECDF.
    * We don't compute the same statistics.
"""

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0112_colour_correction_v8.py'
print('Experiment ID: ' + experiment_id)

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import pickle

# other imports
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import cytometer.data

# # limit number of GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

os.environ['KERAS_BACKEND'] = 'tensorflow'

DEBUG = False

'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(home, 'Data/cytometer_data/deepcytometer_pipeline_v8')

klf14_training_colour_histogram_file = os.path.join(saved_models_dir, 'klf14_exp_0112_training_colour_histogram.npz')

# we want to use only the training images that mostly contain white adipocytes, so that we adjust the colour profile to
# them. But we use the total list of files, that is more up to date, and then filter out the files without adipocytes
# saved_kfolds_filename = 'klf14_b6ntac_exp_0079_generate_kfolds.pickle'
saved_extra_kfolds_filename = 'klf14_b6ntac_exp_0094_generate_extra_training_images.pickle'

# load list of images, and indices for training vs. testing indices

# original dataset used in pipelines up to v6 + extra "other" tissue images
kfold_filename = os.path.join(saved_models_dir, saved_extra_kfolds_filename)
with open(kfold_filename, 'rb') as f:
    aux = pickle.load(f)
file_svg_list = aux['file_list']

# correct home directory
file_svg_list = [x.replace('/users/rittscher/rcasero', home) for x in file_svg_list]
file_svg_list = [x.replace('/home/rcasero', home) for x in file_svg_list]

# number of images
n_im = len(file_svg_list)

'''Colour histograms
'''

# bin edges and centers for the histograms
xbins = np.array(list(range(0, 256)))

# init list to keep statistics computations
val_r_all = []
val_g_all = []
val_b_all = []

# loop files with hand traced contours
for i, file_svg in enumerate(file_svg_list):

    print('file ' + str(i) + '/' + str(len(file_svg_list) - 1) + ': ' + os.path.basename(file_svg))

    # skip images without white adipocytes. We want to estimate the ECDFs from images that are mostly white adipocytes
    cell_contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False,
                                                            minimum_npoints=3)
    if (len(cell_contours) == 0):
        print('... skip')
        continue

    # change file extension from .svg to .tif
    file_tif = file_svg.replace('.svg', '.tif')

    # open histology training image
    im = Image.open(file_tif)

    if DEBUG:
        plt.clf()
        plt.imshow(im)

    # concat intensity values for total output
    val_r_all.append(np.array(im.getchannel('R')).flatten())
    val_g_all.append(np.array(im.getchannel('G')).flatten())
    val_b_all.append(np.array(im.getchannel('B')).flatten())

# concatenate vectors into single vector
val_r_all = np.concatenate(val_r_all)
val_g_all = np.concatenate(val_g_all)
val_b_all = np.concatenate(val_b_all)

# compute the intensity values that correspond to each quantile of an ECDF per colour channel
p = np.linspace(0.0, 1.0, 101)
k = 10  # step for subsampling the image data
val_r_klf14 = scipy.stats.mstats.hdquantiles(val_r_all[0::k], prob=p, axis=0)
val_g_klf14 = scipy.stats.mstats.hdquantiles(val_g_all[0::k], prob=p, axis=0)
val_b_klf14 = scipy.stats.mstats.hdquantiles(val_b_all[0::k], prob=p, axis=0)

# function to map ECDF to intensity values in the Klf14 dataset
f_ecdf_to_val_r_klf14 = scipy.interpolate.interp1d(p, val_r_klf14, fill_value=(0, 255), bounds_error=False)
f_ecdf_to_val_g_klf14 = scipy.interpolate.interp1d(p, val_g_klf14, fill_value=(0, 255), bounds_error=False)
f_ecdf_to_val_b_klf14 = scipy.interpolate.interp1d(p, val_b_klf14, fill_value=(0, 255), bounds_error=False)

if DEBUG:
    plt.clf()
    plt.plot(val_r_klf14, p)
    plt.plot(val_g_klf14, p)
    plt.plot(val_b_klf14, p)
    plt.xlabel('Intensity values', fontsize=14)
    plt.ylabel('ECDF', fontsize=14)
    plt.tick_params(labelsize=14)

# mean and std per colour channel
mean_klf14 = np.array([np.mean(val_r_all), np.mean(val_g_all), np.mean(val_b_all)])
std_klf14 = np.array([np.std(val_r_all), np.std(val_g_all), np.std(val_b_all)])

# save colour histograms
np.savez(klf14_training_colour_histogram_file, p=p,
         val_r_klf14=val_r_klf14, val_g_klf14=val_g_klf14, val_b_klf14=val_b_klf14,
         f_ecdf_to_val_r_klf14=f_ecdf_to_val_r_klf14, f_ecdf_to_val_g_klf14=f_ecdf_to_val_g_klf14, f_ecdf_to_val_b_klf14=f_ecdf_to_val_b_klf14,
         mean_klf14=mean_klf14, std_klf14=std_klf14)
