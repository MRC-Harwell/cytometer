"""
This is where we create the file with the median, Q1 and Q3 curves for the colour histograms of the Klf14 training data
that we use to correct the histology colour before passing it to DeepCytometer.
"""

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0107_colour_correction_v8'
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
import cv2

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
xbins_edge = np.array(list(range(0, 256, 5)))
xbins = (xbins_edge[0:-1] + xbins_edge[1:]) / 2

# init list to keep statistics computations
hist_r_all = []
hist_g_all = []
hist_b_all = []
mean_l_all = []
mean_a_all = []
mean_b_all = []
std_l_all = []
std_a_all = []
std_b_all = []

# loop files with hand traced contours
plt.clf()
for i, file_svg in enumerate(file_svg_list):

    print('file ' + str(i) + '/' + str(len(file_svg_list) - 1))

    # change file extension from .svg to .tif
    file_tif = file_svg.replace('.svg', '.tif')

    # open histology training image
    im = Image.open(file_tif)

    if DEBUG:
        plt.clf()
        plt.imshow(im)

    # make a L*a*b copy of the image
    im_lab = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2LAB).astype('float32')
    (l, a, b) = cv2.split(im_lab)

    # L*a*b statistics
    mean_l_all.append(np.mean(l))
    mean_a_all.append(np.mean(a))
    mean_b_all.append(np.mean(b))
    std_l_all.append(np.std(l))
    std_a_all.append(np.std(a))
    std_b_all.append(np.std(b))

    # histograms for each channel
    plt.subplot(131)
    hist_r, _, _ = plt.hist(np.array(im.getchannel('R')).flatten(), bins=xbins_edge, histtype='step', density=True)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.title('Red', fontsize=16)
    plt.xlabel('Intensity', fontsize=14)
    plt.ylabel('$\sqrt{Density}$', fontsize=14)

    plt.subplot(132)
    hist_g, _, _ = plt.hist(np.array(im.getchannel('G')).flatten(), bins=xbins_edge, histtype='step', density=True)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.title('Green', fontsize=16)
    plt.xlabel('Intensity', fontsize=14)

    plt.subplot(133)
    hist_b, _, _ = plt.hist(np.array(im.getchannel('B')).flatten(), bins=xbins_edge, histtype='step', density=True)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.title('Blue', fontsize=16)
    plt.xlabel('Intensity', fontsize=14)

    hist_r_all.append(hist_r)
    hist_g_all.append(hist_g)
    hist_b_all.append(hist_b)

# stack vectors into matrix
hist_r_all = np.vstack(hist_r_all)
hist_g_all = np.vstack(hist_g_all)
hist_b_all = np.vstack(hist_b_all)

# compute quatiles for each of the bins. This computes a median (50-percentile) histogram for each channel, plus the
# 25- and 75-percentile. We can use the latter as some kind of confidence interval for the colour histogram down the
# line
hist_r_q1, hist_r_q2, hist_r_q3 = scipy.stats.mstats.hdquantiles(hist_r_all, prob=[0.25, 0.5, 0.75], axis=0)
hist_g_q1, hist_g_q2, hist_g_q3 = scipy.stats.mstats.hdquantiles(hist_g_all, prob=[0.25, 0.5, 0.75], axis=0)
hist_b_q1, hist_b_q2, hist_b_q3 = scipy.stats.mstats.hdquantiles(hist_b_all, prob=[0.25, 0.5, 0.75], axis=0)

# compute median histogram's mode for each channel. This gives us the most typical colour in the training dataset
mode_r = xbins[np.argmax(hist_r_q2)]
mode_g = xbins[np.argmax(hist_g_q2)]
mode_b = xbins[np.argmax(hist_b_q2)]

# save colour histograms
klf14_training_colour_histogram_file = os.path.join(saved_models_dir, 'klf14_training_colour_histogram.npz')
np.savez(klf14_training_colour_histogram_file, xbins_edge=xbins_edge, xbins=xbins,
         hist_r_q1=hist_r_q1, hist_r_q2=hist_r_q2, hist_r_q3=hist_r_q3,
         hist_g_q1=hist_g_q1, hist_g_q2=hist_g_q2, hist_g_q3=hist_g_q3,
         hist_b_q1=hist_b_q1, hist_b_q2=hist_b_q2, hist_b_q3=hist_b_q3,
         mode_r=mode_r, mode_g=mode_g, mode_b=mode_b,
         mean_l=np.median(mean_l_all), mean_a=np.median(mean_a_all), mean_b=np.median(mean_b_all),
         std_l=np.median(std_l_all), std_a=np.median(std_a_all), std_b=np.median(std_b_all))

# plot KLF14 training colour histograms
plt.clf()

plt.subplot(131)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.plot(xbins, np.sqrt(hist_r_q2), label='Training median')
plt.fill_between(xbins, np.sqrt(hist_r_q2), np.sqrt(hist_r_q1), alpha=0.5, color='C0', label='Q1-Q3')
plt.legend()
plt.title('Red', fontsize=16)
plt.xlabel('Intensity', fontsize=14)
plt.ylabel('$\sqrt{Density}$', fontsize=14)

plt.subplot(132)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.plot(xbins, np.sqrt(hist_g_q2), label='Training median')
plt.fill_between(xbins, np.sqrt(hist_g_q2), np.sqrt(hist_g_q1), alpha=0.5, color='C0', label='Q1-Q3')
plt.fill_between(xbins, np.sqrt(hist_g_q2), np.sqrt(hist_g_q3), alpha=0.5, color='C0')
plt.legend()
plt.title('Green', fontsize=16)
plt.xlabel('Intensity', fontsize=14)

plt.subplot(133)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.plot(xbins, np.sqrt(hist_b_q2), label='Training median')
plt.fill_between(xbins, np.sqrt(hist_b_q2), np.sqrt(hist_b_q1), alpha=0.5, color='C0', label='Q1-Q3')
plt.fill_between(xbins, np.sqrt(hist_b_q2), np.sqrt(hist_b_q3), alpha=0.5, color='C0')
plt.legend()
plt.title('Blue', fontsize=16)
plt.xlabel('Intensity', fontsize=14)

plt.tight_layout()
