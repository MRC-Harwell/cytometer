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
import pickle
# import json
# import tempfile
#
# # other imports
# import datetime
import openslide
import PIL
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import time
# import random
# import pysto

# # limit number of GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

os.environ['KERAS_BACKEND'] = 'tensorflow'
# import keras
# import keras.backend as K
# import keras_contrib

# from keras.models import Model
# from keras.layers import Input, Conv2D, MaxPooling2D, AvgPool2D, Activation, BatchNormalization

# # for data parallelism in keras models
# from keras.utils import multi_gpu_model

import cytometer.model_checkpoint_parallel
import cytometer.utils
import cytometer.data
# import tensorflow as tf

# # limit GPU memory used
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# set_session(tf.Session(config=config))

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
saved_models_dir = os.path.join(root_data_dir, 'saved_models')
annotations_dir = os.path.join(home, 'Software/AIDA/dist/data/annotations')
rreb1_data_dir = os.path.join(home, 'scan_srv2_cox/Liz Bentley/Grace')

# saved_kfolds_filename = 'klf14_b6ntac_exp_0079_generate_kfolds.pickle'
saved_extra_kfolds_filename = 'klf14_b6ntac_exp_0094_generate_extra_training_images.pickle'

# list of NDPI files to process
ndpi_files_list = [
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 Bat 1 - 2018-11-16 16.10.56.ndpi',
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 Bat 2 - 2018-11-16 16.13.43.ndpi',
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 Bat 3 - 2018-11-16 16.16.24.ndpi',
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 G1 - 2018-11-16 14.58.55.ndpi',
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 G2 - 2018-11-16 15.04.07.ndpi',
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 G3 - 2018-11-16 15.10.27.ndpi',
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 liv - 2018-11-16 16.54.43.ndpi',
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 M1 - 2018-11-16 15.27.25.ndpi',
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 M2 - 2018-11-16 15.33.30.ndpi',
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 M3 - 2018-11-16 15.39.07.ndpi',
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 mus - 2018-11-16 17.04.43.ndpi',
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 P1 - 2018-11-16 15.59.52.ndpi',
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 P2 - 2018-11-16 16.48.47.ndpi',
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 P3 - 2018-11-16 16.06.17.ndpi',
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 pan - 2018-11-16 17.00.28.ndpi',
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 S1 - 2018-11-16 15.45.10.ndpi',
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 S2 - 2018-11-16 15.49.47.ndpi',
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 S3 - 2018-11-16 15.53.46.ndpi'
]

# load list of images, and indices for training vs. testing indices

# original dataset used in pipelines up to v6 + extra "other" tissue images
kfold_filename = os.path.join(saved_models_dir, saved_extra_kfolds_filename)
with open(kfold_filename, 'rb') as f:
    aux = pickle.load(f)
file_svg_list = aux['file_list']
idx_test_all = aux['idx_test']
idx_train_all = aux['idx_train']

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

# init list to keep histogram computations
hist_r_all = []
hist_g_all = []
hist_b_all = []

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

# compute quatiles for each of the bins
hist_r_q1, hist_r_q2, hist_r_q3 = scipy.stats.mstats.hdquantiles(hist_r_all, prob=[0.25, 0.5, 0.75], axis=0)
hist_g_q1, hist_g_q2, hist_g_q3 = scipy.stats.mstats.hdquantiles(hist_g_all, prob=[0.25, 0.5, 0.75], axis=0)
hist_b_q1, hist_b_q2, hist_b_q3 = scipy.stats.mstats.hdquantiles(hist_b_all, prob=[0.25, 0.5, 0.75], axis=0)

if DEBUG:
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

# loop new data NDPI files
for i_file, ndpi_file in enumerate(ndpi_files_list):

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
    aux = np.load(rough_mask_file)
    lores_istissue = aux['lores_istissue']
    lores_istissue0 = aux['lores_istissue0']
    im_downsampled = aux['im_downsampled']
    step = aux['step']
    perc_completed_all = list(aux['perc_completed_all'])
    time_step_all = list(aux['time_step_all'])
    del aux

    time_prev = time.time()

    # get indices for the next histology window to process
    (first_row, last_row, first_col, last_col), \
    (lores_first_row, lores_last_row, lores_first_col, lores_last_col) = \
        cytometer.utils.get_next_roi_to_process(lores_istissue, downsample_factor=downsample_factor,
                                                max_window_size=fullres_box_size,
                                                border=np.round((receptive_field - 1) / 2))

    # load window from full resolution slide
    tile = im.read_region(location=(first_col, first_row), level=0,
                          size=(last_col - first_col, last_row - first_row))
    tile = np.array(tile)
    tile = tile[:, :, 0:3]

    # interpolate coarse tissue segmentation to full resolution
    istissue_tile = lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col]
    istissue_tile = cytometer.utils.resize(istissue_tile, size=(last_col - first_col, last_row - first_row),
                                           resample=PIL.Image.NEAREST)

    # histograms for each channel
    hist_r, _ = np.histogram(tile[:, :, 0].flatten(), bins=xbins_edge, density=True)
    hist_g, _ = np.histogram(tile[:, :, 1].flatten(), bins=xbins_edge, density=True)
    hist_b, _ = np.histogram(tile[:, :, 2].flatten(), bins=xbins_edge, density=True)

    plt.subplot(131)
    plt.plot(xbins, np.sqrt(hist_r), 'k', label='New data')
    plt.legend()

    plt.subplot(132)
    plt.plot(xbins, np.sqrt(hist_g), 'k', label='New data')
    plt.legend()

    plt.subplot(133)
    plt.plot(xbins, np.sqrt(hist_b), 'k', label='New data')
    plt.legend()
