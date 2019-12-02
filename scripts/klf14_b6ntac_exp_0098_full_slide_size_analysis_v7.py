"""
Read annotation files from full slides processed by pipeline v7.
"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0098_full_slide_size_analysis_v7'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import sys
import json
import pickle
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import cytometer.utils

# Filter out INFO & WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # limit number of GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

os.environ['KERAS_BACKEND'] = 'tensorflow'

import time
import openslide
import numpy as np
import matplotlib.pyplot as plt
import glob
from cytometer.utils import rough_foreground_mask, bspline_resample
from cytometer.data import append_paths_to_aida_json_file, write_paths_to_aida_json_file
import PIL
import tensorflow as tf
import keras
from keras import backend as K
from skimage.measure import regionprops
import shutil
import itertools
from shapely.geometry import Polygon
import scipy

# # limit GPU memory used
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# set_session(tf.Session(config=config))

DEBUG = False
SAVE_FIGS = False

root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
data_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
training_dir = os.path.join(home, root_data_dir, 'klf14_b6ntac_training')
seg_dir = os.path.join(home, root_data_dir, 'klf14_b6ntac_seg')
figures_dir = os.path.join(root_data_dir, 'figures')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')
results_dir = os.path.join(root_data_dir, 'klf14_b6ntac_results')
annotations_dir = os.path.join(home, 'Software/AIDA/dist/data/annotations')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')

# rough_foreground_mask() parameters
downsample_factor = 8.0
dilation_size = 25
component_size_threshold = 1e6
hole_size_treshold = 8000


# list of annotation files
json_annotation_files = [
    'KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_exp_0097.json',
    'KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04_exp_0097.json',
    'KLF14-B6NTAC-MAT-17.2c  66-16 C1 - 2016-02-04 11.46.39_exp_0097.json',
    'KLF14-B6NTAC-MAT-18.2b  58-16 C1 - 2016-02-03 11.10.52_exp_0097.json',
    'KLF14-B6NTAC-MAT-18.2d  60-16 C1 - 2016-02-03 13.13.57_exp_0097.json',
    'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_exp_0097.json'
]

for i_file, json_file in enumerate(json_annotation_files):

    # name of corresponding .ndpi file
    ndpi_file = json_file.replace('_exp_0097.json', '.ndpi')

    # add path to file
    json_file = os.path.join(annotations_dir, json_file)
    ndpi_file = os.path.join(data_dir, ndpi_file)

    # open full resolution histology slide
    im = openslide.OpenSlide(ndpi_file)

    # pixel size
    assert (im.properties['tiff.ResolutionUnit'] == 'centimeter')
    xres = 1e-2 / float(im.properties['tiff.XResolution'])
    yres = 1e-2 / float(im.properties['tiff.YResolution'])

    # change pixel size to downsampled size
    xres *= downsample_factor
    yres *= downsample_factor

    # rough segmentation of the tissue in the image
    lores_istissue0, im_downsampled = rough_foreground_mask(ndpi_file, downsample_factor=downsample_factor,
                                                            dilation_size=dilation_size,
                                                            component_size_threshold=component_size_threshold,
                                                            hole_size_treshold=hole_size_treshold,
                                                            return_im=True)

    if DEBUG:
        plt.clf()
        plt.subplot(211)
        plt.imshow(im_downsampled)
        plt.subplot(212)
        plt.imshow(lores_istissue0)

    # parse the json file
    with open(json_file) as f:
        json = json.load(f)

    # list of items (there's a contour in each item)
    items = json['layers'][0]['items']

    # init array for interpolated areas
    areas_grid = np.zeros(shape=lores_istissue0.shape, dtype=np.float32)

    # init lists for contour centroids and areas
    areas_all = []
    centroids_all = []

    # loop items (one contour per item)
    for it in items:

        # extract contour
        c = it['segments']

        # convert to downsampled coordinates
        c = np.array(c) / downsample_factor

        if DEBUG:
            plt.fill(c[:, 0], c[:, 1], fill=False, color='r')

        # compute cell area
        area = Polygon(c).area * xres * yres  # (m^2)
        areas_all.append(area)

        # compute centroid of contour
        centroid = np.mean(c, axis=0)
        centroids_all.append(centroid)

    # interpolate scattered data to regular grid
    idx = lores_istissue0 == 1
    xi = np.transpose(np.array(np.where(idx)))[:, [1, 0]]
    areas_grid[idx] = scipy.interpolate.griddata(centroids_all, np.sqrt(areas_all), xi, method='linear', fill_value=0)

    if DEBUG:
        plt.clf()
        plt.subplot(211)
        plt.imshow(im_downsampled)
        plt.subplot(212)
        plt.imshow(areas_grid, vmax=np.sqrt(20000e-12))

