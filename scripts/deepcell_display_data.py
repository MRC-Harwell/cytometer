#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:23:19 2017

@author: rcasero

deepcell_display_data.py:
    
    Script to generate images of the validation data, using CLAHE (Contrast 
    Limited Adaptive Histogram Equalization) to enhance the images
"""

import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import cytometer
import numpy as np
from pkg_resources import parse_version

import cv2
#import skimage.exposure

# pop out window for plots
if (sys.version_info.major < 3):
    %matplotlib qt
else:
    %matplotlib qt5

# data paths
basedatadir = os.path.normpath(os.path.join(cytometer.__path__[0], '../data/deepcell'))
datadir = os.path.join(basedatadir, 'validation_data')

# list of validation datasets
im_file = [
        '3T3/RawImages/phase.tif',
        'HeLa/RawImages/phase.tif',
        'HeLa_plating/10000K/RawImages/img_channel001_position000_time000000000_z000.tif',
        'HeLa_plating/1250K/RawImages/img_channel001_position000_time000000000_z000.tif',
        'HeLa_plating/20000K/RawImages/img_channel001_position000_time000000000_z000.tif',
        'HeLa_plating/2500K/RawImages/img_channel001_position000_time000000000_z000.tif',
        'HeLa_plating/5000K/RawImages/img_channel001_position000_time000000000_z000.tif',
        'MCF10A/RawImages/phase.tif',
        ]

seg_file = [
        '3T3/Validation/3T3_validation_interior.tif',
        'HeLa/Validation/hela_validation_interior.tif',
        'HeLa_plating/10000K/Validation/10000K_feature_1.png',
        'HeLa_plating/1250K/Validation/1250K_feature_1.png',
        'HeLa_plating/20000K/Validation/20000K_feature_1.png',
        'HeLa_plating/2500K/Validation/2500K_feature_1.png',
        'HeLa_plating/5000K/Validation/5000K_feature_1.png',
        'MCF10A/Validation/MCF10A_validation_interior.tif',
        ]

# create a CLAHE object (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

fig = plt.figure(figsize=(21, 10))
params = {'axes.labelsize': 20,'axes.titlesize':25, 'font.size': 20, 
          'legend.fontsize': 20, 'xtick.labelsize': 20, 'ytick.labelsize': 20}
mpl.rcParams.update(params)

nfile = len(im_file)
for i in range(nfile):
    print(im_file[i])

    fig.suptitle(im_file[i])

    # plot image
    im = plt.imread(os.path.join(datadir, im_file[i]))
    plt.subplot(1,3,1)
    plt.imshow(im)
    plt.title('Original')

    # plot image with enhanced contrast
    im2 = clahe.apply(im)
    plt.subplot(1,3,2)
    plt.imshow(im2)
    plt.title('Enhanced contrast')
    
    # plot segmentation mask
    seg = plt.imread(os.path.join(datadir, seg_file[i]))
    plt.subplot(1,3,3)
    plt.imshow(seg)
    plt.title('Hand segmentation')
    
    # save image
    plt.show()
    fig.savefig(os.path.join(datadir, im_file[i].replace('/', '_')))


