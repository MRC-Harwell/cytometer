"""
Read full .ndpi slides, extract 1001x1001 windows manually selected to increase the dataset randomly created with 0074.

The windows are saved with row_R_col_C, where R, C are the row, col centroid of the image. You can get the offset of the
image from the centroid as offset = centroid - box_half_size = centroid - 500.
"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0094_generate_extra_training_images'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import openslide
import numpy as np
from statistics import mode
import matplotlib.pyplot as plt
import cv2
from random import randint, seed
import tifffile
import glob
from cytometer.utils import rough_foreground_mask
import pickle

DEBUG = False

# dimensions of the box that we use to crop the full histology image
box_size = 1001
box_half_size = int((box_size - 1) / 2)
n_samples = 5

# level that we downsample the image to
level = 4

'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

saved_kfolds_filename = 'klf14_b6ntac_exp_0079_generate_kfolds.pickle'

ndpi_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
seg_dir = os.path.join(root_data_dir, 'klf14_b6ntac_seg')

# explicit list of files, to avoid differences if the files in the directory change
ndpi_files_list = [
    'KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04.ndpi',
    'KLF14-B6NTAC-MAT-18.2d  60-16 C1 - 2016-02-03 13.13.57.ndpi',
    'KLF14-B6NTAC-MAT-18.2b  58-16 C1 - 2016-02-03 11.10.52.ndpi',
    'KLF14-B6NTAC-MAT-17.2c  66-16 C1 - 2016-02-04 11.46.39.ndpi',
    'KLF14-B6NTAC-MAT-17.2f  68-16 C1 - 2016-02-04 15.05.54.ndpi',
    'KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38.ndpi',
    'KLF14-B6NTAC-MAT-18.1a  50-16 C1 - 2016-02-02 09.12.41.ndpi',
    'KLF14-B6NTAC-MAT-18.2g  63-16 C1 - 2016-02-03 16.58.52.ndpi',
    'KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46.ndpi',
    'KLF14-B6NTAC-37.1d PAT 109-16 C1 - 2016-02-15 15.19.08.ndpi',
    'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33.ndpi',
    'KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38.ndpi',
    'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53.ndpi',
    'KLF14-B6NTAC-37.1c PAT 108-16 C1 - 2016-02-15 14.49.45.ndpi',
    'KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52.ndpi',
    'KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32.ndpi',
    'KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11.ndpi',
    'KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06.ndpi',
    'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52.ndpi',
    'KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00.ndpi'
]
ndpi_files_list = [os.path.join(ndpi_dir, x) for x in ndpi_files_list]

# Note: if you want to read the full list of KLF14*.ndpi
# ndpi_files_list = glob.glob(os.path.join(ndpi_dir, 'KLF14*.ndpi'))

# loop downsampled histology slides (this was used to find the coordinates of cropping boxes for the next for loop)
for i_file, ndpi_file in enumerate(ndpi_files_list):

    print('File ' + str(i_file) + '/' + str(len(ndpi_files_list)-1) + ': ' + ndpi_file)

    # load file
    im = openslide.OpenSlide(os.path.join(ndpi_dir, ndpi_file))

    # box for left half of image
    box_corner_col = 0
    box_corner_row = 0
    location = (box_corner_col, box_corner_row)
    size = (int(im.level_dimensions[level][0]/2), int(im.level_dimensions[level][1]))

    # extract tile from full resolution image, downsampled so that we can look for regions of interest
    tile_lo = im.read_region(location=location, level=level, size=size)
    tile_lo = np.array(tile_lo)
    tile_lo = tile_lo[:, :, 0:3]

    if DEBUG:
        plt.clf()
        plt.imshow(tile_lo)

new_outfilename_list = []
for i_file, ndpi_file in enumerate(ndpi_files_list):

    print('File ' + str(i_file) + '/' + str(len(ndpi_files_list)) + ': ' + ndpi_file)

    # load file
    im = openslide.OpenSlide(os.path.join(ndpi_dir, ndpi_file))

    # (x,y)-locations of cropping windows in the coordinates of the downsampled image by a factor of 16
    if i_file == 0:
        location_list = ((1059, 1052), (682, 1174), (918, 1533))
    elif i_file == 1:
        location_list = ((883, 866), (864, 1278), (665, 456))
    elif i_file == 2:
        location_list = ((672, 397), (885, 877), (870, 1303))
    elif i_file == 3:
        location_list = ((549, 615), (1094, 1112), (546, 615))
    elif i_file == 4:
        location_list = ((230, 500), (488, 1435), (1225, 1198))
    elif i_file == 5:
        location_list = ((874, 730), (776, 991), (1947, 1170))
    elif i_file == 6:
        location_list = ((786, 874), (995, 950), (1193, 903), (1080, 224))
    elif i_file == 7:
        location_list = ((686, 986), (846, 633), (1677, 999))
    elif i_file == 8:
        location_list = ((553, 415), (341, 274), (164, 872))
    elif i_file == 9:
        location_list = ((563, 827), (116, 628), (1030, 1230))
    elif i_file == 10:
        location_list = ((632, 1508), (1311, 925), (1210, 334))
    elif i_file == 11:
        location_list = ((1627, 2587), (1532, 1804), (1411, 923), (1430, 850))
    elif i_file == 12:
        location_list = ((1133, 1388), (1113, 430), (372, 1187))
    elif i_file == 13:
        location_list = ((330, 744), (264, 350), (556, 451))
    elif i_file == 14:
        location_list = ((591, 1262), (927, 1170), (1140, 1173))
    elif i_file == 15:
        location_list = ((441, 759), (652, 339), (1097, 266))
    elif i_file == 16:
        location_list = ((467, 763), (1197, 855), (1806, 1253))
    elif i_file == 17:
        location_list = ((475, 895), (680, 1334), (1346, 950))
    elif i_file == 18:
        location_list = ((744, 1483), (1216, 992), (1854, 581))
    elif i_file == 19:
        location_list = ((970, 1852), (432, 657), (1015, 803))

    for j in range(len(location_list)):

        # (x,y)-coordinates of cropping box first corner at 16x downsample level
        location = location_list[j]

        # (x.y)-coordinates in the full resolution histology
        location = np.array(location) * im.level_downsamples[level]

        # extract tile at full resolution
        tile = im.read_region(location=location.astype(np.int), level=0, size=(box_size, box_size))
        tile = np.array(tile)
        tile = tile[:, :, 0:3]

        if DEBUG:
            plt.clf()
            plt.imshow(tile)

        # (row, col)-coordinates in the full resolution histology
        box_corner_col = int(location[0])
        box_corner_row = int(location[1])

        # name of the file with the cropped histology
        outfilename = os.path.basename(ndpi_file)
        outfilename = os.path.splitext(outfilename)[0] + '_row_' + str(box_corner_row).zfill(6) \
                      + '_col_' + str(box_corner_col).zfill(6)
        outfilename = os.path.join(training_dir, outfilename + '.tif')

        # check that file doesn't exist
        if os.path.isfile(outfilename):
            raise FileExistsError('File exists: ' + outfilename)

        new_outfilename_list.append(outfilename)

        # save tile as a tiff file with ZLIB compression (LZMA or ZSTD can't be opened by QuPath)
        print('Saving ' + outfilename)
        tifffile.imsave(outfilename, tile,
                        compress=9,
                        resolution=(int(im.properties["tiff.XResolution"]),
                                    int(im.properties["tiff.YResolution"]),
                                    im.properties["tiff.ResolutionUnit"].upper()))

new_n_klf14 = len(new_outfilename_list)

'''Load folds'''

# load list of images, and indices for training vs. testing indices
contour_model_kfold_filename = os.path.join(saved_models_dir, saved_kfolds_filename)
with open(contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
file_svg_list = aux['file_list']
idx_test_all = aux['idx_test']
idx_train_all = aux['idx_train']

# correct home directory
file_svg_list = [x.replace('/home/rcasero', home) for x in file_svg_list]

# number of images
n_im = len(file_svg_list)

# loop folds
new_outfilename_list_trimmed = [x.split('_row')[0] for x in new_outfilename_list]
new_idx_test_all = []
for k in range(len(idx_test_all)):

    # test indices
    idx_test = idx_test_all[k]

    # file names of the original training data files for this fold
    file_svg_test = np.array(file_svg_list)[idx_test]

    # remove the '_row_*_col_*' strings and remove duplicates
    file_svg_test_trimmed = np.unique([x.split('_row')[0] for x in file_svg_test])

    # find the new training windows that correspond to the current fold
    new_idx_test = np.where(np.isin(new_outfilename_list_trimmed, file_svg_test_trimmed))[0]

    new_idx_test_all.append(new_idx_test)

# split indices into train and test sets
new_idx_train_all = []
for k in range(len(new_idx_test_all)):
    new_idx_train = set(range(len(new_outfilename_list))) - set(new_idx_test_all[k])
    new_idx_train_all.append(np.array(list(new_idx_train)))

# save file list
new_kfolds_filename = os.path.join(saved_models_dir, 'klf14_b6ntac_exp_0094_generate_extra_training_images.pickle')
with open(new_kfolds_filename, 'wb') as f:
    x = {'file_list': new_outfilename_list, 'idx_train': new_idx_train_all, 'idx_test': new_idx_test_all,
         'fold_seed': 0, 'n_klf14': new_n_klf14}
    pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)

# debug folds (make sure that each fold contains the same .ndpi data in the old and new datasets)
if DEBUG:
    for k in range(len(idx_test_all)):

        print('Fold: ' + str(k))

        # file names in old dataset
        idx_test = idx_test_all[k]
        file_svg_test = np.array(file_svg_list)[idx_test]

        # file names in new dataset
        new_idx_test = new_idx_test_all[k]
        new_outfilename = np.array(new_outfilename_list)[new_idx_test]

        print('    Old files')
        for file in file_svg_test:
            print('        ' + os.path.basename(file))
        print('    New files')
        for file in new_outfilename:
            print('        ' + os.path.basename(file))

for i, file in enumerate(new_outfilename_list):

    print('i = ' + str(i) + ': ' + os.path.basename(file))

# i = 7 next window to label