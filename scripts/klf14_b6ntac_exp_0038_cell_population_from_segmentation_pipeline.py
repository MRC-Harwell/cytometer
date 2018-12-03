"""
Load pipeline segmentations from 0036.
Compare the cell area distributions of:

* Hand segmented cells (ground truth)
* Automatically segmented cells that overlap with a ground truth cell and Dice>=0.9
* All automatically segmented cells accepted by quality assessment network

This works with contours from exp 0005 and dmap from exp 0015, which only consider fold 0.
"""

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import pickle
import inspect

# other imports
import PIL
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import watershed
from skimage.measure import regionprops

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

import cytometer.data
import cytometer.utils

# limit GPU memory used
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = False

# number of folds for k-fold cross validation
n_folds = 11

# number of epochs for training
epochs = 20

# area (pixel**2) of the smallest object we accept as a cell (pi * (16 pixel)**2 = 804.2 pixel**2)
smallest_cell_area = 804

# training window length
training_window_len = 401

# threshold for valid/invalid segmentation
valid_threshold = 0.9

'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0038_cell_population_from_segmentation_pipeline'

'''
************************************************************************************************************************
Hand segmented cells (ground truth), no-overlap approximation
************************************************************************************************************************
'''

'''Load data
'''

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(root_data_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# list of all non-overlap original files
im_file_list = glob.glob(os.path.join(training_augmented_dir, 'im_seed_nan_*.tif'))

# read pixel size information
orig_file = os.path.basename(im_file_list[0]).replace('im_seed_nan_', '')
im = PIL.Image.open(os.path.join(training_dir, orig_file))
xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

# load data
full_dataset, full_file_list, full_shuffle_idx = \
    cytometer.data.load_datasets(im_file_list, prefix_from='im', prefix_to=['im', 'lab'], nblocks=1)

# remove borders between cells in the lab_train data. For this experiment, we want labels touching each other
for i in range(full_dataset['lab'].shape[0]):
    full_dataset['lab'][i, :, :, 0] = watershed(image=np.zeros(shape=full_dataset['lab'].shape[1:3],
                                                               dtype=full_dataset['lab'].dtype),
                                                markers=full_dataset['lab'][i, :, :, 0],
                                                watershed_line=False)

# relabel background as "0" instead of "1"
full_dataset['lab'][full_dataset['lab'] == 1] = 0

# plot example of data
if DEBUG:
    i = 0
    plt.clf()
    plt.subplot(121)
    plt.imshow(full_dataset['im'][i, :, :, :])
    plt.subplot(122)
    plt.imshow(full_dataset['lab'][i, :, :, 0])

# loop images
df_gtruth = None
for i in range(full_dataset['lab'].shape[0]):

    # get area of each cell in um^2
    props = regionprops(full_dataset['lab'][i, :, :, 0])
    area = np.array([x['area'] for x in props]) * xres * yres

    # create dataframe with metainformation from mouse
    df_window = cytometer.data.tag_values_with_mouse_info(metainfo, os.path.basename(full_file_list['im'][i]),
                                                          area, values_tag='area', tags_to_keep=['id', 'ko', 'sex'])

    # add a column with the window filename. This is later used in the linear models
    df_window['file'] = os.path.basename(full_file_list['im'][i])

    # create new total dataframe, or concat to existing one
    if df_gtruth is None:
        df_gtruth = df_window
    else:
        df_gtruth = pd.concat([df_gtruth, df_window], axis=0, ignore_index=True)


# make sure that in the boxplots PAT comes before MAT
df_gtruth['ko'] = df_gtruth['ko'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))

# plot boxplots for f/m, PAT/MAT comparison as in Nature Genetics paper
plt.clf()
ax = plt.subplot(121)
df_gtruth[df_gtruth['sex'] == 'f'].boxplot(column='area', by='ko', ax=ax, notch=True)
#ax.set_ylim(0, 2e4)
ax.set_title('female', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
ax = plt.subplot(122)
df_gtruth[df_gtruth['sex'] == 'm'].boxplot(column='area', by='ko', ax=ax, notch=True)
#ax.set_ylim(0, 2e4)
ax.set_title('male', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)

# split data into groups
area_gtruth_f_PAT = df_gtruth['area'][(np.logical_and(df_gtruth['sex'] == 'f', df_gtruth['ko'] == 'PAT'))]
area_gtruth_f_MAT = df_gtruth['area'][(np.logical_and(df_gtruth['sex'] == 'f', df_gtruth['ko'] == 'MAT'))]
area_gtruth_m_PAT = df_gtruth['area'][(np.logical_and(df_gtruth['sex'] == 'm', df_gtruth['ko'] == 'PAT'))]
area_gtruth_m_MAT = df_gtruth['area'][(np.logical_and(df_gtruth['sex'] == 'm', df_gtruth['ko'] == 'MAT'))]

# compute percentile profiles of cell populations
perc = np.linspace(0, 100, num=101)
perc_area_gtruth_f_PAT = np.percentile(area_gtruth_f_PAT, perc)
perc_area_gtruth_f_MAT = np.percentile(area_gtruth_f_MAT, perc)
perc_area_gtruth_m_PAT = np.percentile(area_gtruth_m_PAT, perc)
perc_area_gtruth_m_MAT = np.percentile(area_gtruth_m_MAT, perc)

# plot curve profiles
plt.clf()

ax = plt.subplot(121)
plt.plot(perc, (perc_area_gtruth_f_MAT - perc_area_gtruth_f_PAT) / perc_area_gtruth_f_PAT * 100)
ax.set_ylim(-50, 0)
plt.title('Female', fontsize=16)
plt.xlabel('Population percentile', fontsize=14)
plt.ylabel('Area change from PAT to MAT (%)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)

ax = plt.subplot(122)
plt.plot(perc, (perc_area_gtruth_m_MAT - perc_area_gtruth_m_PAT) / perc_area_gtruth_m_PAT * 100)
ax.set_ylim(-50, 0)
plt.title('Male', fontsize=16)
plt.xlabel('Population percentile', fontsize=14)
plt.ylabel('Area change from PAT to MAT (%)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)


'''
************************************************************************************************************************
Automatically extracted cells
************************************************************************************************************************
'''

'''Directories and filenames
'''

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

saved_contour_model_basename = 'klf14_b6ntac_exp_0034_cnn_contour'  # contour

# script name to identify this experiment
experiment_id = inspect.getfile(inspect.currentframe())
if experiment_id == '<input>':
    experiment_id = 'unknownscript'
else:
    experiment_id = os.path.splitext(os.path.basename(experiment_id))[0]

# list of images, and indices for training vs. testing indices
contour_model_kfold_filename = os.path.join(saved_models_dir, saved_contour_model_basename + '_info.pickle')
with open(contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
im_orig_file_list = aux['file_list']
idx_orig_test_all = aux['idx_test_all']


'''Loop folds
'''

# load CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(root_data_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# loop each fold: we split the data into train vs test, train a model, and compute errors with the
# test data. In each fold, the test data is different
# for i_fold, idx_test in enumerate(idx_test_all):
for i_fold, idx_test in enumerate(idx_orig_test_all):

    '''Load data
    '''

    # split the data list into training and testing lists
    im_test_file_list, _ = cytometer.data.split_list(im_orig_file_list, idx_test)

    # number of testing images
    n_im = len(im_test_file_list)

    # load the correspondences and Dice between pipeline segmentations and ground truth
    correspondence = []
    for i in range(n_im):

        print('\tImage ' + str(i) + '/' + str(n_im-1))

        base_file = os.path.basename(im_test_file_list[i])
        labcorr_file = base_file.replace('im_', 'labcorr_kfold_' + str(i_fold).zfill(2) + '_')
        labcorr_file = os.path.join(training_augmented_dir,
                                    labcorr_file.replace('.tif', '.npy'))
        correspondence.append(np.load(file=labcorr_file))
    correspondence = np.concatenate(correspondence, axis=0)















# models to be used
saved_contour_model_basename = 'klf14_b6ntac_exp_0006_cnn_contour'  # contour
saved_dmap_model_basename = 'klf14_b6ntac_exp_0015_cnn_dmap'  # dmap
saved_quality_model_basename = 'klf14_b6ntac_exp_0029_cnn_qualitynet_sigmoid_masked_segmentation'  # Dice coefficient

# get file name for each model and fold
contour_model_name = saved_contour_model_basename + '*.h5'
dmap_model_name = saved_dmap_model_basename + '*.h5'
quality_model_name = saved_quality_model_basename + '*.h5'

# load k-fold sets that were used to train the models (we assume they are the same for contours and dmaps)
saved_contour_model_kfold_filename = os.path.join(saved_models_dir, saved_contour_model_basename + '_info.pickle')
with open(saved_contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
im_file_list = aux['file_list']
idx_test_all = aux['idx_test_all']

# correct home directory if we are in a different system than what was used to train the models
im_file_list = cytometer.data.change_home_directory(im_file_list, '/users/rittscher/rcasero', home, check_isfile=True)

# in this script, we actually only work with fold 0
fold_i = 0

# split the data into training and testing datasets
im_test_file_list, _ = cytometer.data.split_list(im_file_list, idx_test_all[fold_i])

# load im, seg and mask datasets
test_datasets, _, _ = cytometer.data.load_datasets(im_test_file_list, prefix_from='im',
                                                   prefix_to=['im', 'lab', 'seg', 'mask'], nblocks=1)
im_test = test_datasets['im']
seg_test = test_datasets['seg']
mask_test = test_datasets['mask']
lab_test = test_datasets['lab']
del test_datasets

# remove borders between cells in the lab_train data. For this experiment, we want labels touching each other
for i in range(lab_test.shape[0]):
    lab_test[i, :, :, 0] = watershed(image=np.zeros(shape=lab_test.shape[1:3],
                                                    dtype=lab_test.dtype),
                                     markers=lab_test[i, :, :, 0],
                                     watershed_line=False)

# relabel background as "0" instead of "1"
lab_test[lab_test == 1] = 0

# list of model files to inspect
contour_model_files = glob.glob(os.path.join(saved_models_dir, contour_model_name))
dmap_model_files = glob.glob(os.path.join(saved_models_dir, dmap_model_name))
quality_model_files = glob.glob(os.path.join(saved_models_dir, quality_model_name))

contour_model_file = contour_model_files[fold_i]
dmap_model_file = dmap_model_files[fold_i]
quality_model_file = quality_model_files[fold_i]

# segment the test images
labels, labels_info = cytometer.utils.segmentation_pipeline(im_test,
                                                            contour_model_file,
                                                            dmap_model_file,
                                                            quality_model_file,
                                                            smallest_cell_area=smallest_cell_area)

if DEBUG:
    i = 3
    plt.clf()
    plt.subplot(221)
    plt.imshow(im_test[i, :, :, :])
    plt.title('histo')
    plt.subplot(222)
    plt.imshow(lab_test[i, :, :, 0])
    plt.title('ground truth')
    plt.subplot(223)
    plt.imshow(labels[i, :, :, 0])
    plt.title('pipeline segmentation')

# make copy of automatically segmented labels so that we can remove from them the segmentations without ground truth
labels_with_ground_truth = labels.copy()

# put all the pipeline segmented areas (with corresponding ground truth) into a dataframe
df_test = None

for i in range(im_test.shape[0]):

    # match automatically segmented cells to cells with hand-traced ground truth segmentation
    overlap = cytometer.utils.match_overlapping_labels(lab_test[i, :, :, 0], labels[i, :, :, 0])

    # labels of cells that have a ground truth reference
    lab_with_ground_truth = overlap['lab_test'][overlap['lab_ref'] != 0]

    # remove the automatically segmented cells that have no ground truth
    labels_with_ground_truth[i, :, :, 0][np.isin(labels[i, :, :, 0], lab_with_ground_truth, invert=True)] = 0

    # compute areas of all cells in images
    props_ref = regionprops(lab_test[i, :, :, 0])  # ground truth
    props_test = regionprops(labels[i, :, :, 0])   # pipeline segmentations

    areas_ref = []
    areas_test = []

    # add areas of cells with correspondence to list of areas
    for props in props_test:
        if props['label'] in overlap['lab_test']:
            areas_test.append(props['area'] * xres * yres)

    # create dataframe with metainformation from mouse
    df_window = cytometer.data.tag_values_with_mouse_info(metainfo, os.path.basename(im_test_file_list[i]),
                                                          areas_test, values_tag='area',
                                                          tags_to_keep=['id', 'ko', 'sex'])

    # add a column with the window filename. This is later used in the linear models
    df_window['file'] = os.path.basename(im_test_file_list[i])

    # create new total dataframe, or concat to existing one
    if df_test is None:
        df_test = df_window
    else:
        df_test = pd.concat([df_test, df_window], axis=0, ignore_index=True)

if DEBUG:
    i = 3
    plt.subplot(224)
    plt.imshow(labels_with_ground_truth[i, :, :, 0])
    plt.title('remove segmentations\nwithout ground truth')

# boxplots
plt.clf()
plt.subplot(121)
plt.boxplot(areas_ref)
plt.title('ground truth')
plt.subplot(122)
plt.boxplot(areas_test)
plt.title('pipeline segmentation')



# make sure that in the boxplots PAT comes before MAT
df_test['ko'] = df_test['ko'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))

# # plot boxplots for f/m, PAT/MAT comparison as in Nature Genetics paper
# plt.clf()
# ax = plt.subplot(121)
# df_gtruth[df_gtruth['sex'] == 'f'].boxplot(column='area', by='ko', ax=ax, notch=True)
# #ax.set_ylim(0, 2e4)
# ax.set_title('female', fontsize=16)
# ax.set_xlabel('')
# ax.set_ylabel('area (um^2)', fontsize=14)
# plt.tick_params(axis='both', which='major', labelsize=14)
# ax = plt.subplot(122)
# df_gtruth[df_gtruth['sex'] == 'm'].boxplot(column='area', by='ko', ax=ax, notch=True)
# #ax.set_ylim(0, 2e4)
# ax.set_title('male', fontsize=16)
# ax.set_xlabel('')
# ax.set_ylabel('area (um^2)', fontsize=14)
# plt.tick_params(axis='both', which='major', labelsize=14)

# split data into groups
area_test_f_PAT = df_test['area'][(np.logical_and(df_test['sex'] == 'f', df_test['ko'] == 'PAT'))]
area_test_f_MAT = df_test['area'][(np.logical_and(df_test['sex'] == 'f', df_test['ko'] == 'MAT'))]
area_test_m_PAT = df_test['area'][(np.logical_and(df_test['sex'] == 'm', df_test['ko'] == 'PAT'))]
area_test_m_MAT = df_test['area'][(np.logical_and(df_test['sex'] == 'm', df_test['ko'] == 'MAT'))]

# compute percentile profiles of cell populations
perc = np.linspace(0, 100, num=101)
perc_area_test_f_PAT = np.percentile(area_test_f_PAT, perc)
perc_area_test_f_MAT = np.percentile(area_test_f_MAT, perc)
perc_area_test_m_PAT = np.percentile(area_test_m_PAT, perc)
perc_area_test_m_MAT = np.percentile(area_test_m_MAT, perc)

# plot curve profiles
plt.clf()

ax = plt.subplot(121)
plt.plot(perc, (perc_area_test_f_MAT - perc_area_test_f_PAT) / perc_area_test_f_PAT * 100)
ax.set_ylim(-50, 0)
plt.title('Female', fontsize=16)
plt.xlabel('Population percentile', fontsize=14)
plt.ylabel('Area change from PAT to MAT (%)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)

ax = plt.subplot(122)
plt.plot(perc, (perc_area_test_m_MAT - perc_area_test_m_PAT) / perc_area_test_m_PAT * 100)
ax.set_ylim(-50, 0)
plt.title('Male', fontsize=16)
plt.xlabel('Population percentile', fontsize=14)
plt.ylabel('Area change from PAT to MAT (%)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)

