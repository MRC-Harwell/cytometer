"""
Areas computed in exp 0043. Reusing code to compute ground truth areas from exp 0038.
"""

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import cytometer.utils

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

os.environ['KERAS_BACKEND'] = 'tensorflow'

import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
from cytometer.data import append_paths_to_aida_json_file, write_paths_to_aida_json_file
import PIL
import tensorflow as tf
from skimage.measure import regionprops
from skimage.morphology import watershed
import inspect
import pandas as pd

# limit GPU memory used
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))

import keras

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

saved_contour_model_basename = 'klf14_b6ntac_exp_0034_cnn_contour'
saved_dmap_model_basename = 'klf14_b6ntac_exp_0035_cnn_dmap'
saved_quality_model_basename = 'klf14_b6ntac_exp_0042_cnn_qualitynet_thresholded_sigmoid_pm_1_band_masked_segmentation'
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')

contour_model_name = saved_contour_model_basename + '*.h5'
dmap_model_name = saved_dmap_model_basename + '*.h5'
quality_model_name = saved_quality_model_basename + '*.h5'

# script name to identify this experiment
experiment_id = inspect.getfile(inspect.currentframe())
if experiment_id == '<input>':
    experiment_id = 'unknownscript'
else:
    experiment_id = os.path.splitext(os.path.basename(experiment_id))[0]

# area (pixel**2) of the smallest object we accept as a cell (pi * (16 pixel)**2 = 804.2 pixel**2)
smallest_cell_area = 804

# training window length
training_window_len = 401

# threshold for quality network
quality_threshold = 0.9


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

'''
************************************************************************************************************************
Ground-truth pipeline segmentations with Quality >= 0.9, whether they overlap hand traced cells 
or not.
Cells were segmented in exp 0036. Here, we apply the quality network.
************************************************************************************************************************
'''

quality_threshold = 0.9

'''Inference using all folds
'''

model_name = 'klf14_b6ntac_exp_0041_cnn_qualitynet_thresholded_sigmoid_pm_1_masked_segmentation_model_fold_*.h5'

saved_model_filename = os.path.join(saved_models_dir, model_name)

# list of model files to inspect
model_files = glob.glob(os.path.join(saved_models_dir, model_name))

# list of images, and indices for training vs. testing indices
contour_model_kfold_filename = os.path.join(saved_models_dir, 'klf14_b6ntac_exp_0034_cnn_contour_info.pickle')
with open(contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
im_orig_file_list = aux['file_list']
idx_orig_test_all = aux['idx_test_all']

n_folds = len(model_files)
if (n_folds != len(idx_orig_test_all)):
    raise Exception('Number of folds in dataset and model files don\'t coincide')

# process test data from all folds
n_cells = np.zeros((n_folds, ), np.float32)
df_pipeline = None
for i_fold, idx_test in enumerate(idx_orig_test_all):

    # name of model's file
    model_file = model_files[i_fold]

    print('i_fold = ' + str(i_fold) + ', model = ' + model_file)

    # load quality model
    model = keras.models.load_model(model_file)

    '''Load test data of pipeline segmentations
    '''

    # split the data list into training and testing lists
    im_test_file_list, _ = cytometer.data.split_list(im_orig_file_list, idx_test)

    # number of test images
    n_im = len(im_test_file_list)

    for i in range(n_im):

        # load training dataset
        datasets, _, _ = cytometer.data.load_datasets([im_test_file_list[i]], prefix_from='im',
                                                      prefix_to=['im', 'predlab_kfold_' + str(i_fold).zfill(2)],
                                                      nblocks=1)

        test_im = datasets['im']
        test_predlab = datasets['predlab_kfold_' + str(i_fold).zfill(2)]
        del datasets

        '''Split images into one-cell images, compute true Dice values, and prepare for inference
        '''

        # create one image per cell, and compute true Dice coefficient values, ignoring
        # cells that touch the image borders
        test_onecell_im, test_onecell_testlab, test_onecell_index_list = \
            cytometer.utils.one_image_per_label(test_im, test_predlab,
                                                training_window_len=training_window_len,
                                                smallest_cell_area=smallest_cell_area,
                                                clear_border_lab=True)

        # multiply histology by -1/+1 mask as this is what the quality network expects
        test_aux = 2 * (test_onecell_testlab.astype(np.float32) - 0.5)
        masked_test_onecell_im = test_onecell_im * np.repeat(test_aux,
                                                             repeats=test_onecell_im.shape[3], axis=3)

        '''Assess quality of each cell's segmentation with Quality Network
        '''

        # quality score
        qual = model.predict(masked_test_onecell_im)

        # add number of cells to the total of this fold
        n_cells[i_fold] += len(qual)

        # area of each segmentation
        area = np.sum(test_onecell_testlab, axis=(1, 2, 3)) * xres * yres

        if DEBUG:
            plt.clf()
            plt.subplot(121)
            j = 50
            plt.imshow(test_onecell_im[j, :, :, :])
            plt.contour(test_onecell_testlab[j, :, :, 0], levels=1, colors='green')
            plt.title('Qual = ' + str("{:.2f}".format(qual[j, 0]))
                      + ', area = ' + str("{:.1f}".format(area[j])) + ' $\mu m^2$')
            plt.text(175, 180, '+1', fontsize=14, verticalalignment='top')
            plt.text(100, 75, '-1', fontsize=14, verticalalignment='top', color='black')
            plt.subplot(122)
            i = 80
            plt.imshow(test_onecell_im[j, :, :, :])
            plt.contour(test_onecell_testlab[j, :, :, 0], levels=1, colors='red')
            plt.title('Qual = ' + str("{:.2f}".format(qual[j, 0]))
                      + ', area = ' + str("{:.1f}".format(area[j])) + ' $\mu m^2$')
            plt.text(175, 180, '+1', fontsize=14, verticalalignment='top')
            plt.text(100, 75, '-1', fontsize=14, verticalalignment='top', color='black')

        # create dataframe with metainformation from mouse
        df_window = cytometer.data.tag_values_with_mouse_info(metainfo, os.path.basename(im_test_file_list[i]),
                                                              area,
                                                              values_tag='area', tags_to_keep=['id', 'ko', 'sex'])

        # add a column with the quality values
        df_window['quality'] = qual

        # add a column with the window filename. This is later used in the linear models
        df_window['file'] = os.path.basename(im_test_file_list[i])

        # accumulate results
        if df_pipeline is None:
            df_pipeline = df_window

            # make sure that in the boxplots PAT comes before MAT
            df_pipeline['ko'] = df_pipeline['ko'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'],
                                                                                       ordered=True))
        else:
            df_pipeline = pd.concat([df_pipeline, df_window], axis=0, ignore_index=True)

'''
************************************************************************************************************************
Pipeline automatic extraction applied to full slides (both female, one MAT and one PAT).
Areas computed with exp 0043.
************************************************************************************************************************
'''

'''Load area data
'''

# list of histology files
files_list = glob.glob(os.path.join(data_dir, 'KLF14*.ndpi'))

# "KLF14-B6NTAC-MAT-18.2b  58-16 B3 - 2016-02-03 11.01.43.ndpi"
# file_i = 10; file = files_list[file_i]

# "KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38.ndpi"
file_i = 331
file = files_list[file_i]

print('File ' + str(file_i) + '/' + str(len(files_list)) + ': ' + file)

# name of file to save annotations
annotations_file = os.path.basename(file)
annotations_file = os.path.splitext(annotations_file)[0]
annotations_file = os.path.join(annotations_dir, annotations_file + '.json')

# name of file to save areas and contours
results_file = os.path.basename(file)
results_file = os.path.splitext(results_file)[0]
results_file = os.path.join(results_dir, results_file + '.npz')

# load areas
results = np.load(results_file)
area_full_pipeline_f_PAT = np.concatenate(tuple(results['areas'])) * 1e12

# "KLF14-B6NTAC-MAT-17.1b  45-16 C1 - 2016-02-01 12.23.50.ndpi"
file_i = 55
file = files_list[file_i]

print('File ' + str(file_i) + '/' + str(len(files_list)) + ': ' + file)

# name of file to save annotations
annotations_file = os.path.basename(file)
annotations_file = os.path.splitext(annotations_file)[0]
annotations_file = os.path.join(annotations_dir, annotations_file + '.json')

# name of file to save areas and contours
results_file = os.path.basename(file)
results_file = os.path.splitext(results_file)[0]
results_file = os.path.join(results_dir, results_file + '.npz')

# load areas
results = np.load(results_file)
area_full_pipeline_f_MAT = np.concatenate(tuple(results['areas'])) * 1e12

'''Compare PAT and MAT populations
'''

# plot boxplots
plt.clf()
plt.boxplot((area_full_pipeline_f_PAT, area_full_pipeline_f_MAT), notch=True, labels=('PAT', 'MAT'))

'''
************************************************************************************************************************
Compare ground truth to pipeline cells
************************************************************************************************************************
'''

plt.clf()
plt.subplot(121)
plt.boxplot((area_gtruth_f_PAT, area_full_pipeline_f_PAT), notch=True, labels=('GT', 'Pipeline'))
plt.title('Female PAT')
plt.ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.subplot(122)
plt.boxplot((area_gtruth_f_MAT, area_full_pipeline_f_MAT), notch=True, labels=('GT', 'Pipeline'))
plt.title('Female MAT')
plt.tick_params(axis='both', which='major', labelsize=14)


'''
************************************************************************************************************************
Segment folds of the training data, using the pipeline function.

Here, we take training images as inputs, and pass them to cytometer.utils.segmentation_pipeline().

With this, we replicate the segmentation in exp 0036 + quality control as in 0042 (or in the 
2nd experiment in this script).
************************************************************************************************************************
'''

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(root_data_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# list of images, and indices for training vs. testing indices
contour_model_kfold_filename = os.path.join(saved_models_dir, saved_contour_model_basename + '_info.pickle')
with open(contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
im_orig_file_list = aux['file_list']
idx_orig_test_all = aux['idx_test_all']

# read pixel size information
im = PIL.Image.open(im_orig_file_list[0])
xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

# trained models for all folds
contour_model_files = sorted(glob.glob(os.path.join(saved_models_dir, contour_model_name)))
dmap_model_files = sorted(glob.glob(os.path.join(saved_models_dir, dmap_model_name)))
quality_model_files = sorted(glob.glob(os.path.join(saved_models_dir, quality_model_name)))

df_all = []

for fold_i, idx_test in enumerate(idx_orig_test_all):

    print('Fold = ' + str(fold_i) + '/' + str(len(idx_orig_test_all) - 1))

    '''Load data 
    '''

    # split the data into training and testing datasets
    im_test_file_list, im_train_file_list = cytometer.data.split_list(im_orig_file_list, idx_test)

    # load training dataset
    datasets, _, _ = cytometer.data.load_datasets(im_test_file_list, prefix_from='im',
                                                  prefix_to=['im'], nblocks=1)
    im = datasets['im']
    del datasets

    # number of images
    n_im = im.shape[0]

    # select the models that correspond to current fold
    contour_model_file = contour_model_files[fold_i]
    dmap_model_file = dmap_model_files[fold_i]
    quality_model_file = quality_model_files[fold_i]

    # load models
    contour_model = keras.models.load_model(contour_model_file)
    dmap_model = keras.models.load_model(dmap_model_file)
    quality_model = keras.models.load_model(quality_model_file)

    '''Cell segmentation with quality control
    '''

    # segment histology
    labels, labels_info = \
        cytometer.utils.segmentation_pipeline(im,
                                              contour_model, dmap_model, quality_model,
                                              quality_model_type='-1_1_band',
                                              smallest_cell_area=smallest_cell_area)

    for i in range(im.shape[0]):

        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(im[i, :, :, :])
            plt.subplot(222)
            plt.imshow(im[i, :, :, :])
            plt.contour(labels[i, :, :, 0], levels=np.unique(labels[i, :, :, 0]), colors='C0')

        # list of labels that are on the edges
        lab_edge = cytometer.utils.edge_labels(labels[i, :, :, 0])

        # delete edge cell labels from labels_info
        idx_delete = np.where(np.logical_and(labels_info['im'] == i, np.isin(labels_info['label'], lab_edge)))[0]
        labels_info = np.delete(labels_info, idx_delete)

        # delete edge cells from the segmentation
        labels[i, :, :, 0] = np.logical_not(np.isin(labels[i, :, :, 0], lab_edge)) * labels[i, :, :, 0]

        if DEBUG:
            plt.subplot(223)
            plt.imshow(im[i, :, :, :])
            plt.contour(labels[i, :, :, 0], levels=np.unique(labels[i, :, :, 0]), colors='C0')

        # list of labels that the quality network rejects
        idx_bad = np.logical_and(labels_info['im'] == i, labels_info['quality'] < quality_threshold)
        lab_bad = labels_info['label'][idx_bad]

        # delete bad labels from labels_info
        labels_info = np.delete(labels_info, idx_bad)

        # delete bad cells from the segmentation
        labels[i, :, :, 0] = np.logical_not(np.isin(labels[i, :, :, 0], lab_bad)) * labels[i, :, :, 0]

        if DEBUG:
            plt.subplot(224)
            plt.imshow(im[i, :, :, :])
            plt.contour(labels[i, :, :, 0], levels=np.unique(labels[i, :, :, 0]), colors='C0')

        # compute cell areas
        props = regionprops(labels[i, :, :, 0])
        p_label = [p['label'] for p in props]
        p_area = np.array([p['area'] for p in props])
        areas = p_area * xres * yres  # (m^2)

        # create dataframe with mouse metainformation and area values
        df = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(im_test_file_list[i]),
                                                       values=areas, values_tag='area',
                                                       tags_to_keep=['id', 'ko', 'sex'])

        # concatenate results
        if len(df_all) == 0:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])

plt.clf()
plt.boxplot(areas_all)
