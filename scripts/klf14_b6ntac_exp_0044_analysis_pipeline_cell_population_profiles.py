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
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

os.environ['KERAS_BACKEND'] = 'tensorflow'

import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
from cytometer.data import append_paths_to_aida_json_file
import PIL
import tensorflow as tf
from skimage.measure import regionprops
from skimage.morphology import watershed
from sklearn.metrics import confusion_matrix
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
annotations_dir = os.path.join(home, 'Data/cytometer_data/aida_data_Klf14/annotations')
metainfo_dir = os.path.join(home, 'Data/cytometer_data/klf14')

saved_contour_model_basename = 'klf14_b6ntac_exp_0034_cnn_contour'
saved_dmap_model_basename = 'klf14_b6ntac_exp_0035_cnn_dmap'
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')

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

# thresholds for quality network
dice_threshold = 0.9
quality_threshold = 0.5


'''
************************************************************************************************************************
Hand segmented cells (ground truth), no-overlap approximation
************************************************************************************************************************
'''

'''Load data
'''

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
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
                                                          area, values_tag='area', tags_to_keep=['id', 'ko_parent', 'sex'])

    # add a column with the window filename. This is later used in the linear models
    df_window['file'] = os.path.basename(full_file_list['im'][i])

    # create new total dataframe, or concat to existing one
    if df_gtruth is None:
        df_gtruth = df_window
    else:
        df_gtruth = pd.concat([df_gtruth, df_window], axis=0, ignore_index=True)


# make sure that in the boxplots PAT comes before MAT
df_gtruth['ko_parent'] = df_gtruth['ko_parent'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))

# plot boxplots for f/m, PAT/MAT comparison as in Nature Genetics paper
plt.clf()
ax = plt.subplot(121)
df_gtruth[df_gtruth['sex'] == 'f'].boxplot(column='area', by='ko_parent', ax=ax, notch=True)
#ax.set_ylim(0, 2e4)
ax.set_title('female', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
ax = plt.subplot(122)
df_gtruth[df_gtruth['sex'] == 'm'].boxplot(column='area', by='ko_parent', ax=ax, notch=True)
#ax.set_ylim(0, 2e4)
ax.set_title('male', fontsize=16)
ax.set_xlabel('')
plt.tick_params(axis='both', which='major', labelsize=14)

# split data into groups
area_gtruth_f_PAT = df_gtruth['area'][(np.logical_and(df_gtruth['sex'] == 'f', df_gtruth['ko_parent'] == 'PAT'))]
area_gtruth_f_MAT = df_gtruth['area'][(np.logical_and(df_gtruth['sex'] == 'f', df_gtruth['ko_parent'] == 'MAT'))]
area_gtruth_m_PAT = df_gtruth['area'][(np.logical_and(df_gtruth['sex'] == 'm', df_gtruth['ko_parent'] == 'PAT'))]
area_gtruth_m_MAT = df_gtruth['area'][(np.logical_and(df_gtruth['sex'] == 'm', df_gtruth['ko_parent'] == 'MAT'))]

# compute percentile profiles of cell populations
perc = np.linspace(0, 100, num=101)
perc_area_gtruth_f_PAT = np.percentile(area_gtruth_f_PAT, perc)
perc_area_gtruth_f_MAT = np.percentile(area_gtruth_f_MAT, perc)
perc_area_gtruth_m_PAT = np.percentile(area_gtruth_m_PAT, perc)
perc_area_gtruth_m_MAT = np.percentile(area_gtruth_m_MAT, perc)

'''
************************************************************************************************************************
Exp 0042:

Segment all folds of the training data, using the pipeline function.

Here, we take training images as inputs, and pass them to cytometer.utils.segmentation_pipeline().

With this, we replicate the segmentation in exp 0036 + quality control as in 0042 (or in the 
2nd experiment in this script).

Quality mask: +1 / -1 band of 75 pixels / 0
************************************************************************************************************************
'''

# quality network
saved_quality_model_basename = 'klf14_b6ntac_exp_0042_cnn_qualitynet_thresholded_sigmoid_pm_1_band_masked_segmentation'
quality_model_name = saved_quality_model_basename + '*.h5'

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
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

df_gtruth_pipeline_good = []
df_gtruth_pipeline_bad = []

for i_fold, idx_test in enumerate(idx_orig_test_all):

    print('Fold = ' + str(i_fold) + '/' + str(len(idx_orig_test_all) - 1))

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
    contour_model_file = contour_model_files[i_fold]
    dmap_model_file = dmap_model_files[i_fold]
    quality_model_file = quality_model_files[i_fold]

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

        # compute cell areas
        props = regionprops(labels[i, :, :, 0])
        p_label = [p['label'] for p in props]
        p_area = np.array([p['area'] for p in props])
        areas = p_area * xres * yres  # (m^2)

        # delete bad labels from labels_info
        labels_info = np.delete(labels_info, idx_bad)

        # delete bad cells from the segmentation
        labels[i, :, :, 0] = np.logical_not(np.isin(labels[i, :, :, 0], lab_bad)) * labels[i, :, :, 0]

        if DEBUG:
            plt.subplot(224)
            plt.imshow(im[i, :, :, :])
            plt.contour(labels[i, :, :, 0], levels=np.unique(labels[i, :, :, 0]), colors='C0')

        # split areas into good objects and bad objects
        idx_bad = np.isin(p_label, lab_bad)
        idx_good = np.logical_not(idx_bad)

        # create dataframe with mouse metainformation and area values
        df_bad = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(im_test_file_list[i]),
                                                           values=areas[idx_bad], values_tag='area',
                                                           tags_to_keep=['id', 'ko_parent', 'sex'])
        df_good = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(im_test_file_list[i]),
                                                            values=areas[idx_good], values_tag='area',
                                                            tags_to_keep=['id', 'ko_parent', 'sex'])

        # concatenate results
        if len(df_gtruth_pipeline_good) == 0:
            df_gtruth_pipeline_good = df_good
        else:
            df_gtruth_pipeline_good = pd.concat([df_gtruth_pipeline_good, df_good])
        if len(df_gtruth_pipeline_bad) == 0:
            df_gtruth_pipeline_bad = df_bad
        else:
            df_gtruth_pipeline_bad = pd.concat([df_gtruth_pipeline_bad, df_bad])

# split data into groups
area_gtruth_pipeline_good_f_PAT = df_gtruth_pipeline_good['area'][(np.logical_and(df_gtruth_pipeline_good['sex'] == 'f',
                                                                                  df_gtruth_pipeline_good['ko_parent'] == 'PAT'))]
area_gtruth_pipeline_good_f_MAT = df_gtruth_pipeline_good['area'][(np.logical_and(df_gtruth_pipeline_good['sex'] == 'f',
                                                                                  df_gtruth_pipeline_good['ko_parent'] == 'MAT'))]
area_gtruth_pipeline_good_m_PAT = df_gtruth_pipeline_good['area'][(np.logical_and(df_gtruth_pipeline_good['sex'] == 'm',
                                                                                  df_gtruth_pipeline_good['ko_parent'] == 'PAT'))]
area_gtruth_pipeline_good_m_MAT = df_gtruth_pipeline_good['area'][(np.logical_and(df_gtruth_pipeline_good['sex'] == 'm',
                                                                                  df_gtruth_pipeline_good['ko_parent'] == 'MAT'))]

area_gtruth_pipeline_bad_f_PAT = df_gtruth_pipeline_bad['area'][(np.logical_and(df_gtruth_pipeline_bad['sex'] == 'f',
                                                                                df_gtruth_pipeline_bad['ko_parent'] == 'PAT'))]
area_gtruth_pipeline_bad_f_MAT = df_gtruth_pipeline_bad['area'][(np.logical_and(df_gtruth_pipeline_bad['sex'] == 'f',
                                                                                  df_gtruth_pipeline_bad['ko_parent'] == 'MAT'))]
area_gtruth_pipeline_bad_m_PAT = df_gtruth_pipeline_bad['area'][(np.logical_and(df_gtruth_pipeline_bad['sex'] == 'm',
                                                                                  df_gtruth_pipeline_bad['ko_parent'] == 'PAT'))]
area_gtruth_pipeline_bad_m_MAT = df_gtruth_pipeline_bad['area'][(np.logical_and(df_gtruth_pipeline_bad['sex'] == 'm',
                                                                                  df_gtruth_pipeline_bad['ko_parent'] == 'MAT'))]

# plot results
if DEBUG:
    plt.clf()
    plt.boxplot((area_gtruth_f_PAT, area_gtruth_pipeline_good_f_PAT, area_gtruth_pipeline_bad_f_PAT,
                 area_gtruth_f_MAT, area_gtruth_pipeline_good_f_MAT, area_gtruth_pipeline_bad_f_MAT),
                notch=True, labels=('PAT$_{GT}$', 'PAT$_{GT/P,good}$', 'PAT$_{GT/P,bad}$',
                                    'MAT$_{GT}$', 'MAT$_{GT/P,good}$', 'MAT$_{GT/P,bad}$'),
                positions=(0, 1, 2, 4, 5, 6))
    plt.ylabel('area  ($\mu m^2)$', fontsize=14)
    plt.title('Female')
    plt.tick_params(axis='both', which='major', labelsize=14)

    if SAVE_FIGS:
        plt.savefig(
            os.path.join(figures_dir, 'klf14_b6ntac_exp_0044_area_boxplots_quality_rejection_bias_female.png'))

    plt.clf()
    plt.boxplot((area_gtruth_m_PAT, area_gtruth_pipeline_good_m_PAT, area_gtruth_pipeline_bad_m_PAT,
                 area_gtruth_m_MAT, area_gtruth_pipeline_good_m_MAT, area_gtruth_pipeline_bad_m_MAT),
                notch=True, labels=('PAT$_{GT}$', 'PAT$_{GT/P,good}$', 'PAT$_{GT/P,bad}$',
                                    'MAT$_{GT}$', 'MAT$_{GT/P,good}$', 'MAT$_{GT/P,bad}$'),
                positions=(0, 1, 2, 4, 5, 6))
    plt.ylabel('area  ($\mu m^2)$', fontsize=14)
    plt.title('Male')
    plt.tick_params(axis='both', which='major', labelsize=14)

    if SAVE_FIGS:
        plt.savefig(
            os.path.join(figures_dir, 'klf14_b6ntac_exp_0044_area_boxplots_quality_rejection_bias_male.png'))

'''
************************************************************************************************************************
Exp 0045:

Segment all folds of the training data, using the pipeline function.

Here, we take training images as inputs, and pass them to cytometer.utils.segmentation_pipeline().

With this, we replicate the segmentation in exp 0036 + quality control as in 0045.

Quality mask: +1 / -1 band of 20% equivalent radius / 0
Quality trained with binary cross-entropy.
************************************************************************************************************************
'''

# quality network
saved_quality_model_basename = 'klf14_b6ntac_exp_0045_cnn_qualitynet_thresholded_sigmoid_pm_1_prop_band_masked_segmentation'
quality_model_name = saved_quality_model_basename + '*.h5'

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
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

df_gtruth_pipeline_good = []
df_gtruth_pipeline_bad = []

for i_fold, idx_test in enumerate(idx_orig_test_all):

    print('Fold = ' + str(i_fold) + '/' + str(len(idx_orig_test_all) - 1))

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
    contour_model_file = contour_model_files[i_fold]
    dmap_model_file = dmap_model_files[i_fold]
    quality_model_file = quality_model_files[i_fold]

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
                                              quality_model_type='-1_1_prop_band',
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

        # compute cell areas
        props = regionprops(labels[i, :, :, 0])
        p_label = [p['label'] for p in props]
        p_area = np.array([p['area'] for p in props])
        areas = p_area * xres * yres  # (m^2)

        # delete bad labels from labels_info
        labels_info = np.delete(labels_info, idx_bad)

        # delete bad cells from the segmentation
        labels[i, :, :, 0] = np.logical_not(np.isin(labels[i, :, :, 0], lab_bad)) * labels[i, :, :, 0]

        if DEBUG:
            plt.subplot(224)
            plt.imshow(im[i, :, :, :])
            plt.contour(labels[i, :, :, 0], levels=np.unique(labels[i, :, :, 0]), colors='C0')

        # split areas into good objects and bad objects
        idx_bad = np.isin(p_label, lab_bad)
        idx_good = np.logical_not(idx_bad)

        # create dataframe with mouse metainformation and area values
        df_bad = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(im_test_file_list[i]),
                                                           values=areas[idx_bad], values_tag='area',
                                                           tags_to_keep=['id', 'ko_parent', 'sex'])
        df_good = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(im_test_file_list[i]),
                                                            values=areas[idx_good], values_tag='area',
                                                            tags_to_keep=['id', 'ko_parent', 'sex'])

        # concatenate results
        if len(df_gtruth_pipeline_good) == 0:
            df_gtruth_pipeline_good = df_good
        else:
            df_gtruth_pipeline_good = pd.concat([df_gtruth_pipeline_good, df_good])
        if len(df_gtruth_pipeline_bad) == 0:
            df_gtruth_pipeline_bad = df_bad
        else:
            df_gtruth_pipeline_bad = pd.concat([df_gtruth_pipeline_bad, df_bad])

# split data into groups
area_gtruth_pipeline_good_f_PAT = df_gtruth_pipeline_good['area'][(np.logical_and(df_gtruth_pipeline_good['sex'] == 'f',
                                                                                  df_gtruth_pipeline_good['ko_parent'] == 'PAT'))]
area_gtruth_pipeline_good_f_MAT = df_gtruth_pipeline_good['area'][(np.logical_and(df_gtruth_pipeline_good['sex'] == 'f',
                                                                                  df_gtruth_pipeline_good['ko_parent'] == 'MAT'))]
area_gtruth_pipeline_good_m_PAT = df_gtruth_pipeline_good['area'][(np.logical_and(df_gtruth_pipeline_good['sex'] == 'm',
                                                                                  df_gtruth_pipeline_good['ko_parent'] == 'PAT'))]
area_gtruth_pipeline_good_m_MAT = df_gtruth_pipeline_good['area'][(np.logical_and(df_gtruth_pipeline_good['sex'] == 'm',
                                                                                  df_gtruth_pipeline_good['ko_parent'] == 'MAT'))]

area_gtruth_pipeline_bad_f_PAT = df_gtruth_pipeline_bad['area'][(np.logical_and(df_gtruth_pipeline_bad['sex'] == 'f',
                                                                                df_gtruth_pipeline_bad['ko_parent'] == 'PAT'))]
area_gtruth_pipeline_bad_f_MAT = df_gtruth_pipeline_bad['area'][(np.logical_and(df_gtruth_pipeline_bad['sex'] == 'f',
                                                                                  df_gtruth_pipeline_bad['ko_parent'] == 'MAT'))]
area_gtruth_pipeline_bad_m_PAT = df_gtruth_pipeline_bad['area'][(np.logical_and(df_gtruth_pipeline_bad['sex'] == 'm',
                                                                                  df_gtruth_pipeline_bad['ko_parent'] == 'PAT'))]
area_gtruth_pipeline_bad_m_MAT = df_gtruth_pipeline_bad['area'][(np.logical_and(df_gtruth_pipeline_bad['sex'] == 'm',
                                                                                  df_gtruth_pipeline_bad['ko_parent'] == 'MAT'))]

# plot results
if DEBUG:
    plt.clf()
    plt.boxplot((area_gtruth_f_PAT, area_gtruth_pipeline_good_f_PAT, area_gtruth_pipeline_bad_f_PAT,
                 area_gtruth_f_MAT, area_gtruth_pipeline_good_f_MAT, area_gtruth_pipeline_bad_f_MAT),
                notch=True, labels=('PAT$_{GT}$', 'PAT$_{GT/P,good}$', 'PAT$_{GT/P,bad}$',
                                    'MAT$_{GT}$', 'MAT$_{GT/P,good}$', 'MAT$_{GT/P,bad}$'),
                positions=(0, 1, 2, 4, 5, 6))
    plt.ylabel('area  ($\mu m^2)$', fontsize=14)
    plt.title('Female')
    plt.tick_params(axis='both', which='major', labelsize=14)

    if SAVE_FIGS:
        plt.savefig(
            os.path.join(figures_dir, 'klf14_b6ntac_exp_0044_area_boxplots_quality_rejection_bias_female_quality_prop_band.png'))

    plt.clf()
    plt.boxplot((area_gtruth_m_PAT, area_gtruth_pipeline_good_m_PAT, area_gtruth_pipeline_bad_m_PAT,
                 area_gtruth_m_MAT, area_gtruth_pipeline_good_m_MAT, area_gtruth_pipeline_bad_m_MAT),
                notch=True, labels=('PAT$_{GT}$', 'PAT$_{GT/P,good}$', 'PAT$_{GT/P,bad}$',
                                    'MAT$_{GT}$', 'MAT$_{GT/P,good}$', 'MAT$_{GT/P,bad}$'),
                positions=(0, 1, 2, 4, 5, 6))
    plt.ylabel('area  ($\mu m^2)$', fontsize=14)
    plt.title('Male')
    plt.tick_params(axis='both', which='major', labelsize=14)

    if SAVE_FIGS:
        plt.savefig(
            os.path.join(figures_dir, 'klf14_b6ntac_exp_0044_area_boxplots_quality_rejection_bias_male_quality_prop_band.png'))

'''
************************************************************************************************************************
Exp 0046:

Segment all folds of the training data, using the pipeline function.

Here, we take training images as inputs, and pass them to cytometer.utils.segmentation_pipeline().

With this, we replicate the segmentation in exp 0036 + quality control as in 0046.

Quality mask: +1 / -1 band of 20% equivalent radius / 0
Quality trained with focal loss.
************************************************************************************************************************
'''

# quality network
saved_quality_model_basename = 'klf14_b6ntac_exp_0046_cnn_qualitynet_thresholded_sigmoid_pm_1_prop_band_masked_segmentation_focal_loss'
quality_model_name = saved_quality_model_basename + '*.h5'

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# list of images, and indices for training vs. testing indices
contour_model_kfold_filename = os.path.join(saved_models_dir, saved_contour_model_basename + '_info.pickle')
with open(contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
im_orig_file_list = aux['file_list']
idx_orig_test_all = aux['idx_test_all']

# change home directory
im_orig_file_list = cytometer.data.change_home_directory(im_orig_file_list, home_path_from='/users/rittscher/rcasero',
                                                         home_path_to=home,
                                                         check_isfile=False)

# read pixel size information
im = PIL.Image.open(im_orig_file_list[0])
xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

df_gtruth_pipeline = []

for i_fold, idx_test in enumerate(idx_orig_test_all):

    print('Fold = ' + str(i_fold) + '/' + str(len(idx_orig_test_all) - 1))

    '''Load data 
    '''

    # split the data into training and testing datasets
    im_test_file_list, im_train_file_list = cytometer.data.split_list(im_orig_file_list, idx_test)

    # load training dataset
    datasets, _, _ = cytometer.data.load_datasets(im_test_file_list, prefix_from='im',
                                                  prefix_to=['im', 'lab'], nblocks=1)
    im = datasets['im']
    reflab = datasets['lab']
    del datasets

    # number of images
    n_im = im.shape[0]

    # select the models that correspond to current fold
    contour_model_file = os.path.join(saved_models_dir, saved_contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    dmap_model_file = os.path.join(saved_models_dir, saved_dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    quality_model_file = os.path.join(saved_models_dir, saved_quality_model_basename + '_model_fold_' + str(i_fold) + '.h5')

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
                                              quality_model_type='-1_1_prop_band',
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
            plt.contour(reflab[i, :, :, 0], levels=np.unique(labels[i, :, :, 0]), colors='C1')

        # compute cell areas from non-edge cells
        props = regionprops(labels[i, :, :, 0])
        p_label = [p['label'] for p in props]
        p_area = np.array([p['area'] for p in props])
        areas = p_area * xres * yres  # (m^2)

        # create dataframe: one cell per row, tagged with mouse metainformation
        df = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(im_test_file_list[i]),
                                                       values=areas, values_tag='area',
                                                       tags_to_keep=['id', 'ko_parent', 'sex'])

        # add to dataframe: image index and cell label
        df['im'] = i
        df['label'] = p_label

        # add to dataframe: Dice coefficients
        # get Dice value for each non-edge automatic segmentation. These Dice values are
        # computed by comparing the automatic segmentation to the manual segmentation. When there's no manual
        # segmentation to compare with, we assume Dice = 0. This is not a perfect choice, as sometimes the pipeline may
        # spot a cell that the human operator didn't, but in general we are going to assume that if the human operator
        # didn't hand-traced an object it's because it was not a well formed white adipocyte.
        #
        # We then create a look up table (LUT) so that we can sort the values according to the labels in labels_info
        # efficiently.
        dice_info = cytometer.utils.match_overlapping_labels(labels_test=labels[i, :, :, 0],
                                                             labels_ref=reflab[i, :, :, 0])

        dice_lut = np.zeros(shape=(np.max(labels[i, :, :, 0]) + 1, ))
        dice_lut[dice_info['lab_test']] = dice_info['dice']

        df['dice'] = dice_lut[p_label]

        # add to dataframe: quality scores
        idx = labels_info['im'] == i
        assert(np.all(p_label == labels_info[idx]['label']))
        df['quality'] = labels_info[idx]['quality']

        ## Delete bad segmentations: this is only for display

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

        # concatenate results
        if len(df_gtruth_pipeline) == 0:
            df_gtruth_pipeline = df
        else:
            df_gtruth_pipeline = pd.concat([df_gtruth_pipeline, df])

#np.savez('/tmp/foo.npz', df_gtruth_pipeline=df_gtruth_pipeline)

# split data into groups
idx_good = np.array(df_gtruth_pipeline['quality'] >= quality_threshold)
idx_f = np.array(df_gtruth_pipeline['sex'] == 'f')
idx_pat = np.array(df_gtruth_pipeline['ko_parent'] == 'PAT')

area_gtruth_pipeline_good_f_PAT = df_gtruth_pipeline['area'][idx_good * idx_f * idx_pat]
area_gtruth_pipeline_good_f_MAT = df_gtruth_pipeline['area'][idx_good * idx_f * ~idx_pat]
area_gtruth_pipeline_good_m_PAT = df_gtruth_pipeline['area'][idx_good * ~idx_f * idx_pat]
area_gtruth_pipeline_good_m_MAT = df_gtruth_pipeline['area'][idx_good * ~idx_f * ~idx_pat]

area_gtruth_pipeline_bad_f_PAT = df_gtruth_pipeline['area'][~idx_good * idx_f * idx_pat]
area_gtruth_pipeline_bad_f_MAT = df_gtruth_pipeline['area'][~idx_good * idx_f * ~idx_pat]
area_gtruth_pipeline_bad_m_PAT = df_gtruth_pipeline['area'][~idx_good * ~idx_f * idx_pat]
area_gtruth_pipeline_bad_m_MAT = df_gtruth_pipeline['area'][~idx_good * ~idx_f * ~idx_pat]

# plot results
if DEBUG:
    plt.clf()
    plt.boxplot((area_gtruth_f_PAT, area_gtruth_pipeline_good_f_PAT, area_gtruth_pipeline_bad_f_PAT,
                 area_gtruth_f_MAT, area_gtruth_pipeline_good_f_MAT, area_gtruth_pipeline_bad_f_MAT),
                notch=True, labels=('PAT$_{GT}$', 'PAT$_{GT/P,good}$', 'PAT$_{GT/P,bad}$',
                                    'MAT$_{GT}$', 'MAT$_{GT/P,good}$', 'MAT$_{GT/P,bad}$'),
                positions=(0, 1, 2, 4, 5, 6))
    plt.ylabel('area  ($\mu m^2)$', fontsize=14)
    plt.title('Female')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim((0, 15000))

    if SAVE_FIGS:
        plt.savefig(
            os.path.join(figures_dir, 'klf14_b6ntac_exp_0044_area_boxplots_quality_rejection_bias_female_quality_prop_band_focal_loss.png'))

    plt.clf()
    plt.boxplot((area_gtruth_m_PAT, area_gtruth_pipeline_good_m_PAT, area_gtruth_pipeline_bad_m_PAT,
                 area_gtruth_m_MAT, area_gtruth_pipeline_good_m_MAT, area_gtruth_pipeline_bad_m_MAT),
                notch=True, labels=('PAT$_{GT}$', 'PAT$_{GT/P,good}$', 'PAT$_{GT/P,bad}$',
                                    'MAT$_{GT}$', 'MAT$_{GT/P,good}$', 'MAT$_{GT/P,bad}$'),
                positions=(0, 1, 2, 4, 5, 6))
    plt.ylabel('area  ($\mu m^2)$', fontsize=14)
    plt.title('Male')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim((0, 15000))

    if SAVE_FIGS:
        plt.savefig(
            os.path.join(figures_dir, 'klf14_b6ntac_exp_0044_area_boxplots_quality_rejection_bias_male_quality_prop_band_focal_loss.png'))

# Compute confusion matrix for all cells together
if DEBUG:
    y_true = df_gtruth_pipeline['dice'] >= dice_threshold
    y_pred = df_gtruth_pipeline['quality'] >= quality_threshold

    cytometer.utils.plot_confusion_matrix(y_true, y_pred,
                                          normalize=True,
                                          title='All cells',
                                          xlabel='Predict Quality $\geq$ 0.5',
                                          ylabel='Ground-truth Dice $\geq$ 0.9',
                                          cmap=plt.cm.Blues)

    # confusion matrix for females
    df_gtruth_pipeline_female = df_gtruth_pipeline.loc[df_gtruth_pipeline['sex'] == 'f', :]
    y_true = df_gtruth_pipeline_female['dice'] >= dice_threshold
    y_pred = df_gtruth_pipeline_female['quality'] >= quality_threshold

    cytometer.utils.plot_confusion_matrix(y_true, y_pred,
                                          normalize=True,
                                          title='Female',
                                          xlabel='Predict Quality $\geq$ 0.5',
                                          ylabel='Ground-truth Dice $\geq$ 0.9',
                                          cmap=plt.cm.Blues)

    # confusion matrix for males
    df_gtruth_pipeline_male = df_gtruth_pipeline.loc[df_gtruth_pipeline['sex'] == 'm', :]
    y_true = df_gtruth_pipeline_male['dice'] >= dice_threshold
    y_pred = df_gtruth_pipeline_male['quality'] >= quality_threshold

    cytometer.utils.plot_confusion_matrix(y_true, y_pred,
                                          normalize=True,
                                          title='Male',
                                          xlabel='Predict Quality $\geq$ 0.5',
                                          ylabel='Ground-truth Dice $\geq$ 0.9',
                                          cmap=plt.cm.Blues)

# compute sensitivity and specificity over intervals of cell area
area_intervals = list(range(0, 8000, 250)) + [np.Inf, ]
sensitivity = np.zeros(shape=(len(area_intervals) - 1, ))
specificity = np.zeros(shape=(len(area_intervals) - 1, ))
for i in range(len(area_intervals) - 1):

    df = df_gtruth_pipeline.loc[np.logical_and(df_gtruth_pipeline['area'] >= area_intervals[i],
                                               df_gtruth_pipeline['area'] < area_intervals[i + 1]), :]

    # sensitivity = TP / P
    #  TP = Dice >= 0.9 & quality >= 0.5
    #  P  = Dice >= 0.9
    TP = np.count_nonzero(np.logical_and(df['dice'] >= dice_threshold, df['quality'] >= quality_threshold))
    P = np.count_nonzero(df['dice'] >= dice_threshold)
    if P == 0:
        sensitivity[i] = np.nan
    else:
        sensitivity[i] = TP / P

    # specificity = TN / N
    #  TN = Dice < 0.9 & quality < 0.5
    #  N  = Dice < 0.9
    TN = np.count_nonzero(np.logical_and(df['dice'] < dice_threshold, df['quality'] < quality_threshold))
    N = np.count_nonzero(df['dice'] < dice_threshold)
    if N == 0:
        specificity[i] = np.nan
    else:
        specificity[i] = TN / N

if DEBUG:
    plt.clf()
    area_midpoints = (np.array(area_intervals[0:-1]) + np.array(area_intervals[1:]))/2.0
    plt.plot(area_midpoints, sensitivity, label='Sensitivity')
    plt.plot(area_midpoints, specificity, label='Specificity')
    plt.legend()
    plt.xlabel('area ($\mu m^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)

    if SAVE_FIGS:
        plt.savefig(
            os.path.join(figures_dir, 'klf14_b6ntac_exp_0044_pipeline_sensitivity_specificity.png'))

# compute plot of cell area vs. Dice/quality value

if DEBUG:


    plt.clf()
    plt.subplot(121)
    plt.plot(df_gtruth_pipeline['area'][df_gtruth_pipeline['dice'] < dice_threshold],
             df_gtruth_pipeline['dice'][df_gtruth_pipeline['dice'] < dice_threshold], '.C0')
    plt.plot(df_gtruth_pipeline['area'][df_gtruth_pipeline['dice'] >= dice_threshold],
             df_gtruth_pipeline['dice'][df_gtruth_pipeline['dice'] >= dice_threshold], '.C1')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Cell area', fontsize=14)
    plt.ylabel('Dice', fontsize=14)

    plt.subplot(122)

    # quality < 0.5, dice < 0.9
    idx = (df_gtruth_pipeline['quality'] < quality_threshold) * (df_gtruth_pipeline['dice'] < dice_threshold)
    plt.plot(df_gtruth_pipeline['dice'][idx], df_gtruth_pipeline['quality'][idx], '.C0', label="D<0.9, Q<0.5")
    # quality >= 0.5, dice < 0.9
    idx = (df_gtruth_pipeline['quality'] >= quality_threshold) * (df_gtruth_pipeline['dice'] < dice_threshold)
    plt.plot(df_gtruth_pipeline['dice'][idx], df_gtruth_pipeline['quality'][idx], '.C1', label="D<0.9, Q$\geq$0.5")
    # quality < 0.5, dice >= 0.9
    idx = (df_gtruth_pipeline['quality'] < quality_threshold) * (df_gtruth_pipeline['dice'] >= dice_threshold)
    plt.plot(df_gtruth_pipeline['dice'][idx], df_gtruth_pipeline['quality'][idx], '.C2', label="D$\geq$0.9, Q<0.5")
    # quality >= 0.5, dice >= 0.9
    idx = (df_gtruth_pipeline['quality'] >= quality_threshold) * (df_gtruth_pipeline['dice'] >= dice_threshold)
    plt.plot(df_gtruth_pipeline['dice'][idx], df_gtruth_pipeline['quality'][idx], '.C3', label="D$\geq$0.9, Q$\geq$0.5")

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()
    plt.xlabel('Dice', fontsize=14)
    plt.ylabel('Quality', fontsize=14)

    # plot Dice vs. quality, and colour according to size
    plt.clf()
    plt.scatter(df_gtruth_pipeline['dice'], df_gtruth_pipeline['quality'], c=np.log(df_gtruth_pipeline['area']+1), s=5)
    plt.colorbar()
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Dice', fontsize=14)
    plt.ylabel('Quality', fontsize=14)

'''
************************************************************************************************************************
Exp 0048:

Training all folds of DenseNet for quality assessement of individual cells based on classification of
thresholded Dice coefficient (Dice >= 0.9). Here the loss is binary focal loss.

The reason is to center the decision boundary on 0.9, to get finer granularity around that threshold.

Mask one-cell histology windows with 0/-1/+1 mask. The mask has a band with a width of 20% the equivalent radius
of the cell (equivalent radius is the radius of a circle with the same area as the cell).

We then convert the images to polar coordinates before training.

This is part of a series of experiments with different types of masks: 0039, 0040, 0041, 0042 and 0045.
************************************************************************************************************************
'''

# quality network
saved_quality_model_basename = 'klf14_b6ntac_exp_0048_cnn_qualitynet_prop_band_focal_loss_polar'
quality_model_name = saved_quality_model_basename + '*.h5'

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# list of images, and indices for training vs. testing indices
contour_model_kfold_filename = os.path.join(saved_models_dir, saved_contour_model_basename + '_info.pickle')
with open(contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
im_orig_file_list = aux['file_list']
idx_orig_test_all = aux['idx_test_all']

# change home directory
im_orig_file_list = cytometer.data.change_home_directory(im_orig_file_list, home_path_from='/users/rittscher/rcasero',
                                                         home_path_to=home,
                                                         check_isfile=False)

# read pixel size information
im = PIL.Image.open(im_orig_file_list[0])
xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

df_gtruth_pipeline = []

for i_fold, idx_test in enumerate(idx_orig_test_all):

    print('Fold = ' + str(i_fold) + '/' + str(len(idx_orig_test_all) - 1))

    '''Load data 
    '''

    # split the data into training and testing datasets
    im_test_file_list, im_train_file_list = cytometer.data.split_list(im_orig_file_list, idx_test)

    # load training dataset
    datasets, _, _ = cytometer.data.load_datasets(im_test_file_list, prefix_from='im',
                                                  prefix_to=['im', 'lab'], nblocks=1)
    im = datasets['im']
    reflab = datasets['lab']
    del datasets

    # number of images
    n_im = im.shape[0]

    # select the models that correspond to current fold
    contour_model_file = os.path.join(saved_models_dir, saved_contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    dmap_model_file = os.path.join(saved_models_dir, saved_dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    quality_model_file = os.path.join(saved_models_dir, saved_quality_model_basename + '_model_fold_' + str(i_fold) + '.h5')

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
                                              quality_model_type='-1_1_prop_band',
                                              quality_model_preprocessing='polar',
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
            plt.contour(reflab[i, :, :, 0], levels=np.unique(labels[i, :, :, 0]), colors='C1')

        # compute cell areas from non-edge cells
        props = regionprops(labels[i, :, :, 0])
        p_label = [p['label'] for p in props]
        p_area = np.array([p['area'] for p in props])
        areas = p_area * xres * yres  # (m^2)

        # create dataframe: one cell per row, tagged with mouse metainformation
        df = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(im_test_file_list[i]),
                                                       values=areas, values_tag='area',
                                                       tags_to_keep=['id', 'ko_parent', 'sex'])

        # add to dataframe: image index and cell label
        df['im'] = i
        df['label'] = p_label

        # add to dataframe: Dice coefficients
        # get Dice value for each non-edge automatic segmentation. These Dice values are
        # computed by comparing the automatic segmentation to the manual segmentation. When there's no manual
        # segmentation to compare with, we assume Dice = 0. This is not a perfect choice, as sometimes the pipeline may
        # spot a cell that the human operator didn't, but in general we are going to assume that if the human operator
        # didn't hand-traced an object it's because it was not a well formed white adipocyte.
        #
        # We then create a look up table (LUT) so that we can sort the values according to the labels in labels_info
        # efficiently.
        dice_info = cytometer.utils.match_overlapping_labels(labels_test=labels[i, :, :, 0],
                                                             labels_ref=reflab[i, :, :, 0])

        dice_lut = np.zeros(shape=(np.max(labels[i, :, :, 0]) + 1, ))
        dice_lut[dice_info['lab_test']] = dice_info['dice']

        df['dice'] = dice_lut[p_label]

        # add to dataframe: quality scores
        idx = labels_info['im'] == i
        assert(np.all(p_label == labels_info[idx]['label']))
        df['quality'] = labels_info[idx]['quality']

        ## Delete bad segmentations: this last block of code is only useful to display results

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

        # concatenate results
        if len(df_gtruth_pipeline) == 0:
            df_gtruth_pipeline = df
        else:
            df_gtruth_pipeline = pd.concat([df_gtruth_pipeline, df])

#np.savez('/tmp/foo.npz', df_gtruth_pipeline=df_gtruth_pipeline)

# delete segmentations with Dice < 0.2, because those don't really have a ground truth. For example,
# the segmentation itself may be a good segmentation of a cell, but that cell had no ground truth.
# So instead, the segmentation is just touching the edge of the ground truth of a neighbour cell
idx = df_gtruth_pipeline['dice'] >= 0.2
df_gtruth_pipeline = df_gtruth_pipeline.loc[idx, :]

# split data into groups
idx_good = np.array(df_gtruth_pipeline['quality'] >= quality_threshold)
idx_f = np.array(df_gtruth_pipeline['sex'] == 'f')
idx_pat = np.array(df_gtruth_pipeline['ko_parent'] == 'PAT')

area_gtruth_pipeline_good_f_PAT = df_gtruth_pipeline['area'][idx_good * idx_f * idx_pat]
area_gtruth_pipeline_good_f_MAT = df_gtruth_pipeline['area'][idx_good * idx_f * ~idx_pat]
area_gtruth_pipeline_good_m_PAT = df_gtruth_pipeline['area'][idx_good * ~idx_f * idx_pat]
area_gtruth_pipeline_good_m_MAT = df_gtruth_pipeline['area'][idx_good * ~idx_f * ~idx_pat]

area_gtruth_pipeline_bad_f_PAT = df_gtruth_pipeline['area'][~idx_good * idx_f * idx_pat]
area_gtruth_pipeline_bad_f_MAT = df_gtruth_pipeline['area'][~idx_good * idx_f * ~idx_pat]
area_gtruth_pipeline_bad_m_PAT = df_gtruth_pipeline['area'][~idx_good * ~idx_f * idx_pat]
area_gtruth_pipeline_bad_m_MAT = df_gtruth_pipeline['area'][~idx_good * ~idx_f * ~idx_pat]

# plot results
if DEBUG:
    plt.clf()
    plt.boxplot((area_gtruth_f_PAT, area_gtruth_pipeline_good_f_PAT, area_gtruth_pipeline_bad_f_PAT,
                 area_gtruth_f_MAT, area_gtruth_pipeline_good_f_MAT, area_gtruth_pipeline_bad_f_MAT),
                notch=True, labels=('PAT$_{GT}$', 'PAT$_{GT/P,good}$', 'PAT$_{GT/P,bad}$',
                                    'MAT$_{GT}$', 'MAT$_{GT/P,good}$', 'MAT$_{GT/P,bad}$'),
                positions=(0, 1, 2, 4, 5, 6))
    plt.ylabel('area  ($\mu m^2)$', fontsize=14)
    plt.title('Female')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim((0, 9000))

    if SAVE_FIGS:
        plt.savefig(
            os.path.join(figures_dir, 'klf14_b6ntac_exp_0048_area_boxplots_quality_rejection_bias_female_quality_prop_band_focal_loss.png'))

    plt.clf()
    plt.boxplot((area_gtruth_m_PAT, area_gtruth_pipeline_good_m_PAT, area_gtruth_pipeline_bad_m_PAT,
                 area_gtruth_m_MAT, area_gtruth_pipeline_good_m_MAT, area_gtruth_pipeline_bad_m_MAT),
                notch=True, labels=('PAT$_{GT}$', 'PAT$_{GT/P,good}$', 'PAT$_{GT/P,bad}$',
                                    'MAT$_{GT}$', 'MAT$_{GT/P,good}$', 'MAT$_{GT/P,bad}$'),
                positions=(0, 1, 2, 4, 5, 6))
    plt.ylabel('area  ($\mu m^2)$', fontsize=14)
    plt.title('Male')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim((0, 15000))

    if SAVE_FIGS:
        plt.savefig(
            os.path.join(figures_dir, 'klf14_b6ntac_exp_0048_area_boxplots_quality_rejection_bias_male_quality_prop_band_focal_loss.png'))

# Compute confusion matrix for all cells together
if DEBUG:
    y_true = df_gtruth_pipeline['dice'] >= dice_threshold
    y_pred = df_gtruth_pipeline['quality'] >= quality_threshold

    cytometer.utils.plot_confusion_matrix(y_true, y_pred,
                                          normalize=True,
                                          title='All cells',
                                          xlabel='Predict Quality $\geq$ 0.5',
                                          ylabel='Ground-truth Dice $\geq$ 0.9',
                                          cmap=plt.cm.Blues)

    # confusion matrix for females
    df_gtruth_pipeline_female = df_gtruth_pipeline.loc[df_gtruth_pipeline['sex'] == 'f', :]
    y_true = df_gtruth_pipeline_female['dice'] >= dice_threshold
    y_pred = df_gtruth_pipeline_female['quality'] >= quality_threshold

    cytometer.utils.plot_confusion_matrix(y_true, y_pred,
                                          normalize=True,
                                          title='Female',
                                          xlabel='Predict Quality $\geq$ 0.5',
                                          ylabel='Ground-truth Dice $\geq$ 0.9',
                                          cmap=plt.cm.Blues)

    # confusion matrix for males
    df_gtruth_pipeline_male = df_gtruth_pipeline.loc[df_gtruth_pipeline['sex'] == 'm', :]
    y_true = df_gtruth_pipeline_male['dice'] >= dice_threshold
    y_pred = df_gtruth_pipeline_male['quality'] >= quality_threshold

    cytometer.utils.plot_confusion_matrix(y_true, y_pred,
                                          normalize=True,
                                          title='Male',
                                          xlabel='Predict Quality $\geq$ 0.5',
                                          ylabel='Ground-truth Dice $\geq$ 0.9',
                                          cmap=plt.cm.Blues)

# compute sensitivity and specificity over intervals of cell area
area_intervals = list(range(0, 8000, 250)) + [np.Inf, ]
sensitivity = np.zeros(shape=(len(area_intervals) - 1, ))
specificity = np.zeros(shape=(len(area_intervals) - 1, ))
for i in range(len(area_intervals) - 1):

    df = df_gtruth_pipeline.loc[np.logical_and(df_gtruth_pipeline['area'] >= area_intervals[i],
                                               df_gtruth_pipeline['area'] < area_intervals[i + 1]), :]

    # sensitivity = TP / P
    #  TP = Dice >= 0.9 & quality >= 0.5
    #  P  = Dice >= 0.9
    TP = np.count_nonzero(np.logical_and(df['dice'] >= dice_threshold, df['quality'] >= quality_threshold))
    P = np.count_nonzero(df['dice'] >= dice_threshold)
    if P == 0:
        sensitivity[i] = np.nan
    else:
        sensitivity[i] = TP / P

    # specificity = TN / N
    #  TN = Dice < 0.9 & quality < 0.5
    #  N  = Dice < 0.9
    TN = np.count_nonzero(np.logical_and(df['dice'] < dice_threshold, df['quality'] < quality_threshold))
    N = np.count_nonzero(df['dice'] < dice_threshold)
    if N == 0:
        specificity[i] = np.nan
    else:
        specificity[i] = TN / N

if DEBUG:
    plt.clf()
    area_midpoints = (np.array(area_intervals[0:-1]) + np.array(area_intervals[1:]))/2.0
    plt.plot(area_midpoints, sensitivity, label='Sensitivity')
    plt.plot(area_midpoints, specificity, label='Specificity')
    plt.legend()
    plt.xlabel('area ($\mu m^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)

    if SAVE_FIGS:
        plt.savefig(
            os.path.join(figures_dir, 'klf14_b6ntac_exp_0048_pipeline_sensitivity_specificity.png'))

# compute plot of cell area vs. Dice/quality value

if DEBUG:


    plt.clf()
    plt.subplot(121)
    plt.plot(df_gtruth_pipeline['area'][df_gtruth_pipeline['dice'] < dice_threshold],
             df_gtruth_pipeline['dice'][df_gtruth_pipeline['dice'] < dice_threshold], '.C0')
    plt.plot(df_gtruth_pipeline['area'][df_gtruth_pipeline['dice'] >= dice_threshold],
             df_gtruth_pipeline['dice'][df_gtruth_pipeline['dice'] >= dice_threshold], '.C1')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Cell area', fontsize=14)
    plt.ylabel('Dice', fontsize=14)

    plt.subplot(122)

    # quality < 0.5, dice < 0.9
    idx = (df_gtruth_pipeline['quality'] < quality_threshold) * (df_gtruth_pipeline['dice'] < dice_threshold)
    plt.plot(df_gtruth_pipeline['dice'][idx], df_gtruth_pipeline['quality'][idx], '.C0', label="D<0.9, Q<0.5")
    # quality >= 0.5, dice < 0.9
    idx = (df_gtruth_pipeline['quality'] >= quality_threshold) * (df_gtruth_pipeline['dice'] < dice_threshold)
    plt.plot(df_gtruth_pipeline['dice'][idx], df_gtruth_pipeline['quality'][idx], '.C1', label="D<0.9, Q$\geq$0.5")
    # quality < 0.5, dice >= 0.9
    idx = (df_gtruth_pipeline['quality'] < quality_threshold) * (df_gtruth_pipeline['dice'] >= dice_threshold)
    plt.plot(df_gtruth_pipeline['dice'][idx], df_gtruth_pipeline['quality'][idx], '.C2', label="D$\geq$0.9, Q<0.5")
    # quality >= 0.5, dice >= 0.9
    idx = (df_gtruth_pipeline['quality'] >= quality_threshold) * (df_gtruth_pipeline['dice'] >= dice_threshold)
    plt.plot(df_gtruth_pipeline['dice'][idx], df_gtruth_pipeline['quality'][idx], '.C3', label="D$\geq$0.9, Q$\geq$0.5")

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()
    plt.xlabel('Dice', fontsize=14)
    plt.ylabel('Quality', fontsize=14)

    # plot Dice vs. quality, and colour according to size
    plt.clf()
    plt.scatter(df_gtruth_pipeline['dice'], df_gtruth_pipeline['quality'], c=np.log(df_gtruth_pipeline['area']+1), s=5)
    plt.colorbar()
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Dice', fontsize=14)
    plt.ylabel('Quality', fontsize=14)

'''
************************************************************************************************************************
Exp 0049:

Training all folds of DenseNet for quality assessement of individual cells based on classification of
thresholded Dice coefficient (Dice >= 0.9). Here the loss is binary focal loss.

The reason is to center the decision boundary on 0.9, to get finer granularity around that threshold.

Mask one-cell histology windows with 0/-1/+1 mask. The mask has a band with a width of 20% the equivalent radius
of the cell (equivalent radius is the radius of a circle with the same area as the cell).

The difference of this one with 0046 is that here we remove segmentations with Dice < 0.5 (automatic segmentations that
have poor ground truth) from the training dataset.

This is part of a series of experiments with different types of masks: 0039, 0040, 0041, 0042, 0045, 0046, 0048.
************************************************************************************************************************
'''

# quality network
saved_quality_model_basename = 'klf14_b6ntac_exp_0049_cnn_qualitynet_pm_1_prop_band_focal_loss_dice_geq_0_5'
quality_model_name = saved_quality_model_basename + '*.h5'

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# list of images, and indices for training vs. testing indices
contour_model_kfold_filename = os.path.join(saved_models_dir, saved_contour_model_basename + '_info.pickle')
with open(contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
im_orig_file_list = aux['file_list']
idx_orig_test_all = aux['idx_test_all']

# change home directory
im_orig_file_list = cytometer.data.change_home_directory(im_orig_file_list, home_path_from='/users/rittscher/rcasero',
                                                         home_path_to=home,
                                                         check_isfile=False)

# read pixel size information
im = PIL.Image.open(im_orig_file_list[0])
xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

df_gtruth_pipeline = []

for i_fold, idx_test in enumerate(idx_orig_test_all):

    print('Fold = ' + str(i_fold) + '/' + str(len(idx_orig_test_all) - 1))

    '''Load data 
    '''

    # split the data into training and testing datasets
    im_test_file_list, im_train_file_list = cytometer.data.split_list(im_orig_file_list, idx_test)

    # load training dataset
    datasets, _, _ = cytometer.data.load_datasets(im_test_file_list, prefix_from='im',
                                                  prefix_to=['im', 'lab'], nblocks=1)
    im = datasets['im']
    reflab = datasets['lab']
    del datasets

    # number of images
    n_im = im.shape[0]

    # select the models that correspond to current fold
    contour_model_file = os.path.join(saved_models_dir, saved_contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    dmap_model_file = os.path.join(saved_models_dir, saved_dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    quality_model_file = os.path.join(saved_models_dir, saved_quality_model_basename + '_model_fold_' + str(i_fold) + '.h5')

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
                                              quality_model_type='-1_1_prop_band',
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
            plt.contour(reflab[i, :, :, 0], levels=np.unique(labels[i, :, :, 0]), colors='C1')

        # compute cell areas from non-edge cells
        props = regionprops(labels[i, :, :, 0])
        p_label = [p['label'] for p in props]
        p_area = np.array([p['area'] for p in props])
        areas = p_area * xres * yres  # (m^2)

        # create dataframe: one cell per row, tagged with mouse metainformation
        df = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(im_test_file_list[i]),
                                                       values=areas, values_tag='area',
                                                       tags_to_keep=['id', 'ko_parent', 'sex'])

        # add to dataframe: image index and cell label
        df['im'] = i
        df['label'] = p_label

        # add to dataframe: Dice coefficients
        # get Dice value for each non-edge automatic segmentation. These Dice values are
        # computed by comparing the automatic segmentation to the manual segmentation. When there's no manual
        # segmentation to compare with, we assume Dice = 0. This is not a perfect choice, as sometimes the pipeline may
        # spot a cell that the human operator didn't, but in general we are going to assume that if the human operator
        # didn't hand-traced an object it's because it was not a well formed white adipocyte.
        #
        # We then create a look up table (LUT) so that we can sort the values according to the labels in labels_info
        # efficiently.
        dice_info = cytometer.utils.match_overlapping_labels(labels_test=labels[i, :, :, 0],
                                                             labels_ref=reflab[i, :, :, 0])

        dice_lut = np.zeros(shape=(np.max(labels[i, :, :, 0]) + 1, ))
        dice_lut[dice_info['lab_test']] = dice_info['dice']

        df['dice'] = dice_lut[p_label]

        # add to dataframe: quality scores
        idx = labels_info['im'] == i
        assert(np.all(p_label == labels_info[idx]['label']))
        df['quality'] = labels_info[idx]['quality']

        ## Delete bad segmentations: this last block of code is only useful to display results

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

        # concatenate results
        if len(df_gtruth_pipeline) == 0:
            df_gtruth_pipeline = df
        else:
            df_gtruth_pipeline = pd.concat([df_gtruth_pipeline, df])

#np.savez('/tmp/foo.npz', df_gtruth_pipeline=df_gtruth_pipeline)

# delete segmentations with Dice < 0.2, because those don't really have a ground truth. For example,
# the segmentation itself may be a good segmentation of a cell, but that cell had no ground truth.
# So instead, the segmentation is just touching the edge of the ground truth of a neighbour cell
idx = df_gtruth_pipeline['dice'] >= 0.2
df_gtruth_pipeline = df_gtruth_pipeline.loc[idx, :]

# split data into groups
idx_good = np.array(df_gtruth_pipeline['quality'] >= quality_threshold)
idx_f = np.array(df_gtruth_pipeline['sex'] == 'f')
idx_pat = np.array(df_gtruth_pipeline['ko_parent'] == 'PAT')

area_gtruth_pipeline_good_f_PAT = df_gtruth_pipeline['area'][idx_good * idx_f * idx_pat]
area_gtruth_pipeline_good_f_MAT = df_gtruth_pipeline['area'][idx_good * idx_f * ~idx_pat]
area_gtruth_pipeline_good_m_PAT = df_gtruth_pipeline['area'][idx_good * ~idx_f * idx_pat]
area_gtruth_pipeline_good_m_MAT = df_gtruth_pipeline['area'][idx_good * ~idx_f * ~idx_pat]

area_gtruth_pipeline_bad_f_PAT = df_gtruth_pipeline['area'][~idx_good * idx_f * idx_pat]
area_gtruth_pipeline_bad_f_MAT = df_gtruth_pipeline['area'][~idx_good * idx_f * ~idx_pat]
area_gtruth_pipeline_bad_m_PAT = df_gtruth_pipeline['area'][~idx_good * ~idx_f * idx_pat]
area_gtruth_pipeline_bad_m_MAT = df_gtruth_pipeline['area'][~idx_good * ~idx_f * ~idx_pat]

# plot results
if DEBUG:
    plt.clf()
    plt.boxplot((area_gtruth_f_PAT, area_gtruth_pipeline_good_f_PAT, area_gtruth_pipeline_bad_f_PAT,
                 area_gtruth_f_MAT, area_gtruth_pipeline_good_f_MAT, area_gtruth_pipeline_bad_f_MAT),
                notch=True, labels=('PAT$_{GT}$', 'PAT$_{GT/P,good}$', 'PAT$_{GT/P,bad}$',
                                    'MAT$_{GT}$', 'MAT$_{GT/P,good}$', 'MAT$_{GT/P,bad}$'),
                positions=(0, 1, 2, 4, 5, 6))
    plt.ylabel('area  ($\mu m^2)$', fontsize=14)
    plt.title('Female')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim((0, 9000))

    if SAVE_FIGS:
        plt.savefig(
            os.path.join(figures_dir, 'klf14_b6ntac_exp_0049_area_boxplots_quality_rejection_bias_female_quality_prop_band_focal_loss.png'))

    plt.clf()
    plt.boxplot((area_gtruth_m_PAT, area_gtruth_pipeline_good_m_PAT, area_gtruth_pipeline_bad_m_PAT,
                 area_gtruth_m_MAT, area_gtruth_pipeline_good_m_MAT, area_gtruth_pipeline_bad_m_MAT),
                notch=True, labels=('PAT$_{GT}$', 'PAT$_{GT/P,good}$', 'PAT$_{GT/P,bad}$',
                                    'MAT$_{GT}$', 'MAT$_{GT/P,good}$', 'MAT$_{GT/P,bad}$'),
                positions=(0, 1, 2, 4, 5, 6))
    plt.ylabel('area  ($\mu m^2)$', fontsize=14)
    plt.title('Male')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim((0, 15000))

    if SAVE_FIGS:
        plt.savefig(
            os.path.join(figures_dir, 'klf14_b6ntac_exp_0049_area_boxplots_quality_rejection_bias_male_quality_prop_band_focal_loss.png'))

# Compute confusion matrix for all cells together
if DEBUG:
    y_true = df_gtruth_pipeline['dice'] >= dice_threshold
    y_pred = df_gtruth_pipeline['quality'] >= quality_threshold

    cytometer.utils.plot_confusion_matrix(y_true, y_pred,
                                          normalize=True,
                                          title='All cells',
                                          xlabel='Predict Quality $\geq$ 0.5',
                                          ylabel='Ground-truth Dice $\geq$ 0.9',
                                          cmap=plt.cm.Blues)

    # confusion matrix for females
    df_gtruth_pipeline_female = df_gtruth_pipeline.loc[df_gtruth_pipeline['sex'] == 'f', :]
    y_true = df_gtruth_pipeline_female['dice'] >= dice_threshold
    y_pred = df_gtruth_pipeline_female['quality'] >= quality_threshold

    cytometer.utils.plot_confusion_matrix(y_true, y_pred,
                                          normalize=True,
                                          title='Female',
                                          xlabel='Predict Quality $\geq$ 0.5',
                                          ylabel='Ground-truth Dice $\geq$ 0.9',
                                          cmap=plt.cm.Blues)

    # confusion matrix for males
    df_gtruth_pipeline_male = df_gtruth_pipeline.loc[df_gtruth_pipeline['sex'] == 'm', :]
    y_true = df_gtruth_pipeline_male['dice'] >= dice_threshold
    y_pred = df_gtruth_pipeline_male['quality'] >= quality_threshold

    cytometer.utils.plot_confusion_matrix(y_true, y_pred,
                                          normalize=True,
                                          title='Male',
                                          xlabel='Predict Quality $\geq$ 0.5',
                                          ylabel='Ground-truth Dice $\geq$ 0.9',
                                          cmap=plt.cm.Blues)

# compute sensitivity and specificity over intervals of cell area
area_intervals = list(range(0, 8000, 250)) + [np.Inf, ]
sensitivity = np.zeros(shape=(len(area_intervals) - 1, ))
specificity = np.zeros(shape=(len(area_intervals) - 1, ))
for i in range(len(area_intervals) - 1):

    df = df_gtruth_pipeline.loc[np.logical_and(df_gtruth_pipeline['area'] >= area_intervals[i],
                                               df_gtruth_pipeline['area'] < area_intervals[i + 1]), :]

    # sensitivity = TP / P
    #  TP = Dice >= 0.9 & quality >= 0.5
    #  P  = Dice >= 0.9
    TP = np.count_nonzero(np.logical_and(df['dice'] >= dice_threshold, df['quality'] >= quality_threshold))
    P = np.count_nonzero(df['dice'] >= dice_threshold)
    if P == 0:
        sensitivity[i] = np.nan
    else:
        sensitivity[i] = TP / P

    # specificity = TN / N
    #  TN = Dice < 0.9 & quality < 0.5
    #  N  = Dice < 0.9
    TN = np.count_nonzero(np.logical_and(df['dice'] < dice_threshold, df['quality'] < quality_threshold))
    N = np.count_nonzero(df['dice'] < dice_threshold)
    if N == 0:
        specificity[i] = np.nan
    else:
        specificity[i] = TN / N

if DEBUG:
    plt.clf()
    area_midpoints = (np.array(area_intervals[0:-1]) + np.array(area_intervals[1:]))/2.0
    plt.plot(area_midpoints, sensitivity, label='Sensitivity')
    plt.plot(area_midpoints, specificity, label='Specificity')
    plt.legend()
    plt.xlabel('area ($\mu m^2$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)

    if SAVE_FIGS:
        plt.savefig(
            os.path.join(figures_dir, 'klf14_b6ntac_exp_0049_pipeline_sensitivity_specificity.png'))

# compute plot of cell area vs. Dice/quality value

if DEBUG:


    plt.clf()
    plt.subplot(121)
    plt.plot(df_gtruth_pipeline['area'][df_gtruth_pipeline['dice'] < dice_threshold],
             df_gtruth_pipeline['dice'][df_gtruth_pipeline['dice'] < dice_threshold], '.C0')
    plt.plot(df_gtruth_pipeline['area'][df_gtruth_pipeline['dice'] >= dice_threshold],
             df_gtruth_pipeline['dice'][df_gtruth_pipeline['dice'] >= dice_threshold], '.C1')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Cell area', fontsize=14)
    plt.ylabel('Dice', fontsize=14)

    plt.subplot(122)

    # quality < 0.5, dice < 0.9
    idx = (df_gtruth_pipeline['quality'] < quality_threshold) * (df_gtruth_pipeline['dice'] < dice_threshold)
    plt.plot(df_gtruth_pipeline['dice'][idx], df_gtruth_pipeline['quality'][idx], '.C0', label="D<0.9, Q<0.5")
    # quality >= 0.5, dice < 0.9
    idx = (df_gtruth_pipeline['quality'] >= quality_threshold) * (df_gtruth_pipeline['dice'] < dice_threshold)
    plt.plot(df_gtruth_pipeline['dice'][idx], df_gtruth_pipeline['quality'][idx], '.C1', label="D<0.9, Q$\geq$0.5")
    # quality < 0.5, dice >= 0.9
    idx = (df_gtruth_pipeline['quality'] < quality_threshold) * (df_gtruth_pipeline['dice'] >= dice_threshold)
    plt.plot(df_gtruth_pipeline['dice'][idx], df_gtruth_pipeline['quality'][idx], '.C2', label="D$\geq$0.9, Q<0.5")
    # quality >= 0.5, dice >= 0.9
    idx = (df_gtruth_pipeline['quality'] >= quality_threshold) * (df_gtruth_pipeline['dice'] >= dice_threshold)
    plt.plot(df_gtruth_pipeline['dice'][idx], df_gtruth_pipeline['quality'][idx], '.C3', label="D$\geq$0.9, Q$\geq$0.5")

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()
    plt.xlabel('Dice', fontsize=14)
    plt.ylabel('Quality', fontsize=14)

    # plot Dice vs. quality, and colour according to size
    plt.clf()
    plt.scatter(df_gtruth_pipeline['dice'], df_gtruth_pipeline['quality'], c=np.log(df_gtruth_pipeline['area']+1), s=5)
    plt.colorbar()
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Dice', fontsize=14)
    plt.ylabel('Quality', fontsize=14)

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
Distribution of cell areas in the training data set
************************************************************************************************************************
'''

# we assume that we have computed the areas in the training dataset at the beginning of this script, creating the
# dataframe df_gtruth

area_intervals = list(range(0, 8000, 250)) + [np.Inf, ]
area_midpoints = (np.array(area_intervals[0:-1]) + np.array(area_intervals[1:]))/2.0

if DEBUG:
    plt.clf()
    plt.hist(df_gtruth['area'], bins=area_intervals, density=False)
    plt.xlabel('area ($\mu m^2$)', fontsize=14)
    plt.ylabel('number of cells', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
