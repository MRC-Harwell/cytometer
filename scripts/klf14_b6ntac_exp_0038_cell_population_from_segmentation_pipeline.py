"""
Load pipeline segmentations from 0036.
Compare the cell area distributions of:

* Hand segmented cells (ground truth)
* Automatically segmented cells that overlap with a ground truth cell and Dice>=0.9
* All automatically segmented cells accepted by quality assessment network
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
import seaborn as sns
from skimage.morphology import watershed
from skimage.measure import regionprops

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

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
SAVE_FIGS = False

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


'''
************************************************************************************************************************
Hand segmented cells (ground truth), no-overlap approximation
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
figures_dir = os.path.join(root_data_dir, 'figures')
metainfo_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0038_cell_population_from_segmentation_pipeline'

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
ax.set_ylabel('area (um^2)', fontsize=14)
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
Pipeline segmentations that overlap with hand traced cells
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
metainfo_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper')

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

'''Load data
'''

# load CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(metainfo_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# list of all non-overlap original files
im_file_list = glob.glob(os.path.join(training_augmented_dir, 'im_seed_nan_*.tif'))

# read pixel size information
orig_file = os.path.basename(im_file_list[0]).replace('im_seed_nan_', '')
im = PIL.Image.open(os.path.join(training_dir, orig_file))
xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

# loop each fold: we split the data into train vs test, train a model, and compute errors with the
# test data. In each fold, the test data is different
# for i_fold, idx_test in enumerate(idx_test_all):
correspondence_all = []
df_pipeline_gtruth = None
for i_fold, idx_test in enumerate(idx_orig_test_all):

    # split the data list into training and testing lists
    im_test_file_list, _ = cytometer.data.split_list(im_orig_file_list, idx_test)

    # number of testing images
    n_im = len(im_test_file_list)

    # load the correspondences and Dice between pipeline segmentations and ground truth
    # NOTE: in correspondence, we keep all pipeline and hand traced areas, regardless of Dice
    #       in df_pipeline, we only keep pipeline areas with Dice >= 0.9
    for i in range(n_im):

        # print('\tImage ' + str(i) + '/' + str(n_im-1))

        base_file = os.path.basename(im_test_file_list[i])
        labcorr_file = base_file.replace('im_', 'labcorr_kfold_' + str(i_fold).zfill(2) + '_')
        labcorr_file = os.path.join(training_augmented_dir,
                                    labcorr_file.replace('.tif', '.npy'))
        correspondence = np.load(file=labcorr_file)
        correspondence_all.append(correspondence)

        # convert areas from pixel to um^2
        good_areas = correspondence['area_test'] * xres * yres
        good_areas = good_areas[correspondence['dice'] >= valid_threshold]

        if len(good_areas) == 0:
            continue

        # create dataframe with metainformation from mouse
        df_window = cytometer.data.tag_values_with_mouse_info(metainfo, os.path.basename(im_test_file_list[i]),
                                                              good_areas,
                                                              values_tag='area', tags_to_keep=['id', 'ko_parent', 'sex'])

        # add a column with the window filename. This is later used in the linear models
        df_window['file'] = os.path.basename(im_test_file_list[i])

        # create new total dataframe, or concat to existing one
        if df_pipeline_gtruth is None:
            df_pipeline_gtruth = df_window

            # make sure that in the boxplots PAT comes before MAT
            df_pipeline_gtruth['ko_parent'] = df_pipeline_gtruth['ko_parent'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'],
                                                                                                     ordered=True))

        else:
            df_pipeline_gtruth = pd.concat([df_pipeline_gtruth, df_window], axis=0, ignore_index=True)

# convert from list of arrays to single array
correspondence_all = np.concatenate(correspondence_all, axis=0)

# convert areas from pixel^2 to um^2
area_ref = correspondence_all['area_ref'] * xres * yres
area_test = correspondence_all['area_test'] * xres * yres


'''Validate segmentations' areas quantitatively and with plots
'''

# correlation coeff
rho_all_cells = np.corrcoef(area_ref, area_test)[0, 1]

print('Pearson corr = ' + "{:.3f}".format(rho_all_cells))

# plot cell areas. All pipeline segmentations with ground truth vs. their ground truth
plt.clf()
plt.scatter(area_ref, area_test)
plt.tick_params(labelsize=16)
plt.plot([0, 17500], [0, 17500], 'orange')
plt.xlabel('hand traced area ($\mu^2$)', fontsize=16)
plt.ylabel('pipeline segmentation area ($\mu^2$)', fontsize=16)
plt.text(12000, 85000, r'$\rho = $' + "{:.3f}".format(rho_all_cells), fontsize=16)

if SAVE_FIGS:
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0038_area_pipeline_vs_manual_scatter.png'))

# cells with good Dice coefficient
idx_good_segmentation = correspondence_all['dice'] >= valid_threshold

print('Ratio of good cells = '
      + "{:.1f}".format(100 * np.count_nonzero(idx_good_segmentation) / len(idx_good_segmentation))
      + '%')

# correlation coeff
rho_good_cells = np.corrcoef(area_ref[idx_good_segmentation], area_test[idx_good_segmentation])[0, 1]

print('Pearson corr = ' + "{:.3f}".format(rho_good_cells))

# plot cell areas. Only segmentations with good quality
plt.clf()
plt.scatter(area_ref[idx_good_segmentation], area_test[idx_good_segmentation])
plt.tick_params(labelsize=16)
plt.plot([0, 17500], [0, 17500], 'orange')
plt.xlabel('hand traced area ($\mu^2$)', fontsize=16)
plt.ylabel('pipeline segmentation area ($\mu^2$)', fontsize=16)
plt.text(12000, 9000, r'$\rho = $' + "{:.3f}".format(rho_good_cells), fontsize=16)

if SAVE_FIGS:
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0038_good_area_pipeline_vs_manual_scatter.png'))

'''Population curves
'''

# split data into groups
area_pipeline_gtruth_f_PAT = df_pipeline_gtruth['area'][(np.logical_and(df_pipeline_gtruth['sex'] == 'f',
                                                                        df_pipeline_gtruth['ko_parent'] == 'PAT'))]
area_pipeline_gtruth_f_MAT = df_pipeline_gtruth['area'][(np.logical_and(df_pipeline_gtruth['sex'] == 'f',
                                                                        df_pipeline_gtruth['ko_parent'] == 'MAT'))]
area_pipeline_gtruth_m_PAT = df_pipeline_gtruth['area'][(np.logical_and(df_pipeline_gtruth['sex'] == 'm',
                                                                        df_pipeline_gtruth['ko_parent'] == 'PAT'))]
area_pipeline_gtruth_m_MAT = df_pipeline_gtruth['area'][(np.logical_and(df_pipeline_gtruth['sex'] == 'm',
                                                                        df_pipeline_gtruth['ko_parent'] == 'MAT'))]

# compute percentile profiles of cell populations
perc = np.linspace(0, 100, num=101)
perc_area_pipeline_gtruth_f_PAT = np.percentile(area_pipeline_gtruth_f_PAT, perc)
perc_area_pipeline_gtruth_f_MAT = np.percentile(area_pipeline_gtruth_f_MAT, perc)
perc_area_pipeline_gtruth_m_PAT = np.percentile(area_pipeline_gtruth_m_PAT, perc)
perc_area_pipeline_gtruth_m_MAT = np.percentile(area_pipeline_gtruth_m_MAT, perc)

# plot curve profiles
plt.clf()

ax = plt.subplot(121)
plt.plot(perc, (perc_area_gtruth_f_MAT - perc_area_gtruth_f_PAT) / perc_area_gtruth_f_PAT * 100, label='Hand traced')
plt.plot(perc, (perc_area_pipeline_gtruth_f_MAT - perc_area_pipeline_gtruth_f_PAT) / perc_area_pipeline_gtruth_f_PAT * 100,
         label='Pipeline HT Dice>=0.9')
ax.set_ylim(-50, 0)
plt.title('Female', fontsize=16)
plt.xlabel('Population percentile', fontsize=14)
plt.ylabel('Area change from PAT to MAT (%)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend()

ax = plt.subplot(122)
plt.plot(perc, (perc_area_gtruth_m_MAT - perc_area_gtruth_m_PAT) / perc_area_gtruth_m_PAT * 100, label='Hand traced')
plt.plot(perc, (perc_area_pipeline_gtruth_m_MAT - perc_area_pipeline_gtruth_m_PAT) / perc_area_pipeline_gtruth_m_PAT * 100,
         label='Pipeline HT Dice>=0.9')
ax.set_ylim(-50, 0)
plt.title('Male', fontsize=16)
plt.xlabel('Population percentile', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend()

if SAVE_FIGS:
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0038_population_profiles_hand_traced_pipeline_ht_dice.png'))

'''
************************************************************************************************************************
All pipeline segmentations with Quality >= 0.9, whether they overlap hand traced cells or not
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
                                                              values_tag='area', tags_to_keep=['id', 'ko_parent', 'sex'])

        # add a column with the quality values
        df_window['quality'] = qual

        # add a column with the window filename. This is later used in the linear models
        df_window['file'] = os.path.basename(im_test_file_list[i])

        # accumulate results
        if df_pipeline is None:
            df_pipeline = df_window

            # make sure that in the boxplots PAT comes before MAT
            df_pipeline['ko_parent'] = df_pipeline['ko_parent'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'],
                                                                                       ordered=True))
        else:
            df_pipeline = pd.concat([df_pipeline, df_window], axis=0, ignore_index=True)

'''Print result summaries
'''

if DEBUG:
    sns.set(style="white", color_codes=True)
    sns.jointplot(df_pipeline.area, df_pipeline.quality)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Cell area ($\mu m^2$)', fontsize=16)
    plt.ylabel('Quality score', fontsize=16)

if SAVE_FIGS:
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_inspect_exp_0038_cell_area_vs_quality_jointplot.png'))

'''Population curves
'''

# dataframe of good quality segmentations
df_pipeline_quality = df_pipeline[df_pipeline.quality >= quality_threshold]

# split data into groups
area_pipeline_quality_f_PAT = df_pipeline_quality['area'][(np.logical_and(df_pipeline_quality['sex'] == 'f',
                                                                          df_pipeline_quality['ko_parent'] == 'PAT'))]
area_pipeline_quality_f_MAT = df_pipeline_quality['area'][(np.logical_and(df_pipeline_quality['sex'] == 'f',
                                                                          df_pipeline_quality['ko_parent'] == 'MAT'))]
area_pipeline_quality_m_PAT = df_pipeline_quality['area'][(np.logical_and(df_pipeline_quality['sex'] == 'm',
                                                                          df_pipeline_quality['ko_parent'] == 'PAT'))]
area_pipeline_quality_m_MAT = df_pipeline_quality['area'][(np.logical_and(df_pipeline_quality['sex'] == 'm',
                                                                          df_pipeline_quality['ko_parent'] == 'MAT'))]

# size of each group
print('f/PAT: ' + str(area_pipeline_quality_f_PAT.shape[0]))
print('f/MAT: ' + str(area_pipeline_quality_f_MAT.shape[0]))
print('m/PAT: ' + str(area_pipeline_quality_m_PAT.shape[0]))
print('m/MAT: ' + str(area_pipeline_quality_m_MAT.shape[0]))

# compute percentile profiles of cell populations
perc = np.linspace(0, 100, num=101)
perc_area_pipeline_quality_f_PAT = np.percentile(area_pipeline_quality_f_PAT, perc)
perc_area_pipeline_quality_f_MAT = np.percentile(area_pipeline_quality_f_MAT, perc)
perc_area_pipeline_quality_m_PAT = np.percentile(area_pipeline_quality_m_PAT, perc)
perc_area_pipeline_quality_m_MAT = np.percentile(area_pipeline_quality_m_MAT, perc)

# plot curve profiles
plt.clf()

ax = plt.subplot(121)
plt.plot(perc, (perc_area_gtruth_f_MAT - perc_area_gtruth_f_PAT) / perc_area_gtruth_f_PAT * 100, label='Hand traced')
plt.plot(perc, (perc_area_pipeline_gtruth_f_MAT - perc_area_pipeline_gtruth_f_PAT) / perc_area_pipeline_gtruth_f_PAT * 100,
         label='Pipeline HT Dice>=0.9')
plt.plot(perc, (perc_area_pipeline_quality_f_MAT - perc_area_pipeline_quality_f_PAT) / perc_area_pipeline_quality_f_PAT * 100,
         label='Pipeline Quality>=0.9')
ax.set_ylim(-50, 0)
plt.title('Female', fontsize=16)
plt.xlabel('Population percentile', fontsize=14)
plt.ylabel('Area change from PAT to MAT (%)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend()

ax = plt.subplot(122)
plt.plot(perc, (perc_area_gtruth_m_MAT - perc_area_gtruth_m_PAT) / perc_area_gtruth_m_PAT * 100, label='Hand traced')
plt.plot(perc, (perc_area_pipeline_gtruth_m_MAT - perc_area_pipeline_gtruth_m_PAT) / perc_area_pipeline_gtruth_m_PAT * 100,
         label='Pipeline HT Dice>=0.9')
plt.plot(perc, (perc_area_pipeline_quality_m_MAT - perc_area_pipeline_quality_m_PAT) / perc_area_pipeline_quality_m_PAT * 100,
         label='Pipeline Quality>=0.9')
ax.set_ylim(-50, 0)
plt.title('Male', fontsize=16)
plt.xlabel('Population percentile', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend()

if SAVE_FIGS:
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0038_population_profiles_quality_pipeline.png'))

'''Plot cell populations as histograms
'''

# get range of cell area values
area_min = np.amin(np.concatenate((df_gtruth['area'], df_pipeline_gtruth['area'], df_pipeline_quality['area'])))
area_max = np.amax(np.concatenate((df_gtruth['area'], df_pipeline_gtruth['area'], df_pipeline_quality['area'])))

print('area_min = ' + "{:.1f}".format(area_min) + ' um^2')
print('area_max = ' + "{:.1f}".format(area_max) + ' um^2')

area_bins = np.linspace(0, 18000, 51)
area_bin_centers = (area_bins[0:-1] + area_bins[1:]) / 2.0

# histograms of cell areas
hist_area_gtruth_f_PAT, _ = np.histogram(area_gtruth_f_PAT, bins=area_bins, density=True)
hist_area_gtruth_f_MAT, _ = np.histogram(area_gtruth_f_MAT, bins=area_bins, density=True)
hist_area_gtruth_m_PAT, _ = np.histogram(area_gtruth_m_PAT, bins=area_bins, density=True)
hist_area_gtruth_m_MAT, _ = np.histogram(area_gtruth_m_MAT, bins=area_bins, density=True)

hist_area_pipeline_gtruth_f_PAT, _ = np.histogram(area_pipeline_gtruth_f_PAT, bins=area_bins, density=True)
hist_area_pipeline_gtruth_f_MAT, _ = np.histogram(area_pipeline_gtruth_f_MAT, bins=area_bins, density=True)
hist_area_pipeline_gtruth_m_PAT, _ = np.histogram(area_pipeline_gtruth_m_PAT, bins=area_bins, density=True)
hist_area_pipeline_gtruth_m_MAT, _ = np.histogram(area_pipeline_gtruth_m_MAT, bins=area_bins, density=True)

# plot curves
if DEBUG:
    plt.clf()

    plt.subplot(221)
    plt.plot(area_bin_centers / 1e3, np.zeros(hist_area_gtruth_f_MAT.shape), 'k')
    plt.plot(area_bin_centers / 1e3, hist_area_gtruth_f_PAT, 'C0--')
    plt.plot(area_bin_centers / 1e3, hist_area_gtruth_f_MAT, color='C0')
    plt.plot(area_bin_centers / 1e3, hist_area_pipeline_gtruth_f_PAT, 'C1--')
    plt.plot(area_bin_centers / 1e3, hist_area_pipeline_gtruth_f_MAT, 'C1')
    plt.ylim((-0.0001, 0.0007))
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylabel('pdf', fontsize=14)

    plt.subplot(222)
    plt.plot(area_bin_centers / 1e3, np.zeros(hist_area_gtruth_f_MAT.shape), 'k')
    plt.plot(area_bin_centers / 1e3, hist_area_gtruth_m_PAT, 'C0--')
    plt.plot(area_bin_centers / 1e3, hist_area_gtruth_m_MAT, color='C0')
    plt.plot(area_bin_centers / 1e3, hist_area_pipeline_gtruth_m_PAT, 'C1--')
    plt.plot(area_bin_centers / 1e3, hist_area_pipeline_gtruth_m_MAT, 'C1')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.ylim((-0.0001, 0.0007))
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.subplot(223)
    plt.plot(area_bin_centers / 1e3, np.zeros(hist_area_gtruth_f_MAT.shape), 'k')
    plt.plot(area_bin_centers / 1e3, hist_area_gtruth_f_MAT - hist_area_gtruth_f_PAT, 'C0')
    plt.plot(area_bin_centers / 1e3, hist_area_pipeline_gtruth_f_MAT - hist_area_pipeline_gtruth_f_PAT, 'C1')
    plt.ylim((-0.00025, 0.00025))
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel(r'area ($\times$1000 $\mu m^2$)', fontsize=14)
    plt.ylabel(r'$\Delta$pdf', fontsize=14)

    plt.subplot(224)
    plt.plot(area_bin_centers / 1e3, np.zeros(hist_area_gtruth_f_MAT.shape), 'k')
    plt.plot(area_bin_centers / 1e3, hist_area_gtruth_m_MAT - hist_area_gtruth_m_PAT, 'C0')
    plt.plot(area_bin_centers / 1e3, hist_area_pipeline_gtruth_m_MAT - hist_area_pipeline_gtruth_m_PAT, 'C1')
    plt.ylim((-0.00025, 0.00025))
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel(r'area ($\times$1000 $\mu m^2$)', fontsize=14)

# plot ECDF with confidence internals
if DEBUG:

    # error bars for histograms: http://scikit-hep.org/examples/visual/errorbars.html

    # beta function
    plt.clf()
    alpha = 0.5

    area_out, quantile_out, quantile_ci_lo, quantile_ci_hi = \
        cytometer.utils.ecdf_confidence(area_pipeline_gtruth_f_PAT, num_quantiles=101, equispace='quantiles',
                                        confidence=0.95, estimator_name='beta')
    plt.plot(area_out, quantile_out, color='C0', label='PAT')
    plt.gca().fill_between(area_out, quantile_ci_lo, quantile_ci_hi, alpha=alpha, color='C0')

    area_out, quantile_out, quantile_ci_lo, quantile_ci_hi = \
        cytometer.utils.ecdf_confidence(area_pipeline_gtruth_f_MAT, num_quantiles=101, equispace='quantiles',
                                        confidence=0.95, estimator_name='beta')
    plt.plot(area_out, quantile_out, color='C1', label='MAT')
    plt.gca().fill_between(area_out, quantile_ci_lo, quantile_ci_hi, alpha=alpha, color='C1')

    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel(r'area ($\mu m^2$)', fontsize=14)
    plt.ylabel('percentile', fontsize=14)
    plt.title('ECDF with Beta function confidence bands')

    if SAVE_FIGS:
        plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0038_ecdf_ci_beta.png'))

    # Dvoretzky-Kiefer-Wolfowitz confidence band
    plt.clf()
    alpha = 0.5

    area_out, quantile_out, quantile_ci_lo, quantile_ci_hi = \
        cytometer.utils.ecdf_confidence(area_pipeline_gtruth_f_PAT, num_quantiles=101, equispace='quantiles',
                                        confidence=0.95, estimator_name='DKW')
    plt.plot(area_out, quantile_out, color='C0', label='PAT')
    plt.gca().fill_between(area_out, quantile_ci_lo, quantile_ci_hi, alpha=alpha, color='C0')

    area_out, quantile_out, quantile_ci_lo, quantile_ci_hi = \
        cytometer.utils.ecdf_confidence(area_pipeline_gtruth_f_MAT, num_quantiles=101, equispace='quantiles',
                                        confidence=0.95, estimator_name='DKW')
    plt.plot(area_out, quantile_out, color='C1', label='MAT')
    plt.gca().fill_between(area_out, quantile_ci_lo, quantile_ci_hi, alpha=alpha, color='C1')

    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel(r'area ($\mu m^2$)', fontsize=14)
    plt.ylabel('percentile', fontsize=14)
    plt.title('ECDF with Dvoretzky-Kiefer-Wolfowitz confidence bands')

    if SAVE_FIGS:
        plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0038_ecdf_ci_dkw.png'))

'''Permutation testing of histology differences
'''

# female, no multiple test correction
quantiles_gtruth_f, pval_gtruth_f, _ = cytometer.utils.compare_ecdfs(area_pipeline_gtruth_f_PAT,
                                                                     area_pipeline_gtruth_f_MAT,
                                                                     num_quantiles=101, num_perms=10000)

# female, multiple test correction
_, pvaladj_gtruth_f, _ = cytometer.utils.compare_ecdfs(area_pipeline_gtruth_f_PAT,
                                                       area_pipeline_gtruth_f_MAT,
                                                       num_quantiles=101, num_perms=10000,
                                                       multitest_method='fdr_tsbky')

if DEBUG:

    plt.clf()

    # masked curve
    ym = np.ma.masked_where(np.logical_or(pval_gtruth_f > 0.05, np.isnan(pval_gtruth_f)),
                            (perc_area_gtruth_f_MAT - perc_area_gtruth_f_PAT) / perc_area_gtruth_f_PAT * 100)
    ymadj = np.ma.masked_where(np.logical_or(pvaladj_gtruth_f > 0.05, np.isnan(pvaladj_gtruth_f)),
                            (perc_area_gtruth_f_MAT - perc_area_gtruth_f_PAT) / perc_area_gtruth_f_PAT * 100)

    ax = plt.subplot(211)
    plt.plot(perc, (perc_area_gtruth_f_MAT - perc_area_gtruth_f_PAT) / perc_area_gtruth_f_PAT * 100,
             label='Hand traced')
    plt.plot(perc, ym, linewidth=4, label=r'$p\leq 0.005$')
    ax.set_ylim(-50, 0)
    plt.title('Female', fontsize=16)
    plt.ylabel('Area change from\n PAT to MAT (%)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()

    ax = plt.subplot(212)
    plt.plot(quantiles_gtruth_f * 100, pval_gtruth_f, label='uncorrected')
    plt.plot(quantiles_gtruth_f * 100, pvaladj_gtruth_f, label='Benjamini/Yekutieli')
    plt.semilogy([0, 100], [0.05, 0.05], label='p=0.05')
    plt.semilogy([0, 100], [0.01, 0.01], label='p=0.01')
    plt.semilogy([0, 100], [0.001, 0.001], label='p=0.001')
    plt.xlabel('Population percentile', fontsize=14)
    plt.ylabel('P-value', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()

    if SAVE_FIGS:
        plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0038_permutation_testing_female.png'))

# male, no correction
quantiles_gtruth_m, pval_gtruth_m, _ = cytometer.utils.compare_ecdfs(area_pipeline_gtruth_m_PAT,
                                                                     area_pipeline_gtruth_m_MAT,
                                                                     num_quantiles=101, num_perms=10000)

# male, multiple test correction
_, pvaladj_gtruth_m, _ = cytometer.utils.compare_ecdfs(area_pipeline_gtruth_m_PAT,
                                                       area_pipeline_gtruth_m_MAT,
                                                       num_quantiles=101, num_perms=10000,
                                                       multitest_method='fdr_tsbky')

if DEBUG:

    plt.clf()

    # masked curve
    ym = np.ma.masked_where(np.logical_or(pval_gtruth_m > 0.05, np.isnan(pval_gtruth_m)),
                            (perc_area_gtruth_m_MAT - perc_area_gtruth_m_PAT) / perc_area_gtruth_m_PAT * 100)
    ymadj = np.ma.masked_where(np.logical_or(pvaladj_gtruth_m > 0.05, np.isnan(pvaladj_gtruth_m)),
                               (perc_area_gtruth_m_MAT - perc_area_gtruth_m_PAT) / perc_area_gtruth_m_PAT * 100)

    ax = plt.subplot(211)
    plt.plot(perc, (perc_area_gtruth_m_MAT - perc_area_gtruth_m_PAT) / perc_area_gtruth_m_PAT * 100,
             label='Hand traced')
    plt.plot(perc, ym, linewidth=4, label=r'$p\leq 0.005$')
    ax.set_ylim(-50, 0)
    plt.title('Male', fontsize=16)
    plt.ylabel('Area change from\n PAT to MAT (%)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()

    ax = plt.subplot(212)
    plt.plot(quantiles_gtruth_m * 100, pval_gtruth_m, label='uncorrected')
    plt.plot(quantiles_gtruth_m * 100, pvaladj_gtruth_m, label='Benjamini/Yekutieli')
    plt.semilogy([0, 100], [0.05, 0.05], label='p=0.05')
    plt.semilogy([0, 100], [0.01, 0.01], label='p=0.01')
    plt.semilogy([0, 100], [0.001, 0.001], label='p=0.001')
    plt.xlabel('Population percentile', fontsize=14)
    plt.ylabel('P-value', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()

    if SAVE_FIGS:
        plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0038_permutation_testing_male.png'))

