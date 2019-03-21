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

# list of images, and indices for training vs. testing indices
contour_model_kfold_filename = os.path.join(saved_models_dir, saved_contour_model_basename + '_info.pickle')
with open(contour_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
im_orig_file_list = aux['file_list']
idx_orig_test_all = aux['idx_test_all']

# trained models for all folds
contour_model_files = sorted(glob.glob(os.path.join(saved_models_dir, contour_model_name)))
dmap_model_files = sorted(glob.glob(os.path.join(saved_models_dir, dmap_model_name)))
quality_model_files = sorted(glob.glob(os.path.join(saved_models_dir, quality_model_name)))

for fold_i, idx_test in enumerate(idx_orig_test_all):

    print('Fold = ' + str(fold_i) + '/' + str(len(idx_orig_test_all) - 1))

    '''Load data 
    '''

    # load training dataset
    datasets, _, _ = cytometer.data.load_datasets(im_orig_file_list, prefix_from='im',
                                                  prefix_to=['im', 'lab', 'seg', 'mask'], nblocks=1)
    im = datasets['im']
    seg = datasets['seg']
    mask = datasets['mask']
    reflab = datasets['lab']
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

    '''Cell segmentation, match estimated segmentations to ground truth segmentations and Dice coefficients
    '''

    labels = np.zeros(shape=im.shape[:-1] + (1,), dtype=np.int32)
    labels_borders = np.zeros(shape=im.shape[:-1] + (1,), dtype=np.uint8)
    for i in range(n_im):

        print('\tImage ' + str(i) + '/' + str(n_im-1))

        # segment histology
        labels[i, :, :, :], labels_info = \
            cytometer.utils.segmentation_pipeline(im[i:i+1, :, :, :],
                                                  contour_model, dmap_model, quality_model,
                                                  quality_model_type='-1_1_band',
                                                  smallest_cell_area=smallest_cell_area)

        # run histology image through network
        contour_pred = contour_model.predict(im[i, :, :, :].reshape((1,) + im.shape[1:]))
        dmap_pred = dmap_model.predict(im[i, :, :, :].reshape((1,) + im.shape[1:]))

        # cell segmentation
        labels[i, :, :, 0], labels_borders[i, :, :, 0] \
            = cytometer.utils.segment_dmap_contour(dmap_pred[0, :, :, 0],
                                                   contour=contour_pred[0, :, :, 0],
                                                   border_dilation=0)

        # plot results of cell segmentation
        if DEBUG:
            plt.clf()
            plt.subplot(231)
            plt.imshow(im[i, :, :, :])
            plt.title('histology, i = ' + str(i))
            plt.subplot(232)
            plt.imshow(contour_pred[0, :, :, 0])
            plt.title('predicted contours')
            plt.subplot(233)
            plt.imshow(dmap_pred[0, :, :, 0])
            plt.title('predicted dmap')
            plt.subplot(234)
            plt.imshow(labels[i, :, :, 0])
            plt.title('labels')
            plt.subplot(235)
            plt.imshow(labels_borders[i, :, :, 0])
            plt.title('label borders')
            plt.subplot(236)
            plt.imshow(seg[i, :, :, 0])
            plt.title('ground truth borders')

        # compute quality measure of estimated labels
        qual = cytometer.utils.match_overlapping_labels(labels_test=labels[i, :, :, 0],
                                                        labels_ref=reflab[i, :, :, 0])

        # plot validation of cell segmentation
        if DEBUG:
            labels_qual = cytometer.utils.paint_labels(labels=labels[i, :, :, 0], paint_labs=qual['lab_test'],
                                                       paint_values=qual['dice'])

            plt.clf()
            plt.subplot(221)
            plt.imshow(im[i, :, :, :])
            plt.title('histology, i = ' + str(i))
            plt.subplot(222)
            plt.imshow(border[i, :, :, 0])
            plt.title('ground truth labels')
            plt.subplot(223)
            aux = np.zeros(shape=labels_borders[i, :, :, 0].shape + (3,), dtype=np.float32)
            aux[:, :, 0] = border[i, :, :, 0]
            aux[:, :, 1] = labels_borders[i, :, :, 0]
            aux[:, :, 2] = border[i, :, :, 0]
            plt.imshow(aux)
            plt.title('estimated (green) vs. ground truth (purple)')
            plt.subplot(224)
            aux = np.zeros(shape=labels_borders[i, :, :, 0].shape + (3,), dtype=np.float32)
            aux[:, :, 0] = labels_qual
            aux[:, :, 1] = labels_qual
            aux[:, :, 2] = labels_qual
            aux_r = aux[:, :, 0]
            aux_r[border[i, :, :, 0] == 1.0] = 1.0
            aux[:, :, 0] = aux_r
            aux[:, :, 2] = aux_r
            aux_g = aux[:, :, 1]
            aux_g[labels_borders[i, :, :, 0] == 1.0] = 1.0
            aux[:, :, 1] = aux_g
            plt.imshow(aux, cmap='Greys_r')
            plt.title('Dice coeff')

        # filenames for output files
        base_file = im_orig_file_list[i]
        base_path, base_name = os.path.split(base_file)
        predlab_file = os.path.join(training_augmented_dir,
                                    base_file.replace('im_', 'predlab_kfold_' + str(fold_i).zfill(2) + '_'))
        predseg_file = os.path.join(training_augmented_dir,
                                    base_file.replace('im_', 'predseg_kfold_' + str(fold_i).zfill(2) + '_'))
        labcorr_file = base_file.replace('im_', 'labcorr_kfold_' + str(fold_i).zfill(2) + '_')
        labcorr_file = os.path.join(training_augmented_dir,
                                    labcorr_file.replace('.tif', '.npy'))

        # save the predicted labels (one per cell)
        im_out = Image.fromarray(labels[i, :, :, 0], mode='I')  # int32
        im_out.save(predlab_file)

        # save the predicted contours
        im_out = Image.fromarray(labels_borders[i, :, :, 0], mode='L')  # uint8
        im_out.save(predseg_file)

        # save the correspondence between labels and Dice coefficients
        np.save(file=labcorr_file, arr=qual)

    '''Data augmentation
    '''

    # augment segmentation results
    for seed in range(augment_factor - 1):

        print('\t* Augmentation round: ' + str(seed + 1) + '/' + str(augment_factor - 1))

        for i in range(n_im):

            print('\t** Image ' + str(i) + '/' + str(n_im - 1))

            # file name of the histology corresponding to the random transformation of the Dice coefficient image
            base_file = im_orig_file_list[i]
            base_path, base_name = os.path.split(base_file)
            transform_file = os.path.join(base_path, base_name.replace('im_', 'transform_')
                                          .replace('.tif', '.pickle').replace('seed_nan', 'seed_' + str(seed).zfill(3)))
            transform = pickle.load(open(transform_file, 'br'))

            # convert transform to skimage format
            transform_skimage = cytometer.utils.keras2skimage_transform(transform=transform, shape=im.shape[1:3])

            # apply transform to reference images
            labels_augmented = warp(labels[i, :, :, 0], transform_skimage.inverse,
                                    order=0, preserve_range=True)
            labels_borders_augmented = warp(labels_borders[i, :, :, 0], transform_skimage.inverse,
                                            order=0, preserve_range=True)

            # apply flips
            if transform['flip_horizontal'] == 1:
                labels_augmented = labels_augmented[:, ::-1]
                labels_borders_augmented = labels_borders_augmented[:, ::-1]
            if transform['flip_vertical'] == 1:
                labels_augmented = labels_augmented[::-1, :]
                labels_borders_augmented = labels_borders_augmented[::-1, :]

            # convert to correct types for display and saving to file
            labels_augmented = labels_augmented.astype(np.int32)
            labels_borders_augmented = labels_borders_augmented.astype(np.uint8)

            if DEBUG:

                # file name of the histology corresponding to the random transformation of the Dice coefficient image
                im_file_augmented = os.path.join(training_augmented_dir, base_file.replace('seed_nan', 'seed_' + str(seed).zfill(3)))

                # load histology
                aux_dataset, _, _ = cytometer.data.load_datasets([im_file_augmented], prefix_from='im', prefix_to=['im'], nblocks=1)

                qual = cytometer.utils.match_overlapping_labels(labels_test=labels[i, :, :, 0],
                                                                labels_ref=reflab[i, :, :, 0])
                labels_qual_augmented = cytometer.utils.paint_labels(labels=labels_augmented, paint_labs=qual['lab_test'],
                                                                     paint_values=qual['dice'])

                # compare randomly transformed histology to corresponding Dice coefficient
                plt.clf()
                plt.subplot(321)
                plt.imshow(aux_dataset['im'][0, :, :, :])
                plt.subplot(322)
                plt.imshow(labels_qual_augmented, cmap='Greys_r')
                plt.subplot(323)
                aux = pystoim.imfuse(aux_dataset['im'][0, :, :, :], labels_qual_augmented)
                plt.imshow(aux, cmap='Greys_r')
                plt.subplot(324)
                plt.imshow(labels_augmented)
                plt.subplot(325)
                plt.imshow(labels_borders_augmented)

            # filenames for the Dice coefficient augmented files
            predlab_file = os.path.join(training_augmented_dir,
                                        base_name.replace('im_seed_nan_',
                                                          'predlab_kfold_' + str(fold_i).zfill(2) + '_seed_' + str(seed).zfill(3) + '_'))
            predseg_file = os.path.join(training_augmented_dir,
                                        base_name.replace('im_seed_nan_',
                                                          'predseg_kfold_' + str(fold_i).zfill(2) + '_seed_' + str(seed).zfill(3) + '_'))

            # save the predicted labels (one per cell)
            im_out = Image.fromarray(labels_augmented.astype(np.int32), mode='I')  # int32
            im_out.save(predlab_file)

            # save the predicted contours
            im_out = Image.fromarray(labels_borders_augmented.astype(np.uint8), mode='L')  # uint8
            im_out.save(predseg_file)
