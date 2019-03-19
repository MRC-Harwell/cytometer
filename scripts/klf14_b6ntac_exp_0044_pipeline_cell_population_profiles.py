"""
Areas computed in exp 0043.
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
Pipeline automatic extraction applied to full slides (both female, one MAT and one PAT)
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

'''Compare populations
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

