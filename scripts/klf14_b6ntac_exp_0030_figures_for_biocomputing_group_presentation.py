import numpy as np
import matplotlib.pyplot as plt
import os

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
fig_dir = os.path.join(root_data_dir, 'figures')

"""Synthetic example of population percentile vs. area ECDF curves
"""

area = np.linspace(0, 20000, 1000)

# plot synthetic examples of ECDFs
plt.clf()
plt.plot(area, -100 + 200 / (1 + np.exp(- area / 5000)))
plt.plot(area, -100 + 200 / (1 + np.exp(- area / 2000)))
plt.legend(['PAT', 'MAT'])
plt.xlabel(r'Cell area ($\mu m^2$)', fontsize=18)
plt.ylabel('ECDF (%)', fontsize=18)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

# areas for the 50% percentile
perc = 50
a_pat = -2000 * np.log(200 / (100 + perc) - 1)
a_mat = -5000 * np.log(200 / (100 + perc) - 1)

plt.plot([0, 20000], [perc, perc], 'k')
plt.plot([0, 20000], [perc, perc], 'k')
plt.plot([a_pat, a_pat], [0, perc], 'k')
plt.plot([a_mat, a_mat], [0, perc], 'k')

# double arrow between area values
plt.annotate(s='', xy=(a_mat, 20), xytext=(a_pat, 20), arrowprops=dict(arrowstyle='<-'))
plt.text(2300, 10, r'$\Delta$area=', fontsize=14)
plt.text(2300, 5, str((a_pat - a_mat) / a_mat * 100) + '%', fontsize=14)

# save figure
plt.savefig(os.path.join(fig_dir, 'klf14_b6ntac_exp_0030_figures_for_biocomputing_group_presentation_synthetic_area_ecdf.png'),
            bbox_inches='tight', pad_inches=0.5)


"""
Histology, distance transformation, etc for pipeline chart
"""

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import pickle

import glob
import numpy as np

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# limit GPU memory used
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))

# Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
# function
import keras
import keras.backend as K
import cytometer.data
import cytometer.models
from cytometer.utils import principal_curvatures_range_image
import matplotlib.pyplot as plt

# specify data format as (n, row, col, channel)
K.set_image_data_format('channels_last')

DEBUG = True

# area (pixel**2) of the smallest object we accept as a cell (pi * (16 pixel)**2 = 804.2 pixel**2)
smallest_cell_area = 804

# training window length
training_window_len = 401

'''Load data
'''

# data paths
training_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')

saved_dmap_model_basename = 'klf14_b6ntac_exp_0015_cnn_dmap'
saved_contour_model_basename = 'klf14_b6ntac_exp_0006_cnn_contour'
saved_dice_model_basename = 'klf14_b6ntac_exp_0029_cnn_qualitynet_sigmoid_masked_segmentation'


dmap_model_name = saved_dmap_model_basename + '*.h5'
contour_model_name = saved_contour_model_basename + '*.h5'
dice_model_name = saved_dice_model_basename + '*.h5'

# load model weights for each fold
dmap_model_files = glob.glob(os.path.join(saved_models_dir, dmap_model_name))
contour_model_files = glob.glob(os.path.join(saved_models_dir, contour_model_name))
dice_model_files = glob.glob(os.path.join(saved_models_dir, dice_model_name))
n_folds = len(dmap_model_files)

# load k-fold sets that were used to train the models
saved_model_kfold_filename = os.path.join(saved_models_dir, saved_dmap_model_basename + '_info.pickle')
with open(saved_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
im_file_list = aux['file_list']
idx_test_all = aux['idx_test_all']

# correct home directory if we are in a different system than what was used to train the models
im_file_list = cytometer.data.change_home_directory(im_file_list, '/users/rittscher/rcasero', home, check_isfile=True)

'''Load model and visualise results
'''

# list of model files to inspect
dmap_model_files = glob.glob(os.path.join(saved_models_dir, dmap_model_name))
contour_model_files = glob.glob(os.path.join(saved_models_dir, contour_model_name))
dice_model_files = glob.glob(os.path.join(saved_models_dir, dice_model_name))

fold_i = 0
dmap_model_file = dmap_model_files[fold_i]
contour_model_file = contour_model_files[fold_i]
dice_model_file = dice_model_files[fold_i]

# split the data into training and testing datasets
im_test_file_list, _ = cytometer.data.split_list(im_file_list, idx_test_all[fold_i])

# load im, seg and mask datasets
test_datasets, _, _ = cytometer.data.load_datasets(im_test_file_list, prefix_from='im',
                                                   prefix_to=['im', 'mask', 'dmap', 'lab'], nblocks=1)
im_test = test_datasets['im']
mask_test = test_datasets['mask']
dmap_test = test_datasets['dmap']
lab_test = test_datasets['lab']
del test_datasets

# load model
dmap_model = keras.models.load_model(dmap_model_file)
contour_model = keras.models.load_model(contour_model_file)
dice_model = keras.models.load_model(dice_model_file)

# set input layer to size of test images
dmap_model = cytometer.models.change_input_size(dmap_model, batch_shape=(None,) + im_test.shape[1:])
contour_model = cytometer.models.change_input_size(contour_model, batch_shape=(None,) + im_test.shape[1:])

# visualise results
i = 0

# plot histology
plt.clf()
plt.axis('off')
plt.imshow(im_test[i, :, :, :])
plt.savefig(os.path.join(fig_dir, 'klf14_b6ntac_exp_0030_figures_for_biocomputing_group_presentation_histology.png'),
            bbox_inches='tight', pad_inches=0)

# run image through network
dmap_test_pred = dmap_model.predict(im_test[i, :, :, :].reshape((1,) + im_test.shape[1:]))

# compute mean curvature from dmap
_, mean_curvature, _, _ = principal_curvatures_range_image(dmap_test_pred[0, :, :, 0], sigma=10)

plt.clf()
plt.axis('off')
plt.imshow(dmap_test_pred[0, :, :, 0])
plt.savefig(os.path.join(fig_dir, 'klf14_b6ntac_exp_0030_figures_for_biocomputing_group_presentation_dmap.png'),
            bbox_inches='tight', pad_inches=0)

plt.clf()
plt.axis('off')
plt.imshow(mean_curvature)
plt.savefig(os.path.join(fig_dir, 'klf14_b6ntac_exp_0030_figures_for_biocomputing_group_presentation_normal_curvature.png'),
            bbox_inches='tight', pad_inches=0)

# run image through network
contour_test_pred = contour_model.predict(im_test[i, :, :, :].reshape((1,) + im_test.shape[1:]))

plt.clf()
plt.axis('off')
plt.imshow(contour_test_pred[0, :, :, 0])
plt.savefig(os.path.join(fig_dir, 'klf14_b6ntac_exp_0030_figures_for_biocomputing_group_contour.png'),
            bbox_inches='tight', pad_inches=0)

# ad hoc segmentation
lab_test_pred, labborder_test_pred = cytometer.utils.segment_dmap_contour(dmap_test_pred[0, :, :, 0],
                                                                          contour=contour_test_pred[0, :, :, 0],
                                                                          sigma=10, min_seed_object_size=50,
                                                                          border_dilation=5)
plt.clf()
plt.axis('off')
plt.imshow(lab_test_pred)
plt.savefig(os.path.join(fig_dir, 'klf14_b6ntac_exp_0030_figures_for_biocomputing_group_labels.png'),
            bbox_inches='tight', pad_inches=0)

plt.clf()
plt.axis('off')
plt.imshow(labborder_test_pred)
plt.savefig(os.path.join(fig_dir, 'klf14_b6ntac_exp_0030_figures_for_biocomputing_group_labelsborders.png'),
            bbox_inches='tight', pad_inches=0)

# extract cells individually

aux_im_test = np.expand_dims(im_test[i, :, :, :], axis=0)
aux_lab_test = np.expand_dims(np.expand_dims(lab_test_pred, axis=2), axis=0)
aux_lab_test_pred = np.expand_dims(np.expand_dims(lab_test_pred, axis=2), axis=0)

test_cell_im, test_cell_testlab, _, test_cell_reflab, test_cell_dice = \
    cytometer.utils.one_image_per_label(dataset_im=aux_im_test,
                                        dataset_lab_ref=aux_lab_test,
                                        dataset_lab_test=aux_lab_test_pred,
                                        training_window_len=training_window_len,
                                        smallest_cell_area=smallest_cell_area)

j = 33
plt.clf()
plt.subplot(121)
plt.axis('off')
plt.imshow(test_cell_im[j, :, :, :])
plt.subplot(122)
plt.axis('off')
plt.imshow(test_cell_testlab[j, :, :, 0])
plt.savefig(os.path.join(fig_dir, 'klf14_b6ntac_exp_0030_figures_for_biocomputing_group_cellseg_' + str(j) + '.png'),
            bbox_inches='tight', pad_inches=0)

j = 12
plt.clf()
plt.subplot(121)
plt.axis('off')
plt.imshow(test_cell_im[j, :, :, :])
plt.subplot(122)
plt.axis('off')
plt.imshow(test_cell_testlab[j, :, :, 0])
plt.savefig(os.path.join(fig_dir, 'klf14_b6ntac_exp_0030_figures_for_biocomputing_group_cellseg_' + str(j) + '.png'),
            bbox_inches='tight', pad_inches=0)

j = 21
plt.clf()
plt.subplot(121)
plt.axis('off')
plt.imshow(test_cell_im[j, :, :, :])
plt.subplot(122)
plt.axis('off')
plt.imshow(test_cell_testlab[j, :, :, 0])
plt.savefig(os.path.join(fig_dir, 'klf14_b6ntac_exp_0030_figures_for_biocomputing_group_cellseg_' + str(j) + '.png'),
            bbox_inches='tight', pad_inches=0)

j = 45
plt.clf()
plt.subplot(121)
plt.axis('off')
plt.imshow(test_cell_im[j, :, :, :])
plt.subplot(122)
plt.axis('off')
plt.imshow(test_cell_testlab[j, :, :, 0])
plt.savefig(os.path.join(fig_dir, 'klf14_b6ntac_exp_0030_figures_for_biocomputing_group_cellseg_' + str(j) + '.png'),
            bbox_inches='tight', pad_inches=0)

# Dice estimation
preddice_test = dice_model.predict(test_cell_im[:, :, :, :] * np.repeat(test_cell_testlab[:, :, :, :], repeats=3, axis=3))

print('j = 45, dice = ' + "{0:.2f}".format(preddice_test[45][0]))
print('j = 12, dice = ' + "{0:.2f}".format(preddice_test[12][0]))
print('j = 33, dice = ' + "{0:.2f}".format(preddice_test[33][0]))
print('j = 21, dice = ' + "{0:.2f}".format(preddice_test[21][0]))
