# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import pickle

# other imports
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import watershed
import pandas as pd

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.backend as K

from sklearn.metrics import roc_curve, auc

import cytometer.data
import cytometer.models
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
figures_dir = os.path.join(root_data_dir, 'figures')

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0041_cnn_qualitynet_thresholded_sigmoid_pm_1_masked_segmentation'

'''Check whether weights in neural networks are non NaNs (very slow!)
'''

if DEBUG:
    model_name = experiment_id + '_model_fold_*.h5'

    saved_model_filename = os.path.join(saved_models_dir, model_name)

    # list of model files to inspect
    model_files = glob.glob(os.path.join(saved_models_dir, model_name))

    n_folds = len(model_files)

    for i_fold in range(n_folds):

        # name of file
        model_file = model_files[i_fold]

        print('i_fold = ' + str(i_fold) + ', model = ' + model_file)

        # load model
        model = keras.models.load_model(model_file)

        # check whether the model has NaN values
        layers_with_nans = cytometer.models.check_model(model)

        if len(layers_with_nans) > 0:
            print('Model with NaNs: ' + model_file)
        else:
            print('OK model: ' + model_file)


'''Inference using all folds
'''

model_name = experiment_id + '_model_fold_*.h5'

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
est0_gt0 = np.zeros((n_folds, ), np.float32)
est0_gt1 = np.zeros((n_folds, ), np.float32)
est1_gt0 = np.zeros((n_folds, ), np.float32)
est1_gt1 = np.zeros((n_folds, ), np.float32)
test_onecell_dice_all = None
qual_all = None
for i_fold, idx_test in enumerate(idx_orig_test_all):

    # name of model's file
    model_file = model_files[i_fold]

    print('i_fold = ' + str(i_fold) + ', model = ' + model_file)

    '''Load test data of pipeline segmentations
    '''

    # split the data list into training and testing lists
    im_test_file_list, _ = cytometer.data.split_list(im_orig_file_list, idx_test)

    # number of test images
    n_im = len(im_test_file_list)

    # load training dataset
    datasets, _, _ = cytometer.data.load_datasets(im_test_file_list, prefix_from='im',
                                                  prefix_to=['im', 'lab', 'seg',
                                                             'predlab_kfold_' + str(i_fold).zfill(2)], nblocks=1)

    test_im = datasets['im']
    test_seg = datasets['seg']
    test_reflab = datasets['lab']
    test_predlab = datasets['predlab_kfold_' + str(i_fold).zfill(2)]
    del datasets

    # remove borders between cells in the lab_train data
    for i in range(test_reflab.shape[0]):
        test_reflab[i, :, :, 0] = watershed(image=np.zeros(shape=test_reflab[i, :, :, 0].shape, dtype=np.uint8),
                                            markers=test_reflab[i, :, :, 0], watershed_line=False)
    # change the background label from 1 to 0
    test_reflab[test_reflab == 1] = 0

    '''Split images into one-cell images, compute true Dice values, and prepare for inference
    '''

    # create one image per cell, and compute true Dice coefficient values
    test_onecell_im, test_onecell_testlab, test_onecell_index_list, test_onecell_reflab, test_onecell_dice = \
        cytometer.utils.one_image_per_label(test_im, test_predlab,
                                            dataset_lab_ref=test_reflab,
                                            training_window_len=training_window_len,
                                            smallest_cell_area=smallest_cell_area)

    if DEBUG:
        plt.clf()
        plt.subplot(121)
        i = 50
        plt.imshow(test_onecell_im[i, :, :, :])
        plt.contour(test_onecell_reflab[i, :, :, 0], levels=1, colors='black')
        plt.contour(test_onecell_testlab[i, :, :, 0], levels=1, colors='green')
        plt.title('Dice = ' + str("{:.2f}".format(test_onecell_dice[i])))
        plt.text(175, 180, '+1', fontsize=14, verticalalignment='top')
        plt.text(100, 75, '0', fontsize=14, verticalalignment='top', color='white')
        plt.subplot(122)
        i = 80
        plt.imshow(test_onecell_im[i, :, :, :])
        plt.contour(test_onecell_reflab[i, :, :, 0], levels=1, colors='black')
        plt.contour(test_onecell_testlab[i, :, :, 0], levels=1, colors='red')
        plt.title('Dice = ' + str("{:.2f}".format(test_onecell_dice[i])))
        plt.text(175, 180, '+1', fontsize=14, verticalalignment='top')
        plt.text(100, 75, '0', fontsize=14, verticalalignment='top', color='white')

    # multiply histology by -1/+1 mask as this is what the quality network expects
    test_aux = 2 * (test_onecell_testlab.astype(np.float32) - 0.5)
    masked_test_onecell_im = test_onecell_im * np.repeat(test_aux,
                                                         repeats=test_onecell_im.shape[3], axis=3)

    '''Assess quality of each cell's segmentation with Quality Network
    '''

    # load model
    model = keras.models.load_model(model_file)

    # quality score
    qual = model.predict(masked_test_onecell_im)

    if DEBUG:
        plt.clf()
        plt.subplot(121)
        i = 50
        plt.imshow(masked_test_onecell_im[i, :, :, :])
        plt.contour(test_onecell_reflab[i, :, :, 0], levels=1, colors='black')
        plt.contour(test_onecell_testlab[i, :, :, 0], levels=1, colors='green')
        plt.title('Dice = ' + str("{:.2f}".format(test_onecell_dice[i]))
                  + ', qual = ' + str("{:.2f}".format(qual[i, 0])))
        plt.text(175, 180, '+1', fontsize=14, verticalalignment='top')
        plt.text(100, 75, '0', fontsize=14, verticalalignment='top', color='white')
        plt.subplot(122)
        i = 80
        plt.imshow(masked_test_onecell_im[i, :, :, :])
        plt.contour(test_onecell_reflab[i, :, :, 0], levels=1, colors='black')
        plt.contour(test_onecell_testlab[i, :, :, 0], levels=1, colors='red')
        plt.title('Dice = ' + str("{:.2f}".format(test_onecell_dice[i]))
                  + ', qual = ' + str("{:.2f}".format(qual[i, 0])))
        plt.text(175, 180, '+1', fontsize=14, verticalalignment='top')
        plt.text(100, 75, '0', fontsize=14, verticalalignment='top', color='white')

    if DEBUG:
        plt.clf()
        plt.scatter(test_onecell_dice, qual)
        plt.tick_params(labelsize=16)
        plt.xlabel('Ground truth Dice coefficient', fontsize=16)
        plt.ylabel('Quality score', fontsize=16)

    # confusion table: estimated vs ground truth
    quality_threshold = 0.9
    est0_gt0[i_fold] = np.count_nonzero(np.logical_and(qual[:, 0] < quality_threshold,
                                                       test_onecell_dice < quality_threshold)) / len(qual)
    est0_gt1[i_fold] = np.count_nonzero(np.logical_and(qual[:, 0] < quality_threshold,
                                                       test_onecell_dice >= quality_threshold)) / len(qual)
    est1_gt0[i_fold] = np.count_nonzero(np.logical_and(qual[:, 0] >= quality_threshold,
                                                       test_onecell_dice < quality_threshold)) / len(qual)
    est1_gt1[i_fold] = np.count_nonzero(np.logical_and(qual[:, 0] >= quality_threshold,
                                                       test_onecell_dice >= quality_threshold)) / len(qual)

    confusion = np.array([["{:.2f}".format(est1_gt0[i_fold]), "{:.2f}".format(est1_gt1[i_fold])],
                          ["{:.2f}".format(est0_gt0[i_fold]), "{:.2f}".format(est0_gt1[i_fold])]])
    df_summary = pd.DataFrame(confusion, columns=['Bad seg.', 'Good seg.'], index=['Good qual.', 'Bad qual.'])
    print(df_summary)

    # accumulate results
    if qual_all is None:
        qual_all = qual
        test_onecell_dice_all = test_onecell_dice
    else:
        qual_all = np.concatenate((qual_all, qual))
        test_onecell_dice_all = np.concatenate((test_onecell_dice_all, test_onecell_dice))


'''Print result summaries
'''

# print % values for each fold
print(np.round(est1_gt1 * 100))  # good segmentation / accept segmentation
print(np.round(est0_gt0 * 100))  # bad / reject
print(np.round(est1_gt0 * 100))  # bad / accept
print(np.round(est0_gt1 * 100))  # good / reject

# plot boxplots for the Good/Bad segmentation vs. Accept/Reject segmentation

if DEBUG:
    plt.clf()
    plt.boxplot([est1_gt1*100, est0_gt0*100, est1_gt0*100, est0_gt1*100],
                labels=['Good/Accept​', 'Bad/Reject​', 'Bad/Accept​', 'Good/Reject​'])
    plt.title('0/+1 mask for quality network', fontsize=18)
    plt.tick_params(labelsize=16)
    plt.xticks(rotation=15)
    plt.ylim(-5, 65)

if SAVE_FIGS:
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_inspect_exp_0041_boxplots_confusion_matrices.png'))


if DEBUG:
    plt.clf()
    plt.scatter(test_onecell_dice_all, qual_all)
    plt.tick_params(labelsize=16)
    plt.xlabel('Ground truth Dice coefficient', fontsize=16)
    plt.ylabel('Quality score', fontsize=16)

if SAVE_FIGS:
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_inspect_exp_0041_dice_vs_quality_scatter.png'))

# compute ROC
fpr, tpr, thr = roc_curve(test_onecell_dice_all >= valid_threshold, qual_all)
roc_auc = auc(fpr, tpr)

# set the quality threshold so that the False Positive Rate <= 10%
idx = 159
quality_threshold = thr[idx]
# quality_threshold = 0.9

# plot ROC
if DEBUG:
    plt.clf()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.scatter(fpr[idx], tpr[idx],
             label='Quality threshold = %0.2f\nFPR = %0.2f, TPR = %0.2f' % (quality_threshold, fpr[idx], tpr[idx]))
    plt.tick_params(labelsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="lower right")

if SAVE_FIGS:
    plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_inspect_exp_0041_roc.png'))

# aggregated confusion table: estimated vs ground truth
est0_gt0_all = np.count_nonzero(np.logical_and(qual_all[:, 0] < quality_threshold,
                                               test_onecell_dice_all < quality_threshold)) / len(qual_all)
est0_gt1_all = np.count_nonzero(np.logical_and(qual_all[:, 0] < quality_threshold,
                                               test_onecell_dice_all >= quality_threshold)) / len(qual_all)
est1_gt0_all = np.count_nonzero(np.logical_and(qual_all[:, 0] >= quality_threshold,
                                               test_onecell_dice_all < quality_threshold)) / len(qual_all)
est1_gt1_all = np.count_nonzero(np.logical_and(qual_all[:, 0] >= quality_threshold,
                                               test_onecell_dice_all >= quality_threshold)) / len(qual_all)

confusion = np.array([["{:.2f}".format(est1_gt0_all), "{:.2f}".format(est1_gt1_all)],
                      ["{:.2f}".format(est0_gt0_all), "{:.2f}".format(est0_gt1_all)]])
df_summary = pd.DataFrame(confusion, columns=['Bad seg.', 'Good seg.'], index=['Good qual.', 'Bad qual.'])
print(df_summary)

'''Plot metrics and convergence
'''

log_filename = os.path.join(saved_models_dir, experiment_id + '.log')

if os.path.isfile(log_filename):

    # read Keras output
    df_list = cytometer.data.read_keras_training_output(log_filename)

    # plot metrics with every iteration
    plt.clf()
    for df in df_list:
        plt.subplot(211)
        loss_plot, = plt.semilogy(df.index, df.loss, label='loss')
        epoch_ends = np.concatenate((np.where(np.diff(df.epoch))[0], [len(df.epoch)-1, ]))
        epoch_ends_plot1, = plt.semilogy(epoch_ends, df.loss[epoch_ends], 'ro', label='end of epoch')
        plt.legend(handles=[loss_plot, epoch_ends_plot1])
        plt.subplot(212)
        regr_mae_plot, = plt.plot(df.index, df.acc, label='acc')
        regr_mae_epoch_ends_plot2, = plt.plot(epoch_ends, df.acc[epoch_ends], 'ro', label='end of epoch')
        plt.legend(handles=[regr_mae_plot, regr_mae_epoch_ends_plot2])

