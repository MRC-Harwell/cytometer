'''
Generate k-folds split of training and testing data.

* KLF14 data hand segmented by Ramon Casero.
* C3H data hand segmented by Ginaluca Cientanni.

Data is split into 10 folds, separately for KLF14 and C3H, to account for the large difference in the number of files
(there are more C3H files, but actually more KLF14 segmented objects).

KLF14: 61 files
    Cells: 2152
    Other: 100
    Brown: 1
C3H: 248 files
    Cells: 1110
    Other: 185
    Brown: 0
'''

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0080_generate_kfolds'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import glob
import cytometer.data
import numpy as np
import pickle

# number of folds to split the data into
n_folds = 10

# data paths
klf14_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')
klf14_training_dir = os.path.join(klf14_root_data_dir, 'klf14_b6ntac_training')

c3h_root_data_dir = os.path.join(home, 'Data/cytometer_data/c3h')
c3h_training_dir = os.path.join(c3h_root_data_dir, 'c3h_hfd_training')

saved_models_dir = os.path.join(klf14_root_data_dir, 'saved_models')

'''Prepare folds'''

# we are interested only in .tif files for which we created hand segmented contours
im_svg_file_list = glob.glob(os.path.join(klf14_training_dir, '*.svg')) \
                   + glob.glob(os.path.join(c3h_training_dir, '*.svg'))

# extract contours
contours = {'cell': [], 'other': [], 'brown': []}
for i, file in enumerate(im_svg_file_list):
    contours['cell'].append(len(cytometer.data.read_paths_from_svg_file(file, tag='Cell')))
    contours['other'].append(len(cytometer.data.read_paths_from_svg_file(file, tag='Other')))
    contours['brown'].append(len(cytometer.data.read_paths_from_svg_file(file, tag='Brown')))
contours['cell'] = np.array(contours['cell'])
contours['other'] = np.array(contours['other'])
contours['brown'] = np.array(contours['brown'])

# inspect number of hand segmented objects
n_klf14 = len(glob.glob(os.path.join(klf14_training_dir, '*.svg')))
n_c3h = len(glob.glob(os.path.join(c3h_training_dir, '*.svg')))
print('KLF14: ' + str(n_klf14) + ' files')
print('    Cells: ' + str(np.sum(contours['cell'][0:n_klf14])))
print('    Other: ' + str(np.sum(contours['other'][0:n_klf14])))
print('    Brown: ' + str(np.sum(contours['brown'][0:n_klf14])))
print('C3H: ' + str(n_c3h) + ' files')
print('    Cells: ' + str(np.sum(contours['cell'][n_klf14:])))
print('    Other: ' + str(np.sum(contours['other'][n_klf14:])))
print('    Brown: ' + str(np.sum(contours['brown'][n_klf14:])))

# number of images
n_orig_im = len(im_svg_file_list)

# split SVG files into training and testing for k-folds. We split the KLF14 and C3H datasets separately because C3H has
# many more files than KLF14
idx_orig_train_klf14, idx_orig_test_klf14 = cytometer.data.split_file_list_kfolds(
    im_svg_file_list[0:n_klf14], n_folds, ignore_str='_row_.*', fold_seed=0, save_filename=None)
idx_orig_train_c3h, idx_orig_test_c3h = cytometer.data.split_file_list_kfolds(
    im_svg_file_list[n_klf14:], n_folds, ignore_str='_row_.*', fold_seed=0, save_filename=None)

# concatenate KLF14 and C3H sets (correct the C3H indices so that they refer to the whole im_svg_file_list)
idx_orig_train_all = [np.concatenate((x, (y + n_klf14))) for x, y in zip(idx_orig_train_klf14, idx_orig_train_c3h)]
idx_orig_test_all = [np.concatenate((x, (y + n_klf14))) for x, y in zip(idx_orig_test_klf14, idx_orig_test_c3h)]

# save folds
kfold_info_filename = os.path.join(saved_models_dir, experiment_id + '.pickle')
with open(kfold_info_filename, 'wb') as f:
    x = {'file_list': im_svg_file_list, 'idx_train': idx_orig_train_all, 'idx_test': idx_orig_test_all,
         'fold_seed': 0, 'n_klf14': n_klf14, 'n_c3h': n_c3h}
    pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)

# inspect number of hand segmented objects per fold
for k in range(n_folds):
    print('Fold: ' + str(k))
    print('    Train:')
    print('        Cells: ' + str(np.sum(contours['cell'][idx_orig_train_all[k]])))
    print('        Other: ' + str(np.sum(contours['other'][idx_orig_train_all[k]])))
    print('        Brown: ' + str(np.sum(contours['brown'][idx_orig_train_all[k]])))
    print('    Test:')
    print('        Cells: ' + str(np.sum(contours['cell'][idx_orig_test_all[k]])))
    print('        Other: ' + str(np.sum(contours['other'][idx_orig_test_all[k]])))
    print('        Brown: ' + str(np.sum(contours['brown'][idx_orig_test_all[k]])))

# inspect dataset origin in each fold
for k in range(n_folds):
    print('Fold: ' + str(k))
    print('    Train:')
    print('        KLF14: ' + str(np.count_nonzero(idx_orig_train_all[k] < n_klf14)))
    print('        C3H: ' + str(np.count_nonzero(idx_orig_train_all[k] >= n_klf14)))
    print('    Test:')
    print('        KLF14: ' + str(np.count_nonzero(idx_orig_test_all[k] < n_klf14)))
    print('        C3H: ' + str(np.count_nonzero(idx_orig_test_all[k] >= n_klf14)))
