'''
Generate k-folds split of training and testing data.

* KLF14 data hand segmented by Ramon Casero.

Data is split into 10 folds.

KLF14: 61 files
    Cells: 2152
    Other: 100
    Brown: 1
'''

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0079_generate_kfolds'

DEBUG = False

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

saved_models_dir = os.path.join(klf14_root_data_dir, 'saved_models')

'''Prepare folds'''

# we are interested only in .tif files for which we created hand segmented contours
#
# Note: These training images were generated with script 0076.
#
# Note: The line below is the code we used the first time we ran this script, as an easy way to select all existing
# files at the time. However, since then, we have created more .svg files. For the sake of reproducibility of results,
# this line is now commented out, and the list of files is provided explicitly below
# im_svg_file_list = glob.glob(os.path.join(klf14_training_dir, '*.svg'))

# Note: Of these 61 files, only 55 contain cell segmentations
im_svg_file_list = [
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11_row_019220_col_061724.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-37.1d PAT 109-16 C1 - 2016-02-15 15.19.08_row_012172_col_049588.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_023100_col_009220.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-17.2f  68-16 C1 - 2016-02-04 15.05.54_row_013348_col_019316.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11_row_021012_col_057844.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00_row_019228_col_015060.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04_row_026108_col_068956.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.2b  58-16 C1 - 2016-02-03 11.10.52_row_006268_col_013820.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38_row_009588_col_028676.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06_row_013300_col_055476.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_row_008212_col_015364.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_006124_col_082236.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_row_015004_col_010364.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52_row_012404_col_054316.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_row_014980_col_027052.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46_row_021804_col_035412.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32_row_031172_col_025996.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_row_021812_col_022916.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.2d  60-16 C1 - 2016-02-03 13.13.57_row_024076_col_020404.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_row_019556_col_057972.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_row_011004_col_005988.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.1a  50-16 C1 - 2016-02-02 09.12.41_row_028156_col_018596.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_row_005348_col_019844.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_007436_col_019092.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.2b  58-16 C1 - 2016-02-03 11.10.52_row_013820_col_057052.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-17.2f  68-16 C1 - 2016-02-04 15.05.54_row_026132_col_012148.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11_row_016260_col_058300.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_row_010732_col_016692.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46_row_004124_col_012524.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06_row_011852_col_071620.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-17.2c  66-16 C1 - 2016-02-04 11.46.39_row_030820_col_022204.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32_row_034628_col_040116.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11_row_010084_col_058476.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_031860_col_033476.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_row_006652_col_061724.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32_row_035948_col_041492.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00_row_016812_col_017484.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-37.1d PAT 109-16 C1 - 2016-02-15 15.19.08_row_016068_col_007276.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.2d  60-16 C1 - 2016-02-03 13.13.57_row_018380_col_063068.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32_row_013012_col_019820.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-37.1c PAT 108-16 C1 - 2016-02-15 14.49.45_row_007372_col_008556.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52_row_005428_col_058372.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_016756_col_063692.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38_row_021796_col_055852.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32_row_006412_col_012484.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52_row_013604_col_024644.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_row_012828_col_018388.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52_row_014628_col_069148.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_row_027388_col_018468.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52_row_019340_col_017348.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11_row_023236_col_011084.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_row_006900_col_071980.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_024836_col_055124.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46_row_008532_col_009804.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-17.2f  68-16 C1 - 2016-02-04 15.05.54_row_007500_col_050372.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-37.1c PAT 108-16 C1 - 2016-02-15 14.49.45_row_001292_col_004348.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04_row_010716_col_008924.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_row_009236_col_018316.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46_row_017044_col_031228.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-MAT-18.1a  50-16 C1 - 2016-02-02 09.12.41_row_012908_col_010212.svg',
    '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_row_016556_col_010292.svg'
]

# correct home directory
im_svg_file_list = [x.replace('/home/rcasero', home) for x in im_svg_file_list]

if DEBUG:
    for i, file in enumerate(im_svg_file_list):
        print(file)
        print('   Cell: ' + str(len(cytometer.data.read_paths_from_svg_file(file, tag='Cell'))))
        print('   Other: ' + str(len(cytometer.data.read_paths_from_svg_file(file, tag='Other')) +
                                 len(cytometer.data.read_paths_from_svg_file(file, tag='Brown'))))
        print('   Background: ' + str(len(cytometer.data.read_paths_from_svg_file(file, tag='Background'))))

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
print('KLF14: ' + str(n_klf14) + ' files')
print('    Cells: ' + str(np.sum(contours['cell'][0:n_klf14])))
print('    Other: ' + str(np.sum(contours['other'][0:n_klf14])))
print('    Brown: ' + str(np.sum(contours['brown'][0:n_klf14])))

# number of images
n_orig_im = len(im_svg_file_list)

# split SVG files into training and testing for k-folds
idx_orig_train_klf14, idx_orig_test_klf14 = cytometer.data.split_file_list_kfolds(
    im_svg_file_list[0:n_klf14], n_folds, ignore_str='_row_.*', fold_seed=0, save_filename=None)

# rename variables
idx_orig_train_all = idx_orig_train_klf14
idx_orig_test_all = idx_orig_test_klf14

# save folds
kfold_info_filename = os.path.join(saved_models_dir, experiment_id + '.pickle')
with open(kfold_info_filename, 'wb') as f:
    x = {'file_list': im_svg_file_list, 'idx_train': idx_orig_train_all, 'idx_test': idx_orig_test_all,
         'fold_seed': 0, 'n_klf14': n_klf14}
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
    print('    Test:')
    print('        KLF14: ' + str(np.count_nonzero(idx_orig_test_all[k] < n_klf14)))
