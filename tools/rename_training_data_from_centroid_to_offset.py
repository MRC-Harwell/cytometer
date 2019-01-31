# quick and dirty script to change the row, col offset values in filenames from the centre of the window,
# to the first pixel of the window

import os
import shutil

dir_from = '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training_augmented.bak'
dir_to = '/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training_augmented'

files = [f for f in os.listdir(dir_from) if os.path.isfile(os.path.join(dir_from, f))]

for file in files:

    if file == 'Features.TXT':
        continue

    file_basename, file_ext = os.path.splitext(file)  # remove extension
    file_basename = file_basename.split('_')
    row_i = file_basename.index('row')
    row_offset = str(int(file_basename[row_i + 1]) - 500).zfill(6)
    col_i = file_basename.index('col')
    col_offset = str(int(file_basename[col_i + 1]) - 500).zfill(6)

    file_basename[row_i + 1] = row_offset
    file_basename[col_i + 1] = col_offset

    outfile_basename = "_".join(file_basename) + file_ext

    shutil.copy(os.path.join(dir_from, file), os.path.join(dir_to, outfile_basename))
