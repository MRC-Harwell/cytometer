import os
import glob

training_data_dir = '/home/rcasero/Software/cytometer/data/klf14_b6ntac_training'

mat_file_list = glob.glob(os.path.join(training_data_dir, '*-MAT-*.xcf'))
pat_file_list = glob.glob(os.path.join(training_data_dir, '*-PAT-*.xcf'))

for file in mat_file_list:

    file