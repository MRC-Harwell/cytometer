import csv
import glob
import pandas as pd
from pathlib import Path
home = str(Path.home())
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import openslide
import PIL

DEBUG = False


''' Defining watershed algorithm
========================================================================================================================
'''

def watershed_segmentation(file, x_res, y_res):

    image = Image.open(file)
    pixels = np.array(image)

    local_maxi = peak_local_max(image=pixels, min_distance=20, indices=False)

    markers = ndi.label(local_maxi)[0]

    labels = watershed(-pixels, markers)

    pixel_res = x_res * y_res
    unique_cells = np.unique(labels, return_counts=True)
    area_list = unique_cells[1] * pixel_res

    return np.array(area_list)

''' Declaring data directories and initialising pixel variables
========================================================================================================================
'''

root_data_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
base_dir = os.path.join(home, 'Data/cytometer_data/c3h')
training_data_dir = os.path.join(home, 'Data/cytometer_data/c3h/c3h_hfd_training')
training_nooverlap_data_dir = os.path.join(home, 'Data/cytometer_data/c3h/c3h_hfd_training_non_overlap')
training_augmented_dir = os.path.join(home, 'Data/cytometer_data/c3h/c3h_hfd_training_augmented')
saved_models_dir = os.path.join(home, 'Data/cytometer_data/c3h/saved_models')

original_file = os.path.join(root_data_dir, 'C3HHM 3095.5a 353-16 C1 - 2016-04-15 11.54.24.ndpi')

# open original image
im = openslide.OpenSlide(original_file)

# compute pixel size in the original image, where patches were taken from
if im.properties['tiff.ResolutionUnit'].lower() == 'centimeter':
    x_res = 1e-2 / float(im.properties['tiff.XResolution'])  # meters / pixel
    y_res = 1e-2 / float(im.properties['tiff.YResolution'])  # meters / pixel
else:
    raise ValueError('Only centimeter units implemented')

with open(os.path.join(base_dir, 'c3h_hfd_meta_info.csv'), 'r') as f:
    reader = csv.DictReader(f, skipinitialspace=True)
    c3h_info = []
    for row in reader:
        c3h_info.append(row)

f.close()

''' load non-overlapping segmentations, compute cell areas and create dataframe
========================================================================================================================
'''

# list of mouse IDs
c3h_ids = [x['mouse_id'] for x in c3h_info]

file_list = glob.glob(os.path.join(training_nooverlap_data_dir, '*.tif'))

# create empty dataframe to host the data
df_no = pd.DataFrame(data={'area': [], 'mouse_id': [], 'sex': [], 'diet': [], 'image_id': []})

for file in file_list:

    # image ID
    image_id = os.path.basename(file)
    image_id = os.path.splitext(image_id)[-2]
    image_id = image_id.replace('-', ' ', 1)
    image_id = image_id.replace(' ', '', 2)

    # get mouse ID from the file name
    mouse_id = None
    for x in c3h_ids:
        if x in image_id:
            mouse_id = x
            break
    if mouse_id is None:
        raise ValueError('Filename does not seem to correspond to any known mouse ID: ' + file)

    # index of mouse ID
    idx = c3h_ids.index(mouse_id)

    # sex and diet for this mouse
    mouse_sex = c3h_info[idx]['sex']
    mouse_diet = c3h_info[idx]['diet']

    # load file with the watershed non-overlapping labels
    im = PIL.Image.open(file)
    im = im.getchannel(0)

    if DEBUG:
        plt.clf()
        plt.imshow(im)
        plt.show()

    # number of pixels in each label
    areas = np.array(im.histogram(), dtype=np.float32)

    # remove cell contour and background labels
    CONTOUR = 0
    BACKGROUND = 1
    areas = areas[BACKGROUND+1:]

    # remove labels with no pixels (cells that are completely covered by other cells)
    areas = areas[areas != 0]

    # compute areas (m^2) from number of pixels
    areas *= x_res * y_res

    # convert areas to um^2
    areas *= 1e12

    # add to dataframe: area, image id, mouse id, sex, KO
    for i, a in enumerate(areas):
        if a == 0.0:
            print('Warning! Area == 0.0: index ' + str(i) + ':' + image_id)
        df_no = df_no.append({'area': a, 'mouse_id': mouse_id, 'sex': mouse_sex, 'diet': mouse_diet, 'image_id': image_id},
                             ignore_index=True)


''' dmap estimates produced by network
========================================================================================================================
'''

# file_list = glob.glob(os.path.join(training_data_dir, '*.tif'))

# create empty dataframe to host the data
df = pd.DataFrame(data={'area': [], 'mouse_id': [], 'sex': [], 'image_id': []})
print(df)

# read all dmap files, and categorise them into hfd/lfd and f/m
for file in file_list:

    # image ID
    image_id = os.path.basename(file)
    image_id = os.path.splitext(image_id)[-2]
    image_id = image_id.replace('-', ' ', 1)
    image_id = image_id.replace(' ', '', 2)


    # get mouse ID from the file name
    mouse_id = None
    for x in c3h_ids:
        if x in image_id:
            mouse_id = x
            break
    if mouse_id is None:
        raise ValueError('Filename does not seem to correspond to any known mouse ID: ' + file)

    # index of mouse ID
    idx = c3h_ids.index(mouse_id)

    # metainformation for this mouse
    mouse_sex = c3h_info[idx]['sex']
    mouse_diet = c3h_info[idx]['diet']

    # watershed
    areas = watershed_segmentation(file, x_res=1, y_res=1)

    # add to dataframe: area, image id, mouse id, sex, diet
    for i, a in enumerate(areas):
        if a == 0.0:
            print('Warning! Area == 0.0: index ' + str(i) + ':' + image_id)
        df = df.append({'area': a, 'mouse_id': mouse_id, 'sex': mouse_sex, 'diet': mouse_diet, 'image_id': image_id},
                       ignore_index=True)


''' split full dataset into smaller datasets for different groups 
========================================================================================================================
'''

# split dataset into groups
df_no_f = df_no.loc[df_no.sex == 'f', ('area', 'diet', 'image_id', 'mouse_id')]
df_no_m = df_no.loc[df_no.sex == 'm', ('area', 'diet', 'image_id', 'mouse_id')]

df_no_f_h = df_no_f.loc[df_no_f.diet == 'h', ('area', 'image_id', 'mouse_id')]
df_no_f_l = df_no_f.loc[df_no_f.diet == 'l', ('area', 'image_id', 'mouse_id')]
df_no_m_h = df_no_m.loc[df_no_m.diet == 'h', ('area', 'image_id', 'mouse_id')]
df_no_m_l = df_no_m.loc[df_no_m.diet == 'l', ('area', 'image_id', 'mouse_id')]

''' boxplots of each image
========================================================================================================================
'''

# # plot cell area boxplots for each individual image
df_no.boxplot(column='area', by='image_id', vert=False)
#
# plot boxplots for each individual image, split into f/m groups
plt.clf()
ax = plt.subplot(211)
df_no_f.boxplot(column='area', by='image_id', vert=False, ax=ax)
plt.title('female')
#
# ax = plt.subplot(212)
# df_m.boxplot(column='area', by='image_id', vert=False, ax=ax)
# plt.title('male')
#
# # plot boxplots for each individual image, split into MAT/PAT groups
# plt.clf()
# ax = plt.subplot(211)
# df_MAT.boxplot(column='area', by='image_id', vert=False, ax=ax)
# plt.title('MAT')
#
# ax = plt.subplot(212)
# df_PAT.boxplot(column='area', by='image_id', vert=False, ax=ax)
# plt.title('PAT')
#
# # plot boxplots for f/m, h/l comparison (just to check scales etc are okay)
plt.clf()
ax = plt.subplot(121)
df_no_f.boxplot(column='area', by='diet', ax=ax, notch=True)
ax.set_ylim(0, 2e4)
ax.set_title('female', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
ax = plt.subplot(122)
df_no_m.boxplot(column='area', by='diet', ax=ax, notch=True)
ax.set_ylim(0, 2e4)
ax.set_title('male', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)

# same boxplots without outliers
plt.clf()
ax = plt.subplot(121)
df_no_f.boxplot(column='area', by='diet', ax=ax, showfliers=False, notch=True)
ax.set_ylim(0, 2e4)
ax.set_title('female', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
ax = plt.subplot(122)
df_no_m.boxplot(column='area', by='diet', ax=ax, showfliers=False, notch=True)
ax.set_ylim(0, 2e4)
ax.set_title('male', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)

