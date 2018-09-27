import pandas as pd
from pathlib import Path
home = str(Path.home())
import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import measure
import PIL
# cross-platform home directory
from pathlib import Path
home = str(Path.home())

# PyCharm automatically adds cytometer to the python path, but this doesn't happen if the script is run
# with "python scriptname.py"
import glob
import numpy as np
import openslide
import csv
import pickle
import cytometer.models

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# limit GPU memory used
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

# Note: you need to use my branch of keras with the new functionality, that allows element-wise weights of the loss
# function
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AvgPool2D, Activation
import keras.backend as K
import cytometer.data
from cytometer.utils import principal_curvatures_range_image
import matplotlib.pyplot as plt

K.set_image_data_format('channels_last')

DEBUG = False


''' Defining CNN model
========================================================================================================================
'''


def fcn_sherrah2016_regression(input_shape, for_receptive_field=False):

    input = Input(shape=input_shape, dtype='float32', name='input_image')

    x = Conv2D(filters=32, kernel_size=(5, 5), strides=1, dilation_rate=1, padding='same')(input)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(3, 3), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)

    x = Conv2D(filters=96, kernel_size=(5, 5), strides=1, dilation_rate=2, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(5, 5), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, dilation_rate=4, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(9, 9), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(9, 9), strides=1, padding='same')(x)

    x = Conv2D(filters=196, kernel_size=(3, 3), strides=1, dilation_rate=8, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
        x = AvgPool2D(pool_size=(17, 17), strides=1, padding='same')(x)
    else:
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(17, 17), strides=1, padding='same')(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), strides=1, dilation_rate=16, padding='same')(x)
    if for_receptive_field:
        x = Activation('linear')(x)
    else:
        x = Activation('relu')(x)

    # dimensionality reduction layer
    main_output = Conv2D(filters=1, kernel_size=(1, 1), strides=1, dilation_rate=1, padding='same',
                         name='main_output')(x)

    return Model(inputs=input, outputs=main_output)

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
df_no = pd.DataFrame(data={'area': [], 'mouse_id': [], 'sex': [], 'diet': [],
                           'image_id': [], 'training/test': []})

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
        plt.imshow(np.array(im))
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
        df_no = df_no.append({'area': a, 'mouse_id': mouse_id, 'sex': mouse_sex,
                              'diet': mouse_diet, 'image_id': image_id, 'training/test': 'training'},
                             ignore_index=True)


''' Run test images through trained network, extract areas, create dataframe with area for each data group
========================================================================================================================
'''

saved_model_basename = 'c3h_hfd_exp_0000_cnn_dmap'

model_name = saved_model_basename + '*.h5'

# load model weights for each fold
model_files = glob.glob(os.path.join(saved_models_dir, model_name))
n_folds = len(model_files)

# load k-fold sets that were used to train the models
saved_model_kfold_filename = os.path.join(saved_models_dir, saved_model_basename + '_info.pickle')
with open(saved_model_kfold_filename, 'rb') as f:
    aux = pickle.load(f)
im_file_list = aux['file_list']
idx_test_all = aux['idx_test_all']

# correct home directory if we are in a different system than what was used to train the models
im_file_list = cytometer.data.change_home_directory(im_file_list,
                                                    '/users/rittscher/rcasero/Dropbox/c3h/c3h_hfd_training_augmented',
                                                    training_augmented_dir,
                                                    check_isfile=True)

# Run test images through the trained network
# list of model files to inspect
model_files = glob.glob(os.path.join(saved_models_dir, model_name))

# create empty dataframe to host the data
df = pd.DataFrame(data={'area': [], 'mouse_id': [], 'sex': [], 'diet': [], 'image_id': [], 'training/test': []})
print(df)

for fold_i, model_file in enumerate(model_files):

    # split the data into training and testing datasets
    im_test_file_list, _ = cytometer.data.split_list(im_file_list, idx_test_all[fold_i])

    # load im, seg and mask datasets
    test_datasets, _, _ = cytometer.data.load_datasets(im_test_file_list, prefix_from='im',
                                                       prefix_to=['im', 'mask', 'dmap'], nblocks=1)
    im_test = test_datasets['im']
    mask_test = test_datasets['mask']
    dmap_test = test_datasets['dmap']
    del test_datasets

    # load model
    model = fcn_sherrah2016_regression(input_shape=im_test.shape[1:])
    model.load_weights(model_file)
    model = cytometer.models.change_input_size(model, batch_shape=(None, 1001, 1001, 3))

    for i in range(im_test.shape[0]):

        im_test_it = im_test[i, :, :, :].reshape((1,) + im_test.shape[1:])

        # run image through network
        dmap_test_pred = model.predict(im_test_it)

        # compute mean curvature from dmap
        _, mean_curvature, _, _ = principal_curvatures_range_image(dmap_test_pred[0, :, :, 0], sigma=10)

        # reshape for watershed
        dmap_test_pred = dmap_test_pred[0, :, :, 0]

        # from dmaps calculate area using watershed method
        local_maxi = peak_local_max(image=dmap_test_pred, min_distance=40, indices=False)
        markers = ndi.label(local_maxi)[0]
        labels = watershed(-dmap_test_pred, markers)

        # eliminate cell areas which cross boundary of image
        cell_list = measure.regionprops(labels)
        area_list = []
        boundary_cell_list = []

        for k in range(len(cell_list)):
            if cell_list[k]['bbox'][0] != 0 and cell_list[k]['bbox'][1] != 0 and\
                    cell_list[k]['bbox'][2] != 1000 and cell_list[k]['bbox'][3] != 1000:
                # print('this cell is fine')
                area_from_list = int(cell_list[i]['area']) * (x_res * y_res) * 1e12
                area_list.append(area_from_list)
            else:
                # print('this cell touches boundary')
                boundary_cell_list.append(k)


        # # get area of each individual cell
        # unique_cells = np.unique(labels, return_counts=True)
        # cell_number = len(cell_list)
        # area_list = cell_list[] * (x_res * y_res) * 1e12


        #
        # plt.clf()
        # plt.subplot(211)
        # plt.imshow(dmap_test_pred)

        #
        # plt.subplot(212)
        # plt.cla()
        # plt.imshow(labels.astype('uint32'))

        # plt.clf()
        # plt.subplot(221)
        # plt.imshow(im_test_it[1:])
        # plt.title('histology, i = ' + str(i))
        # plt.subplot(222)
        # plt.imshow(dmap_test[i, :, :, 0])
        # plt.title('ground truth dmap')
        # plt.subplot(223)
        # plt.imshow(dmap_test_pred)
        # plt.title('predicted dmap')
        # plt.subplot(224)
        # plt.imshow(mean_curvature)
        # plt.title('mean curvature of dmap')

        # to plot segmentation
        # plt.subplot(111)
        # plt.cla()
        # plt.imshow(labels.astype('uint32'))

        # image ID
        image_id = os.path.basename(im_test_file_list[i])
        image_id = os.path.splitext(image_id)[-2]
        image_id = image_id.replace('im_seed_nan_', '')
        image_id = image_id.replace('-', ' ', 1)
        image_id = image_id.replace(' ', '', 2)

        # get mouse ID from the file name
        mouse_id = None
        for x in c3h_ids:
            if x in image_id:
                mouse_id = x
                break
        if mouse_id is None:
            raise ValueError('Filename does not seem to correspond to any known mouse ID: ' + im_test_file_list[i])

        # index of mouse ID
        idx = c3h_ids.index(mouse_id)

        # metainformation for this mouse
        mouse_sex = c3h_info[idx]['sex']
        mouse_diet = c3h_info[idx]['diet']

        # add to dataframe: area, image id, mouse id, sex, diet
        for i, a in enumerate(area_list):
            if a == 0.0:
                print('Warning! Area == 0.0: index ' + str(i) + ':' + image_id)
            df = df.append(
                {'area': a, 'mouse_id': mouse_id, 'sex': mouse_sex,
                 'diet': mouse_diet, 'image_id': image_id, 'training/test': 'test'},
                ignore_index=True)



''' split full dataset into smaller datasets for different groups 
========================================================================================================================
'''

# split dataset into groups
df_no_f = df_no.loc[df_no.sex == 'f', ('area', 'diet', 'image_id', 'mouse_id')]
df_no_m = df_no.loc[df_no.sex == 'm', ('area', 'diet', 'image_id', 'mouse_id')]

df_no_f_h = df_no_f.loc[df_no_f.diet == 'hfd', ('area', 'image_id', 'mouse_id')]
df_no_f_l = df_no_f.loc[df_no_f.diet == 'lfd', ('area', 'image_id', 'mouse_id')]
df_no_m_h = df_no_m.loc[df_no_m.diet == 'hfd', ('area', 'image_id', 'mouse_id')]
df_no_m_l = df_no_m.loc[df_no_m.diet == 'lfd', ('area', 'image_id', 'mouse_id')]

df_f = df.loc[df.sex == 'f', ('area', 'diet', 'image_id', 'mouse_id')]
df_m = df.loc[df.sex == 'm', ('area', 'diet', 'image_id', 'mouse_id')]

df_f_h = df_f.loc[df_f.diet == 'hfd', ('area', 'image_id', 'mouse_id')]
df_f_l = df_f.loc[df_f.diet == 'lfd', ('area', 'image_id', 'mouse_id')]
df_m_h = df_m.loc[df_m.diet == 'hfd', ('area', 'image_id', 'mouse_id')]
df_m_l = df_m.loc[df_m.diet == 'lfd', ('area', 'image_id', 'mouse_id')]

# merged training and test dataframes to compare test vs training

both_df = [df, df_no]
merge = pd.concat(both_df)

# by sex
merge_f = merge.loc[merge.sex == 'f', ('area', 'diet', 'image_id', 'mouse_id', 'training/test')]
merge_m = merge.loc[merge.sex == 'm', ('area', 'diet', 'image_id', 'mouse_id', 'training/test')]

# by diet
merge_f_h = merge_f.loc[merge_f.diet == 'hfd', ('area', 'image_id', 'mouse_id', 'training/test')]
merge_f_l = merge_f.loc[merge_f.diet == 'lfd', ('area', 'image_id', 'mouse_id', 'training/test')]
merge_m_h = merge_m.loc[merge_m.diet == 'hfd', ('area', 'image_id', 'mouse_id', 'training/test')]
merge_m_l = merge_m.loc[merge_m.diet == 'lfd', ('area', 'image_id', 'mouse_id', 'training/test')]




''' boxplots of each image
========================================================================================================================
'''

# # plot cell area boxplots for each individual image
# df_no.boxplot(column='area', by='image_id', vert=False)
#
# # plot boxplots for each individual image, split into f/m groups
# plt.clf()
# ax = plt.subplot(211)
# df_no_f.boxplot(column='area', by='image_id', vert=False, ax=ax)
# plt.title('female')
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

# plot boxplots for f/m, h/l comparison (just to check scales etc are okay)
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

# same boxplots without outliers (training)
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

# same boxplots without outliers (test)
plt.clf()
ax = plt.subplot(121)
df_f.boxplot(column='area', by='diet', ax=ax, showfliers=False, notch=True)
ax.set_ylim(0, 1.25e4)
ax.set_title('female', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('Cell area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
ax = plt.subplot(122)
df_m.boxplot(column='area', by='diet', ax=ax, showfliers=False, notch=True)
ax.set_ylim(0, 1.25e4)
ax.set_title('male', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('Cell area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)

# boxplot comparing female training vs female test
plt.clf()
ax = plt.subplot(121)
merge_f_h.boxplot(column='area', by='training/test', ax=ax, showfliers=False, notch=True)
ax.set_ylim(0, 2e4)
ax.set_title('female training', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
ax = plt.subplot(122)
merge_f_l.boxplot(column='area', by='training/test', ax=ax, showfliers=False, notch=True)
ax.set_ylim(0, 2e4)
ax.set_title('female test', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)

# boxplot comparing female training vs female test
plt.clf()
ax = plt.subplot(122)
merge_f_l.boxplot(column='area', by='training/test', ax=ax, showfliers=False, notch=True)
ax.set_ylim(0, 2e4)
ax.set_title('female lfd', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
ax = plt.subplot(121)
merge_f_h.boxplot(column='area', by='training/test', ax=ax, showfliers=False, notch=True)
ax.set_ylim(0, 2e4)
ax.set_title('female hfd', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)

# boxplot comparing male training vs male test
plt.clf()
ax = plt.subplot(122)
merge_m_l.boxplot(column='area', by='training/test', ax=ax, showfliers=False, notch=True)
ax.set_ylim(0, 1.5e4)
ax.set_title('male lfd', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
ax = plt.subplot(121)
merge_m_h.boxplot(column='area', by='training/test', ax=ax, showfliers=False, notch=True)
ax.set_ylim(0, 1.5e4)
ax.set_title('male hfd', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)