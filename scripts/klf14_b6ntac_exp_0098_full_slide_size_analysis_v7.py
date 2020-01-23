"""
Read annotation files from full slides processed by pipeline v7.
"""

# script name to identify this experiment
experiment_id = 'klf14_b6ntac_exp_0098_full_slide_size_analysis_v7'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import sys
import ujson
import pickle
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import cytometer.utils
import tensorflow as tf
from PIL import Image, ImageDraw
import pandas as pd
import scipy.stats as stats

# Filter out INFO & WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# limit number of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

os.environ['KERAS_BACKEND'] = 'tensorflow'

import openslide
import numpy as np
import matplotlib.pyplot as plt
from cytometer.utils import rough_foreground_mask
import cytometer.data
import itertools
from shapely.geometry import Polygon
import scipy

LIMIT_GPU_MEM = False

# limit GPU memory used
if LIMIT_GPU_MEM:
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
# figures_dir = os.path.join(root_data_dir, 'figures')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
saved_models_dir = os.path.join(root_data_dir, 'saved_models')
results_dir = os.path.join(root_data_dir, 'klf14_b6ntac_results')
annotations_dir = os.path.join(home, 'Software/AIDA/dist/data/annotations')
training_augmented_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_augmented')

# k-folds file
saved_kfolds_filename = 'klf14_b6ntac_exp_0094_generate_extra_training_images.pickle'

# rough_foreground_mask() parameters
downsample_factor = 8.0
dilation_size = 25
component_size_threshold = 1e6
hole_size_treshold = 8000


# list of annotation files
json_annotation_files = [
    'KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38_exp_0097_corrected.json',
    'KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06_exp_0097_corrected.json',
    'KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00_exp_0097_corrected.json',
    'KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_exp_0097_corrected.json',
    'KLF14-B6NTAC-37.1c PAT 108-16 C1 - 2016-02-15 14.49.45_exp_0097_corrected.json',
    'KLF14-B6NTAC-37.1d PAT 109-16 C1 - 2016-02-15 15.19.08_exp_0097_corrected.json',
    'KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46_exp_0097_corrected.json',
    'KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04_exp_0097_corrected.json',
    'KLF14-B6NTAC-MAT-17.2c  66-16 C1 - 2016-02-04 11.46.39_exp_0097_corrected.json',
    'KLF14-B6NTAC-MAT-18.1a  50-16 C1 - 2016-02-02 09.12.41_exp_0097_corrected.json',
    'KLF14-B6NTAC-MAT-18.2b  58-16 C1 - 2016-02-03 11.10.52_exp_0097_corrected.json',
    'KLF14-B6NTAC-MAT-18.2d  60-16 C1 - 2016-02-03 13.13.57_exp_0097_corrected.json',
    'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_exp_0097_corrected.json',
    'KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11_exp_0097_corrected.json',
    'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_exp_0097_corrected.json',
    'KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32_exp_0097_corrected.json',
    'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33_exp_0097_corrected.json'
]

# load svg files from manual dataset
saved_kfolds_filename = os.path.join(saved_models_dir, saved_kfolds_filename)
with open(saved_kfolds_filename, 'rb') as f:
    aux = pickle.load(f)
file_svg_list = aux['file_list']# load list of images, and indices for training vs. testing indices

# correct home directory in file paths
file_svg_list = cytometer.data.change_home_directory(list(file_svg_list), '/users/rittscher/rcasero', home,
                                                     check_isfile=True)

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(root_data_dir, 'klf14_b6ntac_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

########################################################################################################################
## Colourmap for AIDA
########################################################################################################################

# loop files with hand traced contours
manual_areas_f = []
manual_areas_m = []
for i, file_svg in enumerate(file_svg_list):

    print('file ' + str(i) + '/' + str(len(file_svg_list) - 1) + ': ' + os.path.basename(file_svg))

    # change file extension from .svg to .tif
    file_tif = file_svg.replace('.svg', '.tif')

    # open histology training image
    im = Image.open(file_tif)

    # read pixel size information
    xres = 0.0254 / im.info['dpi'][0] * 1e6  # um
    yres = 0.0254 / im.info['dpi'][1] * 1e6  # um

    # read the ground truth cell contours in the SVG file. This produces a list [contour_0, ..., contour_N-1]
    # where each contour_i = [(X_0, Y_0), ..., (X_P-1, X_P-1)]
    contours = cytometer.data.read_paths_from_svg_file(file_svg, tag='Cell', add_offset_from_filename=False,
                                                       minimum_npoints=3)

    # create dataframe for this image
    df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(file_svg),
                                                          values=[i,], values_tag='i',
                                                          tags_to_keep=['id', 'ko', 'sex'])

    # mouse ID as a string
    id = df_common['id'].values[0]
    sex = df_common['sex'].values[0]
    ko = df_common['ko'].values[0]

    # compute cell area
    if sex == 'f':
        manual_areas_f.append([Polygon(c).area * xres * yres for c in contours])  # (um^2)
    elif sex == 'm':
        manual_areas_m.append([Polygon(c).area * xres * yres for c in contours])  # (um^2)
    else:
        raise ValueError('Wrong sex value')


manual_areas_f = list(itertools.chain.from_iterable(manual_areas_f))
manual_areas_m = list(itertools.chain.from_iterable(manual_areas_m))

# compute function to map between cell areas and [0.0, 1.0], that we can use to sample the colourmap uniformly according
# to area quantiles
f_area2quantile_f = cytometer.data.area2quantile(manual_areas_f)
f_area2quantile_m = cytometer.data.area2quantile(manual_areas_m)

# load AIDA's colourmap
cm = cytometer.data.aida_colourmap()

########################################################################################################################
## Annotation file loop
########################################################################################################################


# loop annotations files
for i_file, json_file in enumerate(json_annotation_files):

    print('JSON annotations file: ' + os.path.basename(json_file))

    # name of corresponding .ndpi file
    ndpi_file = json_file.replace('_exp_0097_corrected.json', '.ndpi')
    kernel_file = os.path.splitext(ndpi_file)[0]

    # add path to file
    json_file = os.path.join(annotations_dir, json_file)
    ndpi_file = os.path.join(data_dir, ndpi_file)

    # open full resolution histology slide
    im = openslide.OpenSlide(ndpi_file)

    # pixel size
    assert (im.properties['tiff.ResolutionUnit'] == 'centimeter')
    xres = 1e-2 / float(im.properties['tiff.XResolution'])
    yres = 1e-2 / float(im.properties['tiff.YResolution'])

    # change pixel size to downsampled size
    xres *= downsample_factor
    yres *= downsample_factor

    # rough segmentation of the tissue in the image
    lores_istissue0, im_downsampled = rough_foreground_mask(ndpi_file, downsample_factor=downsample_factor,
                                                            dilation_size=dilation_size,
                                                            component_size_threshold=component_size_threshold,
                                                            hole_size_treshold=hole_size_treshold,
                                                            return_im=True)

    if DEBUG:
        plt.clf()
        plt.subplot(211)
        plt.imshow(im_downsampled)
        plt.subplot(212)
        plt.imshow(lores_istissue0)

    # list of items (there's a contour in each item)
    contours = cytometer.data.aida_get_contours(json_file, layer_name='White adipocyte.*')

    # init array for interpolated quantiles
    quantiles_grid = np.zeros(shape=lores_istissue0.shape, dtype=np.float32)

    # init array for mask where there are segmentations
    areas_mask = Image.new("1", lores_istissue0.shape[::-1], "black")
    draw = ImageDraw.Draw(areas_mask)

    # init lists for contour centroids and areas
    areas_all = []
    centroids_all = []

    # loop items (one contour per item)
    for c in contours:

        # convert to downsampled coordinates
        c = np.array(c) / downsample_factor

        if DEBUG:
            plt.fill(c[:, 0], c[:, 1], fill=False, color='r')

        # compute cell area
        area = Polygon(c).area * xres * yres  # (m^2)
        areas_all.append(area)

        # compute centroid of contour
        centroid = np.mean(c, axis=0)
        centroids_all.append(centroid)

        # add object described by contour to mask
        draw.polygon(list(c.flatten()), outline="white", fill="white")

    # convert mask
    areas_mask = np.array(areas_mask, dtype=np.bool)

    areas_all = np.array(areas_all) * 1e12

    # interpolate scattered area data to regular grid
    idx = areas_mask
    xi = np.transpose(np.array(np.where(idx)))[:, [1, 0]]
    quantiles_grid[idx] = scipy.interpolate.griddata(centroids_all, areas_all, xi, method='linear', fill_value=0)

    if DEBUG:
        plt.clf()
        plt.subplot(211)
        plt.imshow(im_downsampled)
        plt.axis('off')
        plt.subplot(212)
        plt.imshow(quantiles_grid)

    # create dataframe for this image
    df_common = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(ndpi_file),
                                                          values=[i,], values_tag='i',
                                                          tags_to_keep=['id', 'ko', 'sex'])

    # mouse ID as a string
    id = df_common['id'].values[0]
    sex = df_common['sex'].values[0]
    ko = df_common['ko'].values[0]

    # convert area values to quantiles
    if sex == 'f':
        quantiles_grid = f_area2quantile_f(quantiles_grid)
    elif sex == 'm':
        quantiles_grid = f_area2quantile_m(quantiles_grid)
    else:
        raise ValueError('Wrong sex value')

    # make background white in the plot
    quantiles_grid[~areas_mask] = np.nan

    if DEBUG:
        plt.clf()
        plt.subplot(211)
        plt.imshow(im_downsampled)
        plt.axis('off')
        plt.subplot(212)
        # plt.imshow(areas_grid, vmin=0.0, vmax=1.0, cmap='gnuplot2')
        plt.imshow(quantiles_grid, vmin=0.0, vmax=1.0, cmap=cm)
        cbar = plt.colorbar(shrink=1.0)
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.set_ylabel('Cell area quantile', rotation=90, fontsize=14)
        plt.axis('off')
        plt.tight_layout()

    if DEBUG:
        plt.clf()
        plt.hist(areas_all, bins=50, density=True, histtype='step')

    # plot cell areas for paper
    plt.clf()
    plt.imshow(quantiles_grid, vmin=0.0, vmax=1.0, cmap=cm)
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, kernel_file + '_exp_0098_cell_segmentation.png'),
                bbox_inches='tight')

# colourmap plot
a = np.array([[0,1]])
plt.figure(figsize=(9, 1.5))
img = plt.imshow(a, cmap=cm)
plt.gca().set_visible(False)
cax = plt.axes([0.1, 0.2, 0.8, 0.6])
cbar = plt.colorbar(orientation='horizontal', cax=cax)
cbar.ax.tick_params(labelsize=14)
plt.title('Cell area quantile (w.r.t. manual dataset)', rotation=0, fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'exp_0098_aida_colourmap.png'), bbox_inches='tight')

# plot area distributions
plt.clf()
aq_f = stats.mstats.hdquantiles(np.array(manual_areas_f) * 1e-3, prob=np.linspace(0, 1, 11), axis=0)
for a in aq_f:
    plt.plot([a, a], [0, 0.55], 'r', linewidth=3)
plt.hist(np.array(manual_areas_f) * 1e-3, histtype='stepfilled', bins=50, density=True, linewidth=4, zorder=0)
plt.tick_params(labelsize=14)
plt.xlabel('Cell area ($\cdot 10^3\ \mu m^2$)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'exp_0098_dist_quantiles_manual_f.png'), bbox_inches='tight')

plt.clf()
aq_m = stats.mstats.hdquantiles(np.array(manual_areas_m) * 1e-3, prob=np.linspace(0, 1, 11), axis=0)
for a in aq_m:
    plt.plot([a, a], [0, 0.28], 'r', linewidth=3)
plt.hist(np.array(manual_areas_m) * 1e-3, histtype='stepfilled', bins=50, density=True, linewidth=4, zorder=0)
plt.tick_params(labelsize=14)
plt.xlabel('Cell area ($\cdot 10^3\ \mu m^2$)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'exp_0098_dist_quantiles_manual_m.png'), bbox_inches='tight')
