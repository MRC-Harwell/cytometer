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
import json
import pickle
sys.path.extend([os.path.join(home, 'Software/cytometer')])
import cytometer.utils
from PIL import Image, ImageDraw
from matplotlib.colors import ListedColormap

# Filter out INFO & WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # limit number of GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

os.environ['KERAS_BACKEND'] = 'tensorflow'

import openslide
import numpy as np
import matplotlib.pyplot as plt
from cytometer.utils import rough_foreground_mask
import itertools
from shapely.geometry import Polygon
import scipy
import scipy.stats as stats

# # limit GPU memory used
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# set_session(tf.Session(config=config))

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
    'KLF14-B6NTAC-MAT-18.2b  58-16 C1 - 2016-02-03 11.10.52_exp_0097.json',
    'KLF14-B6NTAC-MAT-18.2d  60-16 C1 - 2016-02-03 13.13.57_exp_0097.json',
    'KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38_exp_0097.json',
    'KLF14-B6NTAC-MAT-17.2c  66-16 C1 - 2016-02-04 11.46.39_exp_0097.json',
    'KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04_exp_0097.json',
    'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53_exp_0097.json',
    'KLF14-B6NTAC-37.1c PAT 108-16 C1 - 2016-02-15 14.49.45_exp_0097.json',
    'KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46_exp_0097.json',
    'KLF14-B6NTAC-37.1d PAT 109-16 C1 - 2016-02-15 15.19.08_exp_0097.json',
    'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52_exp_0097.json',
    'KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38_exp_0097.json'
]

# load svg files from manual dataset
saved_kfolds_filename = os.path.join(saved_models_dir, saved_kfolds_filename)
with open(saved_kfolds_filename, 'rb') as f:
    aux = pickle.load(f)
file_svg_list = aux['file_list']# load list of images, and indices for training vs. testing indices

# correct home directory in file paths
file_svg_list = cytometer.data.change_home_directory(list(file_svg_list), '/users/rittscher/rcasero', home,
                                                     check_isfile=True)

########################################################################################################################
## Colourmap for AIDA
########################################################################################################################

# loop files with hand traced contours
manual_areas_all = []
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

    # compute cell area
    manual_areas_all.append([Polygon(c).area * xres * yres for c in contours])  # (um^2)

manual_areas_all = list(itertools.chain.from_iterable(manual_areas_all))

# compute function to map between cell areas and [0.0, 1.0], that we can use to sample the colourmap uniformly according
# to area quantiles
quantiles = np.linspace(0.0, 1.0, 11)
areas_by_quantiles = stats.mstats.hdquantiles(manual_areas_all, prob=quantiles)
f_area2colour = scipy.interpolate.interp1d(areas_by_quantiles.data, quantiles, bounds_error=False, fill_value=(0.0, 1.0))

if DEBUG:
    plt.clf()
    fig = plt.hist(manual_areas_all, bins=50, density=True, histtype='step')
    for x in areas_by_quantiles.data:
        plt.plot([x, x], [0, fig[0].max()], 'k')

# create colourmap that mirrors AIDA's colourmap
import colorsys
cm = [colorsys.hls_to_rgb(h, l=0.69, s=0.44) + (1,) for h in np.linspace(np.sqrt(20e3 * 1e-12)/360, 315/360, 100)]
cm[0] = (1.0, 1.0, 1.0, 1.0)
# make 0 quantile white for the background of the image
cm = ListedColormap(cm)

########################################################################################################################
## Annotation file loop
########################################################################################################################


# loop annotations files
for i_file, json_file in enumerate(json_annotation_files):

    print('JSON annotations file: ' + os.path.basename(json_file))

    # name of corresponding .ndpi file
    ndpi_file = json_file.replace('_exp_0097.json', '.ndpi')
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

    # parse the json file
    with open(json_file) as f:
        json_data = json.load(f)

    # list of items (there's a contour in each item)
    items = json_data['layers'][0]['items']

    # init array for interpolated areas
    areas_grid = np.zeros(shape=lores_istissue0.shape, dtype=np.float32)

    # init array for mask where there are segmentations
    areas_mask = Image.new("1", lores_istissue0.shape[::-1], "black")
    draw = ImageDraw.Draw(areas_mask)

    # init lists for contour centroids and areas
    areas_all = []
    centroids_all = []

    # loop items (one contour per item)
    for it in items:

        # extract contour
        c = it['segments']

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

        # rasterise object described by contour
        draw.polygon(list(c.flatten()), outline="white", fill="white")

    # convert mask
    areas_mask = np.array(areas_mask, dtype=np.bool)

    areas_all = np.array(areas_all) * 1e12

    # convert area values to quantiles
    q = f_area2colour(areas_all)

    # interpolate scattered data to regular grid
    idx = areas_mask
    xi = np.transpose(np.array(np.where(idx)))[:, [1, 0]]
    areas_grid[idx] = scipy.interpolate.griddata(centroids_all, q, xi, method='linear', fill_value=0)

    if DEBUG:
        plt.clf()
        plt.subplot(211)
        plt.imshow(im_downsampled)
        plt.axis('off')
        plt.subplot(212)
        # plt.imshow(areas_grid, vmin=0.0, vmax=1.0, cmap='gnuplot2')
        plt.imshow(areas_grid, vmin=0.0, vmax=1.0, cmap=cm)
        cbar = plt.colorbar(shrink=0.8)
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.set_ylabel('Cell area quantile\n(w.r.t. manual dataset distribution)', rotation=90, fontsize=14)
        plt.axis('off')
        plt.tight_layout()

    if DEBUG:
        plt.clf()
        plt.hist(areas_all, bins=50, density=True, histtype='step')

    # plot cell areas for paper
    plt.clf()
    plt.imshow(areas_grid, vmin=0.0, vmax=1.0, cmap=cm)
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
