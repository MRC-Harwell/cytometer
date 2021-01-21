"""
Analysis for automatically segmented cells.

Part of the annotations were computed in experiment 0003 and part in experiment 0004.
"""

# script name to identify this experiment
experiment_id = 'rreb1_tm1b_exp_0005_population_analysis_v8'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import sys
sys.path.extend([os.path.join(home, 'Software/cytometer')])

# post-processing parameters
min_area = 203 / 2  # (pix^2) smaller objects are rejected
# max_area = 44879 * 3  # (pix^2) larger objects are rejected

xres_ref = 0.4538234626730202
yres_ref = 0.4537822752643282
min_area_um2 = min_area * xres_ref * yres_ref
# max_area_um2 = max_area * xres_ref * yres_ref
max_area_um2 = 25e3

# list of NDPI files that were processed
ndpi_files_list = [
'RREB1-TM1B-B6N-IC-1.1a 1132-18 G1 - 2019-02-20 09.56.50.ndpi',
'RREB1-TM1B-B6N-IC-1.1a 1132-18 M1 - 2019-02-20 09.48.06.ndpi',
'RREB1-TM1B-B6N-IC-1.1a  1132-18 P1 - 2019-02-20 09.29.29.ndpi',
'RREB1-TM1B-B6N-IC-1.1a  1132-18 S1 - 2019-02-20 09.21.24.ndpi',
'RREB1-TM1B-B6N-IC-1.1b 1133-18 G1 - 2019-02-20 12.31.18.ndpi',
'RREB1-TM1B-B6N-IC-1.1b 1133-18 M1 - 2019-02-20 12.15.25.ndpi',
'RREB1-TM1B-B6N-IC-1.1b 1133-18 P3 - 2019-02-20 11.51.52.ndpi',
'RREB1-TM1B-B6N-IC-1.1b 1133-18 S1 - 2019-02-20 11.31.44.ndpi',
'RREB1-TM1B-B6N-IC-1.1c  1129-18 G1 - 2019-02-19 14.10.46.ndpi',
'RREB1-TM1B-B6N-IC-1.1c  1129-18 M2 - 2019-02-19 13.58.32.ndpi',
'RREB1-TM1B-B6N-IC-1.1c  1129-18 P1 - 2019-02-19 12.41.11.ndpi',
'RREB1-TM1B-B6N-IC-1.1c  1129-18 S1 - 2019-02-19 12.28.03.ndpi',
'RREB1-TM1B-B6N-IC-1.1e 1134-18 G2 - 2019-02-20 14.43.06.ndpi',
'RREB1-TM1B-B6N-IC-1.1e 1134-18 P1 - 2019-02-20 13.59.56.ndpi',
'RREB1-TM1B-B6N-IC-1.1f  1130-18 G1 - 2019-02-19 15.51.35.ndpi',
'RREB1-TM1B-B6N-IC-1.1f  1130-18 M2 - 2019-02-19 15.38.01.ndpi',
'RREB1-TM1B-B6N-IC-1.1f  1130-18 S1 - 2019-02-19 14.39.24.ndpi',
'RREB1-TM1B-B6N-IC-1.1g  1131-18 G1 - 2019-02-19 17.10.06.ndpi',
'RREB1-TM1B-B6N-IC-1.1g  1131-18 M1 - 2019-02-19 16.53.58.ndpi',
'RREB1-TM1B-B6N-IC-1.1g  1131-18 P1 - 2019-02-19 16.37.30.ndpi',
'RREB1-TM1B-B6N-IC-1.1g  1131-18 S1 - 2019-02-19 16.21.16.ndpi',
'RREB1-TM1B-B6N-IC-1.1h 1135-18 G3 - 2019-02-20 15.46.52.ndpi',
'RREB1-TM1B-B6N-IC-1.1h 1135-18 M1 - 2019-02-20 15.30.26.ndpi',
'RREB1-TM1B-B6N-IC-1.1h 1135-18 P1 - 2019-02-20 15.06.59.ndpi',
'RREB1-TM1B-B6N-IC-1.1h 1135-18 S1 - 2019-02-20 14.56.47.ndpi',
'RREB1-TM1B-B6N-IC-2.1a  1128-18 G1 - 2019-02-19 12.04.29.ndpi',
'RREB1-TM1B-B6N-IC-2.1a  1128-18 M2 - 2019-02-19 11.26.46.ndpi',
'RREB1-TM1B-B6N-IC-2.1a  1128-18 P1 - 2019-02-19 11.01.39.ndpi',
'RREB1-TM1B-B6N-IC-2.1a  1128-18 S1 - 2019-02-19 11.59.16.ndpi',
'RREB1-TM1B-B6N-IC-2.2a 1124-18 G1 - 2019-02-18 10.15.04.ndpi',
'RREB1-TM1B-B6N-IC-2.2a 1124-18 M3 - 2019-02-18 10.12.54.ndpi',
'RREB1-TM1B-B6N-IC-2.2a 1124-18 P2 - 2019-02-18 09.39.46.ndpi',
'RREB1-TM1B-B6N-IC-2.2a 1124-18 S1 - 2019-02-18 09.09.58.ndpi',
'RREB1-TM1B-B6N-IC-2.2b 1125-18 G1 - 2019-02-18 12.35.37.ndpi',
'RREB1-TM1B-B6N-IC-2.2b 1125-18 P1 - 2019-02-18 11.16.21.ndpi',
'RREB1-TM1B-B6N-IC-2.2b 1125-18 S1 - 2019-02-18 11.06.53.ndpi',
'RREB1-TM1B-B6N-IC-2.2d 1137-18 S1 - 2019-02-21 10.59.23.ndpi',
'RREB1-TM1B-B6N-IC-2.2e 1126-18 G1 - 2019-02-18 14.58.55.ndpi',
'RREB1-TM1B-B6N-IC-2.2e 1126-18 M1- 2019-02-18 14.50.13.ndpi',
'RREB1-TM1B-B6N-IC-2.2e 1126-18 P1 - 2019-02-18 14.13.24.ndpi',
'RREB1-TM1B-B6N-IC-2.2e 1126-18 S1 - 2019-02-18 14.05.58.ndpi',
'RREB1-TM1B-B6N-IC-5.1a 0066-19 G1 - 2019-02-21 15.26.24.ndpi',
'RREB1-TM1B-B6N-IC-5.1a 0066-19 M1 - 2019-02-21 15.04.14.ndpi',
'RREB1-TM1B-B6N-IC-5.1a 0066-19 P1 - 2019-02-21 14.39.43.ndpi',
'RREB1-TM1B-B6N-IC-5.1a 0066-19 S1 - 2019-02-21 14.04.12.ndpi',
'RREB1-TM1B-B6N-IC-5.1b 0067-19 P1 - 2019-02-21 16.32.24.ndpi',
'RREB1-TM1B-B6N-IC-5.1b 0067-19 S1 - 2019-02-21 16.00.37.ndpi',
'RREB1-TM1B-B6N-IC-5.1b 67-19 G1 - 2019-02-21 17.29.31.ndpi',
'RREB1-TM1B-B6N-IC-5.1b 67-19 M1 - 2019-02-21 17.04.37.ndpi',
'RREB1-TM1B-B6N-IC-5.1c  68-19 G2 - 2019-02-22 09.43.59.ndpi',
'RREB1-TM1B-B6N-IC- 5.1c 68 -19 M2 - 2019-02-22 09.27.30.ndpi',
'RREB1-TM1B-B6N-IC -5.1c 68 -19 peri3 - 2019-02-22 09.08.26.ndpi',
'RREB1-TM1B-B6N-IC- 5.1c 68 -19 sub2 - 2019-02-22 08.39.12.ndpi',
'RREB1-TM1B-B6N-IC-5.1d  69-19 G2 - 2019-02-22 15.13.08.ndpi',
'RREB1-TM1B-B6N-IC-5.1d  69-19 M1 - 2019-02-22 14.39.12.ndpi',
'RREB1-TM1B-B6N-IC-5.1d  69-19 Peri1 - 2019-02-22 12.00.19.ndpi',
'RREB1-TM1B-B6N-IC-5.1d  69-19 sub1 - 2019-02-22 11.44.13.ndpi',
'RREB1-TM1B-B6N-IC-5.1e  70-19 G3 - 2019-02-25 10.34.30.ndpi',
'RREB1-TM1B-B6N-IC-5.1e  70-19 M1 - 2019-02-25 09.53.00.ndpi',
'RREB1-TM1B-B6N-IC-5.1e  70-19 P2 - 2019-02-25 09.27.06.ndpi',
'RREB1-TM1B-B6N-IC-5.1e  70-19 S1 - 2019-02-25 08.51.26.ndpi',
'RREB1-TM1B-B6N-IC-7.1a  71-19 G1 - 2019-02-25 12.27.06.ndpi',
'RREB1-TM1B-B6N-IC-7.1a  71-19 P1 - 2019-02-25 11.31.30.ndpi',
'RREB1-TM1B-B6N-IC-7.1a  71-19 S1 - 2019-02-25 11.03.59.ndpi'
]

########################################################################################################################
## Compute cell populations from automatically segmented images in two depots: SQWAT and GWAT:
##   Cell area histograms
##   HD quantiles of cell areas
## The results are saved, so in later sections, it's possible to just read them for further analysis.
## GENERATES DATA USED IN FOLLOWING SECTIONS
########################################################################################################################

import matplotlib.pyplot as plt
import cytometer.data
import shapely
import scipy.stats as stats
import openslide
import numpy as np
import scipy.stats
import sklearn.neighbors, sklearn.model_selection
import pandas as pd
# from mlxtend.evaluate import permutation_test
# from statsmodels.stats.multitest import multipletests
# import math
import PIL
import warnings

# directories
annotations_dir = os.path.join(home, 'Data/cytometer_data/aida_data_Rreb1_tm1b/annotations')
histology_dir = os.path.join(home, 'scan_srv2_cox/Liz Bentley/Grace/RREB1 Feb19')
dataframe_dir = os.path.join(home, 'GoogleDrive/Research/20200826_Rreb1_Grace')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20200826_Rreb1_Grace/figures')
metainfo_dir = os.path.join(home, 'Data/cytometer_data/rreb1')

# we are not using cell overlap in this study, thus, we use the 'auto' method (watershed)
method = 'auto'

DEBUG = False
SAVEFIG = False

# 0.05, 0.1 , 0.15, 0.2, ..., 0.9 , 0.95
quantiles = np.linspace(0, 1, 21)

# bins for the cell population histograms
area_bin_edges = np.linspace(min_area_um2, max_area_um2, 201)
area_bin_centers = (area_bin_edges[0:-1] + area_bin_edges[1:]) / 2.0

# data file with extra info for the dataframe (quantiles and histograms bins)
dataframe_areas_extra_filename = os.path.join(dataframe_dir, 'rreb1_tm1b_exp_0005_dataframe_areas_extra.npz')

# if the file doesn't exist, save it
if not os.path.isfile(dataframe_areas_extra_filename):
    np.savez(dataframe_areas_extra_filename, quantiles=quantiles, area_bin_edges=area_bin_edges,
             area_bin_centers=area_bin_centers)

# dataframe with histograms and smoothed histograms of cell populations in each slide
dataframe_areas_filename = os.path.join(dataframe_dir, 'rreb1_tm1b_exp_0005_dataframe_areas_' + method + '.pkl')

if os.path.isfile(dataframe_areas_filename):

    # load dataframe with cell population quantiles and histograms
    df_all = pd.read_pickle(dataframe_areas_filename)

else:

    # CSV file with metainformation of all mice
    metainfo_csv_file = os.path.join(metainfo_dir, 'rreb1_tm1b_meta_info.csv')
    metainfo = pd.read_csv(metainfo_csv_file)

    # keep only the last part of the ID in 'Animal Identifier', so that it can be found in the filename
    # RREB1-TM1B-B6N-IC/5.1c -> 5.1c
    metainfo['Animal Identifier'] = [x.replace('RREB1-TM1B-B6N-IC/', '') for x in metainfo['Animal Identifier']]

    # rename columns to make them easier to use in statsmodels
    metainfo = metainfo.rename(
        columns={'Weight (g)': 'Weight', 'Gonadal_AT (g)': 'Gonadal', 'Mesenteric_AT (g)': 'Mesenteric',
                 'PAT+RPAT (g)': 'PAT', 'Brown_AT (g)': 'Brown', 'SAT (g)': 'SAT'})

    # make sure that in the boxplots Rreb1-tm1b:WT comes before Rreb1-tm1b:Het, and female before male
    metainfo['Sex'] = metainfo['Sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
    metainfo['Genotype'] = metainfo['Genotype'].astype(
        pd.api.types.CategoricalDtype(categories=['Rreb1-tm1b:WT', 'Rreb1-tm1b:Het'], ordered=True))

    # mean mouse body weight (female and male)
    mean_bw_f = metainfo[metainfo['Sex'] == 'f']['Weight'].mean()
    mean_bw_m = metainfo[metainfo['Sex'] == 'm']['Weight'].mean()

    # dataframe to keep all results, one row per annotations file
    df_all = pd.DataFrame()

    # get filenames of the annotations. Note that they come from two experiments, 0003 or 0004, so we have to check from
    # which one
    json_annotation_files = []
    for i_file, file in enumerate(ndpi_files_list):

        print('File ' + str(i_file) + '/' + str(len(json_annotation_files)-1) + ': '
              + os.path.basename(file))

        json_file = os.path.join(annotations_dir, file.replace('.ndpi', '_exp_0003_auto_aggregated.json'))
        if os.path.isfile(json_file):
            print('\tExperiment 0003')
            json_annotation_files.append(json_file)
        else:
            json_file = os.path.join(annotations_dir, file.replace('.ndpi', '_exp_0004_auto_aggregated.json'))
            if os.path.isfile(json_file):
                print('\tExperiment 0004')
                json_annotation_files.append(json_file)
            else:
                warnings.warn('Annotations file not found')

    # process annotations files
    for i_file, (ndpi_file, json_file) in enumerate(zip(ndpi_files_list, json_annotation_files)):

        print('File ' + str(i_file) + '/' + str(len(json_annotation_files)-1) + ': '
              + os.path.basename(json_file))

        if not os.path.isfile(json_file):
            warnings.warn('Missing annotations file')
            continue

        # open full resolution histology slide
        im = openslide.OpenSlide(os.path.join(histology_dir, os.path.basename(ndpi_file)))

        # pixel size
        assert (im.properties['tiff.ResolutionUnit'] == 'centimeter')
        xres = float(im.properties['openslide.mpp-x'])  # um/pixel
        yres = float(im.properties['openslide.mpp-y'])  # um/pixel

        # figure out what depot these cells belong to
        aux = ndpi_file.replace('RREB1-TM1B-B6N-IC', '')
        is_gonadal = any([s in aux for s in ['G1', 'G2', 'G3']])
        is_perineal = any([s in aux for s in ['P1', 'P2', 'P3', 'peri', 'Peri']])
        is_subcutaneous = any([s in aux for s in ['S1', 'S2', 'S3', 'sub', 'Sub']])
        is_mesenteric = any([s in aux for s in ['M1', 'M2', 'M3']])
        n_matches = np.count_nonzero([is_gonadal, is_perineal, is_subcutaneous, is_mesenteric])
        if n_matches != 1:
            raise ValueError(['Filename matches in metainfo table: ' + str(n_matches)])
        if is_gonadal:
             depot_label = 'Gonadal'
        elif is_perineal:
            depot_label = 'PAT'  # perineal + retroperineal
        elif is_subcutaneous:
            depot_label = 'SAT'
        elif is_mesenteric:
            depot_label = 'Mesenteric'

        aux = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(json_file),
                                                        tags_to_keep=['Animal Identifier', 'Sex', 'Genotype',
                                                                      'Weight', 'Gonadal', 'PAT',
                                                                      'SAT', 'Mesenteric'],
                                                        id_tag='Animal Identifier')
        df = aux.drop(labels=['Gonadal', 'PAT', 'SAT', 'Mesenteric'], axis='columns')
        df['depot'] = depot_label
        df['depot_weight'] = aux[depot_label]

        # load contours and their confidence measure from annotation file
        cells, props = cytometer.data.aida_get_contours(json_file, layer_name='White adipocyte.*', return_props=True)

        # compute areas of the cells (um^2)
        areas = np.array([shapely.geometry.Polygon(cell).area for cell in cells]) * xres * yres  # um^2

        # smooth out histogram
        kde = sklearn.neighbors.KernelDensity(bandwidth=100, kernel='gaussian').fit(areas.reshape(-1, 1))
        log_dens = kde.score_samples(area_bin_centers.reshape(-1, 1))
        pdf = np.exp(log_dens)

        # compute mode
        df['area_smoothed_mode'] = area_bin_centers[np.argmax(pdf)]

        # compute areas at population quantiles
        areas_at_quantiles = stats.mstats.hdquantiles(areas, prob=quantiles, axis=0)
        df['area_at_quantiles'] = [areas_at_quantiles]

        # compute histograms with area binning
        histo, _ = np.histogram(areas, bins=area_bin_edges, density=True)
        df['histo'] = [list(histo)]

        # smoothed histogram
        df['smoothed_histo'] = [list(pdf)]

        if DEBUG:
            plt.clf()
            plt.plot(1e-3 * area_bin_centers, df['histo'].to_numpy()[0], label='Areas')
            plt.plot(1e-3 * area_bin_centers, df['smoothed_histo'].to_numpy()[0], label='Kernel')
            plt.plot([df['area_smoothed_mode'] * 1e-3, df['area_smoothed_mode'] * 1e-3],
                     [0, np.array(df['smoothed_histo'].to_numpy()[0]).max()], 'k', label='Mode')
            plt.legend()
            plt.xlabel('Area ($10^3 \cdot \mu m^2$)', fontsize=14)

        # add results to total dataframe
        df_all = pd.concat([df_all, df], ignore_index=True)

    # save dataframe with data from both depots for current method (auto or corrected)
    df_all.to_pickle(dataframe_areas_filename)

########################################################################################################################
## Import packages and auxiliary functions common to all analysis sections
## Load metainfo and cell population data
## USED IN PAPER
########################################################################################################################

import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
# import ast
import scipy.stats as stats
import statsmodels.api as sm
# import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import seaborn as sns
# import openslide
# import cytometer.data
import cytometer.stats
# import shapely

# directories
annotations_dir = os.path.join(home, 'Data/cytometer_data/aida_data_Rreb1_tm1b/annotations')
histology_dir = os.path.join(home, 'scan_srv2_cox/Liz Bentley/Grace/RREB1 Feb19')
dataframe_dir = os.path.join(home, 'GoogleDrive/Research/20200826_Rreb1_Grace')
figures_dir = os.path.join(home, 'GoogleDrive/Research/20200826_Rreb1_Grace/figures')
metainfo_dir = os.path.join(home, 'Data/cytometer_data/rreb1')

DEBUG = False

method = 'auto'

# CSV file with metainformation of all mice
metainfo_csv_file = os.path.join(metainfo_dir, 'rreb1_tm1b_meta_info.csv')
metainfo = pd.read_csv(metainfo_csv_file)

# keep only the last part of the ID in 'Animal Identifier', so that it can be found in the filename
# RREB1-TM1B-B6N-IC/5.1c -> 5.1c
metainfo['Animal Identifier'] = [x.replace('RREB1-TM1B-B6N-IC/', '') for x in metainfo['Animal Identifier']]

# rename columns to make them easier to use in statsmodels
metainfo = metainfo.rename(columns={'Weight (g)':'Weight', 'Gonadal_AT (g)':'Gonadal', 'Mesenteric_AT (g)':'Mesenteric',
                                    'PAT+RPAT (g)':'PAT', 'Brown_AT (g)':'Brown', 'SAT (g)':'SAT'})

# make sure that in the boxplots Rreb1-tm1b:WT comes before Rreb1-tm1b:Het, and female before male
metainfo['Sex'] = metainfo['Sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
metainfo['Genotype'] = metainfo['Genotype'].astype(
    pd.api.types.CategoricalDtype(categories=['Rreb1-tm1b:WT', 'Rreb1-tm1b:Het'], ordered=True))

# load dataframe with cell population quantiles and histograms
dataframe_areas_filename = os.path.join(dataframe_dir, 'rreb1_tm1b_exp_0005_dataframe_areas_' + method + '.pkl')
df_all = pd.read_pickle(dataframe_areas_filename)
df_all = df_all.reset_index()

df_all['Sex'] = df_all['Sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
df_all['Genotype'] = df_all['Genotype'].astype(
    pd.api.types.CategoricalDtype(categories=['Rreb1-tm1b:WT', 'Rreb1-tm1b:Het'], ordered=True))

# rename columns to make them easier to use in statsmodels
df_all = df_all.rename(columns={'Weight (g)':'Weight', 'Gonadal_AT (g)':'Gonadal', 'Mesenteric_AT (g)':'Mesenteric',
                                'PAT+RPAT (g)': 'PAT', 'Brown_AT (g)': 'Brown', 'SAT (g)': 'SAT'})

# WARNING! Remove SAT population for female WT 2.2a, because the histology contains very few white adipocytes
idx = (df_all['Animal Identifier'] == '2.2a') & (df_all['depot'] == 'SAT')
df_all.drop(df_all.index[idx], axis='rows', inplace=True)

# load extra info needed for the histograms
dataframe_areas_extra_filename = os.path.join(dataframe_dir, 'rreb1_tm1b_exp_0005_dataframe_areas_extra.npz')
with np.load(dataframe_areas_extra_filename) as aux:
    quantiles = aux['quantiles']
    area_bin_edges = aux['area_bin_edges']
    area_bin_centers = aux['area_bin_centers']

## auxiliary functions

# def read_contours_compute_areas(metainfo, json_annotation_files_dict, depot, method='corrected'):
# 
#     # dataframe to keep all results, one row per cell
#     df_all = pd.DataFrame()
# 
#     # list of annotation files for this depot
#     json_annotation_files = json_annotation_files_dict[depot]
# 
#     # modify filenames to select the particular segmentation we want (e.g. the automatic ones, or the manually refined ones)
#     json_annotation_files = [x.replace('.json', '_exp_0106_' + method + '_aggregated.json') for x in
#                              json_annotation_files]
#     json_annotation_files = [os.path.join(annotations_dir, x) for x in json_annotation_files]
# 
#     for i_file, json_file in enumerate(json_annotation_files):
# 
#         print('File ' + str(i_file) + '/' + str(len(json_annotation_files) - 1) + ': '
#               + os.path.basename(json_file))
# 
#         if not os.path.isfile(json_file):
#             print('Missing annotations file')
#             continue
# 
#         # open full resolution histology slide
#         ndpi_file = json_file.replace('_exp_0106_' + method + '_aggregated.json', '.ndpi')
#         ndpi_file = os.path.join(ndpi_dir, os.path.basename(ndpi_file))
#         im = openslide.OpenSlide(ndpi_file)
# 
#         # pixel size
#         assert (im.properties['tiff.ResolutionUnit'] == 'centimeter')
#         xres = float(im.properties['openslide.mpp-x'])  # um/pixel
#         yres = float(im.properties['openslide.mpp-y'])  # um/pixel
# 
#         # create dataframe for this image
#         if depot == 'sqwat':
#             df = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(json_file),
#                                                            values=[depot, ], values_tag='depot',
#                                                            tags_to_keep=['id', 'ko_parent', 'sex', 'genotype',
#                                                                          'BW', 'SC'])
#             df.rename(columns={'SC': 'DW'}, inplace=True)
#         elif depot == 'gwat':
#             df = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(json_file),
#                                                            values=[depot, ], values_tag='depot',
#                                                            tags_to_keep=['id', 'ko_parent', 'sex', 'genotype',
#                                                                          'BW', 'gWAT'])
#             df.rename(columns={'gWAT': 'DW'}, inplace=True)
#         else:
#             raise RuntimeError('Unknown depot type')
# 
#         # load contours and their confidence measure from annotation file
#         cells, props = cytometer.data.aida_get_contours(json_file, layer_name='White adipocyte.*',
#                                                         return_props=True)
# 
#         # compute areas of the cells (um^2)
#         areas = np.array([shapely.geometry.Polygon(cell).area for cell in cells]) * xres * yres  # (um^2)
#         df = df.reindex(df.index.repeat(len(areas)))
#         df.loc[:, 'area'] = areas
# 
#         # add results to total dataframe
#         df_all = pd.concat([df_all, df], ignore_index=True)
# 
#     return df_all


## Analyse cell populations from automatically segmented images in two depots: SQWAT and GWAT:
## smoothed histograms
## USED IN THE PAPER
########################################################################################################################

depot = 'Gonadal'
# depot = 'PAT'  # perineal + retroperineal
# depot = 'SAT'
# depot = 'Mesenteric'

if SAVEFIG:
    plt.clf()

    # f PAT
    df = df_all[(df_all['depot'] == depot) & (df_all['Sex'] == 'f') & (df_all['Genotype'] == 'Rreb1-tm1b:WT')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())

    plt.subplot(221)
    lineObjects = plt.plot(area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.legend(lineObjects, df['Animal Identifier'])
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.text(0.65, 0.9, 'female WT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xticks([0, 10, 20])
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # f MAT
    df = df_all[(df_all['depot'] == depot) & (df_all['Sex'] == 'f') & (df_all['Genotype'] == 'Rreb1-tm1b:Het')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())

    plt.subplot(222)
    lineObjects = plt.plot(area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.legend(lineObjects, df['Animal Identifier'])
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.text(0.65, 0.9, 'female Het', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xticks([0, 10, 20])
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # m PAT
    df = df_all[(df_all['depot'] == depot) & (df_all['Sex'] == 'm') & (df_all['Genotype'] == 'Rreb1-tm1b:WT')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())

    plt.subplot(223)
    lineObjects = plt.plot(area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.legend(lineObjects, df['Animal Identifier'])
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.text(0.69, 0.9, 'male WT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xticks([0, 10, 20])
    plt.xlim(-1.2, max_area_um2 * 1e-3)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)

    # m MAT
    df = df_all[(df_all['depot'] == depot) & (df_all['Sex'] == 'm') & (df_all['Genotype'] == 'Rreb1-tm1b:Het')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())

    plt.subplot(224)
    lineObjects = plt.plot(area_bin_centers * 1e-3, np.transpose(histo) / histo.max().max())
    plt.legend(lineObjects, df['Animal Identifier'])
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.xticks([0, 10, 20])
    plt.text(0.65, 0.9, 'male Het', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)

    plt.suptitle(depot, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(figures_dir, 'rreb1_tm1b_exp_0005_paper_figures_smoothed_histo_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'rreb1_tm1b_exp_0005_paper_figures_smoothed_histo_' + depot + '.svg'))

if SAVEFIG:
    plt.clf()

    # f PAT
    df = df_all[(df_all['depot'] == depot) & (df_all['Sex'] == 'f') & (df_all['Genotype'] == 'Rreb1-tm1b:WT')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)

    plt.subplot(221)
    plt.fill_between(area_bin_centers * 1e-3, histo_q1[0,] / histo_q3.max(), histo_q3[0,] / histo_q3.max(),
                     alpha=0.5, color='C0')
    plt.plot(area_bin_centers * 1e-3, histo_q2[0,] / histo_q3.max(), 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, right=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.text(0.9, 0.9, 'female PAT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # f MAT
    df = df_all[(df_all['depot'] == depot) & (df_all['Sex'] == 'f') & (df_all['Genotype'] == 'Rreb1-tm1b:Het')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)

    plt.subplot(222)
    plt.fill_between(area_bin_centers * 1e-3, histo_q1[0,] / histo_q3.max(), histo_q3[0,] / histo_q3.max(),
                     alpha=0.5, color='C0')
    plt.plot(area_bin_centers * 1e-3, histo_q2[0,] / histo_q3.max(), 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.text(0.9, 0.9, 'female MAT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # m PAT
    df = df_all[(df_all['depot'] == depot) & (df_all['Sex'] == 'm') & (df_all['Genotype'] == 'Rreb1-tm1b:WT')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)

    plt.subplot(223)
    plt.fill_between(area_bin_centers * 1e-3, histo_q1[0,] / histo_q3.max(), histo_q3[0,] / histo_q3.max(),
                     alpha=0.5, color='C0')
    plt.plot(area_bin_centers * 1e-3, histo_q2[0,] / histo_q3.max(), 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)
    plt.text(0.9, 0.9, 'male PAT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    # m MAT
    df = df_all[(df_all['depot'] == depot) & (df_all['Sex'] == 'm') & (df_all['Genotype'] == 'Rreb1-tm1b:Het')]
    df = df.reset_index()
    histo = np.array(df['smoothed_histo'].tolist())
    histo_q1 = stats.mstats.hdquantiles(histo, prob=0.25, axis=0)
    histo_q2 = stats.mstats.hdquantiles(histo, prob=0.50, axis=0)
    histo_q3 = stats.mstats.hdquantiles(histo, prob=0.75, axis=0)

    plt.subplot(224)
    plt.fill_between(area_bin_centers * 1e-3, histo_q1[0,] / histo_q3.max(), histo_q3[0,] / histo_q3.max(),
                     alpha=0.5, color='C0')
    plt.plot(area_bin_centers * 1e-3, histo_q2[0,] / histo_q3.max(), 'C0', linewidth=2)
    plt.tick_params(axis='y', left=False, labelleft=False, reset=True)
    plt.tick_params(labelsize=14)
    plt.xlabel('Area ($\cdot 10^3\ \mu m^2$)', fontsize=14)
    plt.text(0.9, 0.9, 'male MAT', fontsize=14, transform=plt.gca().transAxes, va='top', ha='right')
    plt.xlim(-1.2, max_area_um2 * 1e-3)

    plt.suptitle(depot, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(figures_dir, 'rreb1_tm1b_exp_0005_paper_figures_smoothed_histo_quartiles_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'rreb1_tm1b_exp_0005_paper_figures_smoothed_histo_quartiles_' + depot + '.svg'))

## numerical quartiles and CIs associated to the histograms

idx_q1 = np.where(quantiles == 0.25)[0][0]
idx_q2 = np.where(quantiles == 0.50)[0][0]
idx_q3 = np.where(quantiles == 0.75)[0][0]

print('Depot: ' + depot)

# f WT
df = df_all[(df_all['depot'] == depot) & (df_all['Sex'] == 'f') & (df_all['Genotype'] == 'Rreb1-tm1b:WT')]
df = df.reset_index()
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())

# compute mean value and CIs in 10^3 um^2 units
q1_mean, q2_mean, q3_mean = areas_at_quantiles.mean(axis=0)[[idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_lo, q2_ci_lo, q3_ci_lo = stats.mstats.hdquantiles(areas_at_quantiles, prob=0.025, axis=0).data[0, [idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_hi, q2_ci_hi, q3_ci_hi = stats.mstats.hdquantiles(areas_at_quantiles, prob=0.975, axis=0).data[0, [idx_q1, idx_q2, idx_q3]] * 1e-3

print('f WT')
print('\t' + '{0:.2f}'.format(q1_mean) + ' (' + '{0:.2f}'.format(q1_ci_lo) + ', ' + '{0:.2f}'.format(q1_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q2_mean) + ' (' + '{0:.2f}'.format(q2_ci_lo) + ', ' + '{0:.2f}'.format(q2_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q3_mean) + ' (' + '{0:.2f}'.format(q3_ci_lo) + ', ' + '{0:.2f}'.format(q3_ci_hi) + ')')

# f Het
df = df_all[(df_all['depot'] == depot) & (df_all['Sex'] == 'f') & (df_all['Genotype'] == 'Rreb1-tm1b:Het')]
df = df.reset_index()
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())

# compute mean value and CIs in 10^3 um^2 units
q1_mean, q2_mean, q3_mean = areas_at_quantiles.mean(axis=0)[[idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_lo, q2_ci_lo, q3_ci_lo = stats.mstats.hdquantiles(areas_at_quantiles, prob=0.025, axis=0).data[0, [idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_hi, q2_ci_hi, q3_ci_hi = stats.mstats.hdquantiles(areas_at_quantiles, prob=0.975, axis=0).data[0, [idx_q1, idx_q2, idx_q3]] * 1e-3

print('f Het')
print('\t' + '{0:.2f}'.format(q1_mean) + ' (' + '{0:.2f}'.format(q1_ci_lo) + ', ' + '{0:.2f}'.format(q1_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q2_mean) + ' (' + '{0:.2f}'.format(q2_ci_lo) + ', ' + '{0:.2f}'.format(q2_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q3_mean) + ' (' + '{0:.2f}'.format(q3_ci_lo) + ', ' + '{0:.2f}'.format(q3_ci_hi) + ')')

# m WT
df = df_all[(df_all['depot'] == depot) & (df_all['Sex'] == 'm') & (df_all['Genotype'] == 'Rreb1-tm1b:WT')]
df = df.reset_index()
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())

# compute mean value and CIs in 10^3 um^2 units
q1_mean, q2_mean, q3_mean = areas_at_quantiles.mean(axis=0)[[idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_lo, q2_ci_lo, q3_ci_lo = stats.mstats.hdquantiles(areas_at_quantiles, prob=0.025, axis=0).data[0, [idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_hi, q2_ci_hi, q3_ci_hi = stats.mstats.hdquantiles(areas_at_quantiles, prob=0.975, axis=0).data[0, [idx_q1, idx_q2, idx_q3]] * 1e-3

print('m WT')
print('\t' + '{0:.2f}'.format(q1_mean) + ' (' + '{0:.2f}'.format(q1_ci_lo) + ', ' + '{0:.2f}'.format(q1_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q2_mean) + ' (' + '{0:.2f}'.format(q2_ci_lo) + ', ' + '{0:.2f}'.format(q2_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q3_mean) + ' (' + '{0:.2f}'.format(q3_ci_lo) + ', ' + '{0:.2f}'.format(q3_ci_hi) + ')')

# m Het
df = df_all[(df_all['depot'] == depot) & (df_all['Sex'] == 'm') & (df_all['Genotype'] == 'Rreb1-tm1b:Het')]
df = df.reset_index()
areas_at_quantiles = np.array(df['area_at_quantiles'].to_list())

# compute mean value and CIs in 10^3 um^2 units
q1_mean, q2_mean, q3_mean = areas_at_quantiles.mean(axis=0)[[idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_lo, q2_ci_lo, q3_ci_lo = stats.mstats.hdquantiles(areas_at_quantiles, prob=0.025, axis=0).data[0, [idx_q1, idx_q2, idx_q3]] * 1e-3
q1_ci_hi, q2_ci_hi, q3_ci_hi = stats.mstats.hdquantiles(areas_at_quantiles, prob=0.975, axis=0).data[0, [idx_q1, idx_q2, idx_q3]] * 1e-3

print('m Het')
print('\t' + '{0:.2f}'.format(q1_mean) + ' (' + '{0:.2f}'.format(q1_ci_lo) + ', ' + '{0:.2f}'.format(q1_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q2_mean) + ' (' + '{0:.2f}'.format(q2_ci_lo) + ', ' + '{0:.2f}'.format(q2_ci_hi) + ')')
print('\t' + '{0:.2f}'.format(q3_mean) + ' (' + '{0:.2f}'.format(q3_ci_lo) + ', ' + '{0:.2f}'.format(q3_ci_hi) + ')')


## Weight ~ Sex

bw_model = sm.RLM.from_formula('Weight ~ C(Sex)', data=metainfo, M=sm.robust.norms.HuberT()).fit()
print(bw_model.summary())

pval_text = 'p=' + '{0:.3e}'.format(bw_model.pvalues['C(Sex)[T.m]']) + \
            ' ' + cytometer.stats.pval_to_asterisk(bw_model.pvalues['C(Sex)[T.m]'])
print(pval_text)

## Weight ~ Sex * Genotype

# bw_model = sm.RLM.from_formula('Weight ~ C(Sex) * C(Genotype)', data=metainfo, M=sm.robust.norms.HuberT()).fit()
bw_model = sm.OLS.from_formula('Weight ~ C(Sex) * C(Genotype)', data=metainfo).fit()
print(bw_model.summary())

df_coeff, df_ci_lo, df_ci_hi, df_pval = \
    cytometer.stats.models_coeff_ci_pval([bw_model],
                                         extra_hypotheses='C(Genotype)[T.Rreb1-tm1b:Het]+C(Sex)[T.m]:C(Genotype)[T.Rreb1-tm1b:Het]')

pval_text = 'β=' + '{0:.3e}'.format(bw_model.params['C(Sex)[T.m]']) + \
            ', p=' + '{0:.3e}'.format(bw_model.pvalues['C(Sex)[T.m]']) + \
            ' ' + cytometer.stats.pval_to_asterisk(bw_model.pvalues['C(Sex)[T.m]'])
print('C(Sex)[T.m]: ' + pval_text)
pval_text = 'β=' + '{0:.3e}'.format(bw_model.params['C(Genotype)[T.Rreb1-tm1b:Het]']) + \
            ', p=' + '{0:.3e}'.format(bw_model.pvalues['C(Genotype)[T.Rreb1-tm1b:Het]']) + \
            ' ' + cytometer.stats.pval_to_asterisk(bw_model.pvalues['C(Genotype)[T.Rreb1-tm1b:Het]'])
print('C(Genotype)[T.Rreb1-tm1b:Het]: ' + pval_text)
pval_text = 'β=' + '{0:.3e}'.format(bw_model.params['C(Sex)[T.m]:C(Genotype)[T.Rreb1-tm1b:Het]']) + \
            ', p=' + '{0:.3e}'.format(bw_model.pvalues['C(Sex)[T.m]:C(Genotype)[T.Rreb1-tm1b:Het]']) + \
            ' ' + cytometer.stats.pval_to_asterisk(bw_model.pvalues['C(Sex)[T.m]:C(Genotype)[T.Rreb1-tm1b:Het]'])
print('C(Sex)[T.m]:C(Genotype)[T.Rreb1-tm1b:Het]: ' + pval_text)
pval_text = 'β=' + '{0:.3e}'.format(df_coeff['C(Genotype)[T.Rreb1-tm1b:Het]+C(Sex)[T.m]:C(Genotype)[T.Rreb1-tm1b:Het]'][0]) + \
            ', p=' + '{0:.3e}'.format(df_pval['C(Genotype)[T.Rreb1-tm1b:Het]+C(Sex)[T.m]:C(Genotype)[T.Rreb1-tm1b:Het]'][0]) + \
            ' ' + cytometer.stats.pval_to_asterisk(df_pval['C(Genotype)[T.Rreb1-tm1b:Het]+C(Sex)[T.m]:C(Genotype)[T.Rreb1-tm1b:Het]'][0])
print('C(Genotype)[T.Rreb1-tm1b:Het]+C(Sex)[T.m]:C(Genotype)[T.Rreb1-tm1b:Het]: ' + pval_text)

if SAVEFIG:
    plt.clf()
    plt.gcf().set_size_inches([5.48, 4.8 ])

    ax = sns.swarmplot(x='Sex', y='Weight', hue='Genotype', data=metainfo, dodge=True)
    plt.xlabel('')
    plt.ylabel('Body weight (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.xticks([0, 1], labels=['Female', 'Male'])
    ax.get_legend().set_title('')
    ax.legend(['WT', 'Het'], loc='upper right', fontsize=12)

    plt.plot([-0.2, -0.2, 0.2, 0.2], [57, 59, 59, 57], 'k', lw=1.5)
    pval_text = '$p$=' + '{0:.2g}'.format(df_pval['C(Genotype)[T.Rreb1-tm1b:Het]'][0]) + \
                ' ' + cytometer.stats.pval_to_asterisk(df_pval['C(Genotype)[T.Rreb1-tm1b:Het]'][0])
    plt.text(0, 59.5, pval_text, ha='center', va='bottom', fontsize=14)
    plt.plot([0.8, 0.8, 1.2, 1.2], [46, 48, 48, 46], 'k', lw=1.5)
    pval_text = '$p$=' + '{0:.2g}'.format(df_pval['C(Genotype)[T.Rreb1-tm1b:Het]+C(Sex)[T.m]:C(Genotype)[T.Rreb1-tm1b:Het]'][0]) + \
                ' ' + cytometer.stats.pval_to_asterisk(df_pval['C(Genotype)[T.Rreb1-tm1b:Het]+C(Sex)[T.m]:C(Genotype)[T.Rreb1-tm1b:Het]'][0])
    plt.text(1, 48.5, pval_text, ha='center', va='bottom', fontsize=14)
    plt.ylim(18, 63)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'rreb1_tm1b_exp_0005_paper_figures_swarm_bw.png'))
    plt.savefig(os.path.join(figures_dir, 'rreb1_tm1b_exp_0005_paper_figures_swarm_bw.svg'))

## depot_weight ~ BW * Genotype

# for convenience
metainfo_f = metainfo[metainfo['Sex'] == 'f']
metainfo_m = metainfo[metainfo['Sex'] == 'm']

# models of depot weight ~ BW * Genotype

# null models (without the Genotype variable)
gonadal_null_model_f = sm.OLS.from_formula('Gonadal ~ Weight', data=metainfo_f).fit()
gonadal_null_model_m = sm.OLS.from_formula('Gonadal ~ Weight', data=metainfo_m).fit()
perineal_null_model_f = sm.OLS.from_formula('PAT ~ Weight', data=metainfo_f).fit()
perineal_null_model_m = sm.OLS.from_formula('PAT ~ Weight', data=metainfo_m).fit()
subcutaneous_null_model_f = sm.OLS.from_formula('SAT ~ Weight', data=metainfo_f).fit()
subcutaneous_null_model_m = sm.OLS.from_formula('SAT ~ Weight', data=metainfo_m).fit()
mesenteric_null_model_f = sm.OLS.from_formula('Mesenteric ~ Weight', data=metainfo_f).fit()
mesenteric_null_model_m = sm.OLS.from_formula('Mesenteric ~ Weight', data=metainfo_m).fit()

gonadal_model_f = sm.OLS.from_formula('Gonadal ~ Weight * C(Genotype)', data=metainfo_f).fit()
gonadal_model_m = sm.OLS.from_formula('Gonadal ~ Weight * C(Genotype)', data=metainfo_m).fit()
perineal_model_f = sm.OLS.from_formula('PAT ~ Weight * C(Genotype)', data=metainfo_f).fit()
perineal_model_m = sm.OLS.from_formula('PAT ~ Weight * C(Genotype)', data=metainfo_m).fit()
subcutaneous_model_f = sm.OLS.from_formula('SAT ~ Weight * C(Genotype)', data=metainfo_f).fit()
subcutaneous_model_m = sm.OLS.from_formula('SAT ~ Weight * C(Genotype)', data=metainfo_m).fit()
mesenteric_model_f = sm.OLS.from_formula('Mesenteric ~ Weight * C(Genotype)', data=metainfo_f).fit()
mesenteric_model_m = sm.OLS.from_formula('Mesenteric ~ Weight * C(Genotype)', data=metainfo_m).fit()

# Likelihood ratio tests of the Genotype variable
print('Likelihood Ratio Test')

print('Female')
lr, pval = cytometer.stats.lrtest(gonadal_null_model_f.llf, gonadal_model_f.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
print('Gonadal: ' + pval_text)

lr, pval = cytometer.stats.lrtest(perineal_null_model_f.llf, perineal_model_f.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
print('Perineal: ' + pval_text)

lr, pval = cytometer.stats.lrtest(subcutaneous_null_model_f.llf, subcutaneous_model_f.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
print('Subcutaneous: ' + pval_text)

lr, pval = cytometer.stats.lrtest(mesenteric_null_model_f.llf, mesenteric_model_f.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
print('Mesenteric: ' + pval_text)

print('Male')
lr, pval = cytometer.stats.lrtest(gonadal_null_model_m.llf, gonadal_model_m.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
print('Gonadal: ' + pval_text)

lr, pval = cytometer.stats.lrtest(perineal_null_model_m.llf, perineal_model_m.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
print('Perineal: ' + pval_text)

lr, pval = cytometer.stats.lrtest(subcutaneous_null_model_m.llf, subcutaneous_model_m.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
print('Subcutaneous: ' + pval_text)

lr, pval = cytometer.stats.lrtest(mesenteric_null_model_m.llf, mesenteric_model_m.llf)
pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
print('Mesenteric: ' + pval_text)

## null models (WT and Het combined)

## linear regressions for WT and Het separately
gonadal_model_f_null = sm.OLS.from_formula('Gonadal ~ Weight', data=metainfo_f).fit()
perineal_model_f_null = sm.OLS.from_formula('PAT ~ Weight', data=metainfo_f).fit()
subcutaneous_model_f_null = sm.OLS.from_formula('SAT ~ Weight', data=metainfo_f).fit()
mesenteric_model_f_null = sm.OLS.from_formula('Mesenteric ~ Weight', data=metainfo_f).fit()

gonadal_model_m_null = sm.OLS.from_formula('Gonadal ~ Weight', data=metainfo_m).fit()
perineal_model_m_null = sm.OLS.from_formula('PAT ~ Weight', data=metainfo_m).fit()
subcutaneous_model_m_null = sm.OLS.from_formula('SAT ~ Weight', data=metainfo_m).fit()
mesenteric_model_m_null = sm.OLS.from_formula('Mesenteric ~ Weight', data=metainfo_m).fit()

# Female WT
idx = (metainfo['Sex'] == 'f') & (metainfo['Genotype'] == 'Rreb1-tm1b:WT')
gonadal_model_f_wt = sm.OLS.from_formula('Gonadal ~ Weight', data=metainfo, subset=idx).fit()
perineal_model_f_wt = sm.OLS.from_formula('PAT ~ Weight', data=metainfo, subset=idx).fit()
subcutaneous_model_f_wt = sm.OLS.from_formula('SAT ~ Weight', data=metainfo, subset=idx).fit()
mesenteric_model_f_wt = sm.OLS.from_formula('Mesenteric ~ Weight', data=metainfo, subset=idx).fit()

# Female Het
idx = (metainfo['Sex'] == 'f') & (metainfo['Genotype'] == 'Rreb1-tm1b:Het')
gonadal_model_f_het = sm.OLS.from_formula('Gonadal ~ Weight', data=metainfo, subset=idx).fit()
perineal_model_f_het = sm.OLS.from_formula('PAT ~ Weight', data=metainfo, subset=idx).fit()
subcutaneous_model_f_het = sm.OLS.from_formula('SAT ~ Weight', data=metainfo, subset=idx).fit()
mesenteric_model_f_het = sm.OLS.from_formula('Mesenteric ~ Weight', data=metainfo, subset=idx).fit()

# Male WT
idx = (metainfo['Sex'] == 'm') & (metainfo['Genotype'] == 'Rreb1-tm1b:WT')
gonadal_model_m_wt = sm.OLS.from_formula('Gonadal ~ Weight', data=metainfo, subset=idx).fit()
perineal_model_m_wt = sm.OLS.from_formula('PAT ~ Weight', data=metainfo, subset=idx).fit()
subcutaneous_model_m_wt = sm.OLS.from_formula('SAT ~ Weight', data=metainfo, subset=idx).fit()
mesenteric_model_m_wt = sm.OLS.from_formula('Mesenteric ~ Weight', data=metainfo, subset=idx).fit()

# Male Het
idx = (metainfo['Sex'] == 'm') & (metainfo['Genotype'] == 'Rreb1-tm1b:Het')
gonadal_model_m_het = sm.OLS.from_formula('Gonadal ~ Weight', data=metainfo, subset=idx).fit()
perineal_model_m_het = sm.OLS.from_formula('PAT ~ Weight', data=metainfo, subset=idx).fit()
subcutaneous_model_m_het = sm.OLS.from_formula('SAT ~ Weight', data=metainfo, subset=idx).fit()
mesenteric_model_m_het = sm.OLS.from_formula('Mesenteric ~ Weight', data=metainfo, subset=idx).fit()

if SAVEFIG:
    plt.clf()
    plt.gcf().set_size_inches([6.4, 9.4])

    plt.subplot(421)
    cytometer.stats.plot_linear_regression(gonadal_model_f_null, metainfo_f, ind_var='Weight',
                                           other_vars={'Sex':'f'}, c='k', line_label='Null')
    cytometer.stats.plot_linear_regression(gonadal_model_f_wt, metainfo_f, ind_var='Weight',
                                           other_vars={'Sex':'f', 'Genotype':'Rreb1-tm1b:WT'}, dep_var='Gonadal',
                                           c='C0', marker='x', line_label='WT')
    cytometer.stats.plot_linear_regression(gonadal_model_f_het, metainfo_f, ind_var='Weight',
                                           other_vars={'Sex':'f', 'Genotype':'Rreb1-tm1b:Het'}, dep_var='Gonadal',
                                           c='C1', marker='+', line_label='Het')
    plt.ylim(0, 5)
    plt.tick_params(labelsize=14)
    plt.title('Female', fontsize=14)
    plt.ylabel('Gonadal\ndepot weight (g)', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)

    plt.subplot(422)
    cytometer.stats.plot_linear_regression(gonadal_model_m_null, metainfo_m, ind_var='Weight',
                                           other_vars={'Sex':'m'}, c='k', line_label='Null')
    cytometer.stats.plot_linear_regression(gonadal_model_m_wt, metainfo_m, ind_var='Weight',
                                           other_vars={'Sex':'m', 'Genotype':'Rreb1-tm1b:WT'}, dep_var='Gonadal',
                                           c='C0', marker='x', line_label='WT')
    cytometer.stats.plot_linear_regression(gonadal_model_m_het, metainfo_m, ind_var='Weight',
                                           other_vars={'Sex':'m', 'Genotype':'Rreb1-tm1b:Het'}, dep_var='Gonadal',
                                           c='C1', marker='+', line_label='Het')
    plt.ylim(0, 5)
    plt.tick_params(labelsize=14)
    plt.title('Male', fontsize=14)

    plt.subplot(423)
    cytometer.stats.plot_linear_regression(perineal_model_f_null, metainfo_f, ind_var='Weight',
                                           other_vars={'Sex':'f'}, c='k', line_label='Null')
    cytometer.stats.plot_linear_regression(perineal_model_f_wt, metainfo_f, ind_var='Weight',
                                           other_vars={'Sex':'f', 'Genotype':'Rreb1-tm1b:WT'}, dep_var='PAT',
                                           c='C0', marker='x', line_label='WT')
    cytometer.stats.plot_linear_regression(perineal_model_f_het, metainfo_f, ind_var='Weight',
                                           other_vars={'Sex':'f', 'Genotype':'Rreb1-tm1b:Het'}, dep_var='PAT',
                                           c='C1', marker='+', line_label='Het')
    plt.ylim(0, 2.2)
    plt.tick_params(labelsize=14)
    plt.ylabel('Perineal\ndepot weight (g)', fontsize=14)

    plt.subplot(424)
    cytometer.stats.plot_linear_regression(perineal_model_m_null, metainfo_m, ind_var='Weight',
                                           other_vars={'Sex':'m'}, c='k', line_label='Null')
    cytometer.stats.plot_linear_regression(perineal_model_m_wt, metainfo_m, ind_var='Weight',
                                           other_vars={'Sex':'m', 'Genotype':'Rreb1-tm1b:WT'}, dep_var='PAT',
                                           c='C0', marker='x', line_label='WT')
    cytometer.stats.plot_linear_regression(perineal_model_m_het, metainfo_m, ind_var='Weight',
                                           other_vars={'Sex':'m', 'Genotype':'Rreb1-tm1b:Het'}, dep_var='PAT',
                                           c='C1', marker='+', line_label='Het')
    plt.ylim(0, 2.2)
    plt.tick_params(labelsize=14)

    plt.subplot(425)
    cytometer.stats.plot_linear_regression(subcutaneous_model_f_null, metainfo_f, ind_var='Weight',
                                           other_vars={'Sex':'f'}, c='k', line_label='Null')
    cytometer.stats.plot_linear_regression(subcutaneous_model_f_wt, metainfo_f, ind_var='Weight',
                                           other_vars={'Sex':'f', 'Genotype':'Rreb1-tm1b:WT'}, dep_var='SAT',
                                           c='C0', marker='x', line_label='WT')
    cytometer.stats.plot_linear_regression(subcutaneous_model_f_het, metainfo_f, ind_var='Weight',
                                           other_vars={'Sex':'f', 'Genotype':'Rreb1-tm1b:Het'}, dep_var='SAT',
                                           c='C1', marker='+', line_label='Het')
    plt.ylim(-0.7, 3.5)
    plt.tick_params(labelsize=14)
    plt.ylabel('Subcutaneous\ndepot weight (g)', fontsize=14)

    plt.subplot(426)
    cytometer.stats.plot_linear_regression(subcutaneous_model_m_null, metainfo_m, ind_var='Weight',
                                           other_vars={'Sex':'m'}, c='k', line_label='Null')
    cytometer.stats.plot_linear_regression(subcutaneous_model_m_wt, metainfo_m, ind_var='Weight',
                                           other_vars={'Sex':'m', 'Genotype':'Rreb1-tm1b:WT'}, dep_var='SAT',
                                           c='C0', marker='x', line_label='WT')
    cytometer.stats.plot_linear_regression(subcutaneous_model_m_het, metainfo_m, ind_var='Weight',
                                           other_vars={'Sex':'m', 'Genotype':'Rreb1-tm1b:Het'}, dep_var='SAT',
                                           c='C1', marker='+', line_label='Het')
    plt.ylim(-0.7, 3.5)
    plt.tick_params(labelsize=14)

    plt.subplot(427)
    cytometer.stats.plot_linear_regression(mesenteric_model_f_null, metainfo_f, ind_var='Weight',
                                           other_vars={'Sex':'f'}, c='k', line_label='Null')
    cytometer.stats.plot_linear_regression(mesenteric_model_f_wt, metainfo_f, ind_var='Weight',
                                           other_vars={'Sex':'f', 'Genotype':'Rreb1-tm1b:WT'}, dep_var='Mesenteric',
                                           c='C0', marker='x', line_label='WT')
    cytometer.stats.plot_linear_regression(mesenteric_model_f_het, metainfo_f, ind_var='Weight',
                                           other_vars={'Sex':'f', 'Genotype':'Rreb1-tm1b:Het'}, dep_var='Mesenteric',
                                           c='C1', marker='+', line_label='Het')
    plt.ylim(0, 2.1)
    plt.tick_params(labelsize=14)
    plt.ylabel('Mesenteric\ndepot weight (g)', fontsize=14)
    plt.xlabel('Body weight (g)', fontsize=14)

    plt.subplot(428)
    cytometer.stats.plot_linear_regression(mesenteric_model_m_null, metainfo_m, ind_var='Weight',
                                           other_vars={'Sex':'m'}, c='k', line_label='Null')
    cytometer.stats.plot_linear_regression(mesenteric_model_m_wt, metainfo_m, ind_var='Weight',
                                           other_vars={'Sex':'m', 'Genotype':'Rreb1-tm1b:WT'}, dep_var='Mesenteric',
                                           c='C0', marker='x', line_label='WT')
    cytometer.stats.plot_linear_regression(mesenteric_model_m_het, metainfo_m, ind_var='Weight',
                                           other_vars={'Sex':'m', 'Genotype':'Rreb1-tm1b:Het'}, dep_var='Mesenteric',
                                           c='C1', marker='+', line_label='Het')
    plt.ylim(0, 2.1)
    plt.tick_params(labelsize=14)
    plt.xlabel('Body weight (g)', fontsize=14)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'rreb1_tm1b_exp_0005_depot_weight_models.png'))
    plt.savefig(os.path.join(figures_dir, 'rreb1_tm1b_exp_0005_depot_weight_models.svg'))



# extract coefficients, errors and p-values from models
df_coeff, df_ci_lo, df_ci_hi, df_pval = \
    cytometer.stats.models_coeff_ci_pval(
        [gonadal_model_f_wt, perineal_model_f_wt, subcutaneous_model_f_wt, mesenteric_model_f_wt,
         gonadal_model_f_het, perineal_model_f_het, subcutaneous_model_f_het, mesenteric_model_f_het,
         gonadal_model_m_wt, perineal_model_m_wt, subcutaneous_model_m_wt, mesenteric_model_m_wt,
         gonadal_model_m_het, perineal_model_m_het, subcutaneous_model_m_het, mesenteric_model_m_het
         ])

# multitest correction using Benjamini-Yekuteli
_, df_corrected_pval, _, _ = multipletests(df_pval.values.flatten(), method='fdr_by', alpha=0.05, returnsorted=False)
df_corrected_pval = pd.DataFrame(df_corrected_pval.reshape(df_pval.shape), columns=df_pval.columns)

# convert p-values to asterisks
df_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval, brackets=False), columns=df_coeff.columns)
df_corrected_asterisk = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_corrected_pval, brackets=False), columns=df_coeff.columns)

if SAVEFIG:
    # spreadsheet for model coefficients and p-values
    cols = ['Intercept', 'Weight']

    df_concat = pd.DataFrame()
    for col in cols:
        df_concat = pd.concat([df_concat, df_coeff[col], df_pval[col], df_asterisk[col],
                               df_corrected_pval[col], df_corrected_asterisk[col]], axis=1)
    df_concat.to_csv(os.path.join(figures_dir, 'rreb1_tm1b_exp_0005_depot_weight_models_coeffs_pvals.csv'))


## one data point per animal
## linear regression analysis of quantile_area ~ DW * Genotype
## USED IN PAPER
########################################################################################################################

## (only mode, 25%-, 50%- and 75%-quantiles for illustration purposes and debugging)

# 0.05, 0.1 , 0.15, 0.2, ..., 0.9 , 0.95
quantiles = np.linspace(0, 1, 21)  #

# indices of the quantiles we are going to model
i_quantiles = [5, 10, 15]  # Q1, Q2, Q3

# for convenience
df_all_f = df_all[df_all['Sex'] == 'f']
df_all_m = df_all[df_all['Sex'] == 'm']

depot = 'Gonadal'
# depot = 'PAT'  # perineal + retroperineal
# depot = 'SAT'
# depot = 'Mesenteric'

# fit linear models to area quantiles
q_models_f_wt = []
q_models_f_het = []
q_models_m_wt = []
q_models_m_het = []
q_models_f_null = []
q_models_m_null = []
for i_q in i_quantiles:

    # choose one area_at_quantile value as the output of the linear model
    df_all['area_at_quantile'] = np.array(df_all['area_at_quantiles'].to_list())[:, i_q]

    # fit WT/Het linear models
    idx = (df_all['Sex'] == 'f') & (df_all['depot'] == depot) & (df_all['Genotype'] == 'Rreb1-tm1b:WT')
    q_model_f_wt = sm.OLS.from_formula('area_at_quantile ~ depot_weight', data=df_all, subset=idx).fit()
    idx = (df_all['Sex'] == 'f') & (df_all['depot'] == depot) & (df_all['Genotype'] == 'Rreb1-tm1b:Het')
    q_model_f_het = sm.OLS.from_formula('area_at_quantile ~ depot_weight', data=df_all, subset=idx).fit()
    idx = (df_all['Sex'] == 'm') & (df_all['depot'] == depot) & (df_all['Genotype'] == 'Rreb1-tm1b:WT')
    q_model_m_wt = sm.OLS.from_formula('area_at_quantile ~ depot_weight', data=df_all, subset=idx).fit()
    idx = (df_all['Sex'] == 'm') & (df_all['depot'] == depot) & (df_all['Genotype'] == 'Rreb1-tm1b:Het')
    q_model_m_het = sm.OLS.from_formula('area_at_quantile ~ depot_weight', data=df_all, subset=idx).fit()

    # fit null models
    idx = (df_all['Sex'] == 'f') & (df_all['depot'] == depot)
    q_model_f_null = sm.OLS.from_formula('area_at_quantile ~ depot_weight', data=df_all, subset=idx).fit()
    idx = (df_all['Sex'] == 'm') & (df_all['depot'] == depot)
    q_model_m_null = sm.OLS.from_formula('area_at_quantile ~ depot_weight', data=df_all, subset=idx).fit()

    # q_model_f = sm.RLM.from_formula('area_q_025 ~ depot_weight * C(Genotype)', data=df_all, subset=idx, M=sm.robust.norms.HuberT()).fit()

    q_models_f_wt.append(q_model_f_wt)
    q_models_f_het.append(q_model_f_het)
    q_models_m_wt.append(q_model_m_wt)
    q_models_m_het.append(q_model_m_het)
    q_models_f_null.append(q_model_f_null)
    q_models_m_null.append(q_model_m_null)

    if DEBUG:
        print(q_model_f_wt.summary())
        print(q_model_f_het.summary())
        print(q_model_m_wt.summary())
        print(q_model_m_het.summary())
        print(q_model_f_null.summary())
        print(q_model_m_null.summary())

# extract coefficients, errors and p-values from quartile models
df_coeff_f, df_ci_lo_f, df_ci_hi_f, df_pval_f = cytometer.stats.models_coeff_ci_pval(q_models_f_wt + q_models_f_het)
df_coeff_m, df_ci_lo_m, df_ci_hi_m, df_pval_m = cytometer.stats.models_coeff_ci_pval(q_models_m_wt + q_models_m_het)

# convert p-values to asterisks
df_asterisk_f = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval_f, brackets=False), columns=df_coeff_f.columns)
df_asterisk_m = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval_m, brackets=False), columns=df_coeff_m.columns)
# df_corrected_asterisk_f = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_corrected_pval_f, brackets=False), columns=df_coeff_f.columns)
# df_corrected_asterisk_m = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_corrected_pval_m, brackets=False), columns=df_coeff_m.columns)


if SAVEFIG:
    # save a table for the summary of findings spreadsheet: "summary_of_WAT_findings"
    cols = ['Intercept', 'C(Genotype)[T.Rreb1-tm1b:Het]',
            'depot_weight', 'depot_weight:C(Genotype)[T.Rreb1-tm1b:Het]']

    df_concat = pd.DataFrame()
    for col in cols:
        df_concat = pd.concat([df_concat, df_coeff_f[col], df_pval_f[col], df_asterisk_f[col]], axis=1)
    df_concat.to_csv(os.path.join(figures_dir, 'rreb1_tm1b_exp_0005_area_at_quantile_models_coeffs_pvals_f_' + depot + '.csv'))

    df_concat = pd.DataFrame()
    for col in cols:
        df_concat = pd.concat([df_concat, df_coeff_m[col], df_pval_m[col], df_asterisk_m[col]], axis=1)
    df_concat.to_csv(os.path.join(figures_dir, 'rreb1_tm1b_exp_0005_area_at_quantile_models_coeffs_pvals_m_' + depot + '.csv'))

# plot
if SAVEFIG:

    plt.clf()
    plt.gcf().set_size_inches([6.4, 7.6])

    plt.subplot(321)
    # Q1 Female
    i = 0  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'f'
    df = df_all_f[df_all_f['depot'] == depot].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(q_models_f_null[i], df, 'depot_weight',
                                           other_vars={'depot': depot, 'Sex': sex}, sy=1e-3, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(q_models_f_wt[i], df, 'depot_weight',
                                           other_vars={'depot': depot, 'Sex': sex, 'Genotype': 'Rreb1-tm1b:WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C0', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(q_models_f_het[i], df, 'depot_weight',
                                           other_vars={'depot': depot, 'Sex': sex, 'Genotype': 'Rreb1-tm1b:Het'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C1', marker='+',
                                           line_label='Het')
    plt.tick_params(labelsize=14)
    plt.ylabel('Area$_{\mathrm{Q1}}$ ($10^3\ \mu m^2$)', fontsize=14)
    plt.title('Female', fontsize=14)
    if depot == 'Gonadal':
        plt.legend(loc='best', fontsize=12)
    if depot == 'Gonadal':
        plt.ylim(-7, 9)
        # plt.text(0.02, 0.98, pval_text, transform=plt.gca().transAxes, va='top', fontsize=12)
    elif depot == 'PAT':
        plt.ylim(0.5, 3.5)
    elif depot == 'SAT':
        plt.ylim(0.8, 2.6)
    elif depot == 'Mesenteric':
        plt.ylim(0.6, 3)

    plt.subplot(322)
    # Q1 Male
    i = 0  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'm'
    df = df_all_m[df_all_m['depot'] == depot].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(q_models_m_null[i], df, 'depot_weight',
                                           other_vars={'depot': depot, 'Sex': sex}, sy=1e-3, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(q_models_m_wt[i], df, 'depot_weight',
                                           other_vars={'depot': depot, 'Sex': sex, 'Genotype': 'Rreb1-tm1b:WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C0', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(q_models_m_het[i], df, 'depot_weight',
                                           other_vars={'depot': depot, 'Sex': sex, 'Genotype': 'Rreb1-tm1b:Het'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C1', marker='+',
                                           line_label='Het')
    plt.tick_params(labelsize=14)
    plt.title('Male', fontsize=14)
    if depot == 'Gonadal':
        plt.ylim(-7, 9)
    elif depot == 'PAT':
        plt.ylim(0.5, 3.5)
    elif depot == 'SAT':
        plt.ylim(0.8, 2.6)
    elif depot == 'Mesenteric':
        plt.ylim(0.6, 3)

    plt.subplot(323)
    # Q2 Female
    i = 1  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'f'
    df = df_all_f[df_all_f['depot'] == depot].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(q_models_f_null[i], df, 'depot_weight',
                                           other_vars={'depot': depot, 'Sex': sex}, sy=1e-3, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(q_models_f_wt[i], df, 'depot_weight',
                                           other_vars={'depot': depot, 'Sex': sex, 'Genotype': 'Rreb1-tm1b:WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C0', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(q_models_f_het[i], df, 'depot_weight',
                                           other_vars={'depot': depot, 'Sex': sex, 'Genotype': 'Rreb1-tm1b:Het'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C1', marker='+',
                                           line_label='Het')
    plt.tick_params(labelsize=14)
    plt.ylabel('Area$_{\mathrm{Q2}}$ ($10^3\ \mu m^2$)', fontsize=14)
    if depot == 'Gonadal':
        plt.ylim(-4, 13)
        # plt.text(0.02, 0.98, pval_text, transform=plt.gca().transAxes, va='top', fontsize=12)
    elif depot == 'PAT':
        plt.ylim(1.5, 7)
    elif depot == 'SAT':
        plt.ylim(1, 5)
    elif depot == 'Mesenteric':
        plt.ylim(1, 4.5)

    plt.subplot(324)
    # Q2 Male
    i = 1  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'm'
    df = df_all_m[df_all_m['depot'] == depot].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(q_models_m_null[i], df, 'depot_weight',
                                           other_vars={'depot': depot, 'Sex': sex}, sy=1e-3, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(q_models_m_wt[i], df, 'depot_weight',
                                           other_vars={'depot': depot, 'Sex': sex, 'Genotype': 'Rreb1-tm1b:WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C0', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(q_models_m_het[i], df, 'depot_weight',
                                           other_vars={'depot': depot, 'Sex': sex, 'Genotype': 'Rreb1-tm1b:Het'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C1', marker='+',
                                           line_label='Het')
    plt.tick_params(labelsize=14)
    if depot == 'Gonadal':
        plt.ylim(-4, 13)
        # plt.text(0.02, 0.98, pval_text, transform=plt.gca().transAxes, va='top', fontsize=12)
    elif depot == 'PAT':
        plt.ylim(1.5, 7)
    elif depot == 'SAT':
        plt.ylim(1, 5)
    elif depot == 'Mesenteric':
        plt.ylim(1, 4.5)

    plt.subplot(325)
    # Q3 Female
    i = 2  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'f'
    df = df_all_f[df_all_f['depot'] == depot].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(q_models_f_null[i], df, 'depot_weight',
                                           other_vars={'depot': depot, 'Sex': sex}, sy=1e-3, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(q_models_f_wt[i], df, 'depot_weight',
                                           other_vars={'depot': depot, 'Sex': sex, 'Genotype': 'Rreb1-tm1b:WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C0', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(q_models_f_het[i], df, 'depot_weight',
                                           other_vars={'depot': depot, 'Sex': sex, 'Genotype': 'Rreb1-tm1b:Het'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C1', marker='+',
                                           line_label='Het')
    plt.tick_params(labelsize=14)
    plt.ylabel('Area$_{\mathrm{Q3}}$ ($10^3\ \mu m^2$)', fontsize=14)
    plt.xlabel('Depot weight (g)', fontsize=14)
    if depot == 'Gonadal':
        plt.ylim(0, 17)
        # plt.text(0.02, 0.98, pval_text, transform=plt.gca().transAxes, va='top', fontsize=12)
    elif depot == 'PAT':
        plt.ylim(2.5, 10)
    elif depot == 'SAT':
        plt.ylim(1.7, 7.6)
    elif depot == 'Mesenteric':
        plt.ylim(1, 7)

    plt.subplot(326)
    # Q3 Male
    i = 2  # quantile index for "i_quantiles"
    i_q = i_quantiles[i]  # quantile index for "quantiles"
    sex = 'm'
    df = df_all_m[df_all_m['depot'] == depot].copy()
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    cytometer.stats.plot_linear_regression(q_models_m_null[i], df, 'depot_weight',
                                           other_vars={'depot': depot, 'Sex': sex}, sy=1e-3, c='k',
                                           line_label='Null')
    cytometer.stats.plot_linear_regression(q_models_m_wt[i], df, 'depot_weight',
                                           other_vars={'depot': depot, 'Sex': sex, 'Genotype': 'Rreb1-tm1b:WT'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C0', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(q_models_m_het[i], df, 'depot_weight',
                                           other_vars={'depot': depot, 'Sex': sex, 'Genotype': 'Rreb1-tm1b:Het'},
                                           dep_var='area_at_quantile', sy=1e-3, c='C1', marker='+',
                                           line_label='Het')
    plt.tick_params(labelsize=14)
    plt.xlabel('Depot weight (g)', fontsize=14)
    if depot == 'Gonadal':
        plt.ylim(0, 17)
        # plt.text(0.02, 0.98, pval_text, transform=plt.gca().transAxes, va='top', fontsize=12)
    elif depot == 'PAT':
        plt.ylim(2.5, 10)
    elif depot == 'SAT':
        plt.ylim(1.7, 7.6)
    elif depot == 'Mesenteric':
        plt.ylim(1, 7)

    depot_title = depot.replace('PAT', 'Perineal').replace('SAT', 'Subcutaneous')
    plt.suptitle(depot_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(figures_dir, 'rreb1_tm1b_exp_0005_area_at_quartile_models_' + depot + '.png'))
    plt.savefig(os.path.join(figures_dir, 'rreb1_tm1b_exp_0005_area_at_quartile_models_' + depot + '.svg'))

## Likelihood Ratio Tests to investigate whether Genotype has an effect

depot = 'Gonadal'
# depot = 'PAT'  # perineal + retroperineal
# depot = 'SAT'
# depot = 'Mesenteric'

# Likelihood ratio tests of the Genotype variable
df = df_all.copy()
print('Likelihood Ratio Test: ' + depot)
print('Female')
for i, i_q in enumerate(i_quantiles):

    print('Quantile: ' + str(quantiles[i_q]))

    idx = (df['Sex'] == 'f') & (df['depot'] == depot)
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    q_model_null = sm.OLS.from_formula('area_at_quantile ~ depot_weight', data=df, subset=idx).fit()
    q_model = sm.OLS.from_formula('area_at_quantile ~ depot_weight * Genotype', data=df, subset=idx).fit()

    lr, pval = cytometer.stats.lrtest(q_model_null.llf, q_model.llf)
    pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    print(depot + ': ' + pval_text)

print('Male')
for i, i_q in enumerate(i_quantiles):

    print('Quantile: ' + str(quantiles[i_q]))

    idx = (df['Sex'] == 'm') & (df['depot'] == depot)
    df['area_at_quantile'] = np.array(df['area_at_quantiles'].to_list())[:, i_q]  # vector of areas at current quantile
    q_model_null = sm.OLS.from_formula('area_at_quantile ~ depot_weight', data=df, subset=idx).fit()
    q_model = sm.OLS.from_formula('area_at_quantile ~ depot_weight * Genotype', data=df, subset=idx).fit()

    lr, pval = cytometer.stats.lrtest(q_model_null.llf, q_model.llf)
    pval_text = 'LR=' + '{0:.2f}'.format(lr) + ', p=' + '{0:.2g}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    print(depot + ': ' + pval_text)




# if SAVEFIG:
#     q = [25, 50, 75]
#
#     plt.clf()
#     plt.gcf().set_size_inches([7.2, 9.5])
#
#     plt.subplot(4,2,1)
#     if depot == 'gwat':
#         ylim = (-1, 20)
#     elif depot == 'sqwat':
#         ylim = (-1, 13)
#     h1, h2 = cytometer.stats.cytometer.stats.plot_model_coeff_compare2(q, df_coeff_f['Intercept'] * 1e-3,
#                               df_ci_lo_f['Intercept'] * 1e-3,
#                               df_ci_hi_f['Intercept'] * 1e-3,
#                               df_pval_f['Intercept'],
#                               df_coeff_f['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
#                               df_ci_lo_f['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
#                               df_ci_hi_f['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
#                               df_pval_f['Intercept+C(ko_parent)[T.MAT]'],
#                               df_corrected_pval_1=df_corrected_pval_f['Intercept'],
#                               df_corrected_pval_2=df_corrected_pval_f['Intercept+C(ko_parent)[T.MAT]'],
#                               ylim=ylim, color_1='C0', color_2='C1', label_1='PAT', label_2='MAT')
#     plt.title('Female', fontsize=14)
#     plt.ylabel(r'$\beta_{0}\ (10^3\ \mu m^2)$', fontsize=14)
#     plt.tick_params(labelsize=14)
#     plt.xticks(q)
#     plt.legend(fontsize=12, loc='upper left')
#     plt.ylim(ylim)
#
#     plt.subplot(4,2,2)
#     if depot == 'gwat':
#         ylim = (-1, 20)
#     elif depot == 'sqwat':
#         ylim = (-1, 13)
#     h1, h2 = cytometer.stats.cytometer.stats.plot_model_coeff_compare2(q, df_coeff_m['Intercept'] * 1e-3,
#                               df_ci_lo_m['Intercept'] * 1e-3,
#                               df_ci_hi_m['Intercept'] * 1e-3,
#                               df_pval_m['Intercept'],
#                               df_coeff_m['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
#                               df_ci_lo_m['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
#                               df_ci_hi_m['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
#                               df_pval_m['Intercept+C(ko_parent)[T.MAT]'],
#                               df_corrected_pval_1=df_corrected_pval_m['Intercept'],
#                               df_corrected_pval_2=df_corrected_pval_m['Intercept+C(ko_parent)[T.MAT]'],
#                               ylim=ylim, color_1='C0', color_2='C1', label_1='PAT', label_2='MAT')
#     plt.title('Male', fontsize=14)
#     plt.tick_params(labelsize=14)
#     plt.xticks(q)
#     plt.ylim(ylim)
#
#     plt.subplot(4,2,3)
#     if depot == 'gwat':
#         ylim = (-3, 9)
#     elif depot == 'sqwat':
#         ylim = (-1, 6)
#     cytometer.stats.plot_model_coeff(q, df_coeff_f['C(ko_parent)[T.MAT]'] * 1e-3,
#                      df_ci_lo_f['C(ko_parent)[T.MAT]'] * 1e-3,
#                      df_ci_hi_f['C(ko_parent)[T.MAT]'] * 1e-3,
#                      df_pval_f['C(ko_parent)[T.MAT]'],
#                      df_corrected_pval=df_corrected_pval_f['C(ko_parent)[T.MAT]'],
#                      ylim=ylim, color='k', label='PAT$\mapsto$MAT')
#     plt.ylabel(r'$\Delta\beta_{0}\ (10^3\ \mu m^2)$', fontsize=14)
#     plt.tick_params(labelsize=14)
#     plt.xticks(q)
#     plt.legend(fontsize=12, loc='upper left')
#     plt.ylim(ylim)
#
#     plt.subplot(4,2,4)
#     if depot == 'gwat':
#         ylim = (-3, 9)
#     elif depot == 'sqwat':
#         ylim = (-1, 6)
#     cytometer.stats.plot_model_coeff(q, df_coeff_m['C(ko_parent)[T.MAT]'] * 1e-3,
#                      df_ci_lo_m['C(ko_parent)[T.MAT]'] * 1e-3,
#                      df_ci_hi_m['C(ko_parent)[T.MAT]'] * 1e-3,
#                      df_pval_m['C(ko_parent)[T.MAT]'],
#                      df_corrected_pval=df_corrected_pval_m['C(ko_parent)[T.MAT]'],
#                      ylim=ylim, color='k', label='PAT$\mapsto$MAT')
#     plt.tick_params(labelsize=14)
#     plt.xticks(q)
#     plt.ylim(ylim)
#
#     plt.subplot(4,2,5)
#     if depot == 'gwat':
#         ylim = (-1.5, 2.5)
#     elif depot == 'sqwat':
#         ylim = (-1.5, 2.5)
#     h1, h2 = cytometer.stats.cytometer.stats.plot_model_coeff_compare2(q, df_coeff_f['DW_BW'] * 1e-5,
#                               df_ci_lo_f['DW_BW'] * 1e-5,
#                               df_ci_hi_f['DW_BW'] * 1e-5,
#                               df_pval_f['DW_BW'],
#                               df_coeff_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                               df_ci_lo_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                               df_ci_hi_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                               df_pval_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'],
#                               df_corrected_pval_1=df_corrected_pval_f['DW_BW'],
#                               df_corrected_pval_2=df_corrected_pval_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'],
#                               ylim=ylim, color_1='C0', color_2='C1', label_1='PAT', label_2='MAT')
#     plt.ylabel(r'$\beta_{DW/BW}\ (10^5\ \mu m^2)$', fontsize=14)
#     plt.tick_params(labelsize=14)
#     plt.xticks(q)
#     plt.ylim(ylim)
#
#     plt.subplot(4,2,6)
#     if depot == 'gwat':
#         ylim = (-1.5, 2.5)
#     elif depot == 'sqwat':
#         ylim = (-1.5, 2.5)
#     h1, h2 = cytometer.stats.cytometer.stats.plot_model_coeff_compare2(q, df_coeff_m['DW_BW'] * 1e-5,
#                               df_ci_lo_m['DW_BW'] * 1e-5,
#                               df_ci_hi_m['DW_BW'] * 1e-5,
#                               df_pval_m['DW_BW'],
#                               df_coeff_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                               df_ci_lo_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                               df_ci_hi_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                               df_pval_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'],
#                               df_corrected_pval_1=df_corrected_pval_m['DW_BW'],
#                               df_corrected_pval_2=df_corrected_pval_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'],
#                               ylim=ylim, color_1='C0', color_2='C1', label_1='PAT', label_2='MAT')
#     plt.tick_params(labelsize=14)
#     plt.xticks(q)
#     plt.ylim(ylim)
#
#     plt.subplot(4,2,7)
#     if depot == 'gwat':
#         ylim = (-2.4, 2)
#     elif depot == 'sqwat':
#         ylim = (-2.6, 2)
#     cytometer.stats.plot_model_coeff(q, df_coeff_f['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                      df_ci_lo_f['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                      df_ci_hi_f['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                      df_pval_f['DW_BW:C(ko_parent)[T.MAT]'],
#                      df_corrected_pval=df_corrected_pval_f['DW_BW:C(ko_parent)[T.MAT]'],
#                      ylim=ylim, color='k', label='PAT$\mapsto$MAT')
#     plt.xlabel('Quantile (%)', fontsize=14)
#     plt.ylabel(r'$\Delta \beta_{DW/BW}\ (10^5\ \mu m^2)$', fontsize=14)
#     plt.tick_params(labelsize=14)
#     plt.xticks(q)
#     plt.ylim(ylim)
#
#     plt.subplot(4,2,8)
#     if depot == 'gwat':
#         ylim = (-2.4, 2)
#     elif depot == 'sqwat':
#         ylim = (-2.6, 2.4)
#     cytometer.stats.plot_model_coeff(q, df_coeff_m['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                      df_ci_lo_m['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                      df_ci_hi_m['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                      df_pval_m['DW_BW:C(ko_parent)[T.MAT]'],
#                      df_corrected_pval=df_corrected_pval_m['DW_BW:C(ko_parent)[T.MAT]'],
#                      ylim=ylim, color='k', label='PAT$\mapsto$MAT')
#     plt.xlabel('Quantile (%)', fontsize=14)
#     plt.tick_params(labelsize=14)
#     plt.xticks(q)
#     plt.ylim(ylim)
#
#     plt.tight_layout()
#
#     if depot == 'gwat':
#         plt.suptitle('Gonadal', fontsize=14)
#     elif depot == 'sqwat':
#         plt.suptitle('Subcutaneous', fontsize=14)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#
#     plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_quartile_dw_bw_linreg_coeffs_' + depot + '.png'))
#     plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_quartile_dw_bw_linreg_coeffs_' + depot + '.svg'))
#
# # 0.1 , 0.2, ..., 0.9
# deciles_idx = list(range(2, 20, 2))
# deciles = quantiles[deciles_idx]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#
# # fit linear models for each decile
# decile_models_f = [sm.RLM.from_formula('area_q_' + '{0:03d}'.format(int(d*100)) + ' ~ DW_BW * C(ko_parent)', data=df_f, M=sm.robust.norms.HuberT()).fit()
#                    for d in deciles]
# decile_models_m = [sm.RLM.from_formula('area_q_' + '{0:03d}'.format(int(d*100)) + ' ~ DW_BW * C(ko_parent)', data=df_m, M=sm.robust.norms.HuberT()).fit()
#                    for d in deciles]
#
# print(decile_models_f[4].summary())
# print(decile_models_m[4].summary())
#
# # extract coefficients, errors and p-values from quartile models
# df_coeff_f, df_ci_lo_f, df_ci_hi_f, df_pval_f = \
#     cytometer.stats.models_coeff_ci_pval(decile_models_f, extra_hypotheses='Intercept + C(ko_parent)[T.MAT], DW_BW + DW_BW:C(ko_parent)[T.MAT]')
# df_coeff_m, df_ci_lo_m, df_ci_hi_m, df_pval_m = \
#     cytometer.stats.models_coeff_ci_pval(decile_models_m, extra_hypotheses='Intercept + C(ko_parent)[T.MAT], DW_BW + DW_BW:C(ko_parent)[T.MAT]')
#
# # multitest correction using Benjamini-Yekuteli
# _, df_corrected_pval_f, _, _ = multipletests(df_pval_f.values.flatten(), method='fdr_by', alpha=0.05, returnsorted=False)
# df_corrected_pval_f = pd.DataFrame(df_corrected_pval_f.reshape(df_pval_f.shape), columns=df_pval_f.columns)
# _, df_corrected_pval_m, _, _ = multipletests(df_pval_m.values.flatten(), method='fdr_by', alpha=0.05, returnsorted=False)
# df_corrected_pval_m = pd.DataFrame(df_corrected_pval_m.reshape(df_pval_m.shape), columns=df_pval_m.columns)
#
# # convert p-values to asterisks
# df_asterisk_f = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval_f, brackets=False), columns=df_coeff_f.columns)
# df_asterisk_m = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_pval_m, brackets=False), columns=df_coeff_m.columns)
# df_corrected_asterisk_f = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_corrected_pval_f, brackets=False), columns=df_coeff_f.columns)
# df_corrected_asterisk_m = pd.DataFrame(cytometer.stats.pval_to_asterisk(df_corrected_pval_m, brackets=False), columns=df_coeff_m.columns)
#
# if SAVEFIG:
#     # save a table for the summary of findings spreadsheet: "summary_of_WAT_findings"
#     cols = ['Intercept', 'Intercept+C(ko_parent)[T.MAT]', 'C(ko_parent)[T.MAT]',
#             'DW_BW', 'DW_BW+DW_BW:C(ko_parent)[T.MAT]', 'DW_BW:C(ko_parent)[T.MAT]']
#
#     df_concat = pd.DataFrame()
#     for col in cols:
#         df_concat = pd.concat([df_concat, df_coeff_f[col], df_pval_f[col], df_asterisk_f[col],
#                                df_corrected_pval_f[col], df_corrected_asterisk_f[col]], axis=1)
#     df_concat.to_csv(os.path.join(figures_dir, 'foo.csv'))
#
#     df_concat = pd.DataFrame()
#     for col in cols:
#         df_concat = pd.concat([df_concat, df_coeff_m[col], df_pval_m[col], df_asterisk_m[col],
#                                df_corrected_pval_m[col], df_corrected_asterisk_m[col]], axis=1)
#     df_concat.to_csv(os.path.join(figures_dir, 'foo.csv'))
#
# if SAVEFIG:
#     plt.clf()
#     plt.gcf().set_size_inches([7.2, 9.5])
#     q = np.array(deciles) * 100
#
#     plt.subplot(4,2,1)
#     if depot == 'gwat':
#         ylim = (-1, 34)
#     elif depot == 'sqwat':
#         ylim = (-1, 20)
#     h1, h2 = cytometer.stats.cytometer.stats.plot_model_coeff_compare2(q, df_coeff_f['Intercept'] * 1e-3,
#                               df_ci_lo_f['Intercept'] * 1e-3,
#                               df_ci_hi_f['Intercept'] * 1e-3,
#                               df_pval_f['Intercept'],
#                               df_coeff_f['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
#                               df_ci_lo_f['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
#                               df_ci_hi_f['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
#                               df_pval_f['Intercept+C(ko_parent)[T.MAT]'],
#                               df_corrected_pval_1=df_corrected_pval_f['Intercept'],
#                               df_corrected_pval_2=df_corrected_pval_f['Intercept+C(ko_parent)[T.MAT]'],
#                               ylim=ylim, color_1='C0', color_2='C1', label_1='PAT', label_2='MAT')
#     plt.title('Female', fontsize=14)
#     plt.ylabel(r'$\beta_{0}\ (10^3\ \mu m^2)$', fontsize=14)
#     plt.tick_params(labelsize=14)
#     plt.legend(fontsize=12, loc='upper left')
#     plt.ylim(ylim)
#
#     plt.subplot(4,2,2)
#     if depot == 'gwat':
#         ylim = (-1, 34)
#     elif depot == 'sqwat':
#         ylim = (-1, 20)
#     h1, h2 = cytometer.stats.cytometer.stats.plot_model_coeff_compare2(q, df_coeff_m['Intercept'] * 1e-3,
#                               df_ci_lo_m['Intercept'] * 1e-3,
#                               df_ci_hi_m['Intercept'] * 1e-3,
#                               df_pval_m['Intercept'],
#                               df_coeff_m['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
#                               df_ci_lo_m['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
#                               df_ci_hi_m['Intercept+C(ko_parent)[T.MAT]'] * 1e-3,
#                               df_pval_m['Intercept+C(ko_parent)[T.MAT]'],
#                               df_corrected_pval_1=df_corrected_pval_m['Intercept'],
#                               df_corrected_pval_2=df_corrected_pval_m['Intercept+C(ko_parent)[T.MAT]'],
#                               ylim=ylim, color_1='C0', color_2='C1', label_1='PAT', label_2='MAT')
#     plt.title('Male', fontsize=14)
#     plt.tick_params(labelsize=14)
#     plt.ylim(ylim)
#
#     plt.subplot(4,2,3)
#     if depot == 'gwat':
#         ylim = (-5, 12)
#     elif depot == 'sqwat':
#         ylim = (-1, 7)
#     cytometer.stats.plot_model_coeff(q, df_coeff_f['C(ko_parent)[T.MAT]'] * 1e-3,
#                      df_ci_lo_f['C(ko_parent)[T.MAT]'] * 1e-3,
#                      df_ci_hi_f['C(ko_parent)[T.MAT]'] * 1e-3,
#                      df_pval_f['C(ko_parent)[T.MAT]'],
#                      df_corrected_pval=df_corrected_pval_f['C(ko_parent)[T.MAT]'],
#                      ylim=ylim, color='k', label='PAT$\mapsto$MAT')
#     plt.ylabel(r'$\Delta\beta_{0}\ (10^3\ \mu m^2)$', fontsize=14)
#     plt.tick_params(labelsize=14)
#     plt.legend(fontsize=12, loc='upper left')
#     plt.ylim(ylim)
#
#     plt.subplot(4,2,4)
#     if depot == 'gwat':
#         ylim = (-5, 12)
#     elif depot == 'sqwat':
#         ylim = (-1, 7)
#     cytometer.stats.plot_model_coeff(q, df_coeff_m['C(ko_parent)[T.MAT]'] * 1e-3,
#                      df_ci_lo_m['C(ko_parent)[T.MAT]'] * 1e-3,
#                      df_ci_hi_m['C(ko_parent)[T.MAT]'] * 1e-3,
#                      df_pval_m['C(ko_parent)[T.MAT]'],
#                      df_corrected_pval=df_corrected_pval_m['C(ko_parent)[T.MAT]'],
#                      ylim=ylim, color='k', label='PAT$\mapsto$MAT')
#     plt.tick_params(labelsize=14)
#     plt.ylim(ylim)
#
#     plt.subplot(4,2,5)
#     if depot == 'gwat':
#         ylim = (-2.5, 3.5)
#     elif depot == 'sqwat':
#         ylim = (-2, 3.5)
#     h1, h2 = cytometer.stats.cytometer.stats.plot_model_coeff_compare2(q, df_coeff_f['DW_BW'] * 1e-5,
#                               df_ci_lo_f['DW_BW'] * 1e-5,
#                               df_ci_hi_f['DW_BW'] * 1e-5,
#                               df_pval_f['DW_BW'],
#                               df_coeff_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                               df_ci_lo_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                               df_ci_hi_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                               df_pval_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'],
#                               df_corrected_pval_1=df_corrected_pval_f['DW_BW'],
#                               df_corrected_pval_2=df_corrected_pval_f['DW_BW+DW_BW:C(ko_parent)[T.MAT]'],
#                               ylim=ylim, color_1='C0', color_2='C1', label_1='PAT', label_2='MAT')
#     plt.ylabel(r'$\beta_{DW/BW}\ (10^5\ \mu m^2)$', fontsize=14)
#     plt.tick_params(labelsize=14)
#     plt.ylim(ylim)
#
#     plt.subplot(4,2,6)
#     if depot == 'gwat':
#         ylim = (-2.5, 3.5)
#     elif depot == 'sqwat':
#         ylim = (-2, 3.5)
#     h1, h2 = cytometer.stats.cytometer.stats.plot_model_coeff_compare2(q, df_coeff_m['DW_BW'] * 1e-5,
#                               df_ci_lo_m['DW_BW'] * 1e-5,
#                               df_ci_hi_m['DW_BW'] * 1e-5,
#                               df_pval_m['DW_BW'],
#                               df_coeff_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                               df_ci_lo_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                               df_ci_hi_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                               df_pval_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'],
#                               df_corrected_pval_1=df_corrected_pval_m['DW_BW'],
#                               df_corrected_pval_2=df_corrected_pval_m['DW_BW+DW_BW:C(ko_parent)[T.MAT]'],
#                               ylim=ylim, color_1='C0', color_2='C1', label_1='PAT', label_2='MAT')
#     plt.tick_params(labelsize=14)
#     plt.ylim(ylim)
#
#     plt.subplot(4,2,7)
#     if depot == 'gwat':
#         ylim = (-4, 3)
#     elif depot == 'sqwat':
#         ylim = (-4, 2.4)
#     cytometer.stats.plot_model_coeff(q, df_coeff_f['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                      df_ci_lo_f['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                      df_ci_hi_f['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                      df_pval_f['DW_BW:C(ko_parent)[T.MAT]'],
#                      df_corrected_pval=df_corrected_pval_f['DW_BW:C(ko_parent)[T.MAT]'],
#                      ylim=ylim, color='k', label='PAT$\mapsto$MAT')
#     plt.xlabel('Quantile (%)', fontsize=14)
#     plt.ylabel(r'$\Delta \beta_{DW/BW}\ (10^5\ \mu m^2)$', fontsize=14)
#     plt.tick_params(labelsize=14)
#     plt.ylim(ylim)
#
#     plt.subplot(4,2,8)
#     if depot == 'gwat':
#         ylim = (-4, 3)
#     elif depot == 'sqwat':
#         ylim = (-4, 2.4)
#     cytometer.stats.plot_model_coeff(q, df_coeff_m['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                      df_ci_lo_m['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                      df_ci_hi_m['DW_BW:C(ko_parent)[T.MAT]'] * 1e-5,
#                      df_pval_m['DW_BW:C(ko_parent)[T.MAT]'],
#                      df_corrected_pval=df_corrected_pval_m['DW_BW:C(ko_parent)[T.MAT]'],
#                      ylim=ylim, color='k', label='PAT$\mapsto$MAT')
#     plt.xlabel('Quantile (%)', fontsize=14)
#     plt.tick_params(labelsize=14)
#     plt.ylim(ylim)
#
#     plt.tight_layout()
#
#     if depot == 'gwat':
#         plt.suptitle('Gonadal', fontsize=14)
#     elif depot == 'sqwat':
#         plt.suptitle('Subcutaneous', fontsize=14)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#
#     plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_decile_dw_bw_linreg_coeffs_' + depot + '.png'))
#     plt.savefig(os.path.join(figures_dir, 'klf14_b6ntac_exp_0110_paper_figures_decile_dw_bw_linreg_coeffs_' + depot + '.svg'))
