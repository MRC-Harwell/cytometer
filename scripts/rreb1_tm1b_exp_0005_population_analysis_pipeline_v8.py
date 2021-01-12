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
max_area = 44879 * 3  # (pix^2) larger objects are rejected

xres_ref = 0.4538234626730202
yres_ref = 0.4537822752643282
min_area_um2 = min_area * xres_ref * yres_ref
max_area_um2 = max_area * xres_ref * yres_ref

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

    # make sure that in the boxplots Rreb1-tm1b:WT comes before Rreb1-tm1b:Het, and female before male
    metainfo['Sex'] = metainfo['Sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
    metainfo['Genotype'] = metainfo['Genotype'].astype(
        pd.api.types.CategoricalDtype(categories=['Rreb1-tm1b:WT', 'Rreb1-tm1b:Het'], ordered=True))

    # mean mouse body weight (female and male)
    mean_bw_f = metainfo[metainfo['Sex'] == 'f']['Weight (g)'].mean()
    mean_bw_m = metainfo[metainfo['Sex'] == 'm']['Weight (g)'].mean()

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
             depot_label = 'Gonadal_AT (g)'
        elif is_perineal:
            depot_label = 'PAT+RPAT (g)'  # perineal + retroperineal
        elif is_subcutaneous:
            depot_label = 'SAT (g)'
        elif is_mesenteric:
            depot_label = 'Mesenteric_AT (g)'

        aux = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=os.path.basename(json_file),
                                                        tags_to_keep=['Animal Identifier', 'Sex', 'Genotype',
                                                                      'Weight (g)', 'Gonadal_AT (g)', 'PAT+RPAT (g)',
                                                                      'SAT (g)', 'Mesenteric_AT (g)'],
                                                        id_tag='Animal Identifier')
        df = aux.drop(labels=['Gonadal_AT (g)', 'PAT+RPAT (g)', 'SAT (g)', 'Mesenteric_AT (g)'], axis='columns')
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
