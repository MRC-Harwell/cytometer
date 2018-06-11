import os
import glob
import numpy as np
from svgpathtools import svg2paths
import matplotlib.pyplot as plt
import openslide
import csv
import pandas as pd
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import mannwhitneyu
from statsmodels.distributions.empirical_distribution import ECDF

DEBUG = False

image_data_dir = '/home/rcasero/data/roger_data'
root_data_dir = '/home/rcasero/Dropbox/klf14'
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')


# Area of Polygon using Shoelace formula
# http://en.wikipedia.org/wiki/Shoelace_formula
# FB - 20120218
# corners must be ordered in clockwise or counter-clockwise direction
def polygon_area(corners):
    n = len(corners)  # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

# extract contour as a list of (X,Y) coordinates
def extract_contour(path, x_res=1.0, y_res=1.0):

    contour = []
    for pt in path:

        # (X, Y) for each point
        contour.append((np.real(pt.start) * x_res, np.imag(pt.start) * y_res))

        if DEBUG:
            plt.plot(*zip(*contour))

    return contour


# extract contours that correspond to non-edge cells in SVG file as list of polygons
def extract_cell_contour_and_compute_area(file, x_res=1.0, y_res=1.0):

    # extract all paths from the SVG file
    paths, attributes = svg2paths(file)

    # loop paths
    areas = []
    for path, attribute in zip(paths, attributes):

        # skip if the countour is not a cell (we also skip edge cells, as they are incomplete, and thus their area
        # is misleading)
        if not attribute['id'].startswith('Cell'):
            continue

        # extract contour polygon from the path object, and compute area
        contour = extract_contour(path, x_res=x_res, y_res=y_res)
        areas.append(polygon_area(contour))

    return np.array(areas)

##################################################################################################################
# main programme
##################################################################################################################

## get pixel size in original images, so that we can compute areas in um^2, instead of pixel^2
## Warning: We are assuming that all images have the same resolution, so we only do this once


# check that all files have the same pixel size
for file in glob.glob(os.path.join(image_data_dir, '*.ndpi')):
    im = openslide.OpenSlide(file)
    print("Xres = " + str(1e-2 / float(im.properties['tiff.XResolution'])) + ', ' +
          "Yres = " + str(1e-2 / float(im.properties['tiff.YResolution'])))

# in practice, we read the pixel size from one file. The reason is that the whole dataset is large, and cannot be
# conveniently stored in the laptop, so when I'm working from home, I have no access to all full original images
original_file = os.path.join(root_data_dir, 'KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38.ndpi')

# open original image
im = openslide.OpenSlide(original_file)

# compute pixel size in the original image, where patches were taken from
if im.properties['tiff.ResolutionUnit'].lower() == 'centimeter':
    x_res = 1e-2 / float(im.properties['tiff.XResolution'])  # meters / pixel
    y_res = 1e-2 / float(im.properties['tiff.YResolution'])  # meters / pixel
else:
    raise ValueError('Only centimeter units implemented')

## read CSV file with female/male labels for mice

with open(os.path.join(root_data_dir, 'klf14_b6ntac_sex_info.csv'), 'r') as f:
    reader = csv.DictReader(f, skipinitialspace=True)
    klf14_info = []
    for row in reader:
        klf14_info.append(row)
f.close()

## read all contour files, and categorise them into MAT/PAT and f/m

# list of mouse IDs
klf14_ids = [x['id'] for x in klf14_info]

file_list = glob.glob(os.path.join(training_data_dir, '*.svg'))

# create empty dataframe to host the data
df = pd.DataFrame(data={'area': [], 'mouse_id': [], 'sex': [], 'ko': [], 'image_id': []})

for file in file_list:

    # image ID
    image_id = os.path.basename(file)
    image_id = os.path.splitext(image_id)[-2]

    # get mouse ID from the file name
    mouse_id = None
    for x in klf14_ids:
        if x in image_id:
            mouse_id = x
            break
    if mouse_id is None:
        raise ValueError('Filename does not seem to correspond to any known mouse ID: ' + file)

    # index of mouse ID
    idx = klf14_ids.index(mouse_id)

    # sex and KO-side for this mouse
    mouse_sex = klf14_info[idx]['sex']
    mouse_ko = klf14_info[idx]['ko']

    # compute areas of all non-edge cells
    areas = extract_cell_contour_and_compute_area(file, x_res=x_res, y_res=y_res)

    # area, image id, mouse id, sex, KO
    for i, a in enumerate(areas):
        if a == 0.0:
            print('Warning! Area == 0.0: index ' + str(i) + ':' + image_id)
        df = df.append({'area': a, 'mouse_id': mouse_id, 'sex': mouse_sex, 'ko': mouse_ko, 'image_id': image_id},
                       ignore_index=True)


# save dataframe to file
#df.to_csv(os.path.join(root_data_dir, 'klf14_b6ntac_cell_areas.csv'))

# split dataset into groups
area_f = df.loc[df.sex == 'f', ('area', 'ko', 'image_id')]
area_m = df.loc[df.sex == 'm', ('area', 'ko', 'image_id')]

area_MAT = df.loc[df.ko == 'MAT', ('area', 'sex', 'image_id')]
area_PAT = df.loc[df.ko == 'PAT', ('area', 'sex', 'image_id')]

# make sure that in the boxplots PAT comes before MAT
area_f['ko'] = area_f['ko'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
area_m['ko'] = area_m['ko'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))

# make sure that in the boxplots f comes before m
area_MAT['sex'] = area_MAT['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
area_PAT['sex'] = area_PAT['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))

# scale area values to um^2
area_f['area'] *= 1e12
area_m['area'] *= 1e12

area_MAT['area'] *= 1e12
area_PAT['area'] *= 1e12

area_f_MAT = area_f.loc[area_f.ko == 'MAT', ('area', 'image_id')]
area_f_PAT = area_f.loc[area_f.ko == 'PAT', ('area', 'image_id')]
area_m_MAT = area_m.loc[area_m.ko == 'MAT', ('area', 'image_id')]
area_m_PAT = area_m.loc[area_m.ko == 'PAT', ('area', 'image_id')]

## boxplots of each image

# plot boxplots for each individual image
plt.clf()
df.boxplot(column='area', by='image_id', vert=False)

## boxplots comparing MAT/PAT and f/m

# plot boxplots for each individual image, split into f/m groups
plt.clf()
ax = plt.subplot(211)
area_f.boxplot(column='area', by='image_id', vert=False, ax=ax)
plt.title('female')

ax = plt.subplot(212)
area_m.boxplot(column='area', by='image_id', vert=False, ax=ax)
plt.title('male')

# plot boxplots for each individual image, split into MAT/PAT groups
plt.clf()
ax = plt.subplot(211)
area_MAT.boxplot(column='area', by='image_id', vert=False, ax=ax)
plt.title('MAT')

ax = plt.subplot(212)
area_PAT.boxplot(column='area', by='image_id', vert=False, ax=ax)
plt.title('PAT')


# plot boxplots
plt.clf()
ax = plt.subplot(121)
area_f.boxplot(column='area', by='ko', ax=ax, notch=True)
ax.set_ylim(0, 2e4)
ax.set_title('female', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
ax = plt.subplot(122)
area_m.boxplot(column='area', by='ko', ax=ax, notch=True)
ax.set_ylim(0, 2e4)
ax.set_title('male', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)

# plot boxplots without outliers
plt.clf()
ax = plt.subplot(121)
area_f.boxplot(column='area', by='ko', ax=ax, showfliers=False, notch=True)
ax.set_ylim(0, 1e4)
ax.set_title('female', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
ax = plt.subplot(122)
area_m.boxplot(column='area', by='ko', ax=ax, showfliers=False, notch=True)
ax.set_ylim(0, 1e4)
ax.set_title('male', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)


# function to estimate PDF of areas using Kernel Density
def compute_and_plot_pdf(ax, area, title, bandwidth=None):
    # compute optimal bandwidth
    params = {'bandwidth': bandwidth}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(area[:, np.newaxis])
    if DEBUG:
        print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

    # compute and plot histogram
    hist, bin_edges, foo = plt.hist(area, bins=100, density=True)

    # compute and plot pdf
    bin_centers = (bin_edges[0:-1] + bin_edges[1:]) / 2.0
    kde = KernelDensity(kernel='gaussian', bandwidth=grid.best_estimator_.bandwidth).fit(area[:, np.newaxis])
    area_log_pdf = kde.score_samples(bin_centers[:, np.newaxis])
    area_pdf = np.exp(area_log_pdf)
    plt.plot(bin_centers, area_pdf)

    # metainfo for plot
    ax.set_xlabel('area (um^2)', fontsize=14)
    ax.set_ylabel('pdf', fontsize=14)
    plt.title(title, fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=12)

    return bin_centers, area_pdf


## plot estimated pdfs separated by f/m, MAT/PAT
plt.clf()

ax = plt.subplot(221)
bin_centers_f_PAT, area_pdf_f_PAT = compute_and_plot_pdf(ax, area_f_PAT['area'], 'f, PAT', bandwidth=np.logspace(2, 3, 200))

ax = plt.subplot(223)
bin_centers_f_MAT, area_pdf_f_MAT = compute_and_plot_pdf(ax, area_f_MAT['area'], 'f, MAT', bandwidth=np.logspace(2, 3, 200))

ax = plt.subplot(222)
bin_centers_m_PAT, area_pdf_m_PAT = compute_and_plot_pdf(ax, area_m_PAT['area'], 'm, PAT', bandwidth=np.logspace(2, 3, 200))

ax = plt.subplot(224)
bin_centers_m_MAT, area_pdf_m_MAT = compute_and_plot_pdf(ax, area_m_MAT['area'], 'm, MAT', bandwidth=np.logspace(2, 3, 200))


## plot pdfs side by side

plt.clf()

ax = plt.subplot(121)
plt.plot(bin_centers_f_PAT, np.exp(area_pdf_f_PAT))
plt.plot(bin_centers_f_MAT, np.exp(area_pdf_f_MAT))
plt.legend(('PAT', 'MAT'))
ax.set_xlabel('area (um^2)', fontsize=18)
ax.set_ylabel('pdf', fontsize=18)
plt.title('female', fontsize=20)
ax.set_xlim(0, 20000)
plt.tick_params(axis='both', which='major', labelsize=16)

ax = plt.subplot(122)
plt.plot(bin_centers_m_PAT, np.exp(area_pdf_m_PAT))
plt.plot(bin_centers_m_MAT, np.exp(area_pdf_m_MAT))
plt.legend(('PAT', 'MAT'))
ax.set_xlabel('area (um^2)', fontsize=18)
ax.set_ylabel('pdf', fontsize=18)
plt.title('male', fontsize=20)
ax.set_xlim(0, 20000)
plt.tick_params(axis='both', which='major', labelsize=16)

## statistical comparison

# Mann–Whitney U test
statistic_f, pvalue_f = mannwhitneyu(area_f_MAT, area_f_PAT, alternative='less')
statistic_m, pvalue_m = mannwhitneyu(area_m_MAT, area_m_PAT, alternative='less')

print('females, statistic: ' + "{0:.1f}".format(statistic_f) + ', p-value: ' + "{0:.2e}".format(pvalue_f))
print('males, statistic: ' + "{0:.1f}".format(statistic_m) + ', p-value: ' + "{0:.2e}".format(pvalue_m))

## compute ECDF


def area_linspace(x, n=100):
    return np.linspace(np.min(x['area']), np.max(x['area']), n)

area_ecdf_f_PAT = ECDF(area_f_PAT['area'])
area_ecdf_f_MAT = ECDF(area_f_MAT['area'])
area_ecdf_m_PAT = ECDF(area_m_PAT['area'])
area_ecdf_m_MAT = ECDF(area_m_MAT['area'])

area_linspace_f_PAT = area_linspace(area_f_PAT)
area_linspace_f_MAT = area_linspace(area_f_MAT)
area_linspace_m_PAT = area_linspace(area_m_PAT)
area_linspace_m_MAT = area_linspace(area_m_MAT)

# plot ECDF curves side by side
plt.clf()

ax = plt.subplot(121)
plt.plot(area_linspace_f_PAT, area_ecdf_f_PAT(area_linspace_f_PAT))
plt.plot(area_linspace_f_MAT, area_ecdf_f_MAT(area_linspace_f_MAT))
plt.legend(('PAT', 'MAT'))
ax.set_xlabel('area (um^2)', fontsize=14)
ax.set_ylabel('ECDF', fontsize=14)
plt.title('female', fontsize=16)
ax.set_xlim(0, 20000)
plt.tick_params(axis='both', which='major', labelsize=16)

ax = plt.subplot(122)
plt.plot(area_linspace_m_PAT, area_ecdf_m_PAT(area_linspace_m_PAT))
plt.plot(area_linspace_m_MAT, area_ecdf_m_MAT(area_linspace_m_MAT))
plt.legend(('PAT', 'MAT'))
ax.set_xlabel('area (um^2)', fontsize=14)
ax.set_ylabel('ECDF', fontsize=14)
plt.title('male', fontsize=16)
ax.set_xlim(0, 20000)
plt.tick_params(axis='both', which='major', labelsize=16)


# ## identification of outliers
#
# # log transform of area data
# df_norm = df.copy()
# df_norm.area = np.log10(df_norm.area)
#
# # plot boxplots
# area_f_norm = df_norm.loc[df.sex == 'f', ('area', 'ko')]
# area_m_norm = df_norm.loc[df.sex == 'm', ('area', 'ko')]
#
# # make sure that in the boxplots PAT comes before MAT
# area_f_norm['ko'] = area_f_norm['ko'].astype(pd.api.types.CategoricalDtype(categories=["PAT", "MAT"], ordered=True))
# area_m_norm['ko'] = area_m_norm['ko'].astype(pd.api.types.CategoricalDtype(categories=["PAT", "MAT"], ordered=True))
#
# plt.clf()
# ax = plt.subplot(121)
# area_f_norm.boxplot(column='area', by='ko', ax=ax, notch=True)
# ax.set_ylim(-10.5, -7.5)
# ax.set_title('female')
# ax.set_xlabel('')
# ax.set_ylabel('log(area)')
# ax = plt.subplot(122)
# area_m_norm.boxplot(column='area', by='ko', ax=ax, notch=True)
# ax.set_ylim(-10.5, -7.5)
# ax.set_title('male')
# ax.set_xlabel('')
# ax.set_ylabel('log(area)')
#
# # split dataset into groups
# area_f_MAT_norm = area_f_norm['area'][area_f_norm['ko'] == 'MAT']
# area_f_PAT_norm = area_f_norm['area'][area_f_norm['ko'] == 'PAT']
# area_m_MAT_norm = area_m_norm['area'][area_m_norm['ko'] == 'MAT']
# area_m_PAT_norm = area_m_norm['area'][area_m_norm['ko'] == 'PAT']
#
#
# # compute limits beyond which we consider data points to be outliers
# def outlier_limits(x):
#     q75, q25 = np.percentile(x, [75, 25])
#     iqr = q75 - q25
#     min = q25 - (iqr * 1.5)
#     max = q75 + (iqr * 1.5)
#     return min, max
#
#
# area_f_MAT_norm_min, area_f_MAT_norm_max = outlier_limits(area_f_MAT_norm)
# area_f_PAT_norm_min, area_f_PAT_norm_max = outlier_limits(area_f_PAT_norm)
# area_m_MAT_norm_min, area_m_MAT_norm_max = outlier_limits(area_m_MAT_norm)
# area_m_PAT_norm_min, area_m_PAT_norm_max = outlier_limits(area_m_PAT_norm)
#
# area_f_MAT_norm_outlier = np.logical_or(area_f_MAT_norm < area_f_MAT_norm_min,
#                                         area_f_MAT_norm > area_f_MAT_norm_max)
# area_f_PAT_norm_outlier = np.logical_or(area_f_PAT_norm < area_f_PAT_norm_min,
#                                         area_f_PAT_norm > area_f_PAT_norm_max)
# area_m_MAT_norm_outlier = np.logical_or(area_m_MAT_norm < area_m_MAT_norm_min,
#                                         area_m_MAT_norm > area_m_MAT_norm_max)
# area_m_PAT_norm_outlier = np.logical_or(area_m_PAT_norm < area_m_PAT_norm_min,
#                                         area_m_PAT_norm > area_m_PAT_norm_max)
#
# # remove outliers from groups
# area_f_MAT = area_f_MAT[np.logical_not(area_f_MAT_norm_outlier)]
# area_f_PAT = area_f_PAT[np.logical_not(area_f_PAT_norm_outlier)]
# area_m_MAT = area_m_MAT[np.logical_not(area_m_MAT_norm_outlier)]
# area_m_PAT = area_m_PAT[np.logical_not(area_m_PAT_norm_outlier)]
#
# ## plot estimated pdfs separated by f/m, MAT/PAT
# plt.clf()
#
# ax = plt.subplot(221)
# bin_centers_f_PAT, area_pdf_f_PAT = compute_and_plot_pdf(ax, area_f_PAT, 'f, PAT', bandwidth=np.logspace(2, 3, 200))
#
# ax = plt.subplot(223)
# bin_centers_f_MAT, area_pdf_f_MAT = compute_and_plot_pdf(ax, area_f_MAT, 'f, MAT', bandwidth=np.logspace(2, 3, 200))
#
# ax = plt.subplot(222)
# bin_centers_m_PAT, area_pdf_m_PAT = compute_and_plot_pdf(ax, area_m_PAT, 'm, PAT', bandwidth=np.logspace(2, 3, 200))
#
# ax = plt.subplot(224)
# bin_centers_m_MAT, area_pdf_m_MAT = compute_and_plot_pdf(ax, area_m_MAT, 'm, MAT', bandwidth=np.logspace(2, 3, 200))
#
#
# ## plot pdfs side by side
#
# plt.clf()
#
# ax = plt.subplot(211)
# plt.plot(bin_centers_f_PAT, np.exp(area_pdf_f_PAT))
# plt.plot(bin_centers_f_MAT, np.exp(area_pdf_f_MAT))
# plt.legend(('PAT', 'MAT'))
# ax.set_xlabel('area (um^2)')
# ax.set_ylabel('pdf')
# plt.title('female')
# ax.set_xlim(0, 20000)
#
# ax = plt.subplot(212)
# plt.plot(bin_centers_m_PAT, np.exp(area_pdf_m_PAT))
# plt.plot(bin_centers_m_MAT, np.exp(area_pdf_m_MAT))
# plt.legend(('PAT', 'MAT'))
# ax.set_xlabel('area (um^2)')
# ax.set_ylabel('pdf')
# plt.title('male')
# ax.set_xlim(0, 20000)
#
# ## statistical comparison
#
# # Mann–Whitney U test
# statistic_f, pvalue_f = mannwhitneyu(area_f_MAT, area_f_PAT, alternative='less')
# statistic_m, pvalue_m = mannwhitneyu(area_m_MAT, area_m_PAT, alternative='less')
#
# print('females, statistic: ' + "{0:.1f}".format(statistic_f) + ', p-value: ' + "{0:.2e}".format(pvalue_f))
# print('males, statistic: ' + "{0:.1f}".format(statistic_m) + ', p-value: ' + "{0:.2e}".format(pvalue_m))

## measure effect size (um^2)

# compute effect as difference of the median areas
effect_f = np.median(area_f_MAT) - np.median(area_f_PAT)
effect_m = np.median(area_m_MAT) - np.median(area_m_PAT)

# area change
print('Female: Median area change from PAT to MAT: ' +
      "{0:.1f}".format(np.median(area_f_MAT) - np.median(area_f_PAT)) + ' um^2 (' +
      "{0:.1f}".format((np.median(area_f_MAT) - np.median(area_f_PAT)) / np.median(area_f_PAT) * 100) + '%)')
print('Male: Median area change from PAT to MAT: ' +
      "{0:.1f}".format(np.median(area_m_MAT) - np.median(area_m_PAT)) + ' um^2 (' +
      "{0:.1f}".format((np.median(area_m_MAT) - np.median(area_m_PAT)) / np.median(area_m_PAT) * 100) + '%)')

# for the median cells areas, compute radii as if cells were circles
radius_f_MAT = np.sqrt(np.median(area_f_MAT) / np.pi)  # (um)
radius_f_PAT = np.sqrt(np.median(area_f_PAT) / np.pi)  # (um)
radius_m_MAT = np.sqrt(np.median(area_m_MAT) / np.pi)  # (um)
radius_m_PAT = np.sqrt(np.median(area_m_PAT) / np.pi)  # (um)

# radius change in percentage
print('Female: Radius change from PAT to MAT: ' +
      "{0:.1f}".format(radius_f_MAT - radius_f_PAT) + ' um ('
      "{0:.1f}".format((radius_f_MAT - radius_f_PAT) / radius_f_PAT * 100) + '%)')
print('Male: Radius change from PAT to MAT: ' +
      "{0:.1f}".format(radius_m_MAT - radius_m_PAT) + ' um ('
      "{0:.1f}".format((radius_m_MAT - radius_m_PAT) / radius_m_PAT * 100) + '%)')

## compare percentiles of the distributions

perc = np.linspace(0, 100, num=100)
perc_f_MAT = np.percentile(area_f_MAT, perc)
perc_f_PAT = np.percentile(area_f_PAT, perc)
perc_m_MAT = np.percentile(area_m_MAT, perc)
perc_m_PAT = np.percentile(area_m_PAT, perc)

# plot curves comparing cell area change at each percentile
plt.clf()
plt.subplot(211)
plt.plot(perc, (perc_f_MAT - perc_f_PAT) / perc_f_PAT * 100)
plt.title('female', fontsize=20)
plt.xlabel('percentile (%)', fontsize=18)
plt.ylabel('change in cell area size from PAT to MAT (%)', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.subplot(212)
plt.plot(perc, (perc_m_MAT - perc_m_PAT) / perc_m_PAT * 100)
plt.title('male', fontsize=20)
plt.xlabel('percentile (%)', fontsize=18)
plt.ylabel('change in cell area size from PAT to MAT (%)', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
