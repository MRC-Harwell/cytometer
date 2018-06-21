import os
import glob
import numpy as np
from svgpathtools import svg2paths
import matplotlib.pyplot as plt
import openslide
import csv
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
import scipy.stats as stats
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
import statsmodels.formula.api as smf
import PIL

DEBUG = False

image_data_dir = '/home/rcasero/data/roger_data'
root_data_dir = '/home/rcasero/Dropbox/klf14'
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')
training_non_overlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')

''' auxiliary functions for area computations from Gimp paths
========================================================================================================================
'''

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


'''main programme
========================================================================================================================
'''

''' checks on the original histology slices, and get pixel size
========================================================================================================================
'''

# check that all histology files have the same pixel size
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

# read CSV file with female/male labels for mice
with open(os.path.join(root_data_dir, 'klf14_b6ntac_sex_info.csv'), 'r') as f:
    reader = csv.DictReader(f, skipinitialspace=True)
    klf14_info = []
    for row in reader:
        klf14_info.append(row)
f.close()

''' load hand traced contours, compute cell areas and create dataframe
========================================================================================================================
'''

# list of mouse IDs
klf14_ids = [x['id'] for x in klf14_info]

file_list = glob.glob(os.path.join(training_data_dir, '*.svg'))

# create empty dataframe to host the data
df = pd.DataFrame(data={'area': [], 'mouse_id': [], 'sex': [], 'ko': [], 'image_id': []})

# read all contour files, and categorise them into MAT/PAT and f/m
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

    # add to dataframe: area, image id, mouse id, sex, KO
    for i, a in enumerate(areas):
        if a == 0.0:
            print('Warning! Area == 0.0: index ' + str(i) + ':' + image_id)
        df = df.append({'area': a, 'mouse_id': mouse_id, 'sex': mouse_sex, 'ko': mouse_ko, 'image_id': image_id},
                       ignore_index=True)

# save dataframe with input data to file
#df.to_csv(os.path.join(root_data_dir, 'klf14_b6ntac_cell_areas.csv'))

''' load non-overlapping segmentations, compute cell areas and create dataframe
========================================================================================================================
'''

file_list = glob.glob(os.path.join(training_non_overlap_data_dir, '*.tif'))

# create empty dataframe to host the data
df_no = pd.DataFrame(data={'area': [], 'mouse_id': [], 'sex': [], 'ko': [], 'image_id': []})

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

    # load file with the whatershed non-overlapping labels
    im = PIL.Image.open(file)
    im = im.getchannel(0)

    if DEBUG:
        plt.clf()
        plt.imshow(im)

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
        df_no = df_no.append({'area': a, 'mouse_id': mouse_id, 'sex': mouse_sex, 'ko': mouse_ko, 'image_id': image_id},
                             ignore_index=True)


''' split full dataset into smaller datasets for different groups 
========================================================================================================================
'''

# split dataset into groups
df_f = df.loc[df.sex == 'f', ('area', 'ko', 'image_id', 'mouse_id')]
df_m = df.loc[df.sex == 'm', ('area', 'ko', 'image_id', 'mouse_id')]

df_no_f = df_no.loc[df_no.sex == 'f', ('area', 'ko', 'image_id', 'mouse_id')]
df_no_m = df_no.loc[df_no.sex == 'm', ('area', 'ko', 'image_id', 'mouse_id')]

df_MAT = df.loc[df.ko == 'MAT', ('area', 'sex', 'image_id', 'mouse_id')]
df_PAT = df.loc[df.ko == 'PAT', ('area', 'sex', 'image_id', 'mouse_id')]

# make sure that in the boxplots PAT comes before MAT
df_f['ko'] = df_f['ko'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
df_m['ko'] = df_m['ko'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))

# make sure that in the boxplots f comes before m
df_MAT['sex'] = df_MAT['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
df_PAT['sex'] = df_PAT['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))

# scale area values to um^2
df_f['area'] *= 1e12
df_m['area'] *= 1e12

df_MAT['area'] *= 1e12
df_PAT['area'] *= 1e12

df_f_MAT = df_f.loc[df_f.ko == 'MAT', ('area', 'image_id', 'mouse_id')]
df_f_PAT = df_f.loc[df_f.ko == 'PAT', ('area', 'image_id', 'mouse_id')]
df_m_MAT = df_m.loc[df_m.ko == 'MAT', ('area', 'image_id', 'mouse_id')]
df_m_PAT = df_m.loc[df_m.ko == 'PAT', ('area', 'image_id', 'mouse_id')]

df_no_f_MAT = df_no_f.loc[df_no_f.ko == 'MAT', ('area', 'image_id', 'mouse_id')]
df_no_f_PAT = df_no_f.loc[df_no_f.ko == 'PAT', ('area', 'image_id', 'mouse_id')]
df_no_m_MAT = df_no_m.loc[df_no_m.ko == 'MAT', ('area', 'image_id', 'mouse_id')]
df_no_m_PAT = df_no_m.loc[df_no_m.ko == 'PAT', ('area', 'image_id', 'mouse_id')]

''' boxplots of each image
========================================================================================================================
'''

# plot cell area boxplots for each individual image
df.boxplot(column='area', by='image_id', vert=False)

# plot boxplots for each individual image, split into f/m groups
plt.clf()
ax = plt.subplot(211)
df_f.boxplot(column='area', by='image_id', vert=False, ax=ax)
plt.title('female')

ax = plt.subplot(212)
df_m.boxplot(column='area', by='image_id', vert=False, ax=ax)
plt.title('male')

# plot boxplots for each individual image, split into MAT/PAT groups
plt.clf()
ax = plt.subplot(211)
df_MAT.boxplot(column='area', by='image_id', vert=False, ax=ax)
plt.title('MAT')

ax = plt.subplot(212)
df_PAT.boxplot(column='area', by='image_id', vert=False, ax=ax)
plt.title('PAT')

# plot boxplots for f/m, PAT/MAT comparison as in Nature Genetics paper
plt.clf()
ax = plt.subplot(121)
df_f.boxplot(column='area', by='ko', ax=ax, notch=True)
ax.set_ylim(0, 2e4)
ax.set_title('female', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
ax = plt.subplot(122)
df_m.boxplot(column='area', by='ko', ax=ax, notch=True)
ax.set_ylim(0, 2e4)
ax.set_title('male', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)

# same boxplots without outliers
plt.clf()
ax = plt.subplot(121)
df_f.boxplot(column='area', by='ko', ax=ax, showfliers=False, notch=True)
ax.set_ylim(0, 1e4)
ax.set_title('female', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
ax = plt.subplot(122)
df_m.boxplot(column='area', by='ko', ax=ax, showfliers=False, notch=True)
ax.set_ylim(0, 1e4)
ax.set_title('male', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('area (um^2)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)

''' cell area PDF estimation
========================================================================================================================
'''

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


# plot estimated pdfs separated by f/m, MAT/PAT
plt.clf()

ax = plt.subplot(221)
bin_centers_f_PAT, area_pdf_f_PAT = compute_and_plot_pdf(ax, df_f_PAT.area, 'f, PAT', bandwidth=np.logspace(2, 3, 200))

ax = plt.subplot(223)
bin_centers_f_MAT, area_pdf_f_MAT = compute_and_plot_pdf(ax, df_f_MAT.area, 'f, MAT', bandwidth=np.logspace(2, 3, 200))

ax = plt.subplot(222)
bin_centers_m_PAT, area_pdf_m_PAT = compute_and_plot_pdf(ax, df_m_PAT.area, 'm, PAT', bandwidth=np.logspace(2, 3, 200))

ax = plt.subplot(224)
bin_centers_m_MAT, area_pdf_m_MAT = compute_and_plot_pdf(ax, df_m_MAT.area, 'm, MAT', bandwidth=np.logspace(2, 3, 200))


# plot pdfs side by side

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

''' cell area ECDF estimation
========================================================================================================================
'''


def area_linspace(x, n=100):
    return np.linspace(np.min(x.area), np.max(x.area), n)


area_ecdf_f_PAT = ECDF(df_f_PAT.area)
area_ecdf_f_MAT = ECDF(df_f_MAT.area)
area_ecdf_m_PAT = ECDF(df_m_PAT.area)
area_ecdf_m_MAT = ECDF(df_m_MAT.area)

area_linspace_f_PAT = area_linspace(df_f_PAT)
area_linspace_f_MAT = area_linspace(df_f_MAT)
area_linspace_m_PAT = area_linspace(df_m_PAT)
area_linspace_m_MAT = area_linspace(df_m_MAT)

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

''' compare distributions to show that female cells are smaller than male cells
========================================================================================================================
'''

# Mann–Whitney U test
statistic_f, pvalue_f = stats.mannwhitneyu(df_f_MAT.area, df_f_PAT.area, alternative='less')
statistic_m, pvalue_m = stats.mannwhitneyu(df_m_MAT.area, df_m_PAT.area, alternative='less')

print('females, statistic: ' + "{0:.1f}".format(statistic_f) + ', p-value: ' + "{0:.2e}".format(pvalue_f))
print('males, statistic: ' + "{0:.1f}".format(statistic_m) + ', p-value: ' + "{0:.2e}".format(pvalue_m))

''' measure effect size (um^2) as change in median cell area
========================================================================================================================
'''

# compute effect as difference of the median areas
effect_f = np.median(df_f_MAT.area) - np.median(df_f_PAT.area)
effect_m = np.median(df_m_MAT.area) - np.median(df_m_PAT.area)

# area change
print('Overlap: Median area change from PAT to MAT:')
print('\tFemale: ' +
      "{0:.1f}".format(effect_f) + ' um^2 (' +
      "{0:.1f}".format(effect_f / np.median(df_f_PAT.area) * 100) + '%)')
print('\tMale: ' +
      "{0:.1f}".format(effect_m) + ' um^2 (' +
      "{0:.1f}".format(effect_m / np.median(df_m_PAT.area) * 100) + '%)')

# for the median cells areas, compute radii as if cells were circles
radius_f_MAT = np.sqrt(np.median(df_f_MAT.area) / np.pi)  # (um)
radius_f_PAT = np.sqrt(np.median(df_f_PAT.area) / np.pi)  # (um)
radius_m_MAT = np.sqrt(np.median(df_m_MAT.area) / np.pi)  # (um)
radius_m_PAT = np.sqrt(np.median(df_m_PAT.area) / np.pi)  # (um)

# radius change in percentage
print('Female: Radius change from PAT to MAT: ' +
      "{0:.1f}".format(radius_f_MAT - radius_f_PAT) + ' um ('
      "{0:.1f}".format((radius_f_MAT - radius_f_PAT) / radius_f_PAT * 100) + '%)')
print('Male: Radius change from PAT to MAT: ' +
      "{0:.1f}".format(radius_m_MAT - radius_m_PAT) + ' um ('
      "{0:.1f}".format((radius_m_MAT - radius_m_PAT) / radius_m_PAT * 100) + '%)')

''' study changes in whole cell population by comparing areas in the same percentiles
========================================================================================================================
'''

perc = np.linspace(0, 100, num=101)
perc_area_f_MAT = np.percentile(df_f_MAT.area, perc)
perc_area_f_PAT = np.percentile(df_f_PAT.area, perc)
perc_area_m_MAT = np.percentile(df_m_MAT.area, perc)
perc_area_m_PAT = np.percentile(df_m_PAT.area, perc)

# plot curves comparing cell area change at each percentile
plt.clf()
plt.subplot(211)
plt.plot(perc, (perc_area_f_MAT - perc_area_f_PAT) / perc_area_f_PAT * 100)
plt.title('female', fontsize=20)
plt.xlabel('percentile (%)', fontsize=18)
plt.ylabel('change in cell area size from PAT to MAT (%)', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.subplot(212)
plt.plot(perc, (perc_area_m_MAT - perc_area_m_PAT) / perc_area_m_PAT * 100)
plt.title('male', fontsize=20)
plt.xlabel('percentile (%)', fontsize=18)
plt.ylabel('change in cell area size from PAT to MAT (%)', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)

''' count how many windows and animals each percentile comes from
========================================================================================================================
'''


def count_windows_animals_in_perc(x, perc):

    # create a bin around each percentile, including a first bin that starts at -inf, and last bin that ends at inf
    bin_edges = np.concatenate(([-np.Inf], (perc[0:-1]+perc[1:])/2, [np.Inf]))

    count_cells = []
    count_windows = []
    count_animals = []

    # loop bins
    for i in range(len(perc)):

        # get cells that belong in current bin according to their area
        x_bin = x[np.logical_and(x.area >= bin_edges[i], x.area < bin_edges[i+1])]

        # count number of cells
        count_cells.append(x_bin.shape[0])

        # count number of different windows those cells come from
        count_windows.append(len(np.unique(x_bin.image_id)))

        # count number of different animals those cells come from
        count_animals.append(len(np.unique(x_bin.mouse_id)))

    return count_cells, count_windows, count_animals


# create larger bins
perc = np.linspace(0, 100, num=21)
perc_area_f_MAT = np.percentile(df_f_MAT.area, perc)
perc_area_f_PAT = np.percentile(df_f_PAT.area, perc)
perc_area_m_MAT = np.percentile(df_m_MAT.area, perc)
perc_area_m_PAT = np.percentile(df_m_PAT.area, perc)

count_cells_f_MAT, count_windows_f_MAT, count_animals_f_MAT = count_windows_animals_in_perc(df_f_MAT, perc_area_f_MAT)
count_cells_f_PAT, count_windows_f_PAT, count_animals_f_PAT = count_windows_animals_in_perc(df_f_PAT, perc_area_f_PAT)
count_cells_m_MAT, count_windows_m_MAT, count_animals_m_MAT = count_windows_animals_in_perc(df_m_MAT, perc_area_m_MAT)
count_cells_m_PAT, count_windows_m_PAT, count_animals_m_PAT = count_windows_animals_in_perc(df_m_PAT, perc_area_m_PAT)

# plot bar charts with number of counts
plt.clf()

plt.subplot(421)
plt.bar(perc, count_cells_f_MAT, width=5, edgecolor='black')
plt.legend(('cells',))
plt.ylabel('f/MAT', fontsize=20)
plt.subplot(422)
plt.bar(perc, count_windows_f_MAT, width=5, edgecolor='black')
plt.bar(perc, count_animals_f_MAT, width=2.5, edgecolor='black')
plt.legend(('windows', 'animals'))

plt.subplot(423)
plt.bar(perc, count_cells_f_PAT, width=5, edgecolor='black')
plt.legend(('cells',))
plt.ylabel('f/PAT', fontsize=20)
plt.subplot(424)
plt.bar(perc, count_windows_f_PAT, width=5, edgecolor='black')
plt.bar(perc, count_animals_f_PAT, width=2.5, edgecolor='black')
plt.legend(('windows', 'animals'))

plt.subplot(425)
plt.bar(perc, count_cells_m_MAT, width=5, edgecolor='black')
plt.legend(('cells',))
plt.ylabel('m/MAT', fontsize=20)
plt.subplot(426)
plt.bar(perc, count_windows_m_MAT, width=5, edgecolor='black')
plt.bar(perc, count_animals_m_MAT, width=2.5, edgecolor='black')
plt.legend(('windows', 'animals'))

plt.subplot(427)
plt.bar(perc, count_cells_m_PAT, width=5, edgecolor='black')
plt.legend(('cells',))
plt.ylabel('m/PAT', fontsize=20)
plt.xlabel('Population percentile (%)', fontsize=18)
plt.subplot(428)
plt.bar(perc, count_windows_m_PAT, width=5, edgecolor='black')
plt.bar(perc, count_animals_m_PAT, width=2.5, edgecolor='black')
plt.legend(('windows', 'animals'))
plt.xlabel('Population percentile (%)', fontsize=18)

''' Linear Mixed Effects Model analysis (boxcox_area ~ sex + ko + random(mouse|window))
========================================================================================================================
'''

print('Normality tests:')
print('===========================================================')
print('area_f: ' + str(stats.normaltest(df_f.area)))
print('area_m: ' + str(stats.normaltest(df_m.area)))
print('sqrt(area_f): ' + str(stats.normaltest(np.sqrt(df_f.area))))
print('sqrt(area_m): ' + str(stats.normaltest(np.sqrt(df_m.area))))
print('log10(area_f): ' + str(stats.normaltest(np.log10(df_f.area))))
print('log10(area_m): ' + str(stats.normaltest(np.log10(df_m.area))))

# Box-Cox transformation of areas (scaled from m^2 to um^2 to avoid numerical errors with the lambda parameter) to make
# data normal
df = df.assign(boxcox_area=stats.boxcox(np.sqrt(df.area * 1e12))[0])
df_m = df_m.assign(boxcox_area=stats.boxcox(np.sqrt(df_m.area * 1e12))[0])

if DEBUG:
    # show that data is now normal
    plt.clf()
    ax = plt.subplot(311)
    prob = stats.probplot(df.area * 1e12, dist=stats.norm, plot=ax)
    plt.title(r'Areas ($\mu m^2$)')
    ax = plt.subplot(312)
    prob = stats.probplot(df.boxcox_area, dist=stats.norm, plot=ax)
    plt.title(r'Box Cox transformation of areas ($\mu m^2$)')
    ax = plt.subplot(313)
    plt.hist(df.boxcox_area)
    plt.tight_layout()

    # show that data is now normal
    plt.clf()
    ax = plt.subplot(311)
    prob = stats.probplot(df_m.area * 1e12, dist=stats.norm, plot=ax)
    plt.title(r'Areas ($\mu m^2$)')
    ax = plt.subplot(312)
    prob = stats.probplot(df_m.boxcox_area, dist=stats.norm, plot=ax)
    plt.title(r'Box Cox transformation of areas ($\mu m^2$)')
    ax = plt.subplot(313)
    plt.hist(df_m.boxcox_area)
    plt.tight_layout()

# for the mixed-effects linear model, we want the KO variable to be ordered, so that it's PAT=0, MAT=1 in terms of
# genetic risk, and the sex variable to be ordered in the sense that males have larger cells than females
df['ko'] = df['ko'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))
df['sex'] = df['sex'].astype(pd.api.types.CategoricalDtype(categories=['f', 'm'], ordered=True))
df_m['ko'] = df_m['ko'].astype(pd.api.types.CategoricalDtype(categories=['PAT', 'MAT'], ordered=True))

# Mixed-effects linear model
vc = {'image_id': '0 + C(image_id)'}  # image_id is a random effected nested inside mouse_id
md = smf.mixedlm('boxcox_area ~ sex + ko', vc_formula=vc, re_formula='1', groups='mouse_id', data=df)
mdf = md.fit()
print(mdf.summary())

# Mixed-effects linear model for only males
vc = {'image_id': '0 + C(image_id)'}  # image_id is a random effected nested inside mouse_id
md = smf.mixedlm('boxcox_area ~ ko', vc_formula=vc, re_formula='1', groups='mouse_id', data=df)
mdf = md.fit()
print(mdf.summary())

''' Logistic regression Mixed Effects Model analysis (thresholded_area ~ sex + ko + (1|mouse_id/image_id))
========================================================================================================================
'''

# suggested by George Nicholson's


from rpy2.robjects import r


def r_lme4_glmer(formula, df, family=r('binomial(link="logit")')):

    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    base = importr('base')
    lme4 = importr('lme4')

    pandas2ri.activate()
    r_df = pandas2ri.py2ri(df)

    #control = r('glmerControl(optimizer="Nelder_Mead")')
    control = r('glmerControl(optimizer="bobyqa")')
    model = lme4.glmer(formula, data=r_df, family=family, control=control)
    return base.summary(model)


# threshold values
threshold = np.linspace(np.min(df.area), np.max(df.area), 101)

# loop thresholds
lme4_ko_coeff = np.empty(shape=(len(threshold)))
lme4_sex_coeff = np.empty(shape=(len(threshold)))
lme4_ko_pval = np.empty(shape=(len(threshold)))
lme4_sex_pval = np.empty(shape=(len(threshold)))
lme4_ko_coeff[:] = np.nan
lme4_sex_coeff[:] = np.nan
lme4_ko_pval[:] = np.nan
lme4_sex_pval[:] = np.nan
for i, thr in enumerate(threshold):

    # binarise output variable depending on whether cell size is smaller or larger than the threshold
    df = df.assign(thresholded_area=df.area >= thr)

    # compute GLMM
    try:
        lme4_output = r_lme4_glmer('thresholded_area ~ sex + ko + (1|mouse_id/image_id)', df)
    except:
        continue

    if DEBUG:
        print(lme4_output)

    lme4_sex_coeff[i] = lme4_output.rx2('coefficients')[1]
    lme4_ko_coeff[i] = lme4_output.rx2('coefficients')[2]
    lme4_sex_pval[i] = lme4_output.rx2('coefficients')[10]
    lme4_ko_pval[i] = lme4_output.rx2('coefficients')[11]

# plot coefficients and p-values
plt.clf()
plt.subplot(221)
plt.plot(threshold * 1e12, lme4_sex_coeff)
plt.ylabel(r'$\beta_{sex}$', fontsize=18)
plt.title('sex', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.subplot(222)
plt.plot(threshold * 1e12, lme4_ko_coeff)
plt.ylabel(r'$\beta_{ko}$', fontsize=18)
plt.title('ko', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.subplot(223)
plt.semilogy(threshold * 1e12, lme4_sex_pval)
plt.xlabel(r'$\tau\ (\mu m^2)$', fontsize=18)
plt.ylabel(r'p-value$_{sex}$', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.subplot(224)
plt.semilogy(threshold * 1e12, lme4_ko_pval)
plt.xlabel(r'$\tau\ (\mu m^2)$', fontsize=18)
plt.ylabel(r'p-value$_{ko}$', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()

''' No-overlap analysis: compare overlapping segmentation areas to non-overlapping segmentation areas
========================================================================================================================
'''

# Mann–Whitney U tests to check that each distribution from the overlapping areas is different to the distribution from
# the non-overlapping areas
statistic_f_MAT, pvalue_f_MAT = stats.mannwhitneyu(df_f_MAT.area, df_no_f_MAT.area, alternative='two-sided')
statistic_f_PAT, pvalue_f_PAT = stats.mannwhitneyu(df_f_PAT.area, df_no_f_PAT.area, alternative='two-sided')
statistic_m_MAT, pvalue_m_MAT = stats.mannwhitneyu(df_m_MAT.area, df_no_m_MAT.area, alternative='two-sided')
statistic_m_PAT, pvalue_m_PAT = stats.mannwhitneyu(df_m_PAT.area, df_no_m_PAT.area, alternative='two-sided')

print('f/MAT, statistic: ' + "{0:.1f}".format(statistic_f_MAT) + ', p-value: ' + "{0:.2e}".format(pvalue_f_MAT))
print('f/PAT, statistic: ' + "{0:.1f}".format(statistic_f_PAT) + ', p-value: ' + "{0:.2e}".format(pvalue_f_PAT))
print('m/MAT, statistic: ' + "{0:.1f}".format(statistic_m_MAT) + ', p-value: ' + "{0:.2e}".format(pvalue_m_MAT))
print('m/PAT, statistic: ' + "{0:.1f}".format(statistic_m_PAT) + ', p-value: ' + "{0:.2e}".format(pvalue_m_PAT))

# compute ECDFs for non-overlap areas
area_no_ecdf_f_MAT = ECDF(df_no_f_MAT.area)
area_no_ecdf_f_PAT = ECDF(df_no_f_PAT.area)
area_no_ecdf_m_MAT = ECDF(df_no_m_MAT.area)
area_no_ecdf_m_PAT = ECDF(df_no_m_PAT.area)

area_no_linspace_f_PAT = area_linspace(df_no_f_PAT)
area_no_linspace_f_MAT = area_linspace(df_no_f_MAT)
area_no_linspace_m_PAT = area_linspace(df_no_m_PAT)
area_no_linspace_m_MAT = area_linspace(df_no_m_MAT)

# compute effect as difference of the median areas
effect_no_f_PAT = np.median(df_no_f_PAT.area) - np.median(df_f_PAT.area)
effect_no_f_MAT = np.median(df_no_f_MAT.area) - np.median(df_f_MAT.area)
effect_no_m_PAT = np.median(df_no_m_PAT.area) - np.median(df_m_PAT.area)
effect_no_m_MAT = np.median(df_no_m_MAT.area) - np.median(df_m_MAT.area)

# area change
print('Median area change from overlap to non-overlap:')
print('\tf/PAT: ' +
      "{0:.1f}".format(effect_no_f_PAT) + ' um^2 (' +
      "{0:.1f}".format(effect_no_f_PAT / np.median(df_no_f_PAT.area) * 100) + '%)')
print('\tf/MAT: ' +
      "{0:.1f}".format(effect_no_f_MAT) + ' um^2 (' +
      "{0:.1f}".format(effect_no_f_MAT / np.median(df_no_f_MAT.area) * 100) + '%)')
print('\tm/PAT: ' +
      "{0:.1f}".format(effect_no_m_PAT) + ' um^2 (' +
      "{0:.1f}".format(effect_no_m_PAT / np.median(df_no_m_PAT.area) * 100) + '%)')
print('\tm/MAT: ' +
      "{0:.1f}".format(effect_no_m_MAT) + ' um^2 (' +
      "{0:.1f}".format(effect_no_m_MAT / np.median(df_no_m_MAT.area) * 100) + '%)')

# plot to compare ECDFs of distributions
plt.clf()
plt.subplot(221)
plt.plot(area_linspace_f_PAT, area_ecdf_f_PAT(area_linspace_f_PAT))
plt.plot(area_no_linspace_f_PAT, area_no_ecdf_f_PAT(area_no_linspace_f_PAT))
plt.legend(('overlap', 'no overlap'))
plt.ylabel('Probability')
plt.title('f/PAT')
plt.subplot(222)
plt.plot(area_linspace_f_MAT, area_ecdf_f_MAT(area_linspace_f_MAT))
plt.plot(area_no_linspace_f_MAT, area_no_ecdf_f_MAT(area_no_linspace_f_MAT))
plt.legend(('overlap', 'no overlap'))
plt.title('f/MAT')
plt.subplot(223)
plt.plot(area_linspace_m_PAT, area_ecdf_m_PAT(area_linspace_m_PAT))
plt.plot(area_no_linspace_m_PAT, area_no_ecdf_m_PAT(area_no_linspace_m_PAT))
plt.legend(('overlap', 'no overlap'))
plt.xlabel(r'Area ($\mu m^2$)')
plt.ylabel('Probability')
plt.title('m/PAT')
plt.subplot(224)
plt.plot(area_linspace_m_MAT, area_ecdf_m_MAT(area_linspace_m_MAT))
plt.plot(area_no_linspace_m_MAT, area_no_ecdf_m_MAT(area_no_linspace_m_MAT))
plt.legend(('overlap', 'no overlap'))
plt.xlabel(r'Area ($\mu m^2$)')
plt.title('m/MAT')

''' No-overlap analysis: measure effect size (um^2) as change in median cell area
========================================================================================================================
'''

# compute effect as difference of the median areas
effect_no_f = np.median(df_no_f_MAT.area) - np.median(df_no_f_PAT.area)
effect_no_m = np.median(df_no_m_MAT.area) - np.median(df_no_m_PAT.area)

# area change
print('No-overlap: Median area change from PAT to MAT:')
print('\tFemale: ' +
      "{0:.1f}".format(effect_no_f) + ' um^2 (' +
      "{0:.1f}".format(effect_no_f / np.median(df_no_f_PAT.area) * 100) + '%)')
print('\tMale: ' +
      "{0:.1f}".format(effect_no_m) + ' um^2 (' +
      "{0:.1f}".format(effect_no_m / np.median(df_no_m_PAT.area) * 100) + '%)')

''' No-overlap analysis: study changes in whole cell population by comparing areas in the same percentiles
========================================================================================================================
'''

perc_no = np.linspace(0, 100, num=101)
perc_area_no_f_MAT = np.percentile(df_no_f_MAT.area, perc_no)
perc_area_no_f_PAT = np.percentile(df_no_f_PAT.area, perc_no)
perc_area_no_m_MAT = np.percentile(df_no_m_MAT.area, perc_no)
perc_area_no_m_PAT = np.percentile(df_no_m_PAT.area, perc_no)

# plot curves comparing cell area change at each percentile
plt.clf()
plt.subplot(211)
plt.plot(perc_no, (perc_area_no_f_MAT - perc_area_no_f_PAT) / perc_area_no_f_PAT * 100)
plt.title('female', fontsize=20)
plt.xlabel('percentile (%)', fontsize=18)
plt.ylabel('change in non-overlap cell area size from PAT to MAT (%)', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
ax = plt.subplot(212)
plt.plot(perc_no, (perc_area_no_m_MAT - perc_area_no_m_PAT) / perc_area_no_m_PAT * 100)
plt.title('male', fontsize=20)
plt.xlabel('percentile (%)', fontsize=18)
plt.ylabel('change in non-overlap cell area size from PAT to MAT (%)', fontsize=16)
ax.set_ylim(-30, 0)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()

# plot curves from overlap and no overlap areas together
plt.clf()
plt.subplot(211)
plt.plot(perc_no, (perc_area_no_f_MAT - perc_area_no_f_PAT) / perc_area_no_f_PAT * 100)
plt.plot(perc, (perc_area_f_MAT - perc_area_f_PAT) / perc_area_f_PAT * 100)
plt.legend(('no overlap', 'overlap'))
plt.title('female', fontsize=20)
plt.xlabel('percentile (%)', fontsize=18)
plt.ylabel('change in non-overlap cell area size\nfrom PAT to MAT (%)', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
ax = plt.subplot(212)
plt.plot(perc_no, (perc_area_no_m_MAT - perc_area_no_m_PAT) / perc_area_no_m_PAT * 100)
plt.plot(perc, (perc_area_m_MAT - perc_area_m_PAT) / perc_area_m_PAT * 100)
plt.legend(('no overlap', 'overlap'))
plt.title('male', fontsize=20)
plt.xlabel('percentile (%)', fontsize=18)
plt.ylabel('change in non-overlap cell area size\nfrom PAT to MAT (%)', fontsize=16)
ax.set_ylim(-55, 5)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
