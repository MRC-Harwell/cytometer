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

DEBUG = False

image_data_dir = '/home/rcasero/data/roger_data'
root_data_dir = '/home/rcasero/Dropbox/klf14'
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')


# Area of Polygon using Shoelace formula
# http://en.wikipedia.org/wiki/Shoelace_formula
# FB - 20120218
# corners must be ordered in clockwise or counter-clockwise direction
def polygon_area(corners):
    n = len(corners) # of corners
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

# extract the polygon from the path object, and compute polygon area
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

cell_areas = {'f': {'MAT': np.empty(0), 'PAT': np.empty(0)},
              'm': {'MAT': np.empty(0), 'PAT': np.empty(0)}}

file_list = glob.glob(os.path.join(training_data_dir, '*.svg'))
for file in file_list:

    # get mouse ID from the file name
    mouse_id = None
    for x in klf14_ids:
        if x in os.path.basename(file):
            mouse_id = x
            break
    if mouse_id is None:
        raise ValueError('Filename does not seem to correspond to any known mouse ID: ' + file)

    # index of mouse ID
    idx = klf14_ids.index(mouse_id)

    # sex and KO-side for this mouse
    mouse_sex = klf14_info[idx]['sex']
    mouse_ko  = klf14_info[idx]['ko']


## boxplots of each image

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
    mouse_ko  = klf14_info[idx]['ko']

    # compute areas of all non-edge cells
    areas = extract_cell_contour_and_compute_area(file, x_res=x_res, y_res=y_res)

    # area, image id, mouse id, sex, KO
    for a in areas:
        df = df.append({'area': a, 'mouse_id': mouse_id, 'sex': mouse_sex, 'ko': mouse_ko, 'image_id': image_id},
                       ignore_index=True)


# plot boxplots for each individual image
df.boxplot(column='area', by='image_id', vert=False)


## boxplots comparing MAT/PAT and f/m

area_f = df.loc[df.sex == 'f', ('area', 'ko')]
area_m = df.loc[df.sex == 'm', ('area', 'ko')]

# make sure that in the boxplots PAT comes before MAT
area_f['ko'] = area_f['ko'].astype(pd.api.types.CategoricalDtype(categories=["PAT", "MAT"], ordered=True))
area_m['ko'] = area_m['ko'].astype(pd.api.types.CategoricalDtype(categories=["PAT", "MAT"], ordered=True))

# scale area values to um^2
area_f['area'] *= 1e12
area_m['area'] *= 1e12

# plot boxplots
plt.clf()
ax = plt.subplot(121)
area_f.boxplot(column='area', by='ko', ax=ax, notch=True)
ax.set_ylim(0, 2e4)
ax.set_title('female')
ax.set_xlabel('')
ax.set_ylabel('area (um^2)')
ax = plt.subplot(122)
area_m.boxplot(column='area', by='ko', ax=ax, notch=True)
ax.set_ylim(0, 2e4)
ax.set_title('male')
ax.set_xlabel('')
ax.set_ylabel('area (um^2)')

# plot boxplots without outliers
plt.clf()
ax = plt.subplot(121)
area_f.boxplot(column='area', by='ko', ax=ax, showfliers=False, notch=True)
ax.set_ylim(0, 1e4)
ax.set_title('female')
ax.set_xlabel('')
ax.set_ylabel('area (um^2)')
ax = plt.subplot(122)
area_m.boxplot(column='area', by='ko', ax=ax, showfliers=False, notch=True)
ax.set_ylim(0, 1e4)
ax.set_title('male')
ax.set_xlabel('')
ax.set_ylabel('area (um^2)')


# split dataset into groups
area_f_MAT = area_f['area'][area_f['ko'] == 'MAT']
area_f_PAT = area_f['area'][area_f['ko'] == 'PAT']
area_m_MAT = area_m['area'][area_m['ko'] == 'MAT']
area_m_PAT = area_m['area'][area_m['ko'] == 'PAT']


# function to estimate PDF of areas
def compute_and_plot_pdf(ax, bin_edges, area, title):
    # compute optimal bandwidth
    params = {'bandwidth': np.logspace(-1, 3, 200)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(area[:, np.newaxis])
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
    ax.set_xlabel('area (um^2)')
    ax.set_ylabel('pdf')
    plt.title(title)

    return bin_centers, area_pdf


## plot estimated pdfs separated by f/m, MAT/PAT
plt.clf()

ax = plt.subplot(221)
bin_centers_f_PAT, area_pdf_f_PAT = compute_and_plot_pdf(ax, bin_edges_f_PAT, area_f_PAT, 'f, PAT')

ax = plt.subplot(223)
bin_centers_f_MAT, area_pdf_f_MAT = compute_and_plot_pdf(ax, bin_edges_f_MAT, area_f_MAT, 'f, MAT')

ax = plt.subplot(222)
bin_centers_m_PAT, area_pdf_m_PAT = compute_and_plot_pdf(ax, bin_edges_m_PAT, area_m_PAT, 'm, PAT')

ax = plt.subplot(224)
bin_centers_m_MAT, area_pdf_m_MAT = compute_and_plot_pdf(ax, bin_edges_m_MAT, area_m_MAT, 'm, MAT')



## plot pdfs

plt.clf()

ax = plt.subplot(211)
plt.plot(bin_centers_f_PAT, np.exp(area_pdf_f_PAT))
plt.plot(bin_centers_f_MAT, np.exp(area_pdf_f_MAT))
plt.legend(('PAT', 'MAT'))
ax.set_xlabel('area (um^2)')
ax.set_ylabel('pdf')
plt.title('female')
ax.set_xlim(0, 20000)


ax = plt.subplot(212)
plt.plot(bin_centers_m_PAT, np.exp(area_pdf_m_PAT))
plt.plot(bin_centers_m_MAT, np.exp(area_pdf_m_MAT))
plt.legend(('PAT', 'MAT'))
ax.set_xlabel('area (um^2)')
ax.set_ylabel('pdf')
plt.title('male')
ax.set_xlim(0, 20000)
