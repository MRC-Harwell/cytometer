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

DEBUG = False

# Showing the file

test_name = os.path.join(home, 'Data/cytometer_data/c3h/c3h_hfd_training_augmented/dmap_seed_nan_C3HHM 3095.5c 368-16 C1 - 2016-04-21 16.04.38_row_012780_col_018892.tif')
test = Image.open(test_name)
image = np.array(test)
plt.clf()
plt.subplot(211)
plt.imshow(image)

# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background

local_maxi = peak_local_max(image=image, min_distance=40, indices=True)

plt.plot(local_maxi[:, 1], local_maxi[:, 0], 'xr')

local_maxi = peak_local_max(image=image, min_distance=15, indices=False)

markers = ndi.label(local_maxi)[0]

plt.subplot(212)
plt.imshow(markers)

labels = watershed(-image, markers)

plt.subplot(212)
plt.cla()
plt.imshow(labels.astype('uint32'))

pixel_res = 1e-12
unique_cells = np.unique(labels, return_counts=True)
cell_number = len(unique_cells[0])
x = unique_cells[0]
area_list = unique_cells[1]*pixel_res
bin_size = (max(area_list) - min(area_list))/cell_number

plt.clf()
plt.hist(area_list, bins=np.arange(min(area_list), max(area_list) + bin_size, bin_size))
plt.title("Cell Areas")
plt.ylabel("Frequency")
plt.xlabel("Cell Area (m^2)")
