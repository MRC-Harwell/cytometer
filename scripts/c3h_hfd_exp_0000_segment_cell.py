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

test_name = os.path.join(home, 'Downloads/dmap_test_pred.tif')
test = Image.open(test_name)
image = np.array(test)
plt.imshow(image)
plt.show()



# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
distance = ndi.distance_transform_edt(image, return_distances=True)    # Euclidean distance
min_distance = 50
local_maxi = peak_local_max(distance, min_distance=min_distance, indices=False)
markers = ndi.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=image)

if DEBUG:
    plt.clf()
    plt.imshow(image)
    plt.show()

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral, interpolation='nearest')
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()