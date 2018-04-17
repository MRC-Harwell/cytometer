import os
import openslide
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data_dir = '/home/rcasero/data/roger_data'

for file in os.listdir(data_dir):

    # load file
    im = openslide.OpenSlide(os.path.join(data_dir, file))

    # level for a x4 downsample factor
    level_4 = im.get_best_level_for_downsample(4)

    assert(im.level_downsamples[level_4] == 4.0)

    # get downsampled image
    im_4 = im.read_region(location=(0, 0), level=level_4, size=im.level_dimensions[level_4])

    plt.imshow(im_4)
    plt.pause(.1)

    Kmeans(n_clusters=2, random_state=0).fit()