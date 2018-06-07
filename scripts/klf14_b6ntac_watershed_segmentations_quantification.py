import os
import glob
import matplotlib.pyplot as plt
from PIL import Image


DEBUG = True

root_data_dir = '/home/rcasero/Dropbox/klf14'
training_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training')

# we are interested only in .tif files for which we created hand segmented contours
file_list = glob.glob(os.path.join(training_data_dir, '*.svg'))

# TODO https://github.com/luispedro/python-image-tutorial/blob/master/Segmenting%20cell%20images%20(fluorescent%20microscopy).ipynb

for file in file_list:

    # DEBUG
    file = file_list[4]

    # change file extension from .svg to .tif
    file = file.replace('.svg', '.tif')

    # load image
    im = Image.open(file)

    if DEBUG:
        plt.clf()
        plt.imshow(im)
