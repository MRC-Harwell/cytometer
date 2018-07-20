from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


DEBUG = False


def load_watershed_seg_and_compute_dmap(seg_file_list, background_label=1):
    '''
    Loads a list of segmentation files and computes distance maps to the objects boundaries/background.

    The segmentation file is assumed to have one integer label (2, 3, 4, ...) per object. The background has label 1.
    The boundaries between objects have label 0 (if boundaries exist).

    Those boundaries==0 are important for this function, because distance maps are computed as the distance of any
    pixel to the closest pixel=0.

    As an example, the watershed method will produce boundaries==0 when expanding segmentation labels.
    :param seg_file_list: list of paths to the files with segmentations, in any format that PIL can load
    :param background_label: label that will be considered to correspond to the background and not an object
    :return: (dmap, mask, seg)
        dmap: np.array with one distance map per segmentation file. It provides the Euclidean distance of each pixel
              to the closest background/boundary pixel.
        mask: np.array with one segmentation mask per file. Background pixels = 0. Foreground/boundary pixels = 1.
        seg: np.array with one segmentation per file. The segmentation labels in the input files.
    '''

    if not isinstance(seg_file_list, list):
        raise ValueError('seg_file_list must be a list')
    if len(seg_file_list) == 0:
        return np.empty((1, 0)), np.empty((1, 0)), np.empty((1, 0))

    # get size of first segmentation. All segmentations must have the same size
    seg0 = Image.open(seg_file_list[0])
    seg0_dtype = np.array(seg0).dtype

    # allocate memory for outputs
    seg = np.zeros(shape=(len(seg_file_list), seg0.height, seg0.width), dtype=seg0_dtype)
    mask = np.zeros(shape=(len(seg_file_list), seg0.height, seg0.width), dtype='float32')  # these will be used as weights
    dmap = np.zeros(shape=(len(seg_file_list), seg0.height, seg0.width), dtype='float32')

    # loop segmented files
    for i, seg_file in enumerate(seg_file_list):

        # load segmentation
        seg_aux = np.array(Image.open(seg_file))

        # the mask is 0 where we have background pixels, and 1 everywhere else (foreground)
        mask[i, :, :] = seg_aux != background_label

        # the watershed algorithm sets the background pixels to 1, but in order to compute distance maps, we want them
        # to be 0, so that they become boundaries
        seg_aux[seg_aux == background_label] = 0

        # copy segmentation slice to segmentation array
        seg[i, :, :] = seg_aux

        # plot image
        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(seg_aux, cmap="gray")
            plt.title('Cell labels')
            plt.subplot(222)
            plt.imshow(mask[i, :, :], cmap="gray")
            plt.title('Training weight')

        # compute distance map from very pixel to the closest boundary
        dmap[i, :, :] = ndimage.distance_transform_edt(seg_aux)

        # plot distance map
        if DEBUG:
            plt.subplot(223)
            plt.imshow(dmap[i, :, :])
            plt.title('Distance map')

    # add a dummy channel dimension to comply with Keras format
    mask = mask.reshape((mask.shape + (1,)))
    seg = seg.reshape((seg.shape + (1,)))
    dmap = dmap.reshape((dmap.shape + (1,)))

    return dmap, mask, seg
