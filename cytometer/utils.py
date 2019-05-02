import warnings
import openslide
import cv2
import numpy as np
import six
import matplotlib.pyplot as plt
from PIL import Image
from statistics import mode
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter
from scipy.sparse import dok_matrix
from scipy.interpolate import splprep, splev
from skimage import measure
from skimage.morphology import watershed
from skimage.future.graph import rag_mean_color
from skimage.measure import regionprops
from skimage.segmentation import clear_border
from skimage.transform import EuclideanTransform, AffineTransform, warp, matrix_transform
from skimage.color import rgb2hsv, hsv2rgb
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix
from mahotas.labeled import borders
import networkx as nx
import keras.backend as K
import keras
import tensorflow as tf
from cytometer.models import change_input_size
from cytometer.CDF_confidence import CDF_error_DKW_band, CDF_error_beta
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
from statsmodels.stats.multitest import multipletests

DEBUG = False


def resize(x, size, resample=Image.NEAREST):
    """
    Resize an image in numpy.ndarray format. PIL is used internally for the resizing.

    :param x: numpy.ndarray (row, col) or (row, col, chan) for colour images.
    :param size: (row, col)-tuple with the output size.
    :param resample: (def Image.NEAREST) An optional resampling filter. This can be one of PIL.Image.NEAREST (use
    nearest neighbour), PIL.Image.BILINEAR (linear interpolation), PIL.Image.BICUBIC (cubic spline interpolation), or
    PIL.Image.LANCZOS (a high-quality downsampling filter).
    :return: numpy.ndarray with the resized image.
    """

    if x.ndim < 2 or x.ndim > 3:
        raise ValueError('x.ndims must be 2 or 3')

    if x.dtype == np.float32:
        if x.ndim == 2:
            # convert to PIL image
            x = Image.fromarray(x)

            # resize image
            x = x.resize(size=size, resample=resample)
        else:
            y = np.zeros(shape=size + (x.shape[2],), dtype=x.dtype)
            for chan in range(x.shape[2]):
                # convert to PIL image
                aux = Image.fromarray(x[:, :, chan])

                # resize image
                y[:, :, chan] = aux.resize(size=size, resample=resample)
            x = y
    else:
        # convert to PIL image
        x = Image.fromarray(x)

        # resize image
        x = x.resize(size=size, resample=resample)

    # convert back to numpy array
    return np.array(x)


def clear_mem():
    """GPU garbage collection in Keras with TensorFlow.

    From Otto Stegmaier and Jeremy Howard.
    https://forums.fast.ai/t/gpu-garbage-collection/1976/5
    """

    K.get_session().close()
    sess = K.get_session()
    sess.close()
    # limit mem
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=cfg))
    return


def paint_labels(labels, paint_labs, paint_values):
    """
    Assign values to pixels in image according to their labels. E.g.

    labels = [2, 2, 2, 3, 0]   paint_labs = [0, 1, 2, 3]   paint_values = [.3, .5., .9, .7]
             [2, 2, 0, 3, 1]
             [2, 2, 0, 0, 3]

    out = paint_labels(labels, paint_labs, paint_values)

    out = [.9, .9, .9, .7, .3]
          [.9, .9, .3, .7, .5]
          [.9, .9, .3, .3, .7]

    :param labels: numpy.ndarray (e.g. segmentation of image). We want to assign a value to each pixel according to its
    label. labels needs to be a type that can be used to index an array, x[labels], e.g. np.int32.
    :param paint_labs: List, tuple or 1-D array of label values, e.g. [2, 5, 6, 7, 10, 13]
    :param paint_values: Same size as paint_labs, with values that will be assigned to the labels, e.g.
    [0.34, 0.12, 0.77, 1.22, -2.34, 0.99]
    :return: numpy.ndarray of the same size as labels, where labels have been replaced by their corresponding values.
    """

    # allocate memory for look-up table. This will make painting each pixel with a value more efficient
    max_lab = np.max([np.max(paint_labs), np.max(labels)])
    lut = np.zeros(shape=(max_lab + 1,), dtype=paint_values.dtype)
    lut.fill(np.nan)

    # populate look-up table with the paint values, so that lut[23] = 4.3 means that pixels with label 23 get painted
    # with value 4.3
    lut[paint_labs] = paint_values

    # paint each pixel with a value according to its label
    return lut[labels]


def rough_foreground_mask(filename, downsample_factor=8.0, dilation_size=25, component_size_threshold=1e5,
                          return_im=False):
    """
    Rough segmentation of large segmentation objects in a microscope image with a format that can be read
    by OpenSlice. The objects are darker than the background.

    The function works by first estimating the colour of the background as the mode of all colours. This assumes
    that background pixels are the most numerous and relatively consistent in colour.

    Foreground pixels are selected as those that are one standard deviation darker than the background mode. Then,
    morphological operators are applied to dilate and connect those pixels. Selected objects are those connected
    components larger than a given threashold (1e5 pixels by default).

    :param filename: Path and filename of the microscope image. This file must be in a format understood by OpenSlice.
    :param downsample_factor: (def 8) For speed, the image will be loaded at this downsampled resolution. This
    downsample factor must exist in the multilevel pyramid in the file.
    :param dilation_size: (def 25) Thresholded foreground pixels will be dilated with a (dilation_size, dilation_size)
    kernel.
    :param component_size_threshold: (def 1e5) Minimum number of pixels to consider a connected component as a
    foreground object.
    :param return_im: (def False) Whether to return also the downsampled image in filename.
    :return:
    seg: downsampled segmentation mask.
    [im_downsampled]: if return_im=True, this is the downsampled image in filename.
    """

    # load file
    im = openslide.OpenSlide(filename)

    # level that corresponds to the downsample factor
    downsample_level = im.get_best_level_for_downsample(downsample_factor)

    if im.level_downsamples[downsample_level] != downsample_factor:
        raise ValueError('File does not contain level with downsample factor ' + str(downsample_factor)
                         + '.\nAvailable levels: ' + str(im.level_downsamples))

    # get downsampled image
    im_downsampled = im.read_region(location=(0, 0), level=downsample_level, size=im.level_dimensions[downsample_level])
    im_downsampled = np.array(im_downsampled)
    im_downsampled = im_downsampled[:, :, 0:3]

    if DEBUG:
        plt.clf()
        plt.subplot(211)
        plt.imshow(im_downsampled)

    # reshape image to matrix with one column per colour channel
    im_downsampled_mat = im_downsampled.copy()
    im_downsampled_mat = im_downsampled_mat.reshape((im_downsampled_mat.shape[0] * im_downsampled_mat.shape[1],
                                                     im_downsampled_mat.shape[2]))

    # background colour
    background_colour = []
    for i in range(3):
        background_colour += [mode(im_downsampled_mat[:, i]), ]
    background_colour_std = np.std(im_downsampled_mat, axis=0)

    # threshold segmentation
    seg = np.ones(im_downsampled.shape[0:2], dtype=bool)
    for i in range(3):
        seg = np.logical_and(seg, im_downsampled[:, :, i] < background_colour[i] - background_colour_std[i])
    seg = seg.astype(dtype=np.uint8)
    seg[seg == 1] = 255

    # dilate the segmentation to fill gaps within tissue
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    seg = cv2.dilate(seg, kernel, iterations=1)
    seg = cv2.erode(seg, kernel, iterations=1)

    # find connected components
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(seg)
    lblareas = stats[:, cv2.CC_STAT_AREA]

    # labels of large components, that we assume correspond to tissue areas
    labels_large = np.where(lblareas > component_size_threshold)[0]
    labels_large = list(labels_large)

    # label=0 is the background, so we remove it
    labels_large.remove(0)

    # only set pixels that belong to the large components
    seg = np.zeros(im_downsampled.shape[0:2], dtype=np.uint8)
    for i in labels_large:
        seg[labels == i] = 255

    # # save segmentation as a tiff file (with ZLIB compression)
    # outfilename = os.path.basename(file)
    # outfilename = os.path.splitext(outfilename)[0] + '_seg'
    # outfilename = os.path.join(seg_dir, outfilename + '.tif')
    # tifffile.imsave(outfilename, seg,
    #                 compress=9,
    #                 resolution=(int(im.properties["tiff.XResolution"]) / downsample_factor,
    #                             int(im.properties["tiff.YResolution"]) / downsample_factor,
    #                             im.properties["tiff.ResolutionUnit"].upper()))

    # plot the segmentation
    if DEBUG:
        plt.subplot(212)
        plt.imshow(seg)

    if return_im:
        return seg, im_downsampled
    else:
        return seg


def get_next_roi_to_process(seg, downsample_factor=1.0, max_window_size=[1001, 1001], border=[65, 65]):
    """
    Find a rectangular region of interest (ROI) within an irregularly-shaped mask to pass to a neural
    network or some other processing algorithm. This function can be called repeatedly to process a whole image.

    The choice of the ROI follows several rules:

      * It cannot be larger than a certain size provided by the user.
      * It will be located on the border of the mask (the ROI tends to go from lower rows to higher rows in the mask).
      * It will contain as many mask pixels as possible.
      * A border can be added, to account for the effective receptive field of the neural network, or the tails of a
        filter.

    The ROI is given as a tuple of coordinates for the top-left and bottom-right corners

        (first_row, last_row, first_col, last_col)

    The function also allows for the mask to be a downsampled version of a larger image. Coordinates for both
    the low-resolution and high-resolution windows are then returned.

    Technical note: In order to find candidates to be the top-left corner of the ROI, we convolve the mask (seg) with
    a horizontal-line kernel (k1) and a vertical-line kernel (k2). We then compute the element-wise product
    y = conv2d(seg, k1) * conv2d(seg, k2). Pixels with y > 0

    :param seg: np.ndarray with downsampled segmentation mask.
    :param downsample_factor: (def 1.0) Scalar factor. seg is assumed to have been downsampled by this factor.
    :param max_window_size: (def [1000, 1000]) Vector with (row, column) size of output window. This is
    a maximum size. If the window were to overflow the image, it gets cropped to the image size.
    :param border: (def (65, 65)) Vector with how many (rows, columns) of the output window are a border around
    the region of interest.
    :return: (first_row, last_row, first_col, last_col). If the histology image is im, the ROI found is
    im[first_row:last_row, first_col:last_col].

    (lores_first_row, lores_last_row, lores_first_col, lores_last_col). ROI in the downsampled segmentation.
    """

    if np.count_nonzero(seg) == 0:
        warnings.warn('Empty segmentation')
        return 0, 0, 0, 0

    # convert to np.array so that we can use algebraic operators
    max_window_size = np.array(max_window_size)
    border = np.array(border)

    # convert segmentation mask to [0, 1]
    seg = (seg != 0).astype('int')

    # approximate measures in the downsampled image (we don't round them)
    lores_max_window_size = max_window_size / downsample_factor
    lores_border = border / downsample_factor

    # kernels that flipped correspond to top line and left line. They need to be pre-flipped
    # because the convolution operation internally flips them (two flips cancel each other)
    kernel_top = np.zeros(shape=np.round(lores_max_window_size - 2 * lores_border).astype('int'))
    kernel_top[int((kernel_top.shape[0] - 1) / 2), :] = 1
    kernel_left = np.zeros(shape=np.round(lores_max_window_size - 2 * lores_border).astype('int'))
    kernel_left[:, int((kernel_top.shape[1] - 1) / 2)] = 1

    from scipy.signal import fftconvolve
    seg_top = np.round(fftconvolve(seg, kernel_top, mode='same'))
    seg_left = np.round(fftconvolve(seg, kernel_left, mode='same'))

    # window detections
    detection_idx = np.nonzero(seg_left * seg_top)

    # set top-left corner of the box = top-left corner of first box detected
    lores_first_row = detection_idx[0][0]
    lores_first_col = detection_idx[1][0]

    # first, we look within a window with the maximum size
    lores_last_row = detection_idx[0][0] + lores_max_window_size[0] - 2 * lores_border[0]
    lores_last_col = detection_idx[1][0] + lores_max_window_size[1] - 2 * lores_border[1]

    # second, if the segmentation is smaller than the window, we reduce the window size
    window = seg[lores_first_row:int(np.round(lores_last_row)), lores_first_col:int(np.round(lores_last_col))]

    idx = np.any(window, axis=1)  # reduce rows size
    last_segmented_pixel_len = np.max(np.where(idx))
    lores_last_row = detection_idx[0][0] + np.min((lores_max_window_size[0] - 2 * lores_border[0],
                                                   last_segmented_pixel_len))

    idx = np.any(window, axis=0)  # reduce cols size
    last_segmented_pixel_len = np.max(np.where(idx))
    lores_last_col = detection_idx[1][0] + np.min((lores_max_window_size[1] - 2 * lores_border[1],
                                                   last_segmented_pixel_len))

    if DEBUG:
        plt.clf()
        plt.subplot(221)
        plt.imshow(seg)
        plt.plot([lores_first_col, lores_last_col, lores_last_col, lores_first_col, lores_first_col],
                 [lores_last_row, lores_last_row, lores_first_row, lores_first_row, lores_last_row], 'red')
        plt.subplot(222)
        plt.imshow(seg_top)
        plt.subplot(223)
        plt.imshow(seg_left)
        plt.subplot(224)
        plt.imshow(seg_top * seg_left)
        plt.plot([lores_first_col, lores_last_col, lores_last_col, lores_first_col, lores_first_col],
                 [lores_last_row, lores_last_row, lores_first_row, lores_first_row, lores_last_row], 'red')

    # add a border around the window
    lores_first_row = np.max([0, lores_first_row - lores_border[0]])
    lores_first_col = np.max([0, lores_first_col - lores_border[1]])

    lores_last_row = np.min([seg.shape[0], lores_last_row + lores_border[0]])
    lores_last_col = np.min([seg.shape[1], lores_last_col + lores_border[1]])

    # convert low resolution indices to high resolution
    first_row = np.int(np.round(lores_first_row * downsample_factor))
    last_row = np.int(np.round(lores_last_row * downsample_factor))
    first_col = np.int(np.round(lores_first_col * downsample_factor))
    last_col = np.int(np.round(lores_last_col * downsample_factor))

    # round down indices in downsampled segmentation
    lores_first_row = int(lores_first_row)
    lores_last_row = int(lores_last_row)
    lores_first_col = int(lores_first_col)
    lores_last_col = int(lores_last_col)

    return (first_row, last_row, first_col, last_col), \
           (lores_first_row, lores_last_row, lores_first_col, lores_last_col)


def principal_curvatures_range_image(img, sigma=10):
    """
    Compute Gaussian, Mean and principal curvatures of an image with depth values. Examples of such images
    are topographic maps, range images, depth maps or distance transformations.

    Any of this images can be projected as a Monge patch, a 2D surface embedded in 3D space, f:U->R^3,
    f(x,y) = (x, y, img(x, y)).

    We use a cubic B-spline in tensor-product form to represent the Monge patch by fitting it to the image.
    The reason is that B-splines can be fitted efficiently (scipy.interpolate.RectBivariateSpline), they
    feature compact support, and as they are polynomial functions, the first and second derivatives can be
    easily computed.

    Using these derivatives, there are formulas from elementary Differential Geometry to compute the
    Gaussian, Normal and principal curvatures of the Monge patch.

    Note that these formulas are very sensitive to noise. Thus, we smooth the input image before fitting
    the spline.

    :param img: 2D numpy.ndarray with the distance/depth/range values. The 2D image corresponds to a Monge
    patch, or 2D surface embedded in 3D space of the form (x, y, img(x, y))
    :param sigma: (def sigma=10) Standard deviation in pixels of Gaussian low-pass filtering of the image.
    For sigma=0, no smoothing is performed.
    :return: K, H, k1, k2 = Gaussian curvature, Mean curvature, principal curvature 1, principal
    curvature 2. Each output is an array of the same size as img, with a curvature value per pixel.
    """

    # Note: here x -> rows, y -> columns

    # low-pass filtering
    img = gaussian_filter(img, sigma=sigma)

    # quadratic B-spline interpolation of the image
    sp = RectBivariateSpline(range(img.shape[0]), range(img.shape[1]), img, kx=3, ky=3, s=0)

    # image derivatives and second derivatives
    hx = sp(range(img.shape[0]), range(img.shape[1]), dx=1, grid=True)
    hy = sp(range(img.shape[0]), range(img.shape[1]), dy=1, grid=True)
    hxx = sp(range(img.shape[0]), range(img.shape[1]), dx=2, grid=True)
    hxy = sp(range(img.shape[0]), range(img.shape[1]), dx=1, dy=1, grid=True)
    hyy = sp(range(img.shape[0]), range(img.shape[1]), dy=2, grid=True)

    if DEBUG:
        plt.clf()
        plt.subplot(421)
        plt.imshow(img)
        plt.title('img')
        plt.xlabel('y')
        plt.ylabel('x')
        plt.subplot(422)
        img_resampled = sp(range(img.shape[0]), range(img.shape[1]), grid=True)
        plt.imshow(img_resampled)
        plt.title('B-spline(img)')
        plt.subplot(423)
        plt.imshow(hx)
        plt.title('hx')
        plt.subplot(424)
        plt.imshow(hy)
        plt.title('hy')
        plt.subplot(425)
        plt.imshow(hxx)
        plt.title('hxx')
        plt.subplot(426)
        plt.imshow(hyy)
        plt.title('hyy')
        plt.subplot(427)
        plt.imshow(hxy)
        plt.title('hxy')
        plt.subplot(427)
        plt.imshow(hxy)
        plt.title('hxy')

    # auxiliary products
    hx2 = hx * hx
    hy2 = hy * hy

    # Gaussian curvature
    K = (hxx * hyy - hxy * hxy) / (1 + hx2 + hy2)**2

    # mean curvature
    H = (1 + hx2) * hyy + (1 + hy2) * hxx - 2 * hx * hy * hxy
    H /= 2 * (1 + hx2 + hy2)**1.5

    # principal curvatures
    H2 = H * H
    k1 = H + np.sqrt(H2 - K)
    k2 = H - np.sqrt(H2 - K)

    if DEBUG:
        plt.clf()
        plt.subplot(321)
        plt.imshow(img)
        plt.title('img')
        plt.xlabel('y')
        plt.ylabel('x')
        plt.subplot(323)
        plt.imshow(K)
        plt.title('Gaussian curvature (K)')
        plt.subplot(324)
        plt.imshow(H)
        plt.title('Mean curvature (H)')
        plt.subplot(325)
        plt.imshow(k1.astype(np.float32))
        plt.title('Principal curvature k1')
        plt.subplot(326)
        plt.imshow(k2.astype(np.float32))
        plt.title('Principal curvature k2')

    return K, H, k1, k2


def segment_dmap_contour(dmap, contour=None, sigma=10, min_seed_object_size=50, border_dilation=0):
    """
    Segment cells from a distance transformation image, and optionally, a contour estimate image.

    This function computes the normal curvature of the dmap seen as a Monge patch. The "valleys" in
    the dmap (the cell contours) correspond to higher normal curvature values.

    If provided, the normal curvature is element-wise multiplied by a prior estimate of where the
    cell contours are. This prior estimate can be computed e.g. with a convolutional neural
    network classifier.

    Then a threshold <= 0 is applied to the curvatures (or weighted curvatures). Those pixels are
    considered to belong to the inside of cells. This usually produces separate "islands", one per
    cell, that don't extend all the way to the cells edges. "Islands" with less that
    min_seed_object_size pixels are removed as noise.

    Next, a watershed algorithm is initialised with the "islands", and run on the (weighted) mean
    curvature image to extend the segmentations all the way to the cell borders. This produces
    one different label per cell.

    The borders of each cell are then extracted, and optionally, dilated with a 3x3 kernel of ones,
    "border_dilation" times.

    :param dmap: numpy.ndarray matrix with distance transformation, distance range image,
    topographic map, etc.
    :param contour: (def None) numpy.ndarray matrix, same size as dmap, with an estimate of where the
    contours are.
    :param sigma: (def 10) Standard deviation in pixels of Gaussian blurring applied to the dmap
    before computing the normal curvature. Because normal curvature is a function of second
    derivatives, slight noise in the dmap gets amplified, and produces poor curvature estimates.
    Gaussian blurring removes part of that noise.
    :param min_seed_object_size: (def 50). Objects with fewer pixels than this value will be discarded.
    :param border_dilation: (def 0) Optional dilation of the watershed borders.
    :return: labels, labels_borders
    """

    # check size of inputs
    if dmap.ndim != 2:
        raise ValueError('dmap array must have 2 dimensions')
    if contour is not None and contour.shape != dmap.shape:
        raise ValueError('if provided, contour must have the same shape as dmap')

    # compute mean curvature from dmap
    _, mean_curvature, _, _ = principal_curvatures_range_image(dmap, sigma=sigma)

    # multiply mean curvature by estimated contours
    if contour is not None:
        contour *= mean_curvature
    else:
        contour = mean_curvature.copy()

    # rough segmentation of inner areas
    labels = (mean_curvature <= 0).astype('uint8')

    # label areas with a different label per connected area
    labels = measure.label(labels)

    # remove very small labels (noise)
    labels_prop = measure.regionprops(labels)
    for j in range(1, np.max(labels)):
        # label of region under consideration is not the same as index j
        lab = labels_prop[j]['label']
        if labels_prop[j]['area'] < min_seed_object_size:
            labels[labels == lab] = 0

    # extend labels using watershed
    labels = watershed(mean_curvature, labels)

    # extract borders of watershed regions for plots
    labels_borders = borders(labels)

    # dilate borders for easier visualization
    if border_dilation > 0:
        kernel = np.ones((3, 3), np.uint8)
        labels_borders = cv2.dilate(labels_borders.astype(np.uint8), kernel=kernel, iterations=border_dilation) > 0

    return labels, labels_borders


def match_overlapping_labels(labels_ref, labels_test, allow_repeat_ref=False):
    """
    Match estimated segmentations to ground truth segmentations and compute Dice coefficients.

    This function takes two segmentations, reference and test, and computes how good each test
    label segmentation is, based on how it overlaps the reference segmentation. In a nutshell,
    we find the reference label best aligned to each test label, and compute the Dice
    coefficient as a similarity measure. Note that labels=0 (background) will be ignored.

    We illustrate this with an example.

    Let one of the labels in the test segmentation be 51.

    This test label partly overlaps 3 labels and the background (0) in the reference segmentation:
    [0, 10, 12, 17].

    The number of pixels in the intersection of 51 with each of the other labels is:
    [53600, 29, 17413, 162]  # compute overlap between estimated and ground truth labels
        if dataset_lab_ref is not None:
            lab_correspondence = match_overlapping_labels(labels_ref=dataset_lab_ref[i, :, :, 0],
                                                          labels_test=dataset_lab_test[i, :, :, 0])


    We ignore the background. Therefore, label 51 is best aligned to label 12.

    Let label 51 contain 20,000 pixels, and label 12, 22,000 pixels.

    The Dice coefficient will be 2 * 17413 / (20000 + 22000) = 0.83.


    After all the test to ref labels pairs have been found, the Dice coefficient values are put in a matrix
    dice[test, ref].

    The largest Dice value is chosen, and the correponding (test, ref) correspondence appended to the output.
    Then test and ref and removed from the matrix, so that they cannot be selected again. The next maximum
    Dice value is selected, and so on.

    :param labels_ref: np.ndarray matrix, some integer type. All pixels with the same label
    correspond to the same object.
    :param labels_test: np.ndarray matrix, some integer type. All pixels with the same label
    correspond to the same object.
    :param allow_repeat_ref: (def False) Flag to allow that reference labels can be assigned multiple times. When False,
    each returned pair is unique. When True, 2+ test labels can correspond to the same ref label.
    :return: structured array out:
     out['lab_test']: (N,) np.ndarray with unique list of labels in the test image.
     out['lab_ref']: (N,) np.ndarray with labels that best align with the test labels.
     out['area_test']: (N,) np.ndarray with area of test label in pixels.
     out['area_ref']: (N,) np.ndarray with area of test label in pixels.
     out['dice']: (N,) np.ndarray with Dice coefficient for each pair of corresponding labels.
    """

    # nomenclature to make array slicing more clear
    test = 0
    ref = 1

    # unique labels in the reference and test images, and number of pixels in each label
    labels_test_unique, labels_test_unique_count = np.unique(labels_test, return_counts=True)
    labels_ref_unique, labels_ref_unique_count = np.unique(labels_ref, return_counts=True)

    # remove 0 labels
    idx = labels_test_unique != 0
    labels_test_unique = labels_test_unique[idx]
    labels_test_unique_count = labels_test_unique_count[idx]

    idx = labels_ref_unique != 0
    labels_ref_unique = labels_ref_unique[idx]
    labels_ref_unique_count = labels_ref_unique_count[idx]

    # look up tables to speed up searching for object sizes
    labels_test_unique_count_lut = np.zeros(shape=(np.max(labels_test_unique) + 1, ),
                                            dtype=labels_test_unique_count.dtype)
    labels_ref_unique_count_lut = np.zeros(shape=(np.max(labels_ref_unique) + 1, ),
                                           dtype=labels_ref_unique_count.dtype)

    labels_test_unique_count_lut[labels_test_unique] = labels_test_unique_count
    labels_ref_unique_count_lut[labels_ref_unique] = labels_ref_unique_count

    # form pairs of values between reference labels and test labels. This is going to
    # produce pairs of all overlapping labels, e.g. if label 5 in the test image
    # overlaps with labels 1, 12 and 4 in the reference image,
    # label_pairs_by_pixel = [..., 5,  5, 5, ...] (test)
    #                        [..., 1, 12, 4, ...] (ref)
    aux = np.stack((labels_test.flatten(), labels_ref.flatten()))  # row 0 = TEST, row 1 = REF
    label_pairs_by_pixel, label_pairs_by_pixel_count = np.unique(aux, axis=1, return_counts=True)

    # remove 0 labels
    idx = np.logical_and(label_pairs_by_pixel[test, :] != 0, label_pairs_by_pixel[ref, :] != 0)
    label_pairs_by_pixel = label_pairs_by_pixel[:, idx]
    label_pairs_by_pixel_count = label_pairs_by_pixel_count[idx]

    # matrix to make searchers of label overlap faster, correspondence[test, ref], and to store Dice coefficient
    # values
    dice = dok_matrix((np.max(labels_test_unique) + 1, np.max(labels_ref_unique) + 1),
                      dtype='float32')

    # iterate pairs of overlapping labels and compute Dice coefficient values
    for i in range(label_pairs_by_pixel.shape[1]):
        lab_test = label_pairs_by_pixel[test, i]
        lab_ref = label_pairs_by_pixel[ref, i]

        # to compute the Dice coefficient we need to know:
        # * |A| number of pixels in the test label
        # * |B| number of pixels in the corresponding ref label
        # * |A ∩ B| = intersection_count: number of pixels in the intersection of both labels
        # DICE = 2 * |A ∩ B| / (|A| + |B|)

        a = labels_test_unique_count_lut[lab_test]
        b = labels_ref_unique_count_lut[lab_ref]
        dice[lab_test, lab_ref] = 2 * label_pairs_by_pixel_count[i] / (a + b)

    # prepare output as structured array
    out = np.zeros((0,), dtype=[('lab_test', labels_test_unique.dtype),
                                ('lab_ref', labels_ref_unique.dtype),
                                ('area_test', np.int64),
                                ('area_ref', np.int64),
                                ('dice', dice.dtype)])

    # starting from the highest Dice values, start finding one-to-one correspondences between test and ref labels
    while dice.nnz > 0:

        # convert Dice matrix to a format with faster slicing
        dice = dice.tocsc()

        # labels with the largest Dice coefficient
        idx = dice.argmax()
        (lab_test, lab_ref) = np.unravel_index(idx, dice.shape)

        # areas of the labels
        area_test = labels_test_unique_count_lut[lab_test]
        area_ref = labels_ref_unique_count_lut[lab_ref]

        # add this pair to the output
        out = np.append(out, np.array([(lab_test, lab_ref, area_test, area_ref, dice[lab_test, lab_ref])],
                                      dtype=out.dtype))

        # convert Dice matrix to a format with faster change of sparsity
        dice = dice.tolil()

        # remove the labels from the matrix so that they cannot be selected again
        dice[lab_test, :] = 0
        if not allow_repeat_ref:
            dice[:, lab_ref] = 0

    # check that all Dice values are in [0.0, 1.0]
    assert(all(out['dice'] >= 0.0) and all(out['dice'] <= 1.0))

    return out


def one_image_per_label(dataset_im, dataset_lab_test, dataset_lab_ref=None,
                        training_window_len=401, smallest_cell_area=804, clear_border_lab=False,
                        smallest_dice=0.0, allow_repeat_ref=False):
    """
    Extract a small image centered on each cell of a dataset according to segmentation labels.

    If ground truth labels are provided, only the best segmentation-ground truth matches are considered.
    Then, Dice coefficient values are computed for the matches.

    :param dataset_im: numpy.ndarray (image, width, height, channel). Histology images.
    :param dataset_lab_test: numpy.ndarray (image, width, height, 1). Instance segmentation of the histology
    to be tested. Each label gives the segmentation of one cell. Not all cells need to have been segmented. Label=0
    corresponds to the background and will be ignored.
    :param dataset_lab_ref: (def None) numpy.ndarray (image, width, height, 1). Ground truth instance segmentation of
    the histology. Each label gives the segmentation of one cell. Not all cells need to have been segmented. Label=0
    corresponds to the background and will be ignored.
    :param training_window_len: (def 401) Each cell will be extracted to a (training_window_len, training_window_len)
    window.
    :param smallest_cell_area: (def 804) Labels with less than smallest_cell_area pixels will be ignored as segmentation
    noise.
    :param clear_border_lab: (def False) Ignore labels that touch the edges of the image.
    :param smallest_dice: (def 0.0) Only when dataset_lab_ref != None. Cells that don't achieve minimum Dice value will
    be ignored. By default, smallest_dice=0.0, so this option will be ignored.
    :param allow_repeat_ref: (def False) Only when dataset_lab_ref != None. Flag to allow that reference labels can be
    assigned multiple times. When False, each returned pair is unique. When True, 2+ test labels can correspond to the
    same ref label.
    :return: If dataset_lab_ref is provided,
    (training_windows, testlabel_windows, index_list, reflabel_windows, dice)

    Otherwise,
    (training_windows, testlabel_windows, index_list)

    * training_windows: numpy.ndarray (N, training_window_len, training_window_len, channel). Small windows extracted
    from the histology. Each window is centered around one of N labelled cells.
    * reflabel_windows: numpy.ndarray (N, training_window_len, training_window_len, 1). The ground truth segmentation
    label or mask for the cell in the training window.
    * testlabel_windows: numpy.ndarray (N, training_window_len, training_window_len, 1). The test segmentation label or
    mask for the cell in the training window.
    * index_list: list of (i, lab_test), where i is the image index, and lab_test is the segmentation label.
    * dice: numpy.ndarray (N,). dice[i] is the Dice coefficient between corresponding each label_windows[i, ...] and its
    corresponding ground truth label.
    """

    # (r,c) size of the image
    n_row = dataset_im.shape[1]
    n_col = dataset_im.shape[2]

    if K.image_data_format() != 'channels_last':
        raise ValueError('Only implemented for K.image_data_format() == \'channels_last\'')

    # check that the sizes match for im and lab (number of channels can be different)
    if dataset_im.shape[0:-1] != dataset_lab_test.shape[0:-1]:
        raise ValueError('dataset_im and dataset_lab_test must have the same number of images, and same (w, h)')

    training_windows_list = []
    testlabel_windows_list = []
    index_list = []
    reflabel_windows_list = []
    dice_list = []
    for i in range(dataset_im.shape[0]):

        if DEBUG:
            print('Image ' + str(i) + '/' + str(dataset_im.shape[0] - 1))  # DEBUG

        # remove labels that are touching the edges of the image
        if clear_border_lab:
            dataset_lab_test[i, :, :, 0] = clear_border(dataset_lab_test[i, :, :, 0])

        # compute overlap between estimated and ground truth labels
        if dataset_lab_ref is not None:
            lab_correspondence = match_overlapping_labels(labels_ref=dataset_lab_ref[i, :, :, 0],
                                                          labels_test=dataset_lab_test[i, :, :, 0],
                                                          allow_repeat_ref=allow_repeat_ref)

        # compute bounding boxes for the testing labels (note that the background 0 label is ignored)
        props_test = regionprops(dataset_lab_test[i, :, :, 0], coordinates='rc')

        for p in props_test:

            if DEBUG:
                print('p[\'label\'] = ' + str(p['label']))  # DEBUG

            # simplify nomenclature
            lab_test = p['label']  # test label under consideration
            if dataset_lab_ref is not None:
                lab_ref = lab_correspondence['lab_ref'][lab_correspondence['lab_test'] == lab_test]  # corresponding ground truth label
                # if reference labels are provided, ignore test labels without a correspondence
                if len(lab_ref) == 0:
                    continue
                assert (len(lab_ref) == 1)  # each test label should have corresponding label
                dice = lab_correspondence['dice'][lab_correspondence['lab_test'] == lab_test]  # Dice coefficient
                assert(len(dice) == 1)  # the test label shouldn't be repeated. Thus, we should get only one Dice coeff here
                lab_ref = lab_ref[0]
                dice = dice[0]

                # ignore test labels that don't have a good enough overlap with a reference label
                if dice < smallest_dice:
                    continue

            # we ignore tiny labels as artifacts, as well as tests labels that have no corresponding ground truth
            if p['area'] < smallest_cell_area or (dataset_lab_ref is not None and lab_ref == 0):
                continue

            # record image index and test label for output
            index_list.append((i, lab_test))

            if DEBUG:
                plt.clf()
                plt.subplot(211)
                plt.imshow(dataset_lab_ref[i, :, :, 0] == lab_ref)
                plt.title('i = ' + str(i) + ', Dice = ' + str(dice))
                plt.ylabel('lab_ref = ' + str(lab_ref))
                plt.subplot(212)
                plt.imshow(dataset_lab_test[i, :, :, 0] == lab_test)
                plt.ylabel('lab_test = ' + str(lab_test))

            # width and height of the test label's bounding box. Taking into account: Bounding box
            # (min_row, min_col, max_row, max_col). Pixels belonging to the bounding box are in the half-open interval
            # [min_row; max_row) and [min_col; max_col).
            bbox_min_row = p['bbox'][0]
            bbox_max_row = p['bbox'][2]
            bbox_min_col = p['bbox'][1]
            bbox_max_col = p['bbox'][3]
            bbox_width = bbox_max_col - bbox_min_col
            bbox_height = bbox_max_row - bbox_min_row

            # padding of cell bbox so that it's centered in the larger bbox (i.e. the training window)
            pad_left = int(np.round((training_window_len - bbox_width) / 2.0))
            pad_bottom = int(np.round((training_window_len - bbox_height) / 2.0))

            # (r,c)-coordinates of the larger bbox within dataset['im'][i, :, :, 0]
            # Note: if the bbox is quite close to an edge, the larger bbox may overflow out of the image
            lbox_min_row = bbox_min_row - pad_bottom
            lbox_max_row = lbox_min_row + training_window_len
            lbox_min_col = bbox_min_col - pad_left
            lbox_max_col = lbox_min_col + training_window_len

            # compute correction to the larger bbox to avoid overflowing the image
            delta_min_row = - lbox_min_row if lbox_min_row < 0 else 0
            delta_max_row = n_row - lbox_max_row if lbox_max_row > n_row else 0
            delta_min_col = - lbox_min_col if lbox_min_col < 0 else 0
            delta_max_col = n_col - lbox_max_col if lbox_max_col > n_col else 0

            # apply the correction
            lbox_min_row += delta_min_row
            lbox_max_row += delta_max_row
            lbox_min_col += delta_min_col
            lbox_max_col += delta_max_col

            # array indices for the training window
            i_min_row = delta_min_row
            i_max_row = training_window_len + delta_max_row
            i_min_col = delta_min_col
            i_max_col = training_window_len + delta_max_col

            # check that the larger bbox we extract from 'im' has the same size as the subarray we target in the
            # training window
            assert(lbox_max_row - lbox_min_row == i_max_row - i_min_row)
            assert(lbox_max_col - lbox_min_col == i_max_col - i_min_col)

            # extract histology window
            training_window = np.zeros(shape=(training_window_len, training_window_len, dataset_im.shape[3]),
                                       dtype=dataset_im.dtype)
            training_window[i_min_row:i_max_row, i_min_col:i_max_col, :] = \
                dataset_im[i, lbox_min_row:lbox_max_row, lbox_min_col:lbox_max_col, :]
            training_windows_list.append(training_window)

            # extract label window
            label_window = np.zeros(shape=(training_window_len, training_window_len, dataset_lab_test.shape[3]),
                                    dtype=np.uint8)
            label_window[i_min_row:i_max_row, i_min_col:i_max_col, 0] = \
                dataset_lab_test[i, lbox_min_row:lbox_max_row, lbox_min_col:lbox_max_col, 0] == lab_test
            testlabel_windows_list.append(label_window)

            # extract reference label window
            if dataset_lab_ref is not None:
                reflabel_window = np.zeros(shape=(training_window_len, training_window_len, dataset_lab_ref.shape[3]),
                                           dtype=np.uint8)
                reflabel_window[i_min_row:i_max_row, i_min_col:i_max_col, 0] = \
                    dataset_lab_ref[i, lbox_min_row:lbox_max_row, lbox_min_col:lbox_max_col, 0] == lab_ref
                reflabel_windows_list.append(reflabel_window)

                # save Dice value for later
                dice_list.append(dice)

            if DEBUG:
                plt.clf()
                plt.subplot(221)
                plt.imshow(dataset_im[i, :, :, :])
                plt.contour(dataset_lab_test[i, :, :, 0] == lab_test, levels=1)
                plt.plot([bbox_min_col, bbox_max_col, bbox_max_col, bbox_min_col, bbox_min_col],
                         [bbox_min_row, bbox_min_row, bbox_max_row, bbox_max_row, bbox_min_row], 'r')
                plt.plot([lbox_min_col, lbox_max_col, lbox_max_col, lbox_min_col, lbox_min_col],
                         [lbox_min_row, lbox_min_row, lbox_max_row, lbox_max_row, lbox_min_row], 'g')
                plt.subplot(222)
                plt.imshow(training_window)
                plt.contour(reflabel_window[:, :, 0], levels=1, colors='green', label='ref')
                plt.contour(label_window[:, :, 0], levels=1, label='test')
                plt.title('Dice = ' + str(round(dice_list[-1], 2)))
                plt.subplot(223)
                plt.imshow(dataset_lab_test[i, :, :, 0] == lab_test)
                plt.subplot(224)
                plt.gca().invert_yaxis()
                plt.contour(reflabel_window[:, :, 0], levels=1, colors='green', label='ref')
                plt.contour(label_window[:, :, 0], levels=1, label='test')

    # convert list to array
    if len(training_windows_list) > 0:
        training_windows_list = np.stack(training_windows_list)
    if len(testlabel_windows_list) > 0:
        testlabel_windows_list = np.stack(testlabel_windows_list)
    if len(index_list) > 0:
        index_list = np.stack(index_list)
    if dataset_lab_ref is not None:
        if len(reflabel_windows_list) > 0:
            reflabel_windows_list = np.stack(reflabel_windows_list)
        if len(dice_list) > 0:
            dice_list = np.stack(dice_list)

    if dataset_lab_ref is None:
        return training_windows_list, testlabel_windows_list, index_list
    else:
        return training_windows_list, testlabel_windows_list, index_list, reflabel_windows_list, dice_list


def quality_model_mask(seg, im=None, quality_model_type='0_1', quality_model_type_param=None):
    """
    Compute masks to apply to cell images for quality network.

    :param seg: np.ndarray with one or more segmentations. The expected shape is one of
    (row, col), (row, col, 1), (n_im, row, col, 1), where n_im=number of images.
    :param im: (def None) np.ndarray with one or more images. The expected shape is one of
    (row, col, n_chan), (row, col, n_chan), (n_im, row, col, n_chan), where n_im=number of images,
    n_chan=number of channels (e.g. for RGB, n_chan=3).
    :param quality_model_type: (def '0_1') String with the type of masking used in the quality network.
    * '0_1': mask: 1 within the segmentation, 0 outside.
    * '0_1_prop_band': mask: 1 within the segmentation, 0 outside. The mask gets eroded or dilated with a band with
        thickness proportional to the equivalent radius of the segmentation.
    * '-1_1': mask: 1 within the segmentation, -1 outside.
    * '-1_1_band': mask: 1 within the segmentation, -1 on outside (75-1)/2 pixel band, 0 beyond the band.
    * '-1_1_prop_band': mask: 1 within the segmentation, -1 on outside 20% equivalent radius thick band, 0 beyond the
        band.
    :param quality_model_type_param: (def None) Parameter for some mask types. If the mask type doesn't require a
    parameter, the parameter will be ignored.
    * '0_1':
        No parameter.
    * '0_1_prop_band':
        Scalar with the relative size of the erosion (inc<0.0) or dilation (inc>0.0) of the mask. E.g. inc=-0.15
        corresponds to an erosion of 15% of the equivalent radius.
    * '-1_1':
        No parameter.
    * '-1_1_band':
        No parameter.
    * '-1_1_prop_band':
        No parameter.
    :return:
    If im is None, return masks.
    If im is not None, return masked_im.
    """

    # if images have shape (row, col, chan), reshape as (1, row, col, chan), so that we can
    # write code for the general case (nim, row, col, chan)
    if seg.ndim == 2:
        seg = np.expand_dims(seg, axis=0)
        seg = np.expand_dims(seg, axis=3)
    elif seg.ndim == 3:
        seg = np.expand_dims(seg, axis=0)
    if (im is not None) and im.ndim == 3:
        im = np.expand_dims(im, axis=0)

    # number of segmentations
    n_seg = seg.shape[0]
    if (im is not None) and im.shape[0] != n_seg:
        raise ValueError('im has different number of images than seg')

    # allocate memory for outputs (we only keep a copy of all the masks if we are going to return
    # them; otherwise, if the output is the masked outputs, we don't need to keep them)
    if im is None:
        mask_all = np.zeros(shape=seg.shape, dtype=np.float32)
    else:
        masked_im = im.copy()

    # loop images
    for j in range(n_seg):

        # compute mask
        if quality_model_type == '0_1':

            # mask: 1 within the segmentation, 0 outside
            mask = seg[j, :, :, 0]

        elif quality_model_type == '0_1_prop_band':
            # mask: 1 within the segmentation, 0 outside
            mask = seg[j, :, :, 0]

            # erode or dilate the mask, if parameter provided
            if quality_model_type_param is not None:
                a = np.count_nonzero(seg[j, :, :, 0])  # segmentation area (pix^2)
                r = np.sqrt(a / np.pi)  # equivalent circle's radius
                len_kernel = int(np.ceil(2 * r * np.abs(quality_model_type_param) + 1))

                kernel = np.ones(shape=(len_kernel, len_kernel))
                if quality_model_type_param < 0:
                    mask = cv2.erode(mask, kernel=kernel)
                elif quality_model_type_param == 0:
                    mask = mask.copy()
                else:  # quality_model_type_param > 0
                    mask = cv2.dilate(mask, kernel=kernel)

        elif quality_model_type == '-1_1':
            # mask: 1 within the segmentation, -1 outside
            mask = 2 * (seg[j, :, :, 0].astype(np.float32) - 0.5)
        elif quality_model_type == '-1_1_band':
            # mask: 1 within the segmentation, -1 on outside (75-1)/2 pixel thick band, 0 beyond the band
            mask = cv2.dilate(seg[j, :, :, 0], kernel=np.ones(shape=(75, 75)))
            mask = - mask.astype(np.float32)
            mask[seg[j, :, :, 0] == 1] = 1
        elif quality_model_type == '-1_1_prop_band':
            # mask: 1 within the segmentation, -1 on outside 20% equivalent radius band, 0 beyond the band
            a = np.count_nonzero(seg[j, :, :, 0])  # segmentation area (pix^2)
            r = np.sqrt(a / np.pi)  # equivalent circle's radius
            len_kernel = int(np.ceil(2 * r * 0.20 + 1))

            mask = cv2.dilate(seg[j, :, :, 0], kernel=np.ones(shape=(len_kernel, len_kernel)))
            mask = - mask.astype(np.float32)
            mask[seg[j, :, :, 0] == 1] = 1
        else:
            raise ValueError('Unrecognised quality_model_type: ' + str(quality_model_type))

        if im is None:
            # save current mask for the output
            mask_all[j, :, :, 0] = mask
        else:
            # mask image with segmentation
            mask = np.expand_dims(mask, axis=2)
            masked_im[j, :, :, :] = im[j, :, :, :] * np.repeat(mask, repeats=im.shape[3], axis=2)

        if DEBUG:
            if quality_model_type == '0_1':
                plt.clf()
                plt.subplot(221)
                plt.imshow(im[j, :, :, :])
                plt.title('Single cell', fontsize=16)
                plt.subplot(222)
                plt.imshow(seg[j, :, :, 0])
                plt.title('Multiplicative mask', fontsize=16)
                plt.subplot(223)
                if np.count_nonzero(masked_im[j, :, :, :] >= 0) > 0:
                    plt.imshow(masked_im[j, :, :, :])
                plt.title('Masked cell', fontsize=16)
            elif quality_model_type == '-1_1':
                plt.clf()
                plt.subplot(221)
                plt.imshow(im[j, :, :, :])
                plt.title('Single cell', fontsize=16)
                plt.subplot(222)
                plt.imshow(seg[j, :, :, 0])
                plt.title('Multiplicative mask', fontsize=16)
                plt.subplot(223)
                if np.count_nonzero(masked_im[j, :, :, :] >= 0) > 0:
                    plt.imshow(masked_im[j, :, :, :] * (masked_im[j, :, :, :] >= 0))
                plt.title('Cell with mask = 1', fontsize=16)
                plt.subplot(224)
                if np.count_nonzero(masked_im[j, :, :, :] < 0) > 0:
                    plt.imshow(-masked_im[j, :, :, :] * (masked_im[j, :, :, :] < 0))
                plt.title('Cell with mask = -1', fontsize=16)
            elif quality_model_type in ['-1_1_band', '-1_1_prop_band']:
                plt.clf()
                plt.subplot(221)
                plt.imshow(im[j, :, :, :])
                plt.title('Single cell', fontsize=16)
                plt.subplot(222)
                plt.imshow(mask[:, :, 0])
                plt.title('Multiplicative mask', fontsize=16)
                plt.subplot(223)
                if np.count_nonzero(mask[:, :, 0] > 0) > 0:
                    plt.imshow(masked_im[j, :, :, :] * (np.repeat(mask, repeats=im.shape[3], axis=2) == 1))
                plt.title('Cell with mask = 1', fontsize=16)
                plt.subplot(224)
                if np.count_nonzero(mask[:, :, 0] < 0) > 0:
                    plt.imshow(-masked_im[j, :, :, :] * (np.repeat(mask, repeats=im.shape[3], axis=2) == -1))
                plt.title('Cell with mask = -1', fontsize=16)

    if im is None:
        return mask_all
    else:
        return masked_im


def bounding_box_with_margin(label, inc=0.0, coordinates='xy'):
    """
    Create a square bounding box around a segmentation mask, with optional enlargement/reduction.
    The output is given as (x0, y0, xend, yend) for plotting or (r0, c0, rend, cend) for indexing arrays.

    Note that because we need integers for indexing, the bounding box may not be completely centered on the segmentation
    mask.

    Note also that the bounding box may have negative indices, or indices beyond the image size.

    :param label: 2D numpy.array with segmentation mask.
    :param inc: (def 0.0) Scalar. The size of the box will be increased (or decreased with negative value) 100*inc%.
    E.g. if inc=0.20, the size of the box will be increased by 20% with respect to the smallest box that encloses the
    segmentation.
    :param coordinates: (def 'xy') 'xy' or 'rc'.
    :return: Coordinates of the bottom left and top right corners of the box.
    If coordinates=='xy': (x0, y0, xend, yend): These are the true coordinates of the box corners, and these values
    can be directly used for plotting.
    If coordinates=='rc': (r0, c0, rend, cend): These are rounded row/column indices that can be used for indexing
    an array, e.g. x[r0:rend, c0:cend]. Note that
        cend = xend+1
        rend = yend+1
    (the +1 is necessary because python drops the last index when slicing an array).
    """

    # compute region properties
    props = regionprops((label != 0).astype(np.uint8), coordinates='rc')
    assert (len(props) == 1)

    # box enclosing segmentation
    bbox = props[0]['bbox']

    # ease nomenclature of box corners
    (bbox_r0, bbox_c0, bbox_rend, bbox_cend) = bbox

    # make the box square
    bbox_r_len = bbox_rend - bbox_r0
    bbox_c_len = bbox_cend - bbox_c0
    bbox_len = np.max((bbox_r_len, bbox_c_len))

    if inc != 0.0:
        # increase or decrease the box size by the scalar
        bbox_len = np.round(bbox_len * (1 + inc))

    # bottom left corner of the box
    bbox_r0 -= np.round((bbox_len - bbox_r_len) / 2.0)
    bbox_c0 -= np.round((bbox_len - bbox_c_len) / 2.0)

    # top right corner of the box (+1 so that we can use it for indexing)
    bbox_rend = bbox_r0 + bbox_len
    bbox_cend = bbox_c0 + bbox_len

    if coordinates == 'xy':
        return np.float64(bbox_c0), np.float64(bbox_r0), np.float64(bbox_cend - 1), np.float64(bbox_rend - 1)
    elif coordinates == 'rc':
        return np.int64(bbox_r0), np.int64(bbox_c0), np.int64(bbox_rend), np.int64(bbox_cend)
    else:
        raise ValueError('Unknown "coordinates" value.')


def extract_bbox(im, bbox):
    """
    Crop bounding box from an image. Note that bounding boxes that go beyond the image boundaries are allowed. In that
    case, external pixels will be set to zero.

    :param im: (row, col) or (row, col, channels) np.ndarray image.
    :param bbox: (r0, c0, rend, cend)-tuple with bottom left and top right vertices of the bounding box.
    :return:
    * out: Cropping of the image, as numpy.ndarray with the same number of channels as im.
    """

    # for simplify code, consider 2D images as images with 1 channel
    if im.ndim < 2 or im.ndim > 3:
        raise ValueError('im must be a (row, col) or (row, col, channel) array')
    elif im.ndim == 2:
        DIM2 = True
        im = np.expand_dims(im, axis=2)
    else:
        DIM2 = False

    # easier nomenclature
    r0, c0, rend, cend = bbox

    # initialise output
    out = np.zeros(shape=(rend - r0, cend - c0, im.shape[2]), dtype=im.dtype)

    # if indices are beyond the image limits, we have to account for that when cropping the input
    if r0 < 0:
        r0_out = -r0
        r0 = 0
    else:
        r0_out = 0
    if c0 < 0:
        c0_out = -c0
        c0 = 0
    else:
        c0_out = 0

    if rend > im.shape[0]:
        rend_out = out.shape[0] - (rend - im.shape[0])
        rend = im.shape[0]
    else:
        rend_out = out.shape[0]
    if cend > im.shape[1]:
        cend_out = out.shape[1] - (cend - im.shape[1])
        cend = im.shape[1]
    else:
        cend_out = out.shape[1]

    # crop the image
    out[r0_out:rend_out, c0_out:cend_out, :] = im[r0:rend, c0:cend, :]

    # output has the same number of dimensions as the input
    if DIM2:
        return out[:, :, 0]
    else:
        return out


def segmentation_pipeline(im, contour_model, dmap_model, quality_model,
                          quality_model_type='0_1', quality_model_preprocessing=None,
                          mask=None, smallest_cell_area=804):
    """
    Instance segmentation of cells using the contour + distance transformation pipeline.

    :param im: numpy.ndarray (image, row, col, channel) with RGB histology images.
    :param contour_model: filename or keras model for the contour detection neural network. This is assumed to
    be a convolutional network where the output has the same (row, col) as the input.
    :param dmap_model: filename or keras model for the distance transformation regression neural network.
    This is assumed to be a convolutional network where the output has the same (row, col) as the input.
    :param quality_model: filename or keras model for the quality coefficient estimation neural network.
    This network processes one cell at a time. Networks trained for different types of masking can be used
    by setting the value of quality_model_type.
    :param quality_model_type: (def '0_1') String with the type of masking used in the quality network.
    * '0_1': mask: 1 within the segmentation, 0 outside.
    * '-1_1': mask: 1 within the segmentation, -1 outside.
    * '-1_1_band': mask: 1 within the segmentation, -1 on outside (75-1)/2 pixel band, 0 beyond the band.
    * '-1_1_prop_band': mask: 1 within the segmentation, -1 on outside 20% equivalent radius thick band, 0 beyond the
      band.
    :param quality_model_preprocessing: (def None) Apply a preprocessing step to the single-cell images before they are
    passed to the Quality Network.
    * 'polar': Convert image from (x,y) to polar (rho, theta)=(cols, rows) coordinates.
    :param mask: (def None) If provided, labels that intersect less than 60% with the mask are ignored.
    The mask can be used to skip segmenting background or another type of tissue.
    :param smallest_cell_area: (def 804) Labels with less than smallest_cell_area pixels will be ignored as
    segmentation noise.
    :return: labels, labels_info

    labels: numpy.ndarray of size (image, row, col, 1). Instance segmentation of im. Each label segments a different
    cell.

    labels_info: numpy structured array. One element per cell.
    labels_info['im']: Each element is the index of the image the cell belongs to.
    labels_info['label']: Each element is the label assigned to the cell in that image.
    labels_info['quality']: Each element is a quality value in [0.0, 1.0] estimating the quality of the segmentation
    for that cell. As a rule of thumb, quality >= 0.9 corresponds to an acceptable segmentation.
    """

    # load model if filename provided
    if isinstance(contour_model, six.string_types):
        contour_model = keras.models.load_model(contour_model)
    if isinstance(dmap_model, six.string_types):
        dmap_model = keras.models.load_model(dmap_model)
    if isinstance(quality_model, six.string_types):
        quality_model = keras.models.load_model(quality_model)

    # set input layer to size of images for the convolutional networks.
    contour_model = change_input_size(contour_model, batch_shape=(None,) + im.shape[1:])
    dmap_model = change_input_size(dmap_model, batch_shape=(None,) + im.shape[1:])

    # This doesn't apply to the quality network because that one needs to collapse to one output value. Thus we
    # read the training window size from the network itself
    training_window_len = quality_model.input_shape[1]
    assert(training_window_len == quality_model.input_shape[2])

    # allocate arrays for labels
    labels = np.zeros(shape=im.shape[0:3] + (1,), dtype='int32')
    labels_borders = np.zeros(shape=im.shape[0:3] + (1,), dtype='bool')

    # run images through networks
    one_im_shape = (1,) + im.shape[1:]
    for i in range(im.shape[0]):

        # process one histology image through the two pipeline branches
        one_im = im[i, :, :, :].reshape(one_im_shape)
        contour_pred = contour_model.predict(one_im)
        dmap_pred = dmap_model.predict(one_im)

        # combine pipeline branches for instance segmentation
        labels[i, :, :, 0], labels_borders[i, :, :, 0] = segment_dmap_contour(dmap_pred[0, :, :, 0],
                                                                              contour=contour_pred[0, :, :, 0],
                                                                              border_dilation=0)

        # remove labels that are too small
        aux = labels[i, :, :, 0]
        props = regionprops(aux)
        for p in props:
            if p.area < smallest_cell_area:
                aux[aux == p.label] = 0

        # remove labels that are not substantially within the mask
        if mask is not None:

            # count number of pixels in each label after we multiply by the mask
            props_masked = regionprops(aux * mask[i, :, :])

            # create a lookup table for quick search of masked label area
            max_label = np.max([p.label for p in props])
            area_masked = np.zeros(shape=(max_label + 1,))
            for p_masked in props_masked:
                area_masked[p_masked.label] = p_masked.area

            # check for each original label whether it is at least 60% covered by the mask
            for p in props:
                if area_masked[p.label] < p.area * 0.60:
                    aux[aux == p.label] = 0

        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(one_im[0, :, :, :])
            plt.subplot(222)
            plt.imshow(contour_pred[0, :, :, 0])
            plt.subplot(223)
            plt.imshow(dmap_pred[0, :, :, 0])
            plt.subplot(224)
            plt.imshow(contour_pred[0, :, :, 0] * dmap_pred[0, :, :, 0])

            plt.clf()
            plt.subplot(121)
            plt.imshow(one_im[0, :, :, :])
            plt.subplot(122)
            plt.imshow(labels[i, :, :, 0])

    # split histology images into individual segmented objects
    # Note: smallest_cell_area should be redundant here, because small labels have been removed
    cell_im, cell_seg, cell_index = one_image_per_label(dataset_im=im,
                                                        dataset_lab_test=labels,
                                                        training_window_len=training_window_len,
                                                        smallest_cell_area=smallest_cell_area)

    # if no cells extracted
    if len(cell_im) == 0:
        return [], []

    # compute mask from segmentation, and mask histology images
    cell_im = quality_model_mask(cell_seg, im=cell_im, quality_model_type=quality_model_type)
    if cell_im.ndim == 3:
        cell_im = np.expand_dims(cell_im, axis=0)

    # preprocess images before feeding to quality network
    if quality_model_preprocessing == 'polar':

        # convert input images to polar coordinates
        # shape = (im, row, col, chan)
        # row = Y, col = X
        center = ((cell_im.shape[2] - 1) / 2.0, (cell_im.shape[1] - 1) / 2.0)
        maxRadius = np.sqrt(cell_im.shape[2] ** 2 + cell_im.shape[1] ** 2) / 2
        for i in range(cell_im.shape[0]):
            cell_im[i, :, :, :] = cv2.linearPolar(cell_im[i, :, :, :], center=center, maxRadius=maxRadius,
                                                  flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

    elif quality_model_preprocessing is not None:

        raise ValueError('Unknown quality_model_preprocessing')

    # compute quality of segmented labels
    quality = quality_model.predict(cell_im)

    if DEBUG:
        i = 4

        plt.clf()
        plt.subplot(221)
        plt.imshow(im[i, :, :, :].reshape(one_im_shape)[0, :, :, :])
        plt.subplot(222)
        plt.imshow(labels[i, :, :, 0])
        plt.subplot(223)
        plt.boxplot(quality)
        plt.subplot(224)
        aux = paint_labels(labels[i, :, :, 0], cell_index[cell_index[:, 0] == i, 1],
                           quality[cell_index[:, 0] == i, 0] >= 0.5)
        plt.imshow(aux)

    # prepare output as structured array
    labels_info = np.zeros((len(quality),), dtype=[('im', cell_index.dtype),
                                                   ('label', cell_index.dtype),
                                                   ('quality', quality.dtype)])
    labels_info['im'] = cell_index[:, 0]
    labels_info['label'] = cell_index[:, 1]
    labels_info['quality'] = quality[:, 0]

    return labels, labels_info


def colour_labels_with_receptive_field(labels, receptive_field):
    """
    Take a segmentation where each object has a different label, and colour them with a distance constraint:

    Let c(i) be the center of mass of label i that we have assigned colour k. If we draw a rectangle of size
    receptive_field around c(i), the only object with colour k within the rectangle is i.

    :param labels: 2D np.ndarray that describes a segmentation. All pixels that correspond to the same object have
    the same label.
    :param receptive_field: (width, height) tuple with the size of the receptive field in pixels. If
    receptive_field is a scalar, then it's used for both the width and height.
    :return: colours, coloured_labels:

    colours is a dictionary with pairs {label: colour}.
    coloured_labels: np.ndarray of the same size as labels, with the labels replaced by colours.
    """

    # receptive_field = (width/cols, height/rows)
    if np.isscalar(receptive_field):
        receptive_field = (receptive_field, receptive_field)
    if not isinstance(receptive_field, tuple):
        raise TypeError('receptive_field must be a scalar or a tuple')

    # constant for clearer nomenclature
    no_colour = 0
    background = 0

    # make receptive_field (height/rows, width/cols)
    receptive_field = receptive_field[::-1]

    # half-sizes of the receptive field rectangle
    if receptive_field[0] % 2:
        receptive_field_half = ((receptive_field[0] - 1) / 2,)
    else:
        receptive_field_half = (receptive_field[0] / 2,)
    if receptive_field[1] % 2:
        receptive_field_half += ((receptive_field[1] - 1) / 2,)
    else:
        receptive_field_half += (receptive_field[1] / 2, )

    # compute the center of mass for each labelled region. E.g. centroids_rc[53, :] is the centroid for label 53.
    # Note: if the centroid is in the center of pixel (row=3, col=6), regionprops gives the centroid as
    # index (3.0, 6.0), instead of the physical centre (3.5, 6.5)
    labels_prop = regionprops(labels, coordinates='rc')
    centroids_rc = {}
    for lp in labels_prop:
        centroids_rc[lp['label']] = lp['centroid']

    # compute Region Adjacency Graph (RAG) for labels. Note that we don't care about the mean colour difference
    # between regions. We only care about whether labels are adjacent to others or not
    rag = rag_mean_color(image=labels, labels=labels)

    # node 0 corresponds to the background, and it's ignored in terms of colouring
    if rag.has_node(background):
        rag.remove_node(background)

    # initially, we colour all nodes as 0 (no colour)
    nx.set_node_attributes(rag, no_colour, 'colour')

    # initialize list of subset nodes that can be considered for colouring.
    # we start with only the first node in the list of nodes in the graph
    v_candidates = set([list(rag)[0]])

    while len(v_candidates) > 0:  # iterate the algorithm until all nodes are labelled

        # we are going to colour the first candidate in the list
        v = v_candidates.pop()

        # add uncoloured neighbours to the list of candidates for the next iteration
        for x in rag[v]:
            if rag.nodes[x]['colour'] == no_colour:
                v_candidates.add(x)

        # (row, col) coordinates of the current vertex
        c = centroids_rc[v]

        # rectangle corners with receptive_field size around current node. Note that if the rectangle goes
        # outside of the image, we have to crop it to avoid an index error
        rmin = int(max(0.0, np.round(c[0] - receptive_field_half[0])))
        rmax = int(min(labels.shape[0] - 1.0, np.round(c[0] + receptive_field_half[0])))
        cmin = int(max(0.0, np.round(c[1] - receptive_field_half[1])))
        cmax = int(min(labels.shape[1] - 1.0, np.round(c[1] + receptive_field_half[1])))

        # extract labels that will interfere with the current vertex at training time
        v_interference = np.unique(labels[rmin:rmax+1, cmin:cmax+1])

        # ignore background pixels (label=0)
        v_interference = v_interference[v_interference != background]

        # get the colours of the interfering labels (if they have any)
        colour_interference = nx.get_node_attributes(rag, 'colour')
        colour_interference = [colour_interference[x] for x in v_interference]

        # remove no_colour from the list of colours
        colour_interference = np.array(colour_interference)
        colour_interference = colour_interference[colour_interference != no_colour]

        # assign the lowest possible colour given the existing colours
        if len(colour_interference) == 0:  # no colours assigned to neighbours
            nx.set_node_attributes(rag, {v: 1}, 'colour')
        else:
            # list of reusable colours between 1 and the highest colour
            max_colour = np.max(colour_interference)
            recyclable_colours = list(set(range(1, max_colour + 1)) - set(colour_interference))

            if len(recyclable_colours) == 0:
                # no recyclable colours, so we have to create a new one
                nx.set_node_attributes(rag, {v: max_colour + 1}, 'colour')
            else:
                # we recycle the lowest colour index for the current node
                nx.set_node_attributes(rag, {v: np.min(recyclable_colours)}, 'colour')

        if DEBUG:
            centroids_xy = {}
            for n in rag.nodes():
                centroids_xy[n] = centroids_rc[n][::-1]

            plt.clf()
            plt.subplot(221)
            plt.imshow(labels)
            plt.title('labels')
            plt.subplot(222)
            plt.imshow(labels)
            nx.draw(rag, pos=centroids_xy, node_size=30)
            plt.title('label adjacency graph')
            plt.subplot(223)
            plt.imshow(labels)
            plt.plot(c[1], c[0], 'or')
            plt.plot([cmin, cmax, cmax, cmin, cmin], [rmin, rmin, rmax, rmax, rmin], 'r')
            plt.title('receptive field')

    # list of coloured nodes
    colours = nx.get_node_attributes(rag, 'colour')

    # transfer colours to labels image
    coloured_labels = labels.copy()
    for lab in colours.keys():
        coloured_labels[labels == lab] = colours[lab]

    if DEBUG:
        plt.clf()
        plt.subplot(221)
        plt.imshow(labels)
        plt.title('labels')
        plt.subplot(222)
        plt.imshow(labels)
        nx.draw(rag, pos=centroids_xy, node_size=30)
        plt.title('label adjacency graph')
        plt.subplot(223)
        plt.imshow(coloured_labels, cmap='tab20')
        c = centroids_rc[38]
        plt.plot(c[1], c[0], 'ok')
        rmin = int(max(0.0, np.round(c[0] - receptive_field_half[0])))
        rmax = int(min(labels.shape[0] - 1.0, np.round(c[0] + receptive_field_half[0])))
        cmin = int(max(0.0, np.round(c[1] - receptive_field_half[1])))
        cmax = int(min(labels.shape[1] - 1.0, np.round(c[1] + receptive_field_half[1])))
        plt.plot([cmin, cmax, cmax, cmin, cmin], [rmin, rmin, rmax, rmax, rmin], 'k')
        plt.title('coloured labels')

    return colours, coloured_labels


def keras2skimage_transform(keras_transform, input_shape, output_shape='same'):
    """
    Convert an affine transform from keras to skimage format. This can then be used to apply a
    transformation to an image (transform_im) or point set (transform_coords).

    Note: Currently, the implemented parameters are:
      * scaling ('zx', 'zy')
      * rotation ('theta')
      * translation ('tx', 'ty')
      * shear ('shear' in counter-clockwise degrees)
      * flip around the x-axis ('flip_vertical')
      * flip around the y-axis ('flip_horizontal')
    Ignored:
      * 'channel_shift_intensity'
      * 'brightness'

    Note 2: Rotations in keras are referred to the centre of the image.
    Rotations in skimage are referred to the origin of coordinates (x, y)=(0, 0).

    :param keras_transform: Dictionary with transform in keras format, as returned by
    keras.get_random_transform. E.g.
    {'theta': -43.1, 'tx': 0, 'ty': 0, 'shear': 4.3, 'zx': 1, 'zy': 1, 'flip_horizontal': 0,
     'flip_vertical': 0, 'channel_shift_intensity': None, 'brightness': None}
    :param input_shape: Tuple with (width, height) size of input image.
    :param output_shape: (def 'same')
        * 'same': The output has the same size as the input. Usually, this will crop out parts of the
                  transformed image.
        * 'full': The output will be large enough to contain the full transformed image. In general,
        the output image won't have the same centre as the output image produced with 'same'.
        * Tuple with (height, width) size of output image. Note: This option produces an output with the
        same centre as 'same'.
    :return:
    * transform_skimage: skimage.transform._geometric.ProjectiveTransform with same affine
    transform.
    * output_shape: (height, width)
    """

    # input image's centre
    im_centre = np.array([(input_shape[1] - 1) / 2, (input_shape[0] - 1) / 2])  # (x, y)

    # move image's centre to origin of coordinates
    transform_skimage_center = EuclideanTransform(translation=-im_centre)

    # affine transformation
    transform_skimage_affine = AffineTransform(matrix=None, scale=(keras_transform['zx'], keras_transform['zy']),
                                               rotation=keras_transform['theta'] / 180.0 * np.pi,
                                               shear=keras_transform['shear'] / 180.0 * np.pi,
                                               translation=(keras_transform['tx'], keras_transform['ty']))

    # horizontal flip
    if keras_transform['flip_horizontal'] == 1:
        transform_skimage_horizontal_flip = EuclideanTransform(np.array([[-1, 0, 0],
                                                                         [0, 1, 0],
                                                                         [0, 0, 1]]))
    else:
        transform_skimage_horizontal_flip = EuclideanTransform(None)

    # vertical flip
    if keras_transform['flip_vertical'] == 1:
        transform_skimage_vertical_flip = EuclideanTransform(np.array([[1, 0, 0],
                                                                       [0, -1, 0],
                                                                       [0, 0, 1]]))
    else:
        transform_skimage_vertical_flip = EuclideanTransform(None)

    # composition of all transformations
    transform_skimage = transform_skimage_center \
                        + transform_skimage_affine \
                        + transform_skimage_horizontal_flip \
                        + transform_skimage_vertical_flip \
                        + transform_skimage_center.inverse

    if output_shape == 'full':

        # input image bounding box's corners
        x0 = 0
        y0 = 0
        xend = input_shape[1] - 1
        yend = input_shape[0] - 1
        corner_coords = np.array([[x0, y0],
                                  [xend, y0],
                                  [xend, yend],
                                  [x0, yend]])

        # apply the same image transformation to the input bounding box
        corner_coords = matrix_transform(corner_coords, transform_skimage.params)  # (x, y)

        # bounding box containing the whole output image
        bbox_out = np.array([np.min(corner_coords, axis=0), np.max(corner_coords, axis=0)])  # (x, y)

        # bottom-left and top-right corners of the output bounding box
        bbox_out_min = np.min(bbox_out, axis=0)  # (x, y)
        bbox_out_max = np.max(bbox_out, axis=0)  # (x, y)

        # size of the output image is the size of the output bounding box
        output_shape = np.ceil(bbox_out_max - bbox_out_min + np.array([1, 1]))  # (x, y)
        output_shape = output_shape[::-1]  # (row, col)

        # move bottom left of bounding box to origin of coordinates, so no part of the image will be
        # cropped
        transform_translate = EuclideanTransform(translation=-bbox_out_min)
        transform_skimage = transform_skimage + transform_translate

    elif output_shape == 'same':

        # size of the output image is the size of the input image
        output_shape = input_shape[0:2]

        # no translation correction needed

    else:  # user provides numerical output size

        # correction of centre point if output size is not the same as the input image size
        shape_correction = (np.array(output_shape) - np.array(input_shape[0:2])) / 2.0
        transform_translate = EuclideanTransform(translation=shape_correction)

        transform_skimage = transform_skimage + transform_translate

    if type(output_shape) == np.ndarray:
        output_shape = tuple(output_shape.astype(np.int64))
    return transform_skimage, output_shape


def transform_coords(coords, transform_skimage):
    """
    Apply a scikit.image transformation to a set of point coordinates.

    The transformations applied to point coordinates in this function are consistent to transformations applied to
    an image with transform_im().

    :param coords: (P, 2) np.array, each row has the (x, y) coordinates of a point.
                    Alternatively, list of (x, y) coordinates.
    :param transform_skimage: skimage transform, e.g. created with keras2skimage_transform().
    :return:
    * coords_out: (P, 2) np.array or list of (x, y) coordinates of the transformed points.
    """

    is_list = type(coords) == list
    if is_list:
        coords = np.vstack(coords)

    # apply transformation to coordinates
    coords_out = matrix_transform(coords, transform_skimage.params)

    # cast to input type
    if is_list:
        coords_out = [tuple(x) for x in coords_out]

    return coords_out


def transform_im(im, transform_skimage, output_shape=None, order=1):
    """
    Apply a scikit.image transformation to an image.

    The motivation for this function is that keras apply_transform() doesn't enable nearest neighbour
    interpolation. Thus, when applied to label or segmentation images, its bi-linear interpolation
    creates bogus labels.

    :param im: (row, col, channel) or (row, col)-np.array image.
    :param transform_skimage: skimage transform, e.g. created with keras2skimage_transform()
    :param output_shape: (def None) (height, width) tuple with the size of the output image. By default,
    the shape of the input image is preserved.
    :param order: (def 1) interpolation order. 0: nearest neighbour, 1: bi-linear, 2: bi-quadratic, ..., 5: bi-quintic.
    :return:
    * im_out: np.array with the same shape and dtype as im.
    """

    # apply transformation to image
    im_out = warp(im, transform_skimage.inverse, order=order, preserve_range=True, output_shape=output_shape)

    # correct output type
    im_out = im_out.astype(im.dtype)

    return im_out


def focal_loss(gamma=2., alpha=.25):
    """
    Focal loss implementation by mkocabas (https://github.com/mkocabas/focal-loss-keras) of the loss proposed by Lin et
    al. "Focal Loss for Dense Object Detection", 2017 (https://arxiv.org/abs/1708.02002). Released under the MIT License
    (see focal_loss_LICENSE.txt).

    Only available for tensorflow.
    :param gamma: (def 2.0)
    :param alpha: (def 0.25)
    :return: focal_loss_fixed(y_true, y_pred)
    """

    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


def rescale_intensity(im, ignore_value=None):
    """
    Stretch the pixel intensities of a batch of images to cover the whole dynamic range of the dtype,
    excluding the black background pixels.

    The scaling is performed on the H-channel of the HSV transform of the RGB image.

    :param im: np.ndarray (batch, rows, cols, channel) RGB images.
    :param ignore_value: (def None). Once the image is converted from RGB to HSV, pixels with
    V=ignore_value will be ignored for the intensity scaling. This is useful e.g. when a regular image
    has areas that have been blacked out, and we want to scale the non-blacked-out intensities.
    :return: Array with the same size as im.
    """

    # indices for hue, saturation, value
    # H = 0
    # S = 1
    V = 2

    # loop images
    for i in range(im.shape[0]):

        if DEBUG:
            plt.clf()
            plt.imshow(im[i, :, :, :])

        # convert to Hue, Saturation, Value format
        im_hsv = rgb2hsv(im[i, :, :, :])

        # pixels that are not black background
        im_v = im_hsv[:, :, V]
        if ignore_value is None:
            im_v = minmax_scale(im_v, feature_range=(0.0, 1.0))
        else:
            # scale intensities of non-background area
            idx = im_v != ignore_value
            im_v[idx] = minmax_scale(im_v[idx], feature_range=(0.0, 1.0))

        # convert back to RGB
        im_hsv[:, :, V] = im_v
        im[i, :, :, :] = hsv2rgb(im_hsv)

        if DEBUG:
            plt.clf()
            plt.imshow(im[i, :, :, :])

            plt.clf()
            plt.hist(im[i, :, :, :].flatten())

    return im


def ecdf_confidence(data, num_quantiles=101, equispace='quantiles', confidence=0.95, estimator_name='beta'):
    """
    Compute empirical ECDF with confidence intervals/bands.

    The ECDF is a function that maps quantiles = ECDF(data). The user can choose whether the output is
    equispaced on the data axis or the quantiles axis.

    Derived from plot_CDF_confidence (https://github.com/wfbradley/CDF-confidence/blob/master/CDF_confidence.py).

    :param data: numpy.array with the 1D data to compute the ECDF for.
    :param num_quantiles: (def 101). Number of points the quantiles/data axes are split into for the output.
    :param equispace: 'quantiles' (def), 'data'. Choose which axis is equispaced for the output, data or quantiles.
    :param confidence: (def 0.95) 0.95 = 95-CI means confidence intervals [2.5%, 97.5%].
    :param estimator_name: (def 'beta) 'DKW' (Dvoretzky-Kiefer-Wolfowitz confidence band) or 'beta'.
    :return: data_out, quantile_out, quantile_ci_lo, quantile_ci_hi

    (num_quantiles,) numpy.arrays with a mapping ECDF(data_out[i]) in [quantile_ci_lo[i], quantile_ci_hi[i]], and
    ECDF(data_out[i])
    """

    if len(np.shape(data)) != 1:
        raise NameError('Data must be 1 dimensional')
    if num_quantiles > len(data) + 1:
        num_quantiles = len(data) + 1
    if len(data) < 2:
        raise NameError('Need at least 2 data points')
    if num_quantiles < 3:
        raise NameError('Need num_quantiles > 2')
    if confidence <= 0.0 or confidence >= 1.0:
        raise NameError('"confidence" must be between 0.0 and 1.0')
    ci_lo = (1.0 - confidence) / 2.0
    ci_hi = 1.0 - ci_lo

    # sort the data, to make it more efficient looking for the min and max values
    data = np.sort(data)

    # compute empirical cumulative distribution (the function, not the values)
    ecdf = ECDF(data)

    # what are equispaced, the data axis or the quantile axis points?
    if equispace == 'data':
        # equispaced points in the data axis
        data_out = np.linspace(data[0], data[-1], num_quantiles)
        # corresponding quantile values
        quantile_out = ecdf(data_out)
    elif equispace == 'quantiles':
        # inverse of the ECDF function
        ecdf_inv = monotone_fn_inverter(ecdf, np.unique(data), vectorized=True)
        # equispaced points in the quantile axis
        quantile_out = np.linspace(ecdf(data[0]), 1.0, num_quantiles)
        # corresponding data values
        data_out = ecdf_inv(quantile_out)
    else:
        raise ValueError('"equispace" must be "data" or "quantile"')

    # some estimators give confidence intervals on the *quantiles*, others give intervals on the *data*;
    # which do we have?
    if estimator_name == 'DKW':
        estimator_type = 'quantile'
        cdf_error_function = CDF_error_DKW_band
    elif estimator_name == 'beta':
        estimator_type = 'quantile'
        cdf_error_function = CDF_error_beta
    else:
        raise NameError('Unknown error estimator name: ' + estimator_name)

    # compute interval for each output point
    quantile_ci_lo = np.zeros(shape=(num_quantiles,), dtype=np.float32)
    quantile_ci_hi = np.zeros(shape=(num_quantiles,), dtype=np.float32)
    if estimator_type == 'quantile':
        for i, q in enumerate(quantile_out):
            quantile_ci_lo[i] = cdf_error_function(len(data), q, ci_lo)
            quantile_ci_hi[i] = cdf_error_function(len(data), q, ci_hi)
    else:
        raise NameError('Unknown error estimator type: ' + estimator_type)

    return data_out, quantile_out, quantile_ci_lo, quantile_ci_hi


def compare_ecdfs(x, y, alpha=0.05, num_quantiles=101, num_perms=1000, multitest_method=None):
    """
    Compute p-values for the difference between each percentile point of the empirical cumulative distribution
    functions (ECDFs) of two samples x, y.

    This function allows multiple test adjustment of p-values using statsmodels.stats.multitest.multipletests.

    This function is basically an implementation of permutation testing (see overview in
    http://rasbt.github.io/mlxtend/user_guide/evaluate/permutation_test/#overview), but generating one p-value
    for each percentile.

    For example, if the median(x)=10, median(y)=7, this function computes the p-value, or probability
    that the observed difference is due to mere chance. The median is the 50 percentile. This function
    computes p-values for the differences between the 0, 1, 2, ..., 100 percentiles.

    :param x: numpy.ndarray vector, data set 1.
    :param y: numpy.ndarray vector, data set 2.
    :param alpha: (def 0.05) float. Error rate. Confidence is 1-alpha.
    :param num_quantiles: (def 101) Number of points the quantile range is split into. For 101, the
    quantiles are the percentiles 0%, 1%, 2%, ..., 100%.
    :param num_perms: (def 1000) Number of bootstrap permutations the p-values are estimated from.
    :param multitest_method: (def None) Adjustment of p-values due to multitesting. These are the same
    values as for statsmodels.stats.multitest.multipletests. Currently
    (https://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html),
        - `bonferroni` : one-step correction
        - `sidak` : one-step correction
        - `holm-sidak` : step down method using Sidak adjustments
        - `holm` : step-down method using Bonferroni adjustments
        - `simes-hochberg` : step-up method  (independent)
        - `hommel` : closed method based on Simes tests (non-negative)
        - `fdr_bh` : Benjamini/Hochberg  (non-negative)
        - `fdr_by` : Benjamini/Yekutieli (negative)
        - `fdr_tsbh` : two stage fdr correction (non-negative)
        - `fdr_tsbky` : two stage fdr correction (non-negative)
    :return:
    quantiles: numpy.ndarray vector with quantile values in [0.0, 1.0].
    pval: corresponding p-values for each quantile, whether adjusted or not.
    reject: boolean vector, pval <= alpha_c (where alpha_c is the corrected alpha value due to multiple test
    adjustment).
    """

    def compute_test_statistics(x, y, quantiles):
        """
        Compute test statistic for each quantile.

        :param x: see compare_ecdfs().
        :param y: see compare_ecdfs().
        :param quantiles: vector of quantile values, e.g. [0.00, 0.01, 0.02, ..., 1.00].
        :return: ts (numpy.ndarray vector) with test statistic values, one per quantile.
        """

        # compute ECDF functions for each bootstrap sampling
        x_ecdf_func = ECDF(x)
        y_ecdf_func = ECDF(y)

        # inverse of the ECDF functions
        x = np.unique(x)
        y = np.unique(y)
        x_ecdf_func_inv = monotone_fn_inverter(x_ecdf_func, x, vectorized=True)
        y_ecdf_func_inv = monotone_fn_inverter(y_ecdf_func, y, vectorized=True)

        # small quantile values are outside the interpolation range, so they cannot be used for computations
        idx = quantiles >= np.max([x_ecdf_func(x[0]), y_ecdf_func(y[0])])

        # data values that correspond to the quantiles
        x_data = x_ecdf_func_inv(quantiles[idx])
        y_data = y_ecdf_func_inv(quantiles[idx])

        # compute test statistic for each quantile
        ts = np.full(shape=(1, len(quantiles)), dtype=np.float32, fill_value=np.nan)
        ts[0, idx] = np.abs(x_data - y_data)  # two-sided test

        return ts

    # number of samples
    nx = len(x)
    ny = len(y)

    # equispaced points in the quantile axis
    quantiles = np.linspace(0.0, 1.0, num_quantiles)

    # merge data samples
    xy = np.concatenate((x, y))

    # compute test statistic for each quantile
    t = compute_test_statistics(x, y, quantiles)

    # allocate memory to keep the bootstrap samples of the test statistic
    ts = np.full(shape=(num_perms, num_quantiles), dtype=np.float32, fill_value=np.nan)

    # repeat bootstrap sampling
    for b in range(num_perms):

        # sample with replacement from the merged data vectors
        xs = np.random.choice(xy, size=(nx,), replace=True)
        ys = np.random.choice(xy, size=(ny,), replace=True)

        # compute test statistic for each quantile
        ts[b, :] = compute_test_statistics(xs, ys, quantiles)

        # compare the test statistic of the input to the null-distribution created by the bootstrap sampling
        idx = np.logical_not(np.isnan(ts[b, :]))
        ts[b, idx] = ts[b, idx] >= t[0, idx]

    # p-value = N_(ts >= t) / N_ts
    pval = np.mean(ts, axis=0)

    # p-values under error rate
    reject = pval <= alpha

    # apply multiple test correction
    if multitest_method is not None:
        idx = np.logical_not(np.isnan(pval))
        reject[idx], pval[idx], _, _ = multipletests(pval[idx], alpha=alpha, method=multitest_method,
                                                     is_sorted=False, returnsorted=False)

    return quantiles, pval, reject


def edge_labels(labels):
    """
    Find which labels touch the borders of the image. The background label (0) will be ignored.

    :param labels: 2D numpy.ndarray with segmentation labels.
    :return:
    edge_labels: numpy.ndarray with list of labels.
    """

    if labels.ndim != 2:
        raise ValueError('labels must be a 2D array')

    # labels that touch the top edge of the image
    edge_labels = np.unique(labels[0, :])

    # labels that touch the left edge
    edge_labels = np.unique(np.concatenate((edge_labels, labels[:, 0].flat)))

    # labels that touch the right edge
    edge_labels = np.unique(np.concatenate((edge_labels, labels[:, -1].flat)))

    # labels that touch the bottom edge
    edge_labels = np.unique(np.concatenate((edge_labels, labels[-1, :].flat)))

    # remove label=0, if present
    edge_labels = np.setdiff1d(edge_labels, 0)

    return edge_labels


def bspline_resample(xy, factor=1.0, k=1, is_closed=True):
    """
    Resample a 2D curve using B-spline interpolation.

    Note that repeated consecutive points will be removed, because otherwise splprep() raises an exception.

    :param xy: (N, 2)-np.ndarray with (x,y)=coordinates
    :param factor: (def 1.0) The number of output points is computed as round(N*factor).
    :param k: (def 1) Degree of the spline, e.g. k=1 (linear), k=2 (quadratic), k=3 (cubic).
    :param is_closed: (def True) Treat xy as a closed curve.
    :return:
    xy_out: (M, 2)-np.ndarray with coordinates of the resampled curve.
    """

    # check input
    if type(xy) != np.ndarray or xy.shape[1] != 2:
        raise ValueError('xy must be a 2-column np.ndarray')

    # remove repeated consecutive points
    idx = np.logical_or(np.diff(xy[:, 0]) != 0, np.diff(xy[:, 1]) != 0)
    if is_closed:
        idx = np.concatenate(([np.any(xy[0, :] - xy[-1, :] != 0), ], idx))
    else:
        idx = np.concatenate(([True, ], idx))
    xy = xy[idx, :]

    # for periodic contours, the last point will be ignored, so we need to add a duplicate
    if is_closed:
        xy = np.concatenate((xy, np.expand_dims(xy[0, :], axis=0)), axis=0)

    # parameters for interpolator
    if is_closed:
        per = 1
    else:
        per = 0

    # compute interpolating B-spline
    tck, u = splprep([xy[:, 0], xy[:, 1]], k=k, s=0, per=per)

    # number of output points
    n = int(np.round(xy.shape[0] * factor))

    # resample B-spline
    x_out, y_out = splev(np.linspace(u[0], u[-1], n), tck)
    xy_out = np.column_stack((x_out, y_out))

    if DEBUG:
        plt.clf()
        plt.plot(xy[:, 0], xy[:, 1], 'b')
        plt.plot(xy_out[:, 0], xy_out[:, 1], 'r')

    return xy_out


def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          xlabel=None,
                          ylabel=None,
                          cmap=plt.cm.Blues,
                          colorbar=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Copied from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    on 27 Mar 2019. Small modifications to allow input xlabel, ylabel.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    classes = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    if xlabel is None:
        xlabel = 'Predicted label'
    if ylabel is None:
        ylabel = 'True label'

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax)
        for cbi in cb.ax.yaxis.get_ticklabels():
            cbi.set_size(14)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.tick_params(axis='both', which='major', labelsize=14)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)
    fig.tight_layout()
    return ax
