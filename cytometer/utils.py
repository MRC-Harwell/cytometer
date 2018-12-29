import cv2
import numpy as np
import six
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter
from scipy.sparse import dok_matrix
from skimage import measure
from skimage.morphology import watershed
from skimage.future.graph import rag_mean_color
from skimage.measure import regionprops
from skimage.segmentation import clear_border
from mahotas.labeled import borders
import networkx as nx
from skimage.transform import SimilarityTransform, AffineTransform
from skimage.color import rgb2hsv, hsv2rgb
from sklearn.preprocessing import minmax_scale
import keras.backend as K
import keras
import tensorflow as tf
from cytometer.models import change_input_size

DEBUG = False


def clear_mem():
    """GPU garbage collection in Keras with TensorFlow.

    From Otto Stegmaier and Jeremy Howard.
    https://forums.fast.ai/t/gpu-garbage-collection/1976/5
    """

    K.get_session().close()
    sess = K.get_session()
    sess.close()
    # limit mem
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))
    return


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

    This function computes the normal curvate of the dmap seen as a Monge patch. The "valleys" in
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
    :param sigma: (def 10) Standar deviation in pixels of Gaussian blurring applied to the dmap
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


def match_overlapping_labels(labels_ref, labels_test):
    """
    Match estimated segmentations to ground truth segmentations and compute Dice coefficients.

    This function takes two segmentations, reference and test, and computes how good each test
    label segmentation is, based on how it overlaps the reference segmentation. In a nutshell,
    we find the reference label best aligned to each test label, and compute the Dice
    coefficient as a similarity measure.

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

    # remove correspondences between any test lab and ref 0 label (background)
    idx = label_pairs_by_pixel[ref, :] != 0
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
        dice[:, lab_ref] = 0

    # check that all Dice values are in [0.0, 1.0]
    assert(all(out['dice'] >= 0.0) and all(out['dice'] <= 1.0))

    return out

def one_image_per_label(dataset_im, dataset_lab_test, dataset_lab_ref=None,
                        training_window_len=401, smallest_cell_area=804, clear_border_lab=False):
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
    :return: If dataset_lab_ref is provided,
    (training_windows, testlabel_windows, reflabel_windows, dice)

    Otherwise,
    (training_windows, testlabel_windows)

    training_windows: numpy.ndarray (N, training_window_len, training_window_len, channel). Small windows extracted from
    the histology. Each window is centered around one of N labelled cells.
    reflabel_windows: numpy.ndarray (N, training_window_len, training_window_len, 1). The ground truth segmentation
    label or mask for the cell in the training window.
    testlabel_windows: numpy.ndarray (N, training_window_len, training_window_len, 1). The test segmentation label or
    mask for the cell in the training window.
    dice: numpy.ndarray (N,). dice[i] is the Dice coefficient between corresponding each label_windows[i, ...] and its
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
                                                          labels_test=dataset_lab_test[i, :, :, 0])

        # compute bounding boxes for the testing labels (note that the background 0 label is ignored)
        props = regionprops(dataset_lab_test[i, :, :, 0], coordinates='rc')

        for p in props:

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
                plt.contour(label_window[:, :, 0], levels=1)
                plt.title('Dice = ' + str(round(dice_list[-1], 2)))
                plt.subplot(223)
                plt.imshow(dataset_lab_test[i, :, :, 0] == lab_test)
                plt.subplot(224)
                plt.gca().invert_yaxis()
                plt.contour(reflabel_window[:, :, 0], levels=1, colors='green', label='ref')
                plt.contour(label_window[:, :, 0], levels=1, label='test')

    # convert list to array
    training_windows_list = np.stack(training_windows_list)
    testlabel_windows_list = np.stack(testlabel_windows_list)
    index_list = np.stack(index_list)
    if dataset_lab_ref is not None:
        reflabel_windows_list = np.stack(reflabel_windows_list)
        dice_list = np.stack(dice_list)

    if dataset_lab_ref is None:
        return training_windows_list, testlabel_windows_list, index_list
    else:
        return training_windows_list, testlabel_windows_list, index_list, reflabel_windows_list, dice_list


def segmentation_pipeline(im, contour_model, dmap_model, quality_model, smallest_cell_area=804):
    """
    Instance segmentation of cells using the contour + distance transformation pipeline.
    :param im: numpy.ndarray (image, row, col, channel) with RGB histology images.
    :param contour_model: filename or keras model for the contour detection neural network. This is assumed to
    be a convolutional network where the output has the same (row, col) as the input.
    :param dmap_model: filename or keras model for the distance transformation regression neural network.
    This is assumed to be a convolutional network where the output has the same (row, col) as the input.
    :param quality_model: filename or keras model for the Dice coefficient estimation neural network. This
    network.
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

        # process histology image through the two pipeline branches
        one_im = im[i, :, :, :].reshape(one_im_shape)
        contour_pred = contour_model.predict(one_im)
        dmap_pred = dmap_model.predict(one_im)

        # combine pipeline branches for instance segmentation
        labels[i, :, :, 0], labels_borders[i, :, :, 0] = segment_dmap_contour(dmap_pred[0, :, :, 0],
                                                                              contour=contour_pred[0, :, :, 0],
                                                                              border_dilation=0)

    # split histology images into individual segmented objects
    cell_im, cell_seg, cell_index = one_image_per_label(dataset_im=im,
                                                        dataset_lab_test=labels,
                                                        training_window_len=training_window_len,
                                                        smallest_cell_area=smallest_cell_area)

    # loop segmented objects
    quality = np.zeros(shape=(cell_im.shape[0], ), dtype=np.float32)
    for j in range(cell_im.shape[0]):

        # mask histology with segmentation
        aux = cell_im[j, :, :, :] * np.repeat(cell_seg[j, :, :, :], repeats=3, axis=2)

        # compute quality measure of each histology window
        quality[j] = quality_model.predict(np.expand_dims(aux, axis=0))

    # prepare output as structured array
    labels_info = np.zeros((len(quality),), dtype=[('im', cell_index.dtype),
                                                   ('label', cell_index.dtype),
                                                   ('quality', quality.dtype)])
    labels_info['im'] = cell_index[:, 0]
    labels_info['label'] = cell_index[:, 1]
    labels_info['quality'] = quality

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


def keras2skimage_transform(transform, shape):
    """
    Convert an affine transform from keras to skimage format. This can then be used to apply a transformation
    to an image with skimage.transform.warp.

    Note: Currently, only scaling ('zx', 'zy') and rotation ('theta') are considered. Translation, flips, etc are
    ignored.

    Note 2: This function takes into account that rotations in keras are referred to the centre of the image, but
    skimage rotates around the centre of coordinates.

    :param transform: Dictionary with affine transform in keras format, as returned by keras.get_random_transform.
    :param shape: Tuple with (width, height) size of output image.
    :return: transform_skimage: skimage.transform._geometric.ProjectiveTransform with same affine transform.
    """

    transform_skimage_center = SimilarityTransform(translation=(shape[1] / 2, shape[0] / 2))
    transform_skimage_center_inv = SimilarityTransform(translation=(-shape[1] / 2, -shape[0] / 2))
    transform_skimage_affine = AffineTransform(matrix=None, scale=(transform['zx'], transform['zy']),
                                               rotation=transform['theta'] / 180.0 * np.pi, shear=None,
                                               translation=None)
    transform_skimage = transform_skimage_center_inv + (transform_skimage_affine + transform_skimage_center)

    return transform_skimage


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
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


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
