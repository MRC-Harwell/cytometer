import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter
from skimage import measure
from skimage.morphology import watershed
from skimage.future.graph import rag_mean_color
from skimage.measure import regionprops
from mahotas.labeled import borders
import networkx as nx
from skimage.transform import SimilarityTransform, AffineTransform


DEBUG = False


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
    :param min_seed_object_size: (def 50). Objects
    :param border_dilation:
    :return:
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


def segmentation_quality(labels_ref, labels_test):
    """
    Dice coefficients for each label in a segmented image.

    This function takes two segmentations, reference and test, and computes how good each test
    label segmentation is based on how it overlaps the reference segmentation. In a nutshell,
    we find the reference label best aligned to each test label, and compute the Dice
    coefficient as a similarity measure.

    We illustrate this with an example.

    Let one of the labels in the test segmentation be 51.

    This test label partly overlaps 5 labels in the reference segmentation:
    [ 2,  3, 10, 12, 17].

    The number of pixels in the intersection of 51 with each of the other labels is:
    [    2,   536,    29, 17413,   162]

    Therefore, label 51 is best aligned to label 12.

    Let label 51 contain 20,000 pixels, and label 12, 22,000 pixels.

    The Dice coefficient will be 2 * 17413 / (20000 + 22000) = 0.83.

    :param labels_ref: np.ndarray matrix, some integer type. All pixels with the same label
    correspond to the same object.
    :param labels_test: np.ndarray matrix, some integer type. All pixels with the same label
    correspond to the same object.
    :return: structured array out:
     out['lab_test']: 1-d np.ndarray with unique list of labels in the test image.
     out['lab_ref']: 1-d np.ndarray with labels that best align with the test labels.
     out['dice']: 1-d np.ndarray with Dice coefficient for each pair of corresponding labels.
    """

    # form pairs of values between reference labels and test labels. This is going to
    # produce pairs of all overlapping labels, e.g. if label 5 in the test image
    # overlaps with labels 1, 12 and 4 in the reference image,
    # label_pairs_by_pixel = [..., 5,  5, 5, ...]
    #                        [..., 1, 12, 4, ...]
    aux = np.stack((labels_test.flatten(), labels_ref.flatten()))
    label_pairs_by_pixel, label_pairs_by_pixel_count = np.unique(aux, axis=1, return_counts=True)

    # unique labels in the reference and test images, and number of pixels in each label
    labels_ref_unique, labels_ref_count = np.unique(labels_ref, return_counts=True)
    labels_test_unique, labels_test_count = np.unique(labels_test, return_counts=True)

    # for each of the test labels, find which reference label overlaps it the most
    labels_ref_correspond = np.zeros(shape=labels_test_unique.shape, dtype=labels_ref_unique.dtype)
    intersection_count = np.zeros(shape=labels_test_count.shape, dtype=labels_test_count.dtype)
    dice = np.zeros(shape=labels_test_count.shape, dtype=np.float32)
    for i, lab in enumerate(labels_test_unique):

        # find ref label with most overlap with the current test label
        idx = label_pairs_by_pixel[0] == lab
        idx_max = np.argmax(label_pairs_by_pixel_count[idx])
        labels_ref_correspond[i] = label_pairs_by_pixel[1, idx][idx_max]

        # number of pixels in the intersection of both labels
        if labels_ref_correspond[i] == 0:  # the test label overlaps mostly background
            intersection_count[i] = 0
        else:                              # the test label overlaps mostly a ref label
            intersection_count[i] = label_pairs_by_pixel_count[idx][idx_max]

        # # DEBUG: check the number of intersection pixels
        # print(np.count_nonzero(np.logical_and(labels_test == lab, labels_ref == labels_ref_correspond[i])))

        # to compute the Dice coefficient we need to know:
        # * |A| = labels_test_count[i]: number of pixels in the test label
        # * |B| = b: number of pixels in the corresponding ref label
        # * |A ∩ B| = intersection_count: number of pixels in the intersection of both labels
        # DICE = 2 * |A ∩ B| / (|A| + |B|)
        a = labels_test_count[i]
        b = labels_ref_count[labels_ref_unique == labels_ref_correspond[i]]
        dice[i] = 2 * intersection_count[i] / (a + b)

    # prepare output as structured array
    out = np.zeros((1, len(dice)), dtype=[('lab_test', labels_test_unique.dtype),
                                          ('lab_ref', labels_ref_unique.dtype),
                                          ('dice', dice.dtype)])
    out['lab_test'] = labels_test_unique
    out['lab_ref'] = labels_ref_correspond
    out['dice'] = dice
    return out


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


def one_image_and_dice_per_cell(dataset_im, dataset_lab, dataset_dice, training_window_len=401, smallest_cell_area=804):
    """
    Extract a small image centered on each cell (label) of a dataset, and the corresponding Dice coefficient that gives
    a measure of how well the label segments the cell (the Dice coefficient must have been computed previously,
    typically by comparing the label with some other ground truth label).

    :param dataset_im: numpy.ndarray (image, width, height, channel). Histology images.
    :param dataset_lab: numpy.ndarray (image, width, height, 1). Instance segmentation of the histology. Each label
    gives the segmentation of one cell. Not all cells need to have been segmented. Label=0 corresponds to the background
    and will be ignored.
    :param dataset_dice: numpy.ndarray (image, width, height, 1). Dice coefficients for dataset_lab. All pixels that
    correspond to the same label should have the same Dice coefficient value, but this is sometimes not the case due to
    interpolation errors after data augmentation. Thus, internally a median value is computed.
    :param training_window_len: (def 401) Each cell will be extracted to a (training_window_len, training_window_len)
    window.
    :param smallest_cell_area: (def 804) Labels with less than smallest_cell_area pixels will be ignored as segmentation
    noise.
    :return: training_windows, dice

    training_windows: numpy.ndarray (N, training_window_len, training_window_len, channel). Small windows extracted from
    the histology. Each window is centered around one of N labelled cells.
    label_windows: numpy.ndarray (N, training_window_len, training_window_len, 1). The segmentation label or mask for
    the cell in the training window.
    dice: numpy.ndarray (N,). Dice coefficient for the cell's segmentation.
    """

    # (r,c) size of the image
    n_row = dataset_im.shape[1]
    n_col = dataset_im.shape[2]

    training_windows_list = []
    label_windows_list = []
    dice_list = []
    for i in range(dataset_im.shape[0]):

        # print('Image ' + str(i) + '/' + str(dataset['im'].shape[0] - 1))

        # for convenience, we save a copy of the Dice coefficients for the current image
        dice_aux = dataset_dice[i, :, :, 0]

        # mask labels that have a Dice coefficient value. These are the ones we can use for training. We ignore the
        # other labels, as they have no ground truth to compare against, but by default they get Dice = 0.0
        labels = dataset_lab[i, :, :, 0] * (dataset_dice[i, :, :, 0] > 0.0).astype(np.uint8)
        labels = labels.astype(np.int32)

        # compute bounding boxes for the testing labels (note that the background 0 label is ignored)
        props = regionprops(labels, coordinates='rc')

        for p in props:

            # sometimes we get artifact labels, for really tiny objects. We ignore those
            if p['area'] < smallest_cell_area:
                continue

            # Dice value of the label (note we take the median to ignore small interpolation errors that pick Dice
            # values from adjacent labels)
            dice_list.append(np.median(dice_aux[labels == p['label']]))

            # width and height of the label's bounding box. Taking into account: Bounding box
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
            label_window = np.zeros(shape=(training_window_len, training_window_len, dataset_lab.shape[3]),
                                    dtype=np.uint8)
            label_window[i_min_row:i_max_row, i_min_col:i_max_col, 0] = \
                dataset_lab[i, lbox_min_row:lbox_max_row, lbox_min_col:lbox_max_col, 0] == p['label']
            label_windows_list.append(label_window)


            if DEBUG:
                plt.clf()
                plt.subplot(221)
                plt.imshow(dataset_im[i, :, :, :])
                plt.contour(labels == p['label'], levels=1)
                plt.plot([bbox_min_col, bbox_max_col, bbox_max_col, bbox_min_col, bbox_min_col],
                         [bbox_min_row, bbox_min_row, bbox_max_row, bbox_max_row, bbox_min_row], 'r')
                plt.plot([lbox_min_col, lbox_max_col, lbox_max_col, lbox_min_col, lbox_min_col],
                         [lbox_min_row, lbox_min_row, lbox_max_row, lbox_max_row, lbox_min_row], 'g')
                plt.subplot(222)
                # plt.imshow(labels)
                plt.imshow(dice_aux)
                plt.subplot(223)
                plt.imshow(training_window)
                plt.contour(label_window[:, :, 0], levels=1)
                plt.subplot(224)
                plt.imshow(label_window[:, :, 0])
                plt.title('Dice = ' + str(dice_list[-1]))

    # convert list to array
    training_windows_list = np.stack(training_windows_list)
    label_windows_list = np.stack(label_windows_list)
    dice_list = np.stack(dice_list)

    return training_windows_list, label_windows_list, dice_list
