import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter
from skimage import measure
from skimage.morphology import watershed
from mahotas.labeled import borders


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

# if __name__ == '__main__':
