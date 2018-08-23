import cv2
import numpy as np
import matplotlib.pyplot as plt

DEBUG = False

img = dmap_test_pred[0, :, :, 0]

def principal_curvatures_range_image(img, ksize=3):

    # normalize image intensities to interval [0.0, 1.0]
    imax = np.max(img)
    imin = np.min(img)
    img = (img - imin) / (imax - imin)

    # image gradients
    hx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    hy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)

    # image second derivatives
    hxx = cv2.Sobel(hx, cv2.CV_32F, 1, 0, ksize=ksize)
    hxy = cv2.Sobel(hx, cv2.CV_32F, 0, 1, ksize=ksize)
    hyy = cv2.Sobel(hy, cv2.CV_32F, 0, 1, ksize=ksize)

    if DEBUG:
        plt.clf()
        plt.subplot(321)
        plt.imshow(img)
        plt.title('h')
        plt.subplot(323)
        plt.imshow(hx)
        plt.title('hx')
        plt.subplot(324)
        plt.imshow(hy)
        plt.title('hy')
        plt.subplot(322)
        plt.imshow(hxy)
        plt.title('hxy')
        plt.subplot(325)
        plt.imshow(hxx)
        plt.title('hxx')
        plt.subplot(326)
        plt.imshow(hyy)
        plt.title('hyy')

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
        plt.title('h')
        plt.subplot(322)
        plt.imshow(np.logical_and(k1 > 0, k2 < 0).astype(np.uint8))
        plt.title('Valley detector')
        plt.subplot(323)
        plt.imshow(K, vmin=np.percentile(K, .1), vmax=np.percentile(K, 99.9))
        plt.title('Gaussian curvature (K)')
        plt.subplot(324)
        plt.imshow(H, vmin=np.percentile(H, .1), vmax=np.percentile(H, 99.9))
        plt.title('Mean curvature (H)')
        plt.subplot(325)
        plt.imshow(k1, vmin=np.percentile(k1, .1), vmax=np.percentile(k1, 99.9))
        plt.title('Principal curvature k1')
        plt.subplot(326)
        plt.imshow(k2, vmin=np.percentile(k2, .1), vmax=np.percentile(k2, 99.9))
        plt.title('Principal curvature k2')

        plt.clf()
        plt.subplot(321)
        plt.imshow(img)
        plt.title('h')
        plt.subplot(323)
        plt.imshow(K, vmin=np.percentile(K, .1), vmax=np.percentile(K, 99.9))
        plt.title('Gaussian curvature (K)')
        plt.subplot(324)
        plt.imshow(H, vmin=np.percentile(H, .1), vmax=np.percentile(H, 99.9))
        plt.title('Mean curvature (H)')
        plt.subplot(325)
        plt.imshow(np.sqrt(np.abs(k1)) * np.sign(k1))
        plt.title('Principal curvature k1')
        plt.subplot(326)
        plt.imshow(np.sqrt(np.abs(k2)) * np.sign(k2))
        plt.title('Principal curvature k2')

        plt.clf()
        plt.subplot(321)
        plt.imshow(img)
        plt.title('h')
        plt.subplot(323)
        plt.imshow(K)
        plt.title('Gaussian curvature (K)')
        plt.subplot(324)
        plt.imshow(H)
        plt.title('Mean curvature (H)')
        plt.subplot(325)
        plt.imshow(k1)
        plt.title('Principal curvature k1')
        plt.subplot(326)
        plt.imshow(k2)
        plt.title('Principal curvature k2')

        np.min(K)
        np.max(K)



# if __name__ == '__main__':


