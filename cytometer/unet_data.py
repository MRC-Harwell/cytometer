"""
Derived from data.py by Marko Jocić (jocicmarko) for Ultrasound Nerve Segmentation
https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/data.py

Author: Lorna Nolan
"""

from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread

data_path = 'raw/'

image_rows = 500
image_cols = 500


def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = (len(images) / 2)*6

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tiff'
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img_ud = np.flipud(img)
        img_lr = np.fliplr(img)
        img_90 = np.rot90(img)
        img_180 = np.rot90(img_90)
        img_270 = np.rot90(img_180)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)
        img_mask_ud = np.flipud(img_mask)
        img_mask_lr = np.fliplr(img_mask)
        img_mask_90 = np.rot90(img_mask)
        img_mask_180 = np.rot90(img_mask_90)
        img_mask_270 = np.rot90(img_mask_180)

        # img = np.array([img])
        # img_mask = np.array([img_mask])
        #
        # imgs[i] = img
        # imgs_mask[i] = img_mask

        k = i*6
        imgs[k] = img
        imgs[k + 1] = img_ud
        imgs[k + 2] = img_lr
        imgs[k + 3] = img_90
        imgs[k + 4] = img_180
        imgs[k + 5] = img_270

        imgs_mask[k] = img_mask
        imgs_mask[k + 1] = img_mask_ud
        imgs_mask[k + 2] = img_mask_lr
        imgs_mask[k + 3] = img_mask_90
        imgs_mask[k + 4] = img_mask_180
        imgs_mask[k + 5] = img_mask_270


        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':
    create_train_data()
    create_test_data()
