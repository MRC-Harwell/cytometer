import os
import glob
import datetime
import tifffile
import numpy as np
import pysto.imgproc as pystoim

# use CPU for testing on laptop
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
import cytometer.models as models
from receptivefield.keras import KerasReceptiveField
import matplotlib.pyplot as plt

# limit GPU memory used
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

# allow data parallelism
from keras.utils import multi_gpu_model



DEBUG = False

root_data_dir = '/home/rcasero/Dropbox/klf14'
training_dir = '/home/rcasero/Dropbox/klf14/klf14_b6ntac_training'
training_nooverlap_data_dir = os.path.join(root_data_dir, 'klf14_b6ntac_training_non_overlap')
saved_models_dir = '/home/rcasero/Software/cytometer/saved_models'

'''Load data
'''

K.set_image_data_format('channels_last')

labels_file_list = glob.glob(os.path.join(training_nooverlap_data_dir, '*.tif'))

# allocate arrays for data
im = np.zeros((len(labels_file_list), 1001, 1001, 3), dtype='uint8')
labels = np.zeros((len(labels_file_list), 1001, 1001, 1), dtype='uint8')
mask = np.zeros((len(labels_file_list), 1001, 1001, 1), dtype='uint8')

# labels
LABEL_BACKGROUND = 1

# loop loading the segmentation labels, and convert to a mask of cells vs. no cells
for i, labels_file in enumerate(labels_file_list):

    # corresponding image file
    im_file = labels_file.replace(training_nooverlap_data_dir, training_dir)

    # load image
    im[i, :, :, :] = tifffile.imread(im_file)

    # load labels
    labels[i, :, :, 0] = tifffile.imread(labels_file)[0, :, :]

    # convert to mask of cells vs. no cells
    mask[i, :, :, :] = labels[i, :, :, :] != LABEL_BACKGROUND

    # plot labels
    if DEBUG:
        plt.clf()
        plt.subplot(221)
        plt.imshow(im[i, :, :, :])
        plt.subplot(222)
        plt.imshow(labels[i, :, :, 0])
        plt.subplot(223)
        plt.imshow(mask[i, :, :, 0])

# remove one pixel column and row to that we can split the images into equal (250, 250) blocks
im = im[:, 0:-1, 0:-1, :]
mask = mask[:, 0:-1, 0:-1, :]

# split images into smaller blocks to avoid GPU memory overflows in training
im_slices, im_blocks, _ = pystoim.block_split(im, nblocks=(1, 5, 5, 1))
mask_slices, mask_blocks, _ = pystoim.block_split(mask, nblocks=(1, 5, 5, 1))

im_split = np.concatenate(im_blocks, axis=0)
mask_split = np.concatenate(mask_blocks, axis=0)

'''CNN
'''

# declare network model
model = models.basic_9_conv_8_bnorm_3_maxpool_binary_classifier(input_shape=im_split.shape[1:])
#model.load_weights('/home/rcasero/Software/cytometer/saved_models/2018-07-12T14:58:10.433876_basic_9_conv_8_bnorm_3_maxpool_binary_classifier.h5')

# compile model
parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
tic = datetime.datetime.now()
parallel_model.fit(im_split, mask_split, batch_size=10, epochs=100, validation_split=.1)
toc = datetime.datetime.now()
print('Training duration: ' + str(toc - tic))

# save result (note, we save the template model, not the multiparallel object)
model.save(os.path.join(saved_models_dir, datetime.datetime.utcnow().isoformat() + '_basic_9_conv_8_bnorm_3_maxpool_binary_classifier.h5'))

# visualise results
if DEBUG:
    for i in range(im_split.shape[0]):

        # run image through network
        mask_pred = model.predict(im_split[i, :, :, :].reshape((1,) + im_split.shape[1:]))

        plt.clf()
        plt.subplot(221)
        plt.imshow(im_split[i, :, :, :])
        plt.subplot(222)
        plt.imshow(mask_split[i, :, :, :].reshape((200, 200)))
        plt.subplot(223)
        plt.imshow(mask_pred.reshape((200, 200)) * 255)
        plt.subplot(224)
        a = mask_split[i, :, :, :].reshape((200, 200))
        b = mask_pred.reshape((200, 200)) * 255
        plt.imshow(pystoim.imfuse(a, b))

# estimate receptive field of the model
def model_build_func(input_shape):
    return models.basic_9_conv_8_bnorm_3_maxpool_binary_classifier(input_shape=input_shape,
                                                                   for_receptive_field=True,
                                                                   dilation_rate=2)

rf = KerasReceptiveField(model_build_func, init_weights=True)

rf_params = rf.compute(
    input_shape=(200, 200, 3),
    input_layer='input_image',
    output_layer='main_output'
)
print(rf_params)

plt.clf()
plt.imshow(im_split[i, :, :, :])
rf.plot_rf_grid(custom_image=im_split[i, :, :, :], figsize=(6, 6))
plt.show()

rf.plot_rf_grid(get_default_image(shape, name='doge'))