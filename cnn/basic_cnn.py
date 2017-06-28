#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 18:32:40 2017

@author: rcasero
"""

# check python interpreter version
import sys
if sys.version_info < (3,0,0):
    raise ImportError('Python 3 required')

# configure Keras, to avoid using file ~/.keras/keras.json
import os
import keras
from importlib import reload
os.environ['KERAS_BACKEND'] = 'tensorflow'
reload(keras.backend)
keras.backend.set_image_dim_ordering('tf')

