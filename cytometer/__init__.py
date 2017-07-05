#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 18:21:57 2017

@author: rcasero
"""

from __future__ import absolute_import

# cytometer package version (note: duplicated in /setup.py)
__version__ = '0.1.0'

# check python interpreter version
import sys
import warnings
if sys.version_info.major < 3:
    warnings.warn('Python 3 required', RuntimeWarning)
