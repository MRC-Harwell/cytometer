#!/usr/bin/env python
# This Python file uses the following encoding: utf-8

from setuptools import setup
from setuptools import find_packages

setup(name='cytometer',
      version='0.2.0',
      description='Cell segmentation and quantification',
      author='Ramón Casero',
      author_email='rcasero@gmail.com',
      url='http://phobos.mrch.har.mrc.ac.uk/r.casero/cytometer/',
      license='GPLv3',
      packages=find_packages(),
      data_files=[('data', ['data/*'])],
      install_requires=['python>=3.5', 'keras>=2.1.2', 'pillow>=5',
                        'matplotlib=2.1.2', 'scikit-image>=0.13.1',
                        'scikit-learn>=0.19.1',
                        'openslide-python>=1.1.1', 'tifffile>=0.14.0',
                        'svgpathtools>=1.3.2', 'seaborn>=0.8.1']
     )
