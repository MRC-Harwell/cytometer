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
      install_requires=['python>=3.6', 'keras>=2.2.0', 'pillow>=5',
                        'numpy>=1.14.2', 'pandas>=0.22.0',
                        'matplotlib>=2.1.2', 'scikit-image>=0.14.0',
                        'scikit-learn>=0.19.1', 'scipy>=1.0.0',
                        'openslide-python>=1.1.1', 'tifffile>=0.14.0',
                        'svgpathtools>=1.3.2', 'statsmodels>=0.9.0',
                        'rpy2>=2.9.3', 'tzlocal>=1.5.1', 'six>=1.11.0',
                        'receptivefield>=0.19', 'pysto>=1.4.1',
                        'mahotas>=1.4.4', 'networkx>=2.1']
      )
