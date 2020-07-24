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
      python_requires='>=3.6',
      install_requires=['keras>=2.2.4',
                        'tensorflow-gpu==2.2.0', 'cudnn==7.6.5=cuda10.2_0', 'h5py==2.10.0', 'graphviz==2.40.1',
                        'cython==0.29.21', 'pydot==1.4.1', 'matplotlib==3.2.2', 'pillow==7.2.0', 'scikit-image==0.15.0',
                        'scikit-learn==0.23.1', 'nose==1.3.7', 'pytest==5.4.3', 'setuptools==49.2.0',
                        'opencv-python==4.1.0.25', 'pysto==1.4.1', 'openslide-python==1.1.1', 'seaborn==0.10.0',
                        'statannot==0.2.3', 'tifffile==2019.5.30', 'mahotas==1.4.5', 'networkx==2.4',
                        'svgpathtools==1.3.3', 'receptivefield==0.4.0', 'rpy2==3.0.5', 'mlxtend==0.17.0', 'ujson==1.35',
                        'pandas==1.0.5', 'shapely==1.7.0', 'six==1.12.0', 'statsmodels==0.11.1']
      )
