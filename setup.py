#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

setup(name='cytometer',
      version='0.1.0',
      description='Cell segmentation and quantification',
      author='Ramón Casero',
      author_email='rcasero@gmail.com',
      url='http://phobos.mrch.har.mrc.ac.uk/r.casero/cytometer/',
      license='GPLv3',
      packages=find_packages(),
      data_files=[('data', ['data/*'])],
      install_requires=['keras', 'tensorflow']
     )
