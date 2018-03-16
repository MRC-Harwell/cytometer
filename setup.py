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
      # cytometer_tensorflow
      #install_requires=['python>=3.6',
      #                      'keras>=2.1.2', 'theano>=1.0.1+28.g96c910444',
      #                      'tensorflow-gpu>=1.4.1',
      #                      'pyyaml>=3.12', 'nose-parameterized>=0.6.0']
      # cytometer_theano
      install_requires=['python>=3.5',
                            'keras>=2.1.2', 'theano>=1.0.1+28.g96c910444']
     )
