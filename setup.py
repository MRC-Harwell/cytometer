#!/usr/bin/env python

from distutils.core import setup

setup(name='cytometer',
      version='0.1.0',
      description='Cell segmentation and quantification',
      author='Ramón Casero',
      author_email='rcasero@gmail.com',
      url='http://phobos.mrch.har.mrc.ac.uk/r.casero/cytometer/',
      packages=['cnn'],
      data_files=[('data', ['data/*'])]
     )
