#!/bin/bash
# install_dependencies_v2.sh
#
#    Script to install the dependencies for more up-to-date packages of cytometer, without some of the constraints given
#    by the old version of Keras needed for training the pipeline.
#
#    This script assumes that install_dependencies.sh has been run already.

# This file is part of Cytometer
# Copyright 2021 Medical Research Council
# SPDX-License-Identifier: Apache-2.0
# Author: Ramon Casero <rcasero@gmail.com>

# exit immediately on errors that are not inside an if test, etc.
set -e

########################################################################
# configuration constants

PYTHON_VERSION=3.8
CONDA_LOCAL_ENV=cytometer_tensorflow_v2

########################################################################
## create python local environment if it doesn't exist already

# check whether the environment already exists
if [[ -z "$(conda info --envs | sed '/^#/ d' | cut -f1 -d ' ' | grep -x ${CONDA_LOCAL_ENV})" ]]; then
    tput setaf 1; echo "** Create conda local environment: ${CONDA_LOCAL_ENV}"; tput sgr0
    conda create -y --name ${CONDA_LOCAL_ENV} python=${PYTHON_VERSION}
else
    tput setaf 1; echo "** Conda local environment already exists: ${CONDA_LOCAL_ENV}"; tput sgr0
fi

########################################################################
## install dependencies

#sudo apt install -y default-jdk

conda activate ${CONDA_LOCAL_ENV}

pip install matplotlib==3.4.2 scipy==1.7.0 scikit-learn==0.24.2 statsmodels==0.12.2 seaborn==0.11.1
pip install ujson==4.0.2 mahotas==1.4.11 pysto==1.4.1 svgpathtools==1.4.1
conda install -y shapely==1.7.1
conda install -y --channel conda-forge pyvips==2.1.8
pip install "aicsimageio[czi] @ git+https://github.com/AllenCellModeling/aicsimageio.git"
#pip install slideio
#pip install python-bioformats
