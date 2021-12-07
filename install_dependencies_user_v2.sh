#!/bin/bash
# install_dependencies_user_v2.sh
#
#    Script to install the dependencies for more up-to-date packages of cytometer, without some of the constraints given
#    by the old version of Keras needed for the pipeline. (This environment cannot run the pipeline, though).
#
#    This script assumes that install_dependencies_machine.sh has been run already.

# This file is part of Cytometer
# Copyright 2021 Medical Research Council
# SPDX-License-Identifier: Apache-2.0
# Author: Ramon Casero <rcasero@gmail.com>

# exit immediately on errors that are not inside an if test, etc.
set -e

########################################################################
# configuration constants

PYTHON_VERSION=3.9
CONDA_LOCAL_ENV=cytometer_tensorflow_v2

########################################################################
## install Miniconda locally so that we can use the conda local environment tools,
## and install python packages with pip and conda

# install Miniconda
mkdir -p ~/Downloads

if [[ ! -z `which conda` ]]; then
    /usr/bin/tput setaf 1; echo "** Conda ${MINICONDA_VERSION} package manager already installed"; /usr/bin/tput sgr0
else
    /usr/bin/tput setaf 1; echo "** Installing conda ${MINICONDA_VERSION} package manager"; /usr/bin/tput sgr0
    mkdir -p ~/Dowloads
    pushd ~/Downloads
    # download installer
    if [[ ! -e "Miniconda${MINICONDA_VERSION}-latest-Linux-x86_64.sh" ]];
    then
	wget https://repo.continuum.io/miniconda/Miniconda${MINICONDA_VERSION}-latest-Linux-x86_64.sh
    fi
    # install conda
    chmod u+x Miniconda${MINICONDA_VERSION}-latest-Linux-x86_64.sh
    ./Miniconda${MINICONDA_VERSION}-latest-Linux-x86_64.sh -b -p "$HOME"/Software/miniconda${MINICONDA_VERSION}
    "$HOME"/Software/miniconda${MINICONDA_VERSION}/bin/conda init
    source ~/.bashrc
    popd
fi

########################################################################
## Ubuntu dependencies

# dependency of matlplotlib's backends
sudo apt-get install libgtk-3-dev

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

source activate ${CONDA_LOCAL_ENV}

pip install matplotlib==3.4.2 scipy==1.7.0 scikit-learn==0.24.2 scikit-image==0.18.1 statsmodels==0.12.2
pip install seaborn==0.11.1 openpyxl==3.0.9
pip install ujson==4.0.2 mahotas==1.4.11 pysto==1.4.1 svgpathtools==1.4.1
pip install shapely==1.7.1
pip install pyvips==2.1.15
pip install "aicsimageio[czi] @ git+https://github.com/AllenCellModeling/aicsimageio.git"
pip install wxpython==4.1.1
#pip install slideio
#pip install python-bioformats
