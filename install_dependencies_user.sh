#!/bin/bash
# install_dependencies_machine.sh
#
#    Script to install a new local environment (cytometer_tensorflow) and software to run DeepCytometer pipeline and
#    experiments. This needs to be run for each user.
#
#    This script assumes that the machine has been previously set up with install_dependencies_machine.sh.

# This file is part of Cytometer
# Copyright 2021 Medical Research Council
# SPDX-License-Identifier: Apache-2.0
# Author: Ramon Casero <rcasero@gmail.com>

# exit immediately on errors that are not inside an if test, etc.
set -e

########################################################################
# configuration constants

PYTHON_VERSION=3.7
CONDA_LOCAL_ENV=cytometer_tensorflow

########################################################################
## install Miniconda locally so that we can use the conda local environment tools,
## and install python packages with pip and conda

# install Miniconda
mkdir -p ~/Downloads

if [[ -d "${HOME}/Software/miniconda${MINICONDA_VERSION}" ]]; then
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
 ## create python local environment if it doesn't exist already

# check whether the environment already exists
if [[ -z "$(conda info --envs | sed '/^#/ d' | cut -f1 -d ' ' | grep -x ${CONDA_LOCAL_ENV})" ]]; then
    tput setaf 1; echo "** Create conda local environment: ${CONDA_LOCAL_ENV}"; tput sgr0
    conda create -y --name ${CONDA_LOCAL_ENV} python=${PYTHON_VERSION}
else
    tput setaf 1; echo "** Conda local environment already exists: ${CONDA_LOCAL_ENV}"; tput sgr0
fi

########################################################################
## install keras python packages in the local environment

# switch to the local environment
source activate ${CONDA_LOCAL_ENV}

tput setaf 1; echo "** Install keras python packages in the local conda environment"; tput sgr0

# check that the variable with the path to the local environment is
# set. Note that different versions of conda use different variable
# names for the path
if [[ ! -v CONDA_PREFIX ]];
then
    if [[ ! -v CONDA_ENV_PATH ]];
    then
	    echo "Error! Neither CONDA_PREFIX nor CONDA_ENV_PATH set in this local environment: ${CONDA_LOCAL_ENV}"
	    exit 1
    else
	    CONDA_PREFIX=${CONDA_ENV_PATH}
    fi
fi

# We need tensorflow-gpu 1.15.0 because that's the latest version that still works with our modified Keras 2.2.5
# https://www.tensorflow.org/install/source#gpu
# Version	              Python version	Compiler	Build tools	  cuDNN	CUDA
# tensorflow_gpu-1.15.0	2.7, 3.3-3.7	  GCC 7.3.1	Bazel 0.26.1	7.4	  10.0
echo "** Dependencies for Tensorflow backend"
pip install tensorflow-gpu==1.15.0

# install my own Keras 2.2 version modified to accept partial training data
pip install git+https://${USER}@github.com/rcasero/keras.git

NVIDIA_DRIVER_VERSION=`nvidia-smi --query-gpu=driver_version --format=csv,noheader --id=0`

# table of CUDA version vs. nVidia driver version
# https://docs.nvidia.com/deeplearning/sdk/cudnn-support-matrix/index.html
case ${NVIDIA_DRIVER_VERSION}
in
    450.*|465.*)  # CUDA 10.2

        conda install -y cudnn==7.6.5=cuda10.2_0
        ;;
    *)
        echo "cudnn version for detected nVidia driver version (v. *${NVIDIA_DRIVER_VERSION}*) not implemented"
        echo "You need to edit script ./install_dependencies.sh and add an instruction to install the correct cudnn version for this nVidia driver version"
        exit 1
        ;;
esac

# install dependencies for Keras
pip install h5py==3.3.0             # to save Keras models to disk
pip install graphviz==0.16          # used by visualization utilities to plot model graphs
pip install cython==0.29.23         # dependency of mkl-random/mkl-fft via pydot
pip install pydot==1.4.2            # used by visualization utilities to plot model graphs

########################################################################
## fix bug: missing dynamic libraries link
#
# With CUDA 10.2, tensorflow-gpu 1.15.0, the basic GPU test
#
# import tensorflow as tf
# tf.test.is_gpu_available()
#
# gives the errors
#
# 2020-08-25 12:18:42.705540: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudart.so.10.0'; dlerror: libcudart.so.10.0: cannot open shared object file: No such file or directory
# 2020-08-25 12:18:42.705741: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcublas.so.10.0'; dlerror: libcublas.so.10.0: cannot open shared object file: No such file or directory
# 2020-08-25 12:18:42.705847: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcufft.so.10.0'; dlerror: libcufft.so.10.0: cannot open shared object file: No such file or directory
# 2020-08-25 12:18:42.705948: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcurand.so.10.0'; dlerror: libcurand.so.10.0: cannot open shared object file: No such file or directory
# 2020-08-25 12:18:42.706053: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusolver.so.10.0'; dlerror: libcusolver.so.10.0: cannot open shared object file: No such file or directory
# 2020-08-25 12:18:42.706150: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusparse.so.10.0'; dlerror: libcusparse.so.10.0: cannot open shared object file: No such file or directory
# 2020-08-25 12:18:42.737947: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 2020-08-25 12:18:42.737987: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1641] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
# Skipping registering GPU devices...
#
# The problem is that the required libraries have slightly newer versions.
#
# We can fix the problem by creating symlinks with the expected library names to the lightly newer versions of the
# libraries

case ${UBUNTU_VERSION}
in
    20.04)

        # list of missing libraries
        source_missing_libraries=('libcudart.so.10.0' 'libcublas.so.10.0' 'libcufft.so.10.0' 'libcurand.so.10.0' 'libcusolver.so.10.0' 'libcusparse.so.10.0')
        target_missing_libraries=('libcudart.so.10.2' 'libcublas.so.10' 'libcufft.so.10' 'libcurand.so.10' 'libcusolver.so.10' 'libcusparse.so.10')

        # loop array indices
        for i in "${!source_missing_libraries[@]}"; do

            source_missing_library=${HOME}/Software/miniconda3/envs/${CONDA_LOCAL_ENV}/lib/${source_missing_libraries[i]}
            target_missing_library=${target_missing_libraries[i]}

            # create symlinks
            if [[ -f ${source_missing_library} ]]; then
                echo "Link from ${source_missing_library} to ${target_missing_library} already exists"
            else
                echo "Creating link from ${source_missing_library} to ${target_missing_library}"
                ln -s ${target_missing_library} ${source_missing_library}
            fi
        done
        ;;
esac

########################################################################
## install cytometer python dependencies packages in the local environment

tput setaf 1; echo "** Install cytometer python dependencies in the local conda environment"; tput sgr0

# install other python packages
pip install numpy==1.21.0
pip install git+https://www.github.com/keras-team/keras-contrib.git
pip install matplotlib==3.3.4
pip install scikit-image==0.18.2 scikit-learn==0.24.2
pip install nose==1.3.7 pytest==6.2.4
pip install opencv-python==4.5.2.54 pysto==1.4.1 openslide-python==1.1.2 seaborn==0.11.1 statannot==0.2.3
pip install mahotas==1.4.11 svgpathtools==1.4.1 receptivefield==0.5.0 rpy2==3.4.5
pip install mlxtend==0.18.0 ujson==4.0.2 rasterio==1.2.6
pip install shapely==1.7.1 six==1.16.0 statsmodels==0.12.2
pip install pyvips==2.1.15
pip install "aicsimageio[czi] @ git+https://github.com/AllenCellModeling/aicsimageio.git"

########################################################################
## Install AIDA

tput setaf 1; echo "** Install AIDA"; tput sgr0

cd ${HOME}/Software
if [[ -d "${HOME}/Software/AIDA" ]]; then
    echo "** AIDA already in ${HOME}/Software/AIDA"
else
    echo "** Cloning AIDA from github"
    git clone https://${USER}@github.com/alanaberdeen/AIDA.git

    # fix bug
    cd ${HOME}/Software/AIDA
    sed -i 's/cp aidaLocal/cp -r aidaLocal/g' package.json

    # AIDA build
    cd ${HOME}/Software/AIDA
    npm install
    cd ${HOME}/Software/AIDA/aidaLocal
    npm install
    cd ${HOME}/Software/AIDA
    npm run-script build

    # create data
    mkdir -p ${HOME}/Data/cytometer_data/aida_data
    cd ${HOME}/Software/AIDA/dist
    ln -s ${HOME}/Data/cytometer_data/aida_data data
fi
