#!/bin/bash
# install_dependencies_machine.sh
#
#    Script to install the nVidia drivers and other Ubuntu packages to set up a machine to run DeepCytometer. This needs
#    sudo privileges.
#
#    After this, users can run scripts:
#
#       * install_dependencies_user.sh: create cytometer_tensorflow environment to run the DeepCytometer pipeline and
#         experiments.
#
#       * install_dependencies_user_v2.sh: create cytometer_tensorflow_v2 environment to convert Zeiss CZI files to
#         DeepZoom.

# This file is part of Cytometer
# Copyright 2021 Medical Research Council
# SPDX-License-Identifier: Apache-2.0
# Author: Ramon Casero <rcasero@gmail.com>

# exit immediately on errors that are not inside an if test, etc.
set -e

########################################################################
# configuration constants

MINICONDA_VERSION=3

########################################################################
## Ubuntu packages and other binaries that the python local environment is going to need

# install OpenSlide library
sudo apt install -y openslide-tools

# install LaTeX dependencies for matplotlib
sudo apt install -y texlive-latex-base texlive-latex-extra
sudo apt install -y dvipng

# install GNU R with lme4 module (generilised linear mixed models)
sudo apt install -y r-base r-cran-lme4

# this is so that pyvips can build a libvips binary extension for python (https://github.com/libvips/pyvips)
sudo apt install -y libvips-dev

########################################################################
## nVidia drivers and CUDA

tput setaf 1; echo "** Install CUDA"; tput sgr0

# find out which Ubuntu version this machine is using
UBUNTU_VERSION=`lsb_release -r | tr -d [:blank:] | sed -e "s/^Release://"`
if [[ -z "$UBUNTU_VERSION" ]]; then
    echo "Ubuntu version could not be found"
    exit 1
else
    echo UBUNTU_VERSION=${UBUNTU_VERSION}
fi

# CUDA Toolkit 10.2
case ${UBUNTU_VERSION}
in
    20.04)
        # Set up nVidia repositories
        # https://www.tensorflow.org/install/gpu#install_cuda_with_apt
        tput setaf 1; echo "  ** Warning! CUDA packages are not available for Ubuntu 20.04, so we are going to use the Ubuntu 18.04 ones"; tput sgr0
        pushd ~/Downloads
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
        sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
        sudo dpkg -i cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
        sudo apt update
        wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
        sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
        sudo apt update
        # Install NVIDIA driver (not needed, cuda-10-2 installs it as a dependency)
        # sudo apt-get install --no-install-recommends nvidia-driver-455
        # Install development and runtime libraries (~4GB)
        sudo apt-get install --no-install-recommends \
            cuda-10-2 \
            libcudnn7=7.6.5.32-1+cuda10.2  \
            libcudnn7-dev=7.6.5.32-1+cuda10.2
        # Install TensorRT. Requires that libcudnn7 is installed above.
        sudo apt-get install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.2 \
            libnvinfer-dev=6.0.1-1+cuda10.2 \
            libnvinfer-plugin6=6.0.1-1+cuda10.2
        # prevent these last packages from being upgraded, as cuda 11 is available but we don't want it because it's
        # not compatible with our chosen version of tensorflow
        sudo apt-mark hold libnvinfer-dev libnvinfer-plugin7 libnvinfer7
        popd
        ;;
    *)
        echo "Error: Ubuntu version not recognised: $UBUNTU_VERSION"
        exit 1
        ;;
esac

########################################################################
## Install AIDA dependencies

sudo snap install node --classic --channel=12
sudo apt install libvips-tools
