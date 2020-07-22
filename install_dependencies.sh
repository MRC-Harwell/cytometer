# install_dependencies.sh
#
#    Script to install the dependencies to run cytometer scripts.
#
#    This script installs several Ubuntu packages, and creates two local conda environments:
#      * cytometer: With the latest versions of Keras/Theano/Tensorflow to run cytometer scripts.
#      * DeepCell: Based on Keras 1.1.1 and Theano 0.9.0 to run DeepCell legacy code.

#    Copyright © 2018-2020  Ramón Casero <rcasero@gmail.com>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

#!/bin/bash

# exit immediately on errors that are not inside an if test, etc.
set -e

########################################################################
# configuration constants

MINICONDA_VERSION=3
PYTHON_VERSION=3.6
CONDA_LOCAL_ENV=cytometer_tensorflow
BACKEND=tensorflow

########################################################################
## Ubuntu packages and other binaries that the python local environment is going to need

# install OpenSlide library
sudo apt install -y openslide-tools

# install LaTeX dependencies for matplotlib
sudo apt install -y texlive-latex-base texlive-latex-extra
sudo apt install -y dvipng

# install GNU R with lme4 module (generilised linear mixed models)
sudo apt install -y r-base r-cran-lme4

########################################################################
## install Miniconda so that we can use the conda local environment tools,
## and install python packages with pip and conda

# install Miniconda
mkdir -p ~/Downloads

if [[ -d "${HOME}/Software/miniconda${MINICONDA_VERSION}" ]];
then
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
    set +e
    isInBashrc=`grep  -c "export PATH=${HOME}/Software/miniconda${MINICONDA_VERSION}/bin" ~/.bashrc`
    set -e
    if [[ "$isInBashrc" -eq 0 ]];
    then
	echo "Adding ${HOME}/Software/miniconda${MINICONDA_VERSION}/bin to PATH in ~/.bashrc"
	echo "
# added by pysto/tools/install_miniconda.sh
export PATH=${HOME}/Software/miniconda${MINICONDA_VERSION}/bin:\"\$PATH\"" >> ~/.bashrc
	source ~/.bashrc
    else
	echo "${HOME}/Software/miniconda${VERSION}/bin already in PATH in ~/.bashrc"
    fi
    popd
fi

export PATH=${HOME}/Software/miniconda${MINICONDA_VERSION}/bin:${PATH}

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

case ${UBUNTU_VERSION}
in
    16.04)
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
        sudo mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
        sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
        sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/ /"
        sudo apt-get update
        sudo apt-get -y install cuda
        ;;
    18.04)
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
        sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
        sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
        sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
        sudo apt-get update
        sudo apt-get -y install cuda
        ;;
    20.04)
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
        sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
        sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
        sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
        sudo apt-get update
        sudo apt-get -y install cuda
        ;;
    *)
        echo "Error: Ubuntu version not recognised: $UBUNTU_VERSION"
        exit 1
        ;;
esac

########################################################################
## create python local environment for cytometer

# if the environment doesn't exist, we create a new one. If it does,
# we add the python packages to it

# check whether the environment already exists
if [[ -z "$(conda info --envs | sed '/^#/ d' | cut -f1 -d ' ' | grep -w ${CONDA_LOCAL_ENV})" ]]; then
    tput setaf 1; echo "** Create conda local environment: ${CONDA_LOCAL_ENV}"; tput sgr0
    conda create -y --name ${CONDA_LOCAL_ENV} python=${PYTHON_VERSION}
else
    tput setaf 1; echo "** Conda local environment already exists: ${CONDA_LOCAL_ENV}"; tput sgr0
fi

########################################################################
## install keras python packages in the local environment

tput setaf 1; echo "** Install keras python packages in the local conda environment"; tput sgr0

# switch to the local environment
source activate ${CONDA_LOCAL_ENV}

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

echo "** Dependencies for Tensorflow backend"
pip install tensorflow-gpu==1.13.1 #pyyaml==5.1.1

# install my own Keras 2.2 version modified to accept partial training data
pip install git+https://github.com/rcasero/keras.git

NVIDIA_DRIVER_VERSION=`nvidia-smi --query-gpu=driver_version --format=csv,noheader --id=0`

# table of CUDA version vs. nVidia driver version
# https://docs.nvidia.com/deeplearning/sdk/cudnn-support-matrix/index.html
case ${NVIDIA_DRIVER_VERSION}
in
    450.*)  # CUDA 11.0.x
        # we should be installing a package built for cuda11, but it's not available
        conda install -y cudnn==7.6.5=cuda10.2_0
        ;;
    *)
        echo "cudnn version for detected nVidia driver version (v. *${NVIDIA_DRIVER_VERSION}*) is unknown"
        exit 1
        ;;
esac

# install dependencies for Keras
conda install -y h5py==2.9.0        # to save Keras models to disk
conda install -y graphviz==2.40.1   # used by visualization utilities to plot model graphs
pip install cython==0.29.10         # dependency of mkl-random/mkl-fft via pydot
pip install pydot==1.4.1            # used by visualization utilities to plot model graphs

# for tests
pip install pytest==4.6.3

########################################################################
## install cytometer python dependencies packages in the local environment

tput setaf 1; echo "** Install cytometer python dependencies in the local conda environment"; tput sgr0

# install other python packages
pip install git+https://www.github.com/keras-team/keras-contrib.git  # tested with version 2.0.8
conda install -y matplotlib==3.1.0 pillow==6.0.0
conda install -y scikit-image==0.15.0 scikit-learn==0.21.2 h5py==2.9.0
conda install -y nose==1.3.7 pytest==5.4.3
pip install setuptools==45.0.0
pip install opencv-python==4.1.0.25 pysto==1.4.1 openslide-python==1.1.1 seaborn==0.10.0 statannot==0.2.3
pip install tifffile==2019.5.30 mahotas==1.4.5 networkx==2.3 svgpathtools==1.3.3 receptivefield==0.4.0 rpy2==3.0.5
pip install mlxtend==0.17.0 ujson==1.35
conda install -y pandas==0.24.2 six==1.12.0 statsmodels==0.10.1
