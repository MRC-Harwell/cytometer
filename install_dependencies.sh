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

# configuration constants
MINICONDA_VERSION=3

########################################################################
## python environment for cytometer

# install OpenSlide library
sudo apt install -y openslide-tools

# install LaTeX dependencies for matplotlib
sudo apt install -y texlive-latex-base texlive-latex-extra
sudo apt install -y dvipng

# install GNU R with lme4 module (generilised linear mixed models)
sudo apt install -y r-base r-cran-lme4

# download or update code from python_setup repository
if [[ -d ~/Software/python_setup ]]
then
    tput setaf 1; echo Updating python_setup from github; tput sgr0
    pushd ~/Software/python_setup/
    git pull
    popd
else
    tput setaf 1; echo Cloning python_setup from github; tput sgr0
    pushd ~/Software/
    git clone https://rcasero@github.com/rcasero/python_setup.git
    popd
fi

# install Miniconda
mkdir -p ~/Downloads
~/Software/python_setup/bin/install_miniconda.sh ${MINICONDA_VERSION}
export PATH=${HOME}/Software/miniconda${MINICONDA_VERSION}/bin:${PATH}

UBUNTU_VERSION=`lsb_release -r | tr -d [:blank:] | sed -e "s/^Release://"`
echo UBUNTU_VERSION=$UBUNTU_VERSION

# install nVidia drivers and CUDA
case "$UBUNTU_VERSION"
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

# create or update environment for development with Keras
~/Software/python_setup/bin/install_keras_environment.sh -e cytometer_tensorflow -b tensorflow
source activate cytometer_tensorflow

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

########################################################################
## python environment for DeepCell

#~/Software/python_setup/bin/install_deepcell_environment.sh
