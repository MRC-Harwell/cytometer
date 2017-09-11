# install_dependencies.sh
#
#    Script to install the dependencies to run cytometer scripts.
#
#    This script installs several Ubuntu packages, and creates two local conda environments:
#      * cytometer: With the latest versions of Keras/Theano/Tensorflow to run cytometer scripts.
#      * DeepCell: Based on Keras 1.1.1 and Theano 0.9.0 to run DeepCell legacy code.

#!/bin/bash

tput setaf 1; echo "** NVIDIA drivers"; tput sgr0

# nVIDIA CUDA drivers and toolkit
sudo apt install -y nvidia-cuda-dev nvidia-cuda-toolkit

tput setaf 1; echo "** Build tools"; tput sgr0

# build tools
sudo apt install -y cmake

# BLAS library, development version, so that Theano code can be compiled with it
sudo apt install -y libblas-dev

# Conda package manager
if hash conda 2>/dev/null; then
    tput setaf 1; echo "** Conda already installed"; tput sgr0
else
    tput setaf 1; echo "** Installing conda"; tput sgr0
    # download installer
    if [ ! -e Miniconda3-latest-Linux-x86_64.sh ]
    then
	wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    fi
    # install conda
    chmod u+x Miniconda3-latest-Linux-x86_64.sh
    sudo ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3
    source ~/.bashrc
fi

########################################################################
## python environment for cytometer
if [ -z "$(conda info --envs | sed '/^#/ d' | cut -f1 -d ' ' | grep -w cytometer)" ]; then
    tput setaf 1; echo "** Create conda local environment: cytometer"; tput sgr0
    conda create -y --name cytometer python=3.6
    source activate cytometer

    # install Tensorflow, Theano and keras latest version from source
    pip install tensorflow-gpu pyyaml
    pip install git+https://github.com/fchollet/keras.git --upgrade --no-deps
    pip install git+https://github.com/Theano/Theano.git --upgrade --no-deps
    pip install nose-parameterized
    conda install -y Cython cudnn=6

    # install libgpuarray from source, with python bindings
    cd ~/Software
    if [ -d libgpuarray ]; then # previous version present
	cd libgpuarray
	git pull
    else # no previous version exists
	git clone https://github.com/Theano/libgpuarray.git
	cd libgpuarray
	mkdir Build
    fi
    cd Build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX:PATH=$CONDA_PREFIX
    make install
    cd ..
    python setup.py build_ext -L $CONDA_PREFIX/lib -I $CONDA_PREFIX/include
    python setup.py install --prefix=$CONDA_PREFIX

    # install gcc in conda to avoid CUDA compilation problems
    conda install -y gcc

    # install other python packages
    conda install -y matplotlib pillow spyder
    conda install -y scikit-image scikit-learn h5py
    conda install -y -c conda-forge tifffile mahotas
    conda install -y nose pytest
else
    tput setaf 1; echo "** Conda local environment already exists (...skipping): cytometer"; tput sgr0
fi

########################################################################
## python environment for DeepCell
if [ -z "$(conda info --envs | sed '/^#/ d' | cut -f1 -d ' ' | grep -w DeepCell)" ]; then
    tput setaf 1; echo "** Create conda local environment: DeepCell"; tput sgr0
    conda create -y --name DeepCell python=2.7
    source activate DeepCell

    # install Keras 1
    conda install -y keras=1.1.1 theano=0.9.0
    conda install -y Cython cudnn=5.1 pygpu=0.6.9

    # install gcc in conda to avoid CUDA compilation problems
    conda install -y gcc

    # install other python packages
    conda install -y matplotlib pillow spyder
    conda install -y scikit-image scikit-learn h5py
    conda install -y -c conda-forge tifffile mahotas
    conda install -y nose pytest

    # clear Theano cache. Previous runs of Keras may cause CUDA compilation/version compatibility problems
    theano-cache purge
else
    tput setaf 1; echo "** Conda local environment already exists (...skipping): DeepCell"; tput sgr0
fi
