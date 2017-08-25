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

# python environment
conda create -y --name cytometer python=3.6
source activate cytometer
pip install tensorflow-gpu pyyaml
pip install git+https://github.com/fchollet/keras.git --upgrade --no-deps
pip install git+https://github.com/Theano/Theano.git --upgrade --no-deps
pip install nose-parameterized
conda install -y Cython cudnn=6

cd ~/Software
git clone https://github.com/Theano/libgpuarray.git
cd libgpuarray
mkdir Build
cd Build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX:PATH=$CONDA_PREFIX
make install
cd ..
python setup.py build_ext -L $CONDA_PREFIX/lib -I $CONDA_PREFIX/include
python setup.py install --prefix=$CONDA_PREFIX

conda install -y matplotlib pillow spyder

conda install -y scikit-image scikit-learn h5py
conda install -y -c conda-forge tifffile mahotas

conda install -y nose pytest
