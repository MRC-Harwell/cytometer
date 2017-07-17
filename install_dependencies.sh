#!/bin/bash

# upgrade Ubuntu
sudo apt dist-upgrade

# nVIDIA CUDA drivers and toolkit
sudo apt install nvidia-cuda-dev nvidia-cuda-toolkit

# build tools
sudo apt install cmake

# Conda package manager
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod u+x Miniconda3-latest-Linux-x86_64.sh
sudo ./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# python environment
conda create --name cytometer python=3.6
source activate cytometer
pip install tensorflow-gpu pyyaml
pip install git+https://github.com/fchollet/keras.git --upgrade --no-deps
pip install git+https://github.com/Theano/Theano.git --upgrade --no-deps
pip install nose-parameterized
conda install -y cudnn=6

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
