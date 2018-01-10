# install_dependencies.sh
#
#    Script to install the dependencies to run cytometer scripts.
#
#    This script installs several Ubuntu packages, and creates two local conda environments:
#      * cytometer: With the latest versions of Keras/Theano/Tensorflow to run cytometer scripts.
#      * DeepCell: Based on Keras 1.1.1 and Theano 0.9.0 to run DeepCell legacy code.

#!/bin/bash

tput setaf 1; echo "** NVIDIA drivers"; tput sgr0

## CUDA Toolkit installation. There are several options (uncomment the preferred one):

# Option 1. CUDA Toolkit from the Ubuntu distribution packages
# sudo apt install -y nvidia-cuda-dev nvidia-cuda-toolkit

# Option 2. From Nvidia website: CUDA Toolkit 9.1 for Ubuntu 16.04
# pushd ~/Downloads || exit 1
# wget https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda-repo-ubuntu1604-9-1-local_9.1.85-1_amd64 || exit 1
# sudo dpkg -i cuda-repo-ubuntu1604-9-1-local_9.1.85-1_amd64.deb
# sudo apt-key add /var/cuda-repo-9-1-local/7fa2af80.pub
# sudo apt-get update
# sudo apt-get install cuda
# popd

# Option 3. From Nvidia website: CUDA Toolkit 9.1 for Ubuntu 17.04
pushd ~/Downloads || exit 1
wget https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda-repo-ubuntu1704-9-1-local_9.1.85-1_amd64
sudo dpkg -i cuda-repo-ubuntu1704-9-1-local_9.1.85-1_amd64.deb
sudo apt-key add /var/cuda-repo-9-1-local/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
popd

tput setaf 1; echo "** Build tools"; tput sgr0

# build tools
sudo apt install -y cmake

# for DeepCell
sudo apt install gcc-5 g++-5

# python IDE
sudo snap install pycharm-community --classic

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
else
    tput setaf 1; echo "** Conda local environment already exists (...skipping): cytometer"; tput sgr0
fi

source activate cytometer

# install Tensorflow, Theano and keras latest version from source
pip install tensorflow-gpu pyyaml || exit 1
pip install git+https://github.com/fchollet/keras.git --upgrade --no-deps || exit 1
pip install git+https://github.com/Theano/Theano.git --upgrade --no-deps || exit 1
pip install nose-parameterized || exit 1
conda install -y Cython cudnn=6 mkl-service || exit 1

# install libgpuarray from source, with python bindings
cd ~/Software
if [ -d libgpuarray ]; then # previous version present
    cd libgpuarray
    git pull || exit 1
else # no previous version exists
    git clone https://github.com/Theano/libgpuarray.git || exit 1
    cd libgpuarray
    mkdir Build
fi
cd Build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX:PATH=$CONDA_PREFIX || exit 1
make || exit 1
make install || exit 1
cd ..
python setup.py -q build_ext -L $CONDA_PREFIX/lib -I $CONDA_PREFIX/include || exit 1
python setup.py -q install --prefix=$CONDA_PREFIX || exit 1

# install other python packages
conda install -y matplotlib pillow spyder || exit 1
conda install -y scikit-image scikit-learn h5py || exit 1
conda install -y -c conda-forge tifffile mahotas || exit 1
conda install -y nose pytest || exit 1
pip install opencv-python pysto || exit 1

########################################################################
## python environment for DeepCell
if [ -z "$(conda info --envs | sed '/^#/ d' | cut -f1 -d ' ' | grep -w DeepCell)" ]; then
    tput setaf 1; echo "** Create conda local environment: DeepCell"; tput sgr0
    conda create -y --name DeepCell python=2.7
else
    tput setaf 1; echo "** Conda local environment already exists (...skipping): DeepCell"; tput sgr0
fi

source activate DeepCell

# current latest version of pip (9.0.1) gives "Command 'lsb_release
# -a' returned non-zero exit status 1" error, so we need to downgrade
# to 8.1.2
conda install -y pip=8.1.2 || exit

# install Keras 1
conda install -y keras=1.1.1 theano=0.9.0 || exit 1
conda install -y cudnn=5.1 pygpu=0.6.9 || exit 1

# install other python packages
conda install -y matplotlib pillow spyder || exit 1
conda install -y scikit-image scikit-learn h5py || exit 1
conda install -y -c conda-forge tifffile mahotas || exit 1
conda install -y nose pytest || exit 1
pip install opencv-python pysto || exit 1

# clear Theano cache. Previous runs of Keras may cause CUDA compilation/version compatibility problems
theano-cache purge
