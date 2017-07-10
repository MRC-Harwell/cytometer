Instructions to set up project `cytometer`.
===========================================

# GPU drivers and python package manager

1. Make sure you are using the nVIDIA drivers, instead of `xserver-xorg-video-nouveau`.
1. Install NVIDIA CUDA development files

        sudo apt-get install nvidia-cuda-dev nvidia-cuda-toolkit
1. Install [conda](https://conda.io/docs/intro.html)
   1. Download the [Miniconda bash installer](https://conda.io/miniconda.html) (e.g. for linux 64-bit, `Miniconda2-latest-Linux-x86_64.sh`).
   1. Make the script executable and run it as root

            cd ~/Downloads
            chmod u+x Miniconda2-latest-Linux-x86_64.sh
            sudo ./Miniconda2-latest-Linux-x86_64.sh
   1. When it asks for the install destination, select `/opt/miniconda2`, rather than the default `/home/rcasero/miniconda2`.
   1. "Do you wish the installer to prepend the Miniconda2 install location to PATH in your /home/rcasero/.bashrc ? [yes|no]". Select yes.

## Checking your GPU set-up

1. Check that you have a working GPU

        (cytometer) $ nvidia-smi
        Wed Jun 28 15:20:35 2017
        +-----------------------------------------------------------------------------+
        | NVIDIA-SMI 375.66                 Driver Version: 375.66                    |
        |-------------------------------+----------------------+----------------------+
        | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
        | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
        |===============================+======================+======================|
        |   0  Quadro K4000        Off  | 0000:01:00.0      On |                  N/A |
        | 30%   39C    P8    12W /  87W |   1490MiB /  3016MiB |      0%      Default |
        +-------------------------------+----------------------+----------------------+
                                                                                        
        +-----------------------------------------------------------------------------+
        | Processes:                                                       GPU Memory |
        |  GPU       PID  Type  Process name                               Usage      |
        |=============================================================================|
        |    0      1372    G   /usr/lib/xorg/Xorg                             365MiB |
        |    0      1974    G   compiz                                         154MiB |
        |    0      2283    G   ...el-token=493C1790BE0AE309A3CB57689C7C3E71   146MiB |
        +-----------------------------------------------------------------------------+

# Create `conda` virtual environments

1. Create a conda environment for cytometer
 * Python 3.5 for Theano 0.8.2

            conda create --name cytometer_py35 python=3.5
 * Python 3.6 for Theano 0.9.0

            conda create --name cytometer_py36 python=3.6
1. Activate the conda environment that you intend to run

        source activate cytometer_py35
    or

        source activate cytometer_py36

# Preparing virtual python environment to run `cytometer`

1. Install python dependencies
 * Theano 0.8.2

            # Basic python dependencies
            conda install matplotlib=2.0.2 pillow=4.2.1 spyder
            # We need to install an older version of Theano, because theano.test() segfaults with Theano 0.9.0 (newest version at the time of this writing)
            conda install theano=0.8.2
            # Basic CNN environment dependencies
            conda install keras=2.0.2 tensorflow=1.1.0 tensorflow-gpu=1.1.0 cudnn=5.1 pygpu=0.6.8
            # DeepCell dependencies
            conda install scikit-image=0.13.0 scikit-learn=0.18.2
            conda install -c conda-forge tifffile=0.12.1 mahotas=1.4.3
            # For testing theano
            conda install nose-parameterized=0.5.0
 * or Theano 0.9.0

            # Basic python dependencies
            conda install matplotlib=2.0.2 pillow=4.2.1 spyder
            # theano.test() segfaults with Theano 0.9.0 (newest version at the time of this writing)
            conda install theano=0.9.0
            # Basic CNN environment dependencies
            conda install keras=2.0.2 tensorflow=1.1.0 tensorflow-gpu=1.1.0 cudnn=5.1 pygpu=0.6.8
            # DeepCell dependencies
            conda install scikit-image=0.13.0 scikit-learn=0.18.2
            conda install -c conda-forge tifffile=0.12.1 mahotas=1.4.3
            # For testing theano
            conda install nose-parameterized=0.5.0
1. So that we can have a Keras configuration for DeepCell and another for our project, 
we are not going to use `~/.keras/keras.json`. Instead, we add snippets like this
to the beginning of every python script

       
        import os
        import keras
        os.environ['KERAS_BACKEND'] = 'theano'
        reload(keras.backend)
        keras.backend.set_image_dim_ordering('th')
1. If you want to use Theano as the backend, create a soft link from `~/.theanorc` to the corresponding file, e.g.
if you are running `cytometer/scripts/basic_cnn.py`

        ln -s ~/Software/cytometer/scripts/basic_cnn.theanorc ~/.theanorc

1. If you want to use Tensorflow as the backend, it will use the GPU automatically 
if one is available, you don't need a configuration file
1. In python, choose a backend. E.g. Tensorflow

        import os
        os.environ['KERAS_BACKEND'] = 'tensorflow'
        os.environ['LIBRARY_PATH'] = '/home/rcasero/.conda/envs/cytometer_py36/lib'
        import keras
        keras.backend.set_image_data_format('channels_first') # theano's image format (required by DeepCell)
   or Theano

        import os
        os.environ['KERAS_BACKEND'] = 'theano'
        os.environ['LIBRARY_PATH'] = '/home/rcasero/.conda/envs/cytometer_py36/lib'
        import keras
        keras.backend.set_image_data_format('channels_first') # theano's image format (required by DeepCell)

## Tests

1. To test `pygpu`

        DEVICE=cuda python -c "import pygpu;pygpu.test()"
        pygpu is installed in /home/rcasero/.conda/envs/cytometer/lib/python3.6/site-packages/pygpu
        NumPy version 1.12.1
        NumPy relaxed strides checking option: True
        NumPy is installed in /home/rcasero/.conda/envs/cytometer/lib/python3.6/site-packages/numpy
        Python version 3.6.1 |Continuum Analytics, Inc.| (default, May 11 2017, 13:09:58) [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
        nose version 1.3.7
        *** Testing for Quadro K4000
1. To test `theano`

        PYTHONPATH=~/Software/cytometer python -c 'import theano; theano.test()'

# Install `cytometer`

# Installation

Currently, we have not tested having `cytometer` installed as a python package, 
as we are at a very early stage.

For the time being, we are working by simply cloning the repository to the local
drive.

1. Clone the `cytometer` repository by running the command

        cd ~/Software
        git clone git@phobos.mrch.har.mrc.ac.uk:r.casero/cytometer.git
1. Change directory to the project

        cd cytometer
1. Add the project to the Spyder3 path. Select Tools -> PYTHONPATH manager, click "+", select `/home/rcasero/Software/cytometer`
 * *Note:* Once the package is installed, `/home/rcasero/Software/cytometer/cytometer` will be copied to `~/.conda/envs/cytometer/lib/python3.6/site-packages/`.
PYTHONPATH points to `~/.conda/envs/cytometer/lib/python3.6/site-packages/`, which is why in development PYTHONPATH has to point at `/home/rcasero/Software/cytometer`
instead of `/home/rcasero/Software/cytometer/cytometer`