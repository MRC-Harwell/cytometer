Instructions to set up project `cytometer`.
===========================================

# Table of Contents

   * [Instructions to set up project cytometer.](#instructions-to-set-up-project-cytometer)
   * [Installing the cytometer python code](#installing-the-cytometer-python-code)
   * [Dependencies and local conda environment](#dependencies-and-local-conda-environment)
      * [GPU drivers and python package manager](#gpu-drivers-and-python-package-manager)
         * [Checking your GPU set-up](#checking-your-gpu-set-up)
      * [Create conda virtual environment](#create-conda-virtual-environment)
      * [Preparing virtual python environment to run cytometer](#preparing-virtual-python-environment-to-run-cytometer)
         * [Tests](#tests)
   * [Packaging the cytometer python code](#packaging-the-cytometer-python-code)
   * [Running cytometer python scripts](#running-cytometer-python-scripts)
   * [Cloud computing on Amazon Web Services](#cloud-computing-on-amazon-web-services)
      * [Creating a test instance](#creating-a-test-instance)
      * [Copying files to the remote instance](#copying-files-to-the-remote-instance)

Created by [gh-md-toc](https://github.com/ekalinin/github-markdown-toc)


# Installing the `cytometer` python code

Currently, we have not tested having `cytometer` installed as a python package, 
as we are at a very early stage.

For the time being, we are working by simply cloning the repository to the local
drive.

1. Clone the `cytometer` repository by running the command

        cd ~/Software
        git clone http://r.casero@phobos.mrch.har.mrc.ac.uk/r.casero/cytometer.git
1. Change directory to the project

        cd cytometer

# Dependencies and local conda environment

You can install the dependencies and set up the local conda environment running
the shell script `install_dependencies.sh` (some user interaction required), or 
manually following the steps below

## GPU drivers and python package manager

1. Make sure you are using the nVIDIA drivers, instead of `xserver-xorg-video-nouveau`.
1. Install NVIDIA CUDA development files

        sudo apt-get install nvidia-cuda-dev nvidia-cuda-toolkit
1. Install [conda](https://conda.io/docs/intro.html)
   1. Download the [Miniconda bash installer](https://conda.io/miniconda.html) (e.g. for linux 64-bit, `Miniconda2-latest-Linux-x86_64.sh`).
   1. Make the script executable and run it as root

            cd ~/Downloads
            chmod u+x Miniconda3-latest-Linux-x86_64.sh
            sudo ./Miniconda3-latest-Linux-x86_64.sh
   1. When it asks for the install destination, select `/opt/miniconda2`, rather than the default `/home/rcasero/miniconda2`.
   1. "Do you wish the installer to prepend the Miniconda2 install location to PATH in your /home/rcasero/.bashrc ? [yes|no]". Select yes.

### Checking your GPU set-up

1. Check that you have a working GPU

        $ nvidia-smi
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

## Create `conda` virtual environment

1. Create a conda environment for cytometer
 * Python 3.6 for Keras/Theano master versions

            conda create --name cytometer python=3.6

## Preparing virtual python environment to run `cytometer`

Installing official conda packages for Keras/Theano didn't work for me. Installing
Theano 0.8.2 with Keras 2.0.2 and python 3.5 would fail to `import theano` due to
compilation errors when the GPU was selected, whereas Theano 0.9.0 would make Keras segfault with `model.add()`
using the GPU. Instead, we work with the latest `master` versions of Keras and Theano.

1. Install BLAS library, development version, so that Theano code can be compiled with it

        sudo apt install libblas-dev
1. Install python packages

        # Select local conda environment
        source activate cytometer
        
        # install keras, theano and tensorflow (this way we get their dependencies)
        # As of this writing: keras-2.0.6 numpy-1.13.1 pyyaml-3.12 scipy-0.19.1 six-1.10.0 theano-0.9.0
        # backports.weakref-1.0rc1 bleach-1.5.0 html5lib-0.9999999 markdown-2.6.8 protobuf-3.3.0 tensorflow-1.2.1 tensorflow-gpu-1.2.1 werkzeug-0.12.2
        pip install tensorflow-gpu pyyaml
        
        # Upgrade keras and theano to latest versions
        # As of this writing: Keras-2.0.6, Theano-0.10.0.dev1
        pip install git+https://github.com/fchollet/keras.git --upgrade --no-deps
        pip install git+https://github.com/Theano/Theano.git --upgrade --no-deps
        
        # For theano.test()
        # As of this writing: nose-parameterized-0.6.0
        pip install nose-parameterized
        
        # Tensorflow/Theano GPU dependencies
        # As of this writing: Cython=0.26
        conda install Cython
    If you want to use tensorflow, current package needs cudnn 5.x
    
        conda install cudnn=5
        
    If you want to use theano, current package needs cudnn 6.x
    
        conda install cudnn=6
    Rest of common packages
    
        # Build and install libgpuarray/pygpu from source (we are going to install in the local conda environment)
        # As of this writing: pygpu=0.6.8 libgpuarray=0.6.8
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
        
        # Basic python dependencies
        # As of this writing: matplotlib=2.0.2 pillow=4.2.1 spyder=3.1.4
        conda install -y matplotlib pillow spyder
        
        # DeepCell dependencies
        # As of this writing: scikit-image=0.13.0 scikit-learn=0.18.2 h5py=2.7.0
        # tifffile=0.12.0 mahotas=1.4.3
        conda install -y scikit-image scikit-learn h5py
        conda install -y -c conda-forge tifffile mahotas
        
        # go back to the cytometer directory
        cd ~/Software/cytometer
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
   * *Note:* `nvcc.flags=-D_FORCE_INLINES` needed to avoid `error: ‘memcpy’ was not declared in this scope`
1. If you want to use Tensorflow as the backend, it will use the GPU automatically 
if one is available, you don't need a configuration file
1. In python, choose a backend. E.g. Tensorflow

        import os
        os.environ['KERAS_BACKEND'] = 'tensorflow'
        os.environ['LIBRARY_PATH'] = '/home/rcasero/.conda/envs/cytometer/lib'
        import keras
        keras.backend.set_image_data_format('channels_first') # theano's image format (required by DeepCell)
   or Theano

        import os
        os.environ['KERAS_BACKEND'] = 'theano'
        os.environ['LIBRARY_PATH'] = '/home/rcasero/.conda/envs/cytometer/lib'
        import keras
        keras.backend.set_image_data_format('channels_first') # theano's image format (required by DeepCell)

### Tests

1. To test `pygpu`

        DEVICE=cuda python -c "import pygpu;pygpu.test()"
        pygpu is installed in /home/rcasero/.conda/envs/cytometer/lib/python3.6/site-packages/pygpu
        NumPy version 1.12.1
        NumPy relaxed strides checking option: True
        NumPy is installed in /home/rcasero/.conda/envs/cytometer/lib/python3.6/site-packages/numpy
        Python version 3.6.1 |Continuum Analytics, Inc.| (default, May 11 2017, 13:09:58) [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
        nose version 1.3.7
        *** Testing for Quadro K4000
1. To run `theano` tests

        PYTHONPATH=~/Software/cytometer python -c 'import theano; theano.test()'
1. To check whether you can import keras with theano backend

        PYTHONPATH=~/Software/cytometer python -c 'import os; os.environ["KERAS_BACKEND"] = "theano"; os.environ["LIBRARY_PATH"] = "/home/rcasero/.conda/envs/cytometer/lib"; import keras'

# Packaging the cytometer python code

The `setup.py` and associated files to create a package are in place. You can create a python package with

    cd ~/Software/cytometer
    ./setup.py sdist

# Running cytometer python scripts

You need to set 

* `LD_LIBRARY_PATH` from the shell so that e.g. Theano can find libcudnn.so
 * for some reason, setting `os.environ['LD_LIBRARY_PATH']` from the python script doesn't work).
 * Setting `LD_LIBRARY_PATH` in `~/.bashrc` won't work because when `.bashrc` is read (upon opening a shell), you have not activated a conda environment yet
* `PYTHONPATH` so that python can find the cytometer modules

If you are working in Spyder,

    PYTHONPATH=~/Software/cytometer LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH spyder &

To run the script directly from the shell

    PYTHONPATH=~/Software/cytometer LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python scripts/basic_cnn.py

Or simply set and export the environmental variables once

    export PYTHONPATH=~/Software/cytometer
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

and then call spyder as

    spyder &

or run the script as

    python scripts/basic_cnn.py

# Cloud computing on Amazon Web Services

## Creating a test instance

1. Create [Amazon Web Services](https://aws.amazon.com) account.
2. Log into your [AWS Management Console](https://console.aws.amazon.com/).
3. Choose [Launch a virtual machine](https://us-east-2.console.aws.amazon.com/quickstart/vm/home).
4. Click on "EC2 Instance", "Get Started".
5. Choose a name for the EC2 Instance, e.g. "cytometer-basic-cnn".
6. Select an Operating system, e.g. Ubuntu (Server 16.04 LTS at the time of this writing).
7. Select as instance type "t2.micro" (1 Core vCPU (up to 3.3 GHz), 1 GiB Memory RAM, 8 GB Storage).
8. Create a key pair, by clicking on "Download" and saving the private key `cytometer-basic-cnn.pem` to e.g. `~/Software/cytometer/private_keys`.
 * If you lose the private key, you won't be able to recover it, or connect to the instance.
9. Click on "Create this instance". Wait a bit until the instance is created.
10. Click on "Proceed to EC2 console"
11. Right click on your instance and select "Connect".
12. Enable connections from your machine.
 1. Go to your instance -> Network & Security -> Security Groups.
 2. Go to this intance's group, e.g. "cytometer-basic-cnn" -> Inbound.
 3. Make sure there's a rule

            SSH     TCP     22      YOUR_PUBLIC_IP/32
    where YOUR_PUBLIC_IP is the IP address your machine provides to the external world. The autodected one may be wrong. You can find it by e.g. googling "What's my ip"
13. Follow the instructions to connect with a command line SSH client or a Java SSH client from the browser (the latter no longer supported by the Google Chrome browser). For example, if you are going to use the command line SSH client,
 1. Make the private key not publicly readable
 
            cd ~/Software/cytometer/private_keys
            chmod 400 cytometer-basic-cnn.pem
 2. Connect to the instance

            ssh -i "cytometer-basic-cnn.pem" ubuntu@XXX-XX-XX-XX-XX.us-east-2.compute.amazonaws.com
    where `XXX-XX-XX-XX-XX` is a code specific to your instance.
    
## Copying files to the remote instance

1. Run from the command line something similar to this (this one copies file `install_dependencies.sh` to the remote instance)

        scp -i private_keys/cytometer-basic-cnn.pem install_dependencies.sh  ubuntu@XXX-XX-XX-XX-XX.us-east-2.compute.amazonaws.com:~/
