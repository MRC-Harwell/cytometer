Instructions to set up project `cytometer`.
===========================================

# Table of Contents

   * [Instructions to set up project cytometer.](#instructions-to-set-up-project-cytometer)
   * [Table of Contents](#table-of-contents)
   * [Installing the cytometer python code](#installing-the-cytometer-python-code)
   * [Dependencies and local conda environments](#dependencies-and-local-conda-environments)
      * [Notes on Ubuntu dependencies](#notes-on-ubuntu-dependencies)
   * [Checking that your GPU is correctly set-up](#checking-that-your-gpu-is-correctly-set-up)
   * [Configuring Keras to be used in a python script](#configuring-keras-to-be-used-in-a-python-script)
   * [Configuring Theano to be used in a python script in a conda environment](#configuring-theano-to-be-used-in-a-python-script-in-a-conda-environment)
   * [Configuring Tensorflow to be used in a python script in a conda environment](#configuring-tensorflow-to-be-used-in-a-python-script-in-a-conda-environment)
   * [Testing conda environment from the command line](#testing-conda-environment-from-the-command-line)
   * [Packaging the cytometer python code](#packaging-the-cytometer-python-code)
   * [Running cytometer python scripts](#running-cytometer-python-scripts)
      * [If the IDE is spyder](#if-the-ide-is-spyder)
      * [If the IDE is pycharm-community](#if-the-ide-is-pycharm-community)
      * [To run a python script directly from the shell](#to-run-a-python-script-directly-from-the-shell)
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

# Dependencies and local conda environments

You can install the dependencies and set up the local conda environments running
the shell script [`install_dependencies.sh`](http://phobos/r.casero/cytometer/blob/master/install_dependencies.sh).

This will create two local environments:
* `cytometer`: for the code of this project.
* `DeepCell`: for experiments with the DeepCell architectures by D. Van Valen (Keras 1 / Theano).
   * Installed with [`install_deepcell_environment.sh`](https://github.com/rcasero/pysto/blob/master/tools/install_deepcell_environment.sh)


## Notes on Ubuntu dependencies

The `install_dependencies.sh` script:
* Installs the NVIDIA drivers (`nvidia-387`) rather than the open source nouveau 
  (`xserver-xorg-video-nouveau`), so that we have full access to the NVIDIA 
  graphic card features.
* Uses Miniconda to install conda and create local python environments.
* Installs the CUDA Toolkit from the [Nvidia website](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu), 
rather than using Ubuntu packages.

# Checking that your GPU is correctly set-up

1. Check that you get something similar to this on a terminal

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

# Configuring Keras to be used in a python script

Keras can be configured with file [`~/.keras/keras.json`](https://keras.io/backend/#kerasjson-details), 
but this will set the same configuration for all conda environments.

Alternatively, you can configure Keras in each separate script with code similar
to this:

    import os
    os.environ['KERAS_BACKEND'] = 'theano'
    import keras.backend as K
    K.set_image_dim_ordering('th') # theano's image format
    K.set_floatx('float32')
    K.set_epsilon('1e-07')

# Configuring Theano to be used in a python script in a conda environment

Theano can be configured with file `.theanorc`, but as above, this gives the same
configuration to all conda environments and scripts.

It's also possible to use assignments to `theano.config.<property>`. However, 
some options need to be set in `THEANO_FLAGS` **before** the `import theano` or
`import keras` statement.

In particular, to avoid CUDA/cuDNN compilation errors when we want to use the 
GPU, it's necessary to add something like this

    import os
    os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cuda0,lib.cnmem=0.75,' \
                                 + 'dnn.include_path=' + os.environ['CONDA_PREFIX'] + '/include,' \
                                 + 'dnn.library_path=' + os.environ['CONDA_PREFIX'] + '/lib,' \
                                 + 'gcc.cxxflags=-I/usr/local/cuda-9.1/targets/x86_64-linux/include,' \
                                 + 'nvcc.fastmath=True'
    import theano

# Configuring Tensorflow to be used in a python script in a conda environment


If you want to use Tensorflow as the backend, it will use the GPU automatically 
if one is available, you don't need a configuration file.

# Testing conda environment from the command line

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

The `setup.py` and associated files to create a package are in place. For the moment, you can create a python source package (not exactly what we need to install with `pip`) with

    cd ~/Software/cytometer
    ./setup.py sdist

TODO: [Packaging and Distributing Projects](https://packaging.python.org/tutorials/distributing-packages/).

# Running cytometer python scripts

You need to set 

* `LD_LIBRARY_PATH` from the shell so that e.g. Theano can find libcudnn.so
 * for some reason, setting `os.environ['LD_LIBRARY_PATH']` from the python script doesn't work).
 * Setting `LD_LIBRARY_PATH` in `~/.bashrc` won't work because when `.bashrc` is read (upon opening a shell), you have not activated a conda environment yet
* `PYTHONPATH` so that python can find the cytometer modules

## If the IDE is spyder

You can launch the IDE from a terminal

    PYTHONPATH=~/Software/cytometer LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH spyder &

Alternatively, set and export the environmental variables once

    export PYTHONPATH=~/Software/cytometer:$PYTHONPATH
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

and then call the IDE as

    spyder &

## If the IDE is pycharm-community

You can launch the IDE from a terminal

    PYTHONPATH=~/Software/cytometer LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH pycharm-community &

Alternatively, set and export the environmental variables once

    export PYTHONPATH=~/Software/cytometer:$PYTHONPATH
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

and then call the IDE as

    pycharm-community &
    
You also have to select the python version in File -> Settings -> Project: scripts 
-> Project Interpreter.

For example, "Python 3.6 (cytometer) ~/.conda/envs/cytometer/bin/python"

## To run a python script directly from the shell

You can run the script as

    PYTHONPATH=~/Software/cytometer:$PYTHONPATH LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python scripts/basic_cnn.py

Alternatively, set and export the environmental variables once

    export PYTHONPATH=~/Software/cytometer:$PYTHONPATH
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

and then run the script as

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
