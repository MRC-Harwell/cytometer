Instructions to set up project `cytometer`.
===========================================

Warning! This file is half outdated, and under active rewriting.

# Table of Contents

   * [Instructions to set up project cytometer.](#instructions-to-set-up-project-cytometer)
   * [Table of Contents](#table-of-contents)
   * [Installing the cytometer python code](#installing-the-cytometer-python-code)
   * [Dependencies and local conda environments](#dependencies-and-local-conda-environments)
      * [Notes on Ubuntu dependencies](#notes-on-ubuntu-dependencies)
      * [Activating GPU in laptops](#activating-gpu-in-laptops)
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

1. Clone the `cytometer` repository by running the command

        cd ~/Software
        git clone https://GITUSERNAME@github.com/MRC-Harwell/cytometer.git
1. Change directory to the project

        cd cytometer

# Dependencies and local conda environments

1. Make `install_dependencies_machine.sh` and `install_dependencies_user.sh` executable

        chmod +x install_dependencies_machine.sh install_dependencies_user.sh
1. You only need to set up the machine once with `install_dependencies_machine.sh` (Ubuntu packages, nVidia drivers, etc). You will be asked for a sudo password, so you need to have sudo privileges

        ./install_dependencies_machine.sh
1. You need to create a local conda environment with `install_dependencies_user.sh` for yourself (don't run it with sudo!).

        ./install_dependencies_user.sh
1. The first time you run this it'll give an error message saying that you need to close your terminal for changes to take effect. Close your terminal and open a new one (or maybe run `source ~/.bashrc`). Run again

        ./install_dependencies_user.sh
        
## Notes on Ubuntu dependencies

The `install_dependencies_machine.sh` script:
* Installs the NVIDIA drivers rather than the open source nouveau (`xserver-xorg-video-nouveau`), so that we have full access to the NVIDIA 
  graphic card features.
* Uses Miniconda to install conda and create local python environments.
* Installs the CUDA Toolkit from the [Nvidia website](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu), rather than using Ubuntu packages.

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

Note: In a laptop with a dual graphics chip configuration as explained above, you will get the same output whether the NVIDIA or Intel
chips are selected. Thus, regardless of what `nvidia-smi` says, you still need to make sure you have selected the NVIDIA option in 
`nvidia-settings`.

# Running cytometer python scripts

1. Activate the cytometer local environment

        source activate cytometer_tensorflow
1. Go to the scripts directory

        cd ~/Software/cytometer/scripts
1. Mount or create the data and annotations directories required by the script. This depends on the script that you are going to run and your own machine setup
1. Run the script with `nohup` so you can close the terminal and leave it running, and use `CUDA_VISIBLE_DEVICES` to choose one of the GPUs, e.g. for GPU=0. When the script starts processing a histology slide, it puts a lock on it by creating a `HISTO_FILENAME.lock` file in the annotations directory. This way, you can run multiple instances of the script in parallel, one per GPU, each one processing a different histology slide. All the output from running the script will be redirected to `nohup_${CUDA_VISIBLE_DEVICES}.log`

        export CUDA_VISIBLE_DEVICES=0 && nohup python tbx15_h156n_exp_0003_zeiss_full_slide_pipeline_v8_no_correction.py >> nohup_${CUDA_VISIBLE_DEVICES}.log &
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

For example, "Python 3.7 (cytometer) ~/.conda/envs/cytometer/bin/python"

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

# TODO: Packaging the cytometer python code

The `setup.py` and associated files to create a package are in place. For the moment, you can create a python source package (not exactly what we need to install with `pip`) with

    cd ~/Software/cytometer
    ./setup.py sdist

TODO: [Packaging and Distributing Projects](https://packaging.python.org/tutorials/distributing-packages/).
