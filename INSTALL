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

# Create `conda` virtual environments

1. Create a conda environment for cytometer (python 3.6 at the time of this writing)

       conda create --name cytometer python=3
1. Activate the conda environment that you intend to run

       source activate cytometer

# Checking your GPU set-up

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
