# install_dependencies.sh
#
#    Script to install the dependencies to run cytometer scripts.
#
#    This script installs several Ubuntu packages, and creates two local conda environments:
#      * cytometer: With the latest versions of Keras/Theano/Tensorflow to run cytometer scripts.
#      * DeepCell: Based on Keras 1.1.1 and Theano 0.9.0 to run DeepCell legacy code.

#    Copyright © 2018  Ramón Casero <rcasero@gmail.com>
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

########################################################################
## python environment for cytometer

# install OpenSlide library
sudo apt install -y openslide-tools

# install LaTeX dependencies for matplotlib
sudo apt install -y texlive-latex-base texlive-latex-extra
sudo apt install -y dvipng

# install GNU R with lme4 module (generilised linear mixed models)
sudo apt install -y r-base r-cran-lme4

# create or update environment for development with Keras
~/Software/python_setup/bin/install_keras_environment.sh cytometer

source activate cytometer

# install other python packages
conda install -y matplotlib pillow 
conda install -y scikit-image scikit-learn h5py 
conda install -y nose pytest
pip install opencv-python pysto openslide-python
pip install tifffile mahotas networkx
conda install -y pandas six

########################################################################
## python environment for DeepCell

~/Software/python_setup/bin/install_deepcell_environment.sh
