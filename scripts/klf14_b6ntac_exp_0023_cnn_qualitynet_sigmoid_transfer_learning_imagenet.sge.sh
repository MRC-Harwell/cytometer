#!/bin/bash

# This file is part of Cytometer
# Copyright 2021 Medical Research Council
# SPDX-License-Identifier: Apache-2.0
# Author: Ramon Casero <rcasero@gmail.com>

#$ -P rittscher.prjc -q gpu8.q
#$ -l gpu=3 -l gputype=p100
#$ -cwd -V
#$ -pe shmem 3

echo "job ID: " $JOB_ID
echo "Run on: " `hostname`
echo "Started at: " `date`

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0

source activate cytometer_tensorflow

python klf14_b6ntac_exp_0023_cnn_qualitynet_sigmoid_transfer_learning_imagenet.py

echo "Finished at :"`date`
exit 0
