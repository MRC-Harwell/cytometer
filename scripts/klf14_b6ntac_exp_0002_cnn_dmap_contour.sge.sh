#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q
#$ -l gpu=1 -l gputype=p100
#$ -cwd -V
#$ -pe shmem 1

echo "job ID: " $JOB_ID
echo "Run on: " `hostname`
echo "Started at: " `date`

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0

source activate cytometer_tensorflow

python klf14_b6ntac_exp_0002_cnn_dmap_contour.py

echo "Finished at :"`date`
exit 0
