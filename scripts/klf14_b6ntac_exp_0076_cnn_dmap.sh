#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q
#$ -l gpu=2 -l gputype=p100
#$ -cwd -V
#$ -pe shmem 2

echo "job ID: " $JOB_ID
echo "Run on: " `hostname`
echo "Started at: " `date`

module load Anaconda3/5.1.0

source activate cytometer_tensorflow

python klf14_b6ntac_exp_0076_cnn_dmap.py

echo "Finished at :"`date`
exit 0
