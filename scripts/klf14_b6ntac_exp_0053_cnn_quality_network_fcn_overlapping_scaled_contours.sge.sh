#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q
#$ -l gpu=4 -l gputype=p100
#$ -cwd -V
#$ -pe shmem 4

echo "job ID: " $JOB_ID
echo "Run on: " `hostname`
echo "Started at: " `date`

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0

source activate cytometer_tensorflow

python klf14_b6ntac_exp_0053_cnn_quality_network_fcn_overlapping_scaled_contours.py
rv="$?"

# exit status held by qacct
echo "Main task exit status: ${rv}"

echo "Finished at :"`date`

exit "${rv}"
