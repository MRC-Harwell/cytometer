#!/usr/bin/env bash

# Script to convert the ndpi images that we use in the training dataset to DeepZoom Image (.dzi) format so that they
# can be visualised by AIDA.
#
# Syntax:
#    ./rebb1_full_histology_ndpi_to_dzi.sh

# This file is part of Cytometer
# Copyright 2021 Medical Research Council
# SPDX-License-Identifier: Apache-2.0
# Author: Ramon Casero <rcasero@gmail.com>

histo_dir=$HOME/coxgroup_zeiss_test
dzi_dir=$HOME/Data/cytometer_data/aida_data_Rreb1_tm1b/images

ndpi_list=(
'RREB1-TM1B-B6N-IC-1.1a 1132-18 G1 - 2019-02-20 09.56.50.ndpi'
'RREB1-TM1B-B6N-IC-1.1a 1132-18 M1 - 2019-02-20 09.48.06.ndpi'
'RREB1-TM1B-B6N-IC-1.1a  1132-18 P1 - 2019-02-20 09.29.29.ndpi'
'RREB1-TM1B-B6N-IC-1.1a  1132-18 S1 - 2019-02-20 09.21.24.ndpi'
'RREB1-TM1B-B6N-IC-1.1b 1133-18 G1 - 2019-02-20 12.31.18.ndpi'
'RREB1-TM1B-B6N-IC-1.1b 1133-18 M1 - 2019-02-20 12.15.25.ndpi'
'RREB1-TM1B-B6N-IC-1.1b 1133-18 P3 - 2019-02-20 11.51.52.ndpi'
'RREB1-TM1B-B6N-IC-1.1b 1133-18 S1 - 2019-02-20 11.31.44.ndpi'
'RREB1-TM1B-B6N-IC-1.1c  1129-18 G1 - 2019-02-19 14.10.46.ndpi'
'RREB1-TM1B-B6N-IC-1.1c  1129-18 M2 - 2019-02-19 13.58.32.ndpi'
'RREB1-TM1B-B6N-IC-1.1c  1129-18 P1 - 2019-02-19 12.41.11.ndpi'
'RREB1-TM1B-B6N-IC-1.1c  1129-18 S1 - 2019-02-19 12.28.03.ndpi'
'RREB1-TM1B-B6N-IC-1.1e 1134-18 G2 - 2019-02-20 14.43.06.ndpi'
'RREB1-TM1B-B6N-IC-1.1e 1134-18 P1 - 2019-02-20 13.59.56.ndpi'
'RREB1-TM1B-B6N-IC-1.1f  1130-18 G1 - 2019-02-19 15.51.35.ndpi'
'RREB1-TM1B-B6N-IC-1.1f  1130-18 M2 - 2019-02-19 15.38.01.ndpi'
'RREB1-TM1B-B6N-IC-1.1f  1130-18 S1 - 2019-02-19 14.39.24.ndpi'
'RREB1-TM1B-B6N-IC-1.1g  1131-18 G1 - 2019-02-19 17.10.06.ndpi'
'RREB1-TM1B-B6N-IC-1.1g  1131-18 M1 - 2019-02-19 16.53.58.ndpi'
'RREB1-TM1B-B6N-IC-1.1g  1131-18 P1 - 2019-02-19 16.37.30.ndpi'
'RREB1-TM1B-B6N-IC-1.1g  1131-18 S1 - 2019-02-19 16.21.16.ndpi'
'RREB1-TM1B-B6N-IC-1.1h 1135-18 G3 - 2019-02-20 15.46.52.ndpi'
'RREB1-TM1B-B6N-IC-1.1h 1135-18 M1 - 2019-02-20 15.30.26.ndpi'
'RREB1-TM1B-B6N-IC-1.1h 1135-18 P1 - 2019-02-20 15.06.59.ndpi'
'RREB1-TM1B-B6N-IC-1.1h 1135-18 S1 - 2019-02-20 14.56.47.ndpi'
'RREB1-TM1B-B6N-IC-2.1a  1128-18 G1 - 2019-02-19 12.04.29.ndpi'
'RREB1-TM1B-B6N-IC-2.1a  1128-18 M2 - 2019-02-19 11.26.46.ndpi'
'RREB1-TM1B-B6N-IC-2.1a  1128-18 P1 - 2019-02-19 11.01.39.ndpi'
'RREB1-TM1B-B6N-IC-2.1a  1128-18 S1 - 2019-02-19 11.59.16.ndpi'
'RREB1-TM1B-B6N-IC-2.2a 1124-18 G1 - 2019-02-18 10.15.04.ndpi'
'RREB1-TM1B-B6N-IC-2.2a 1124-18 M3 - 2019-02-18 10.12.54.ndpi'
'RREB1-TM1B-B6N-IC-2.2a 1124-18 P2 - 2019-02-18 09.39.46.ndpi'
'RREB1-TM1B-B6N-IC-2.2a 1124-18 S1 - 2019-02-18 09.09.58.ndpi'
'RREB1-TM1B-B6N-IC-2.2b 1125-18 G1 - 2019-02-18 12.35.37.ndpi'
'RREB1-TM1B-B6N-IC-2.2b 1125-18 P1 - 2019-02-18 11.16.21.ndpi'
'RREB1-TM1B-B6N-IC-2.2b 1125-18 S1 - 2019-02-18 11.06.53.ndpi'
'RREB1-TM1B-B6N-IC-2.2d 1137-18 S1 - 2019-02-21 10.59.23.ndpi'
'RREB1-TM1B-B6N-IC-2.2e 1126-18 G1 - 2019-02-18 14.58.55.ndpi'
'RREB1-TM1B-B6N-IC-2.2e 1126-18 M1- 2019-02-18 14.50.13.ndpi'
'RREB1-TM1B-B6N-IC-2.2e 1126-18 P1 - 2019-02-18 14.13.24.ndpi'
'RREB1-TM1B-B6N-IC-2.2e 1126-18 S1 - 2019-02-18 14.05.58.ndpi'
'RREB1-TM1B-B6N-IC-5.1a 0066-19 G1 - 2019-02-21 15.26.24.ndpi'
'RREB1-TM1B-B6N-IC-5.1a 0066-19 M1 - 2019-02-21 15.04.14.ndpi'
'RREB1-TM1B-B6N-IC-5.1a 0066-19 P1 - 2019-02-21 14.39.43.ndpi'
'RREB1-TM1B-B6N-IC-5.1a 0066-19 S1 - 2019-02-21 14.04.12.ndpi'
'RREB1-TM1B-B6N-IC-5.1b 0067-19 P1 - 2019-02-21 16.32.24.ndpi'
'RREB1-TM1B-B6N-IC-5.1b 0067-19 S1 - 2019-02-21 16.00.37.ndpi'
'RREB1-TM1B-B6N-IC-5.1b 67-19 G1 - 2019-02-21 17.29.31.ndpi'
'RREB1-TM1B-B6N-IC-5.1b 67-19 M1 - 2019-02-21 17.04.37.ndpi'
'RREB1-TM1B-B6N-IC-5.1c  68-19 G2 - 2019-02-22 09.43.59.ndpi'
'RREB1-TM1B-B6N-IC- 5.1c 68 -19 M2 - 2019-02-22 09.27.30.ndpi'
'RREB1-TM1B-B6N-IC -5.1c 68 -19 peri3 - 2019-02-22 09.08.26.ndpi'
'RREB1-TM1B-B6N-IC- 5.1c 68 -19 sub2 - 2019-02-22 08.39.12.ndpi'
'RREB1-TM1B-B6N-IC-5.1d  69-19 G2 - 2019-02-22 15.13.08.ndpi'
'RREB1-TM1B-B6N-IC-5.1d  69-19 M1 - 2019-02-22 14.39.12.ndpi'
'RREB1-TM1B-B6N-IC-5.1d  69-19 Peri1 - 2019-02-22 12.00.19.ndpi'
'RREB1-TM1B-B6N-IC-5.1d  69-19 sub1 - 2019-02-22 11.44.13.ndpi'
'RREB1-TM1B-B6N-IC-5.1e  70-19 G3 - 2019-02-25 10.34.30.ndpi'
'RREB1-TM1B-B6N-IC-5.1e  70-19 M1 - 2019-02-25 09.53.00.ndpi'
'RREB1-TM1B-B6N-IC-5.1e  70-19 P2 - 2019-02-25 09.27.06.ndpi'
'RREB1-TM1B-B6N-IC-5.1e  70-19 S1 - 2019-02-25 08.51.26.ndpi'
'RREB1-TM1B-B6N-IC-7.1a  71-19 G1 - 2019-02-25 12.27.06.ndpi'
'RREB1-TM1B-B6N-IC-7.1a  71-19 P1 - 2019-02-25 11.31.30.ndpi'
'RREB1-TM1B-B6N-IC-7.1a  71-19 S1 - 2019-02-25 11.03.59.ndpi'
)

for ndpi_file in "${ndpi_list[@]}"
do
  if [[ -f "$histo_dir"/"$ndpi_file" ]]; then
    echo "NDPI file found: $ndpi_file"
  else
    tput setaf 1; echo "--> NDPI file not found: $ndpi_file"; tput sgr0
    continue
  fi

  # create output DeepZoom file from input NDPI filename
  dzi_file=${ndpi_file%.ndpi}
  #echo DZI_FILE = "$dzi_dir"/"$dzi_file".dzi

  if [[ ! -f "$dzi_dir"/"$dzi_file".dzi ]]; then
    echo -e "\tConverting NDPI to DeepZoom..."
    vips dzsave "$histo_dir"/"$ndpi_file" "$dzi_dir"/"$dzi_file"
  else
    echo -e "\tSkipping... DeepZoom already exists"
  fi
done
