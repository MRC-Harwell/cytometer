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
dzi_dir=$HOME/Data/cytometer_data/aida_data_Rreb1_tm1b_zeiss/images

histo_list=(
'___00000022___00000436_02052021-TBX15del964-pWAT-0028.czi'
'___00000022___00001449_27042021-TBX15del964-iWAT-0015.czi'
'___00000022___00001641_28042021-TBX15del964-gWAT-0020.czi'
)

for histo_file in "${histo_list[@]}"
do
  if [[ -f "$histo_dir"/"$histo_file" ]]; then
    echo "CZI file found: $histo_file"
  else
    tput setaf 1; echo "--> CZI file not found: $histo_file"; tput sgr0
    continue
  fi

  # create output DeepZoom file from input NDPI filename
  dzi_file=${histo_file%.czi}
  #echo DZI_FILE = "$dzi_dir"/"$dzi_file".dzi

  if [[ ! -f "$dzi_dir"/"$dzi_file".dzi ]]; then
    echo -e "\tConverting CZI to DeepZoom..."
    vips dzsave "$histo_dir"/"$histo_file" "$dzi_dir"/"$dzi_file"
  else
    echo -e "\tSkipping... DeepZoom already exists"
  fi
done
