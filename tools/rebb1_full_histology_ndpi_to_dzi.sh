#!/usr/bin/env bash

# Script to convert the ndpi images that we use in the training dataset to DeepZoom Image (.dzi) format so that they
# can be visualised by AIDA.
#
# Syntax:
#    ./rebb1_full_histology_ndpi_to_dzi.sh

ndpi_dir=$HOME/scan_srv2_cox/"Liz Bentley"/Grace
dzi_dir=$HOME/Software/AIDA/dist/data/images

ndpi_list=(
    'RREB1-TM1B-B6N-IC-1.1a  1132-18 G1 - 2018-11-16 14.58.55.ndpi'
)

for ndpi_file in "${ndpi_list[@]}"
do
  dzi_file=${ndpi_file%.ndpi}
  if [ ! -f "$dzi_dir"/"$dzi_file".dzi ]; then
    echo "Processing: " "$ndpi_file"
    vips dzsave "$ndpi_dir"/"$ndpi_file" "$dzi_dir"/"$dzi_file"
  else
    echo "Skipping: " "$ndpi_file"
  fi
done
