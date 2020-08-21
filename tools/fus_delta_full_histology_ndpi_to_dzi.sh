#!/usr/bin/env bash

# Script to convert the ndpi images that we use in the training dataset to DeepZoom Image (.dzi) format so that they
# can be visualised by AIDA.
#
# Syntax:
#    ./fus_delta_full_histology_ndpi_to_dzi.sh

ndpi_dir=${HOME}/AAcevedo_images/FUS-DELTA-DBA-B6/"Histology scans"/"iWAT scans"
dzi_dir=${HOME}/Data/cytometer_data/aida_data_Fus_Delta/images

ndpi_list=(
#'FUS-DELTA-DBA-B6-IC 14.4a BF1 1052-18 - 2018-09-18 10.16.32.ndpi'
#'FUS-DELTA-DBA-B6-IC 14.4a BF2 1052-18 - 2018-09-18 10.19.35.ndpi'
'FUS-DELTA-DBA-B6-IC 14.4a WF1 1052-18 - 2018-09-18 10.08.42.ndpi'
'FUS-DELTA-DBA-B6-IC 14.4a WF2 1052-18 - 2018-09-18 10.12.56.ndpi'
'FUS-DELTA-DBA-B6-IC-13.3a  1276-18 1 - 2018-12-03 12.18.01.ndpi'
'FUS-DELTA-DBA-B6-IC-13.3a  1276-18 2 - 2018-12-03 12.22.52.ndpi'
'FUS-DELTA-DBA-B6-IC-13.3a  1276-18 3 - 2018-12-03 12.27.21.ndpi'
'FUS-DELTA-DBA-B6-IC-13.3b  1273-18 1 - 2018-12-03 11.11.51.ndpi'
'FUS-DELTA-DBA-B6-IC-13.3b  1273-18 2 - 2018-12-03 11.17.25.ndpi'
'FUS-DELTA-DBA-B6-IC-13.3b  1273-18 3 - 2018-12-03 11.22.28.ndpi'
'FUS-DELTA-DBA-B6-IC-13.3c  1274-18 1 - 2018-12-03 11.27.49.ndpi'
'FUS-DELTA-DBA-B6-IC-13.3c  1274-18 2 - 2018-12-03 11.32.18.ndpi'
'FUS-DELTA-DBA-B6-IC-13.3c  1274-18 3 - 2018-12-03 11.36.56.ndpi'
'FUS-DELTA-DBA-B6-IC-13.4a  1285-18 1 - 2018-12-04 10.23.00.ndpi'
'FUS-DELTA-DBA-B6-IC-13.4a  1285-18 2 - 2018-12-04 10.28.45.ndpi'
'FUS-DELTA-DBA-B6-IC-13.4a  1285-18 3 - 2018-12-04 10.34.10.ndpi'
'FUS-DELTA-DBA-B6-IC-13.4c  1286-18 1 - 2018-12-04 10.40.01.ndpi'
'FUS-DELTA-DBA-B6-IC-13.4c  1286-18 2 - 2018-12-04 10.43.30.ndpi'
'FUS-DELTA-DBA-B6-IC-13.4c  1286-18 3 - 2018-12-04 10.47.03.ndpi'
'FUS-DELTA-DBA-B6-IC-14.2a  1287-18 1 - 2018-12-04 11.24.57.ndpi'
'FUS-DELTA-DBA-B6-IC-14.2a  1287-18 2 - 2018-12-04 11.28.34.ndpi'
'FUS-DELTA-DBA-B6-IC-14.2a  1287-18 3 - 2018-12-04 11.31.57.ndpi'
'FUS-DELTA-DBA-B6-IC-14.2b  1282-18 1 - 2018-12-03 16.31.18.ndpi'
'FUS-DELTA-DBA-B6-IC-14.2b  1282-18 2 - 2018-12-03 16.35.50.ndpi'
'FUS-DELTA-DBA-B6-IC-14.2b  1282-18 3 - 2018-12-03 16.41.01.ndpi'
'FUS-DELTA-DBA-B6-IC-14.3a  1271-18 1 - 2018-12-03 10.18.57.ndpi'
'FUS-DELTA-DBA-B6-IC-14.3a  1271-18 2 - 2018-12-03 10.23.49.ndpi'
'FUS-DELTA-DBA-B6-IC-14.3a  1271-18 3 - 2018-12-03 10.28.18.ndpi'
'FUS-DELTA-DBA-B6-IC-14.3b  1277-18 1 - 2018-12-03 14.35.52.ndpi'
'FUS-DELTA-DBA-B6-IC-14.3b  1277-18 2 - 2018-12-03 14.40.04.ndpi'
'FUS-DELTA-DBA-B6-IC-14.3b  1277-18 3 - 2018-12-03 14.44.04.ndpi'
'FUS-DELTA-DBA-B6-IC-14.3c  1279-18 1 - 2018-12-03 15.34.30.ndpi'
'FUS-DELTA-DBA-B6-IC-14.3c  1279-18 2 - 2018-12-03 15.40.01.ndpi'
'FUS-DELTA-DBA-B6-IC-14.3c  1279-18 3 - 2018-12-03 15.45.23.ndpi'
'FUS-DELTA-DBA-B6-IC-14.3d  1283-18 1 - 2018-12-04 09.36.48.ndpi'
'FUS-DELTA-DBA-B6-IC-14.3d  1283-18 2 - 2018-12-04 09.41.57.ndpi'
'FUS-DELTA-DBA-B6-IC-14.3d  1283-18 3 - 2018-12-04 09.46.20.ndpi'
'FUS-DELTA-DBA-B6-IC-14.4b  1284-18 1 - 2018-12-04 09.51.13.ndpi'
'FUS-DELTA-DBA-B6-IC-14.4b  1284-18 2 - 2018-12-04 09.56.04.ndpi'
'FUS-DELTA-DBA-B6-IC-14.4b  1284-18 3 - 2018-12-04 10.00.57.ndpi'
'FUS-DELTA-DBA-B6-IC-14.4c  1272-18 1 - 2018-12-03 10.32.48.ndpi'
'FUS-DELTA-DBA-B6-IC-14.4c  1272-18 2 - 2018-12-03 10.37.19.ndpi'
'FUS-DELTA-DBA-B6-IC-14.4c  1272-18 3 - 2018-12-03 10.42.03.ndpi'
'FUS-DELTA-DBA-B6-IC-14.4f  1280-18 1 - 2018-12-03 15.50.33.ndpi'
'FUS-DELTA-DBA-B6-IC-14.4f  1280-18 2 - 2018-12-03 15.56.08.ndpi'
'FUS-DELTA-DBA-B6-IC-14.4f  1280-18 3 - 2018-12-03 16.01.52.ndpi'
'FUS-DELTA-DBA-B6-IC-14.4g  1281-18 1 - 2018-12-03 16.14.50.ndpi'
'FUS-DELTA-DBA-B6-IC-14.4g  1281-18 2 - 2018-12-03 16.20.21.ndpi'
'FUS-DELTA-DBA-B6-IC-14.4g  1281-18 3 - 2018-12-03 16.25.37.ndpi'
'FUS-DELTA-DBA-B6-IC-15.2f  1275-18 1 - 2018-12-03 11.58.30.ndpi'
'FUS-DELTA-DBA-B6-IC-15.2f  1275-18 2 - 2018-12-03 12.05.43.ndpi'
'FUS-DELTA-DBA-B6-IC-15.2f  1275-18 3 - 2018-12-03 12.11.46.ndpi'
'FUS-DELTA-DBA-B6-IC-15.2g  1278-18 1 - 2018-12-03 14.48.11.ndpi'
'FUS-DELTA-DBA-B6-IC-15.2g  1278-18 2 - 2018-12-03 14.53.36.ndpi'
'FUS-DELTA-DBA-B6-IC-15.2g  1278-18 3 - 2018-12-03 14.58.55.ndpi'
)

for ndpi_file in "${ndpi_list[@]}"
do
  if [[ -f "$ndpi_dir"/"$ndpi_file" ]]; then
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
    vips dzsave "$ndpi_dir"/"$ndpi_file" "$dzi_dir"/"$dzi_file"
  else
    echo -e "\tSkipping... DeepZoom already exists"
  fi
done
