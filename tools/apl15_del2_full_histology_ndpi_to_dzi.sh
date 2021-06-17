#!/usr/bin/env bash

# Script to convert ndpi images to DeepZoom Image (.dzi) format so that they can be visualised by AIDA.
#
# Syntax:
#    ./apl15_del2_full_histology_ndpi_to_dzi.sh

# This file is part of Cytometer
# Copyright 2021 Medical Research Council
# SPDX-License-Identifier: Apache-2.0
# Author: Ramon Casero <rcasero@gmail.com>

ndpi_dir=$HOME/scan_srv2_cox/'Ying Bai'/'For Ramon'
dzi_dir=$HOME/Data/cytometer_data/aida_data_Apl15_del2/images

ndpi_list=(
'APL15-DEL2-EM1-B6N 31.1a 696-19 Gwat 1 - 2019-08-19 12.44.49.ndpi'
'APL15-DEL2-EM1-B6N 31.1a 696-19 Gwat 3 - 2019-08-19 12.57.53.ndpi'
'APL15-DEL2-EM1-B6N 31.1a 696-19 Gwat 6 - 2019-08-19 13.16.35.ndpi'
'APL15-DEL2-EM1-B6N 31.1a 696-19 Iwat 1 - 2019-08-19 13.54.09.ndpi'
'APL15-DEL2-EM1-B6N 31.1a 696-19 Iwat 3 - 2019-08-19 15.29.54.ndpi'
'APL15-DEL2-EM1-B6N 31.1a 696-19 Iwat 6 - 2019-08-19 15.52.25.ndpi'
'APL15-DEL2-EM1-B6N 32.2b 695-19 Gwat 1 - 2019-08-16 13.04.34.ndpi'
'APL15-DEL2-EM1-B6N 32.2b 695-19 Gwat 3 - 2019-08-16 13.18.24.ndpi'
'APL15-DEL2-EM1-B6N 32.2b 695-19 Gwat 6 - 2019-08-16 13.37.00.ndpi'
'APL15-DEL2-EM1-B6N 32.2b 695-19 Iwat 1 - 2019-08-19 12.01.13.ndpi'
'APL15-DEL2-EM1-B6N 32.2b 695-19 Iwat 3 - 2019-08-19 12.13.50.ndpi'
'APL15-DEL2-EM1-B6N 32.2b 695-19 Iwat 6 - 2019-08-19 12.31.59.ndpi'
'APL15-DEL2-EM1-B6N 34.1b 693-19 Gwat 1 - 2019-08-15 11.11.40.ndpi'
'APL15-DEL2-EM1-B6N 34.1b 693-19 Gwat 3 - 2019-08-15 11.24.16.ndpi'
'APL15-DEL2-EM1-B6N 34.1b 693-19 Gwat 6 - 2019-08-15 11.45.59.ndpi'
'APL15-DEL2-EM1-B6N 34.1b 693-19 Iwat 1 - 2019-08-15 11.55.14.ndpi'
'APL15-DEL2-EM1-B6N 34.1b 693-19 Iwat 3 - 2019-08-15 13.31.37.ndpi'
'APL15-DEL2-EM1-B6N 34.1b 693-19 Iwat 6 - 2019-08-15 12.22.16.ndpi'
'APL15-DEL2-EM1-B6N 35.2b 701-19 Gwat 1 - 2019-08-22 09.47.48.ndpi'
'APL15-DEL2-EM1-B6N 35.2b 701-19 Gwat 3 - 2019-08-22 10.59.34.ndpi'
'APL15-DEL2-EM1-B6N 35.2b 701-19 Gwat 6 - 2019-08-22 10.15.02.ndpi'
'APL15-DEL2-EM1-B6N 35.2b 701-19 Iwat 1 - 2019-08-22 11.12.17.ndpi'
'APL15-DEL2-EM1-B6N 35.2b 701-19 Iwat 3 - 2019-08-22 11.24.19.ndpi'
'APL15-DEL2-EM1-B6N 35.2b 701-19 Iwat 6 - 2019-08-22 11.41.44.ndpi'
'APL15-DEL2-EM1-B6N 35.2c 694-19 Gwat 1 - 2019-08-15 13.41.17.ndpi'
'APL15-DEL2-EM1-B6N 35.2c 694-19 Gwat 3 - 2019-08-15 13.48.15.ndpi'
'APL15-DEL2-EM1-B6N 35.2c 694-19 Gwat 6 - 2019-08-15 14.03.44.ndpi'
'APL15-DEL2-EM1-B6N 35.2c 694-19 Iwat 1 - 2019-08-16 11.13.41.ndpi'
'APL15-DEL2-EM1-B6N 35.2c 694-19 Iwat 3 - 2019-08-16 11.25.11.ndpi'
'APL15-DEL2-EM1-B6N 35.2c 694-19 Iwat 6 - 2019-08-16 11.42.18.ndpi'
'APL15-DEL2-EM1-B6N 37.1b 697-19 Gwat 1 - 2019-08-20 11.40.02.ndpi'
'APL15-DEL2-EM1-B6N 37.1b 697-19 Gwat 3 - 2019-08-20 12.44.02.ndpi'
'APL15-DEL2-EM1-B6N 37.1b 697-19 Gwat 6 - 2019-08-20 13.03.14.ndpi'
'APL15-DEL2-EM1-B6N 37.1b 697-19 Iwat 1 - 2019-08-20 14.08.10.ndpi'
'APL15-DEL2-EM1-B6N 37.1b 697-19 Iwat 3 - 2019-08-20 14.22.06.ndpi'
'APL15-DEL2-EM1-B6N 37.1b 697-19 Iwat 6 - 2019-08-20 14.41.48.ndpi'
'APL15-DEL2-EM1-B6N 38.1c 700-19 Gwat 1 - 2019-08-21 11.59.38.ndpi'
'APL15-DEL2-EM1-B6N 38.1c 700-19 Gwat 3 - 2019-08-21 16.23.02.ndpi'
'APL15-DEL2-EM1-B6N 38.1c 700-19 Gwat 6 - 2019-08-21 16.43.02.ndpi'
'APL15-DEL2-EM1-B6N 38.1c 700-19 Iwat 1 - 2019-08-22 08.36.58.ndpi'
'APL15-DEL2-EM1-B6N 38.1c 700-19 Iwat 3 - 2019-08-22 08.49.27.ndpi'
'APL15-DEL2-EM1-B6N 38.1c 700-19 Iwat 6 - 2019-08-22 09.08.29.ndpi'
'APL15-DEL2-EM1-B6N 38.1d 698-19 Gwat 1 - 2019-08-20 15.05.02.ndpi'
'APL15-DEL2-EM1-B6N 38.1d 698-19 Gwat 3 - 2019-08-20 15.18.30.ndpi'
'APL15-DEL2-EM1-B6N 38.1d 698-19 Gwat 6 - 2019-08-20 15.37.51.ndpi'
'APL15-DEL2-EM1-B6N 38.1d 698-19 Iwat 1 - 2019-08-21 08.59.51.ndpi'
'APL15-DEL2-EM1-B6N 38.1d 698-19 Iwat 3 - 2019-08-21 09.13.35.ndpi'
'APL15-DEL2-EM1-B6N 38.1d 698-19 Iwat 6 - 2019-08-21 09.32.24.ndpi'
'APL15-DEL2-EM1-B6N 38.1e 699-19 Gwat 1 - 2019-08-21 09.44.01.ndpi'
'APL15-DEL2-EM1-B6N 38.1e 699-19 Gwat 3 - 2019-08-21 09.57.22.ndpi'
'APL15-DEL2-EM1-B6N 38.1e 699-19 Gwat 6 - 2019-08-21 10.15.39.ndpi'
'APL15-DEL2-EM1-B6N 38.1e 699-19 Iwat 1 - 2019-08-21 11.16.47.ndpi'
'APL15-DEL2-EM1-B6N 38.1e 699-19 Iwat 3 - 2019-08-21 11.29.58.ndpi'
'APL15-DEL2-EM1-B6N 38.1e 699-19 Iwat 6 - 2019-08-21 11.49.53.ndpi'
'APL15-DEL2_EM1_B6N 34.1e 692-19 Gwat 1 - 2019-08-14 16.48.21.ndpi'
'APL15-DEL2_EM1_B6N 34.1e 692-19 Gwat 3 - 2019-08-14 17.09.59.ndpi'
'APL15-DEL2_EM1_B6N 34.1e 692-19 Gwat 6 - 2019-08-14 17.28.37.ndpi'
'APL15-DEL2_EM1_B6N 34.1e 692-19 Iwat 1 - 2019-08-15 09.49.41.ndpi'
'APL15-DEL2_EM1_B6N 34.1e 692-19 Iwat 3 - 2019-08-15 10.02.15.ndpi'
'APL15-DEL2_EM1_B6N 34.1e 692-19 Iwat 6 - 2019-08-15 10.20.19.ndpi'
'APL15-DEL2_EM1_B6N 35.1a 690-19 Gwat 1 - 2019-08-14 12.06.19.ndpi'
'APL15-DEL2_EM1_B6N 35.1a 690-19 Gwat 3 - 2019-08-14 12.22.40.ndpi'
'APL15-DEL2_EM1_B6N 35.1a 690-19 Gwat 6 - 2019-08-14 12.42.00.ndpi'
'APL15-DEL2_EM1_B6N 35.1a 690-19 Iwat 1 - 2019-08-14 12.53.04.ndpi'
'APL15-DEL2_EM1_B6N 35.1a 690-19 Iwat 3 - 2019-08-14 13.04.25.ndpi'
'APL15-DEL2_EM1_B6N 35.1a 690-19 Iwat 6 - 2019-08-14 13.22.36.ndpi'
'APL15-DEL2_EM1_B6N 38.1a 691-19 Gwat 1 - 2019-08-14 13.48.53.ndpi'
'APL15-DEL2_EM1_B6N 38.1a 691-19 Gwat 3 - 2019-08-14 14.02.42.ndpi'
'APL15-DEL2_EM1_B6N 38.1a 691-19 Gwat 6 - 2019-08-14 14.24.10.ndpi'
'APL15-DEL2_EM1_B6N 38.1a 691-19 Iwat 1 - 2019-08-14 15.34.14.ndpi'
'APL15-DEL2_EM1_B6N 38.1a 691-19 Iwat 3 - 2019-08-14 15.44.04.ndpi'
'APL15-DEL2_EM1_B6N 38.1a 691-19 Iwat 6 - 2019-08-14 15.59.53.ndpi'
'ARL15-DEL2-EM1-B6N-29.1E Gwat 689-19 1 - 2019-08-08 11.51.43.ndpi'
'ARL15-DEL2-EM1-B6N-29.1E Gwat 689-19 3 - 2019-08-08 12.04.20.ndpi'
'ARL15-DEL2-EM1-B6N-29.1E Gwat 689-19 6 - 2019-08-08 12.24.13.ndpi'
'ARL15-DEL2-EM1-B6N-29.1E Iwat 689-19 1 - 2019-08-08 12.33.17.ndpi'
'ARL15-DEL2-EM1-B6N-29.1E Iwat 689-19 3 - 2019-08-12 16.08.24.ndpi'
'ARL15-DEL2-EM1-B6N-29.1E Iwat 689-19 6 - 2019-08-12 16.30.33.ndpi'
'ARL15-DEL2-EM1-B6N-32.1f Gwat 688-19 1 - 2019-08-08 10.07.15.ndpi'
'ARL15-DEL2-EM1-B6N-32.1f Gwat 688-19 3 - 2019-08-08 10.23.58.ndpi'
'ARL15-DEL2-EM1-B6N-32.1f Gwat 688-19 6 - 2019-08-08 10.46.56.ndpi'
'ARL15-DEL2-EM1-B6N-32.1f Iwat 688-19 1 - 2019-08-08 11.13.12.ndpi'
'ARL15-DEL2-EM1-B6N-32.1f Iwat 688-19 3 - 2019-08-08 11.31.23.ndpi'
'ARL15-DEL2-EM1-B6N-32.1f Iwat 688-19 6 - 2019-08-08 11.43.11.ndpi'
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
