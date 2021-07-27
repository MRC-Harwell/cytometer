#!/usr/bin/env bash

# Script to aggregate the ndpi images that we use for the cytometer paper into a 7z file that we can
# upload to Zenodo
#
# Syntax:
#    ./klf14_prepare_ndpi_for_zenodo.sh

# This file is part of Cytometer
# Copyright 2021 Medical Research Council
# SPDX-License-Identifier: Apache-2.0
# Author: Ramon Casero <rcasero@gmail.com>

ndpi_dir=$HOME/scan_srv2_cox/"Maz Yon"
compressed_file=histology.7z

ndpi_list=(
'KLF14-B6NTAC-MAT-18.3b  223-16 C2 - 2016-02-26 10.35.52.ndpi'
'KLF14-B6NTAC-36.1a PAT 96-16 C1 - 2016-02-10 16.12.38.ndpi'
'KLF14-B6NTAC-36.1b PAT 97-16 C1 - 2016-02-10 17.38.06.ndpi'
'KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00.ndpi'
'KLF14-B6NTAC 36.1d PAT 99-16 C1 - 2016-02-11 11.48.31.ndpi'
'KLF14-B6NTAC 36.1e PAT 100-16 C1 - 2016-02-11 14.06.56.ndpi'
'KLF14-B6NTAC 36.1f PAT 101-16 C1 - 2016-02-11 15.23.06.ndpi'
'KLF14-B6NTAC 36.1g PAT 102-16 C1 - 2016-02-11 17.20.14.ndpi'
'KLF14-B6NTAC 36.1h PAT 103-16 C1 - 2016-02-12 10.15.22.ndpi'
'KLF14-B6NTAC 36.1i PAT 104-16 C1 - 2016-02-12 12.14.38.ndpi'
'KLF14-B6NTAC 36.1j PAT 105-16 C1 - 2016-02-12 14.33.33.ndpi'
'KLF14-B6NTAC 37.1a PAT 106-16 C1 - 2016-02-12 16.21.00.ndpi'
'KLF14-B6NTAC-37.1b PAT 107-16 C1 - 2016-02-15 11.43.31.ndpi'
'KLF14-B6NTAC-37.1c PAT 108-16 C1 - 2016-02-15 14.49.45.ndpi'
'KLF14-B6NTAC-37.1d PAT 109-16 C1 - 2016-02-15 15.19.08.ndpi'
'KLF14-B6NTAC-37.1e PAT 110-16 C1 - 2016-02-15 17.33.11.ndpi'
'KLF14-B6NTAC-37.1g PAT 112-16 C1 - 2016-02-16 13.33.09.ndpi'
'KLF14-B6NTAC-37.1h PAT 113-16 C1 - 2016-02-16 15.14.09.ndpi'
'KLF14-B6NTAC-38.1e PAT 94-16 C1 - 2016-02-10 12.13.10.ndpi'
'KLF14-B6NTAC-38.1f PAT 95-16 C1 - 2016-02-10 14.41.44.ndpi'
'KLF14-B6NTAC-MAT-16.2a  211-16 C1 - 2016-02-17 11.46.42.ndpi'
'KLF14-B6NTAC-MAT-16.2b  212-16 C1 - 2016-02-17 12.49.00.ndpi'
'KLF14-B6NTAC-MAT-16.2c  213-16 C1 - 2016-02-17 14.51.18.ndpi'
'KLF14-B6NTAC-MAT-16.2d  214-16 C1 - 2016-02-17 16.02.46.ndpi'
'KLF14-B6NTAC-MAT-16.2e  215-16 C1 - 2016-02-18 09.19.26.ndpi'
'KLF14-B6NTAC-MAT-16.2f  216-16 C1 - 2016-02-18 10.28.27.ndpi'
'KLF14-B6NTAC-MAT-17.1a  44-16 C1 - 2016-02-01 11.14.17.ndpi'
'KLF14-B6NTAC-MAT-17.1b  45-16 C1 - 2016-02-01 12.23.50.ndpi'
'KLF14-B6NTAC-MAT-17.1c  46-16 C1 - 2016-02-01 14.02.04.ndpi'
'KLF14-B6NTAC-MAT-17.1d  47-16 C1 - 2016-02-01 15.25.53.ndpi'
'KLF14-B6NTAC-MAT-17.1e  48-16 C1 - 2016-02-01 16.27.05.ndpi'
'KLF14-B6NTAC-MAT-17.1f  49-16 C1 - 2016-02-01 17.51.46.ndpi'
'KLF14-B6NTAC-MAT-17.2a  64-16 C1 - 2016-02-04 09.17.52.ndpi'
'KLF14-B6NTAC-MAT-17.2b  65-16 C1 - 2016-02-04 10.24.22.ndpi'
'KLF14-B6NTAC-MAT-17.2c  66-16 C1 - 2016-02-04 11.46.39.ndpi'
'KLF14-B6NTAC-MAT-17.2d  67-16 C1 - 2016-02-04 12.34.32.ndpi'
'KLF14-B6NTAC-MAT-17.2f  68-16 C1 - 2016-02-04 15.05.54.ndpi'
'KLF14-B6NTAC-MAT-17.2g  69-16 C1 - 2016-02-04 16.15.05.ndpi'
'KLF14-B6NTAC-MAT-18.1a  50-16 C1 - 2016-02-02 09.12.41.ndpi'
'KLF14-B6NTAC-MAT-18.1b  51-16 C1 - 2016-02-02 09.59.16.ndpi'
'KLF14-B6NTAC-MAT-18.1c  52-16 C1 - 2016-02-02 12.26.58.ndpi'
'KLF14-B6NTAC-MAT-18.1d  53-16 C1 - 2016-02-02 14.32.03.ndpi'
'KLF14-B6NTAC-MAT-18.1e  54-16 C1 - 2016-02-02 15.26.33.ndpi'
'KLF14-B6NTAC-MAT-18.1f  55-16 C1 - 2016-02-02 16.14.30.ndpi'
'KLF14-B6NTAC-MAT-18.2a  57-16 C1 - 2016-02-03 09.10.17.ndpi'
'KLF14-B6NTAC-MAT-18.2b  58-16 C1 - 2016-02-03 11.10.52.ndpi'
'KLF14-B6NTAC-MAT-18.2c  59-16 C1 - 2016-02-03 11.56.52.ndpi'
'KLF14-B6NTAC-MAT-18.2d  60-16 C1 - 2016-02-03 13.13.57.ndpi'
'KLF14-B6NTAC-MAT-18.2e  61-16 C1 - 2016-02-03 14.19.35.ndpi'
'KLF14-B6NTAC-MAT-18.2f  62-16 C1 - 2016-02-03 15.46.15.ndpi'
'KLF14-B6NTAC-MAT-18.2g  63-16 C1 - 2016-02-03 16.58.52.ndpi'
'KLF14-B6NTAC-MAT-18.3b  223-16 C1 - 2016-02-26 09.18.44.ndpi'
'KLF14-B6NTAC-MAT-18.3c  218-16 C1 - 2016-02-18 13.12.09.ndpi'
'KLF14-B6NTAC-MAT-18.3d  224-16 C1 - 2016-02-26 11.13.53.ndpi'
'KLF14-B6NTAC-MAT-19.1a  56-16 C1 - 2016-02-02 17.23.31.ndpi'
'KLF14-B6NTAC-MAT-19.2b  219-16 C1 - 2016-02-18 15.41.38.ndpi'
'KLF14-B6NTAC-MAT-19.2c  220-16 C1 - 2016-02-18 17.03.38.ndpi'
'KLF14-B6NTAC-MAT-19.2e  221-16 C1 - 2016-02-25 14.00.14.ndpi'
'KLF14-B6NTAC-MAT-19.2f  217-16 C1 - 2016-02-18 11.48.16.ndpi'
'KLF14-B6NTAC-MAT-19.2g  222-16 C1 - 2016-02-25 15.13.00.ndpi'
'KLF14-B6NTAC-PAT-36.3a  409-16 C1 - 2016-03-15 10.18.46.ndpi'
'KLF14-B6NTAC-PAT-36.3b  412-16 C1 - 2016-03-15 14.37.55.ndpi'
'KLF14-B6NTAC-PAT-36.3d  416-16 C1 - 2016-03-16 14.44.11.ndpi'
'KLF14-B6NTAC-PAT-37.2a  406-16 C1 - 2016-03-14 12.01.56.ndpi'
'KLF14-B6NTAC-PAT-37.2b  410-16 C1 - 2016-03-15 11.24.20.ndpi'
'KLF14-B6NTAC-PAT-37.2c  407-16 C1 - 2016-03-14 14.13.54.ndpi'
'KLF14-B6NTAC-PAT-37.2d  411-16 C1 - 2016-03-15 12.42.26.ndpi'
'KLF14-B6NTAC-PAT-37.2e  408-16 C1 - 2016-03-14 16.23.30.ndpi'
'KLF14-B6NTAC-PAT-37.2f  405-16 C1 - 2016-03-14 10.58.34.ndpi'
'KLF14-B6NTAC-PAT-37.2g  415-16 C1 - 2016-03-16 11.47.52.ndpi'
'KLF14-B6NTAC-PAT-37.2h  418-16 C1 - 2016-03-16 17.01.17.ndpi'
'KLF14-B6NTAC-PAT-37.3a  413-16 C1 - 2016-03-15 15.54.12.ndpi'
'KLF14-B6NTAC-PAT-37.3c  414-16 C1 - 2016-03-15 17.15.41.ndpi'
'KLF14-B6NTAC-PAT-37.4a  417-16 C1 - 2016-03-16 15.55.32.ndpi'
'KLF14-B6NTAC-PAT-37.4b  419-16 C1 - 2016-03-17 10.22.54.ndpi'
'KLF14-B6NTAC-PAT-39.1h  453-16 C1 - 2016-03-17 11.38.04.ndpi'
'KLF14-B6NTAC-PAT-39.2d  454-16 C1 - 2016-03-17 14.33.38.ndpi'
#
'KLF14-B6NTAC-36.1a PAT 96-16 B1 - 2016-02-10 15.32.31.ndpi'
'KLF14-B6NTAC-36.1b PAT 97-16 B1 - 2016-02-10 17.15.16.ndpi'
'KLF14-B6NTAC-36.1c PAT 98-16 B1 - 2016-02-10 18.32.40.ndpi'
'KLF14-B6NTAC 36.1d PAT 99-16 B1 - 2016-02-11 11.29.55.ndpi'
'KLF14-B6NTAC 36.1e PAT 100-16 B1 - 2016-02-11 12.51.11.ndpi'
'KLF14-B6NTAC 36.1f PAT 101-16 B1 - 2016-02-11 14.57.03.ndpi'
'KLF14-B6NTAC 36.1g PAT 102-16 B1 - 2016-02-11 16.12.01.ndpi'
'KLF14-B6NTAC 36.1h PAT 103-16 B1 - 2016-02-12 09.51.08.ndpi'
'KLF14-B6NTAC 36.1i PAT 104-16 B1 - 2016-02-12 11.37.56.ndpi'
'KLF14-B6NTAC 36.1j PAT 105-16 B1 - 2016-02-12 14.08.19.ndpi'
'KLF14-B6NTAC 37.1a PAT 106-16 B1 - 2016-02-12 15.33.02.ndpi'
'KLF14-B6NTAC-37.1b PAT 107-16 B1 - 2016-02-15 11.25.20.ndpi'
'KLF14-B6NTAC-37.1c PAT 108-16 B1 - 2016-02-15 12.33.10.ndpi'
'KLF14-B6NTAC-37.1d PAT 109-16 B1 - 2016-02-15 15.03.44.ndpi'
'KLF14-B6NTAC-37.1e PAT 110-16 B1 - 2016-02-15 16.16.06.ndpi'
'KLF14-B6NTAC-37.1g PAT 112-16 B1 - 2016-02-16 12.02.07.ndpi'
'KLF14-B6NTAC-37.1h PAT 113-16 B1 - 2016-02-16 14.53.02.ndpi'
'KLF14-B6NTAC-38.1e PAT 94-16 B1 - 2016-02-10 11.35.53.ndpi'
'KLF14-B6NTAC-38.1f PAT 95-16 B1 - 2016-02-10 14.16.55.ndpi'
'KLF14-B6NTAC-MAT-16.2a  211-16 B1 - 2016-02-17 11.21.54.ndpi'
'KLF14-B6NTAC-MAT-16.2b  212-16 B1 - 2016-02-17 12.33.18.ndpi'
'KLF14-B6NTAC-MAT-16.2c  213-16 B1 - 2016-02-17 14.01.06.ndpi'
'KLF14-B6NTAC-MAT-16.2d  214-16 B1 - 2016-02-17 15.43.57.ndpi'
'KLF14-B6NTAC-MAT-16.2e  215-16 B1 - 2016-02-17 17.14.16.ndpi'
'KLF14-B6NTAC-MAT-16.2f  216-16 B1 - 2016-02-18 10.05.52.ndpi'
'KLF14-B6NTAC-MAT-17.1a  44-16 B1 - 2016-02-01 09.19.20.ndpi'
'KLF14-B6NTAC-MAT-17.1b  45-16 B1 - 2016-02-01 12.05.15.ndpi'
'KLF14-B6NTAC-MAT-17.1c  46-16 B1 - 2016-02-01 13.01.30.ndpi'
'KLF14-B6NTAC-MAT-17.1d  47-16 B1 - 2016-02-01 15.11.42.ndpi'
'KLF14-B6NTAC-MAT-17.1e  48-16 B1 - 2016-02-01 16.01.09.ndpi'
'KLF14-B6NTAC-MAT-17.1f  49-16 B1 - 2016-02-01 17.12.31.ndpi'
'KLF14-B6NTAC-MAT-17.2a  64-16 B1 - 2016-02-04 08.57.34.ndpi'
'KLF14-B6NTAC-MAT-17.2b  65-16 B1 - 2016-02-04 10.06.00.ndpi'
'KLF14-B6NTAC-MAT-17.2c  66-16 B1 - 2016-02-04 11.14.28.ndpi'
'KLF14-B6NTAC-MAT-17.2d  67-16 B1 - 2016-02-04 12.20.20.ndpi'
'KLF14-B6NTAC-MAT-17.2f  68-16 B1 - 2016-02-04 14.01.40.ndpi'
'KLF14-B6NTAC-MAT-17.2g  69-16 B1 - 2016-02-04 15.52.52.ndpi'
'KLF14-B6NTAC-MAT-18.1a  50-16 B1 - 2016-02-02 08.49.06.ndpi'
'KLF14-B6NTAC-MAT-18.1b  51-16 B1 - 2016-02-02 09.46.31.ndpi'
'KLF14-B6NTAC-MAT-18.1c  52-16 B1 - 2016-02-02 11.24.31.ndpi'
'KLF14-B6NTAC-MAT-18.1d  53-16 B1 - 2016-02-02 14.11.37.ndpi'
'KLF14-B6NTAC-MAT-18.1e  54-16 B1 - 2016-02-02 15.06.05.ndpi'
'KLF14-B6NTAC-MAT-18.2a  57-16 B1 - 2016-02-03 08.54.27.ndpi'
'KLF14-B6NTAC-MAT-18.2b  58-16 B1 - 2016-02-03 09.58.06.ndpi'
'KLF14-B6NTAC-MAT-18.2c  59-16 B1 - 2016-02-03 11.41.32.ndpi'
'KLF14-B6NTAC-MAT-18.2d  60-16 B1 - 2016-02-03 12.56.49.ndpi'
'KLF14-B6NTAC-MAT-18.2e  61-16 B1 - 2016-02-03 14.02.25.ndpi'
'KLF14-B6NTAC-MAT-18.2f  62-16 B1 - 2016-02-03 15.00.17.ndpi'
'KLF14-B6NTAC-MAT-18.2g  63-16 B1 - 2016-02-03 16.40.37.ndpi'
'KLF14-B6NTAC-MAT-18.3b  223-16 B1 - 2016-02-25 16.53.42.ndpi'
'KLF14-B6NTAC-MAT-18.3c  218-16 B1 - 2016-02-18 12.51.46.ndpi'
'KLF14-B6NTAC-MAT-18.3d  224-16 B1 - 2016-02-26 10.48.56.ndpi'
'KLF14-B6NTAC-MAT-19.1a  56-16 B1 - 2016-02-02 16.57.46.ndpi'
'KLF14-B6NTAC-MAT-19.2b  219-16 B1 - 2016-02-18 14.21.50.ndpi'
'KLF14-B6NTAC-MAT-19.2c  220-16 B1 - 2016-02-18 16.40.48.ndpi'
'KLF14-B6NTAC-MAT-19.2e  221-16 B1 - 2016-02-25 13.15.27.ndpi'
'KLF14-B6NTAC-MAT-19.2f  217-16 B1 - 2016-02-18 11.23.22.ndpi'
'KLF14-B6NTAC-MAT-19.2g  222-16 B1 - 2016-02-25 14.51.57.ndpi'
'KLF14-B6NTAC-PAT-36.3a  409-16 B1 - 2016-03-15 09.24.54.ndpi'
'KLF14-B6NTAC-PAT-36.3b  412-16 B1 - 2016-03-15 14.11.47.ndpi'
'KLF14-B6NTAC-PAT-36.3d  416-16 B1 - 2016-03-16 14.22.04.ndpi'
'KLF14-B6NTAC-PAT-37.2a  406-16 B1 - 2016-03-14 11.46.47.ndpi'
'KLF14-B6NTAC-PAT-37.2b  410-16 B1 - 2016-03-15 11.12.01.ndpi'
'KLF14-B6NTAC-PAT-37.2c  407-16 B1 - 2016-03-14 12.54.55.ndpi'
'KLF14-B6NTAC-PAT-37.2d  411-16 B1 - 2016-03-15 12.01.13.ndpi'
'KLF14-B6NTAC-PAT-37.2e  408-16 B1 - 2016-03-14 16.06.43.ndpi'
'KLF14-B6NTAC-PAT-37.2f  405-16 B1 - 2016-03-14 09.49.45.ndpi'
'KLF14-B6NTAC-PAT-37.2g  415-16 B1 - 2016-03-16 11.04.45.ndpi'
'KLF14-B6NTAC-PAT-37.2h  418-16 B1 - 2016-03-16 16.42.16.ndpi'
'KLF14-B6NTAC-PAT-37.3a  413-16 B1 - 2016-03-15 15.31.26.ndpi'
'KLF14-B6NTAC-PAT-37.3c  414-16 B1 - 2016-03-15 16.49.22.ndpi'
'KLF14-B6NTAC-PAT-37.4a  417-16 B1 - 2016-03-16 15.25.38.ndpi'
'KLF14-B6NTAC-PAT-37.4b  419-16 B1 - 2016-03-17 09.10.42.ndpi'
'KLF14-B6NTAC-PAT-38.1a  90-16 B1 - 2016-02-04 17.27.42.ndpi'
'KLF14-B6NTAC-PAT-39.1h  453-16 B1 - 2016-03-17 11.15.50.ndpi'
'KLF14-B6NTAC-PAT-39.2d  454-16 B1 - 2016-03-17 12.16.06.ndpi'
#
'KLF14-B6NTAC-37.1f PAT 111-16 C2 - 2016-02-16 11.26 (1).ndpi'
#
'KLF14-B6NTAC-PAT-37.2b  410-16 C3 - 2016-03-15 11.34.14.ndpi'
'KLF14-B6NTAC-PAT-37.2c  407-16 C3 - 2016-03-14 14.22.59.ndpi'
'KLF14-B6NTAC-PAT-37.2d  411-16 C3 - 2016-03-15 12.49.31.ndpi'
#
'KLF14-B6NTAC-PAT 37.2b 410-16 B4 - 2020-02-14 10.22.36.ndpi'
'KLF14-B6NTAC-PAT 37.2b 410-16 C4 - 2020-02-14 10.27.23.ndpi'
'KLF14-B6NTAC-PAT 37.2c 407-16 B4 - 2020-02-14 10.19.12.ndpi'
'KLF14-B6NTAC-PAT 37.2c 407-16 C4 - 2020-02-14 10.15.57.ndpi'
'KLF14-B6NTAC-PAT 37.2d 411-16 B4 - 2020-02-14 10.30.19.ndpi'
'KLF14-B6NTAC-PAT 37.2d 411-16 C4 - 2020-02-14 10.34.10.ndpi'
#
'KLF14-B6NTAC-MAT-19.2c  220-16 C2 - 2016-02-18 17.08.43.ndpi'
'KLF14-B6NTAC-MAT-19.2c  220-16 C3 - 2016-02-18 17.13.08.ndpi'
)

pushd "${ndpi_dir}" || exit
# aggregate NDPI files into a non-compressed 7z file. Compression will only reduce the final size from 27GB to 21GB, but
# increase processing time by a lot, which will be incovenient for people downloading the dataset
7z a -m0=Copy ${compressed_file} "${ndpi_list[@]}"
popd
