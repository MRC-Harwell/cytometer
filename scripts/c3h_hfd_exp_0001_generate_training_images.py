"""
Read full .ndpi slides, rough segmentation of tissue areas, random selection of centroids, extract
1001x1001 windows around centroids.

The windows are saved with row_R_col_C, where R, C are the row, col centroid of the image. You can get the offset of the
image from the centroid as offset = centroid - box_half_size = centroid - 500.

We include two version of the code:
    * OLD_CODE=True: This is how things were done for the KLF14 training dataset. There's a bug because instead of using
      the high resolution centroid we used the low resolution centroid. This still created a valid training dataset,
      though.
    * OLD_CODE=False: This is how we would do things in future.
"""

# script name to identify this experiment
experiment_id = 'c3h_hfd_exp_0001_generate_training_images'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import openslide
import numpy as np
from statistics import mode
import matplotlib.pyplot as plt
import cv2
from random import randint, seed
import tifffile
import glob
from cytometer.utils import rough_foreground_mask

DEBUG = False

ndpi_dir = os.path.join(home, 'scan_srv2_cox/Maz Yon')
training_dir = os.path.join(home, 'Data/cytometer_data/c3h/c3h_hfd_training')
seg_dir = os.path.join(home, 'Data/cytometer_data/c3h/c3h_hfd_seg')
downsample_factor = 8.0

# use the old version of the code that was used for the KLF14 experiments. The new function rough_foreground_mask()
# produces a similar result, but because there are a few different pixels in "seg", the randomly selected training
# windows would be different.
OLD_CODE = True

box_size = 1001
box_half_size = int((box_size - 1) / 2)
n_samples = 5

# explicit list of files, to avoid different results if the files in the directory change
hand_segmented_files_list = [
    'C3H-H339.7a  363-16 A1 - 2016-04-20 12.15.22_row_001108_col_007204.xcf',
    'C3H-H339.7a  363-16 A1 - 2016-04-20 12.15.22_row_010356_col_031316.xcf',
    'C3H-H339.7a  363-16 A1 - 2016-04-20 12.15.22_row_015708_col_015356.xcf',
    'C3H-H339.7a  363-16 A1 - 2016-04-20 12.15.22_row_022156_col_061124.xcf',
    'C3H-H339.7a  363-16 A1 - 2016-04-20 12.15.22_row_034164_col_004740.xcf',
    'C3H-H339.7a  363-16 B1 - 2016-04-20 15.11.45_row_011876_col_016100.xcf',
    'C3H-H339.7a  363-16 B1 - 2016-04-20 15.11.45_row_011988_col_062884.xcf',
    'C3H-H339.7a  363-16 B1 - 2016-04-20 15.11.45_row_017668_col_078636.xcf',
    'C3H-H339.7a  363-16 B2 - 2016-04-20 15.18.06_row_006452_col_054324.xcf',
    'C3H-H339.7a  363-16 B2 - 2016-04-20 15.18.06_row_013076_col_051580.xcf',
    'C3H-H339.7a  363-16 B2 - 2016-04-20 15.18.06_row_013212_col_009276.xcf',
    'C3H-H339.7a  363-16 B3 - 2016-04-20 15.51.46_row_004916_col_013532.xcf',
    'C3H-H339.7a  363-16 B3 - 2016-04-20 15.51.46_row_006076_col_067876.xcf',
    'C3H-H339.7a  363-16 B3 - 2016-04-20 15.51.46_row_021548_col_056020.xcf',
    'C3H-H 339.7a 363-16 C1 - 2016-04-21 16.59.12_row_009124_col_018468.xcf',
    'C3H-H 339.7a 363-16 C1 - 2016-04-21 16.59.12_row_014316_col_033076.xcf',
    'C3H-H 339.7a 363-16 C1 - 2016-04-21 16.59.12_row_023468_col_038236.xcf',
    'C3H-H 339.7a 363-16 C1 - 2016-04-21 16.59.12_row_025244_col_049676.xcf',
    'C3H-H 339.7a 363-16 C1 - 2016-04-21 16.59.12_row_031844_col_026540.xcf',
    'C3H-H 339.7a 363-16 C2 - 2016-04-21 17.04.17_row_004132_col_016716.xcf',
    'C3H-H 339.7a 363-16 C2 - 2016-04-21 17.04.17_row_007724_col_032468.xcf',
    'C3H-H 339.7a 363-16 C2 - 2016-04-21 17.04.17_row_008804_col_037300.xcf',
    'C3H-H 339.7a 363-16 C2 - 2016-04-21 17.04.17_row_024876_col_051692.xcf',
    'C3H-H 339.7a 363-16 C2 - 2016-04-21 17.04.17_row_029460_col_030948.xcf',
    'C3H-H 339.7a 363-16 C3 - 2016-04-21 17.08.56_row_008956_col_033436.xcf',
    'C3H-H 339.7a 363-16 C3 - 2016-04-21 17.08.56_row_010380_col_028228.xcf',
    'C3H-H 339.7a 363-16 C3 - 2016-04-21 17.08.56_row_015476_col_015116.xcf',
    'C3H-H 339.7a 363-16 C3 - 2016-04-21 17.08.56_row_030468_col_020868.xcf',
    'C3H-H339.7b  364-16 B1 - 2016-04-20 16.08.23_row_010964_col_025444.xcf',
    'C3H-H339.7b  364-16 B1 - 2016-04-20 16.08.23_row_011964_col_028612.xcf',
    'C3H-H339.7b  364-16 B1 - 2016-04-20 16.08.23_row_018572_col_001668.xcf',
    'C3H-H339.7b  364-16 B1 - 2016-04-20 16.08.23_row_028924_col_061308.xcf',
    'C3H-H339.7b  364-16 B2 - 2016-04-20 16.01.42_row_023148_col_024828.xcf',
    'C3H-H339.7b  364-16 B2 - 2016-04-20 16.01.42_row_023348_col_054316.xcf',
    'C3H-H339.7b  364-16 B2 - 2016-04-20 16.01.42_row_037012_col_055588.xcf',
    'C3H-H339.7b  364-16 B2 - 2016-04-20 16.01.42_row_040540_col_060388.xcf',
    'C3H-H339.7b  364-16 B2 - 2016-04-20 16.01.42_row_041708_col_062068.xcf',
    'C3H-H339.7b  364-16 B3 - 2016-04-20 16.15.48_row_008148_col_066780.xcf',
    'C3H-H339.7b  364-16 B3 - 2016-04-20 16.15.48_row_013284_col_026892.xcf',
    'C3H-H 339.7b 364-16 C1 - 2016-04-21 17.22.26_row_004196_col_010444.xcf',
    'C3H-H 339.7b 364-16 C1 - 2016-04-21 17.22.26_row_019740_col_051124.xcf',
    'C3H-H 339.7b 364-16 C1 - 2016-04-21 17.22.26_row_041604_col_009940.xcf',
    'C3H-H 339.7b 364-16 C2 - 2016-04-21 17.27.49_row_003860_col_014548.xcf',
    'C3H-H 339.7b 364-16 C2 - 2016-04-21 17.27.49_row_006956_col_017156.xcf',
    'C3H-H 339.7b 364-16 C2 - 2016-04-21 17.27.49_row_026316_col_036004.xcf',
    'C3H-H 339.7b 364-16 C2 - 2016-04-21 17.27.49_row_034100_col_035172.xcf',
    'C3H-H 339.7b 364-16 C3 - 2016-04-21 17.32.50_row_010276_col_007420.xcf',
    'C3H-H 339.7b 364-16 C3 - 2016-04-21 17.32.50_row_016740_col_048612.xcf',
    'C3H-H 339.7b 364-16 C3 - 2016-04-21 17.32.50_row_023500_col_052116.xcf',
    'C3H-H 339.7b 364-16 C3 - 2016-04-21 17.32.50_row_035636_col_006684.xcf',
    'C3H-H339.7c  365-16 B1 - 2016-04-20 17.18.09_row_011740_col_068644.xcf',
    'C3H-H339.7c  365-16 B1 - 2016-04-20 17.18.09_row_020620_col_015532.xcf',
    'C3H-H339.7c  365-16 B2 - 2016-04-20 17.24.48_row_009092_col_014740.xcf',
    'C3H-H339.7c  365-16 B2 - 2016-04-20 17.24.48_row_013428_col_077740.xcf',
    'C3H-H339.7c  365-16 B2 - 2016-04-20 17.24.48_row_034476_col_054076.xcf',
    'C3H-H339.7c  365-16 B2 - 2016-04-20 17.24.48_row_035108_col_058684.xcf',
    'C3H-H 339.7c 365-16 C1 - 2016-04-21 17.37.25_row_006100_col_047468.xcf',
    'C3H-H 339.7c 365-16 C1 - 2016-04-21 17.37.25_row_017404_col_011548.xcf',
    'C3H-H 339.7c 365-16 C1 - 2016-04-21 17.37.25_row_027132_col_070556.xcf',
    'C3H-H 339.7c 365-16 C1 - 2016-04-21 17.37.25_row_027836_col_043988.xcf',
    'C3H-H 339.7c 365-16 C1 - 2016-04-21 17.37.25_row_032036_col_025964.xcf',
    'C3H-H 339.7c 365-16 C2 - 2016-04-21 17.51.33_row_003252_col_028980.xcf',
    'C3H-H 339.7c 365-16 C2 - 2016-04-21 17.51.33_row_004580_col_031548.xcf',
    'C3H-H 339.7c 365-16 C2 - 2016-04-21 17.51.33_row_036716_col_023692.xcf',
    'C3H-H 339.7c 365-16 C3 - 2016-04-21 17.57.40_row_004132_col_050644.xcf',
    'C3H-H 339.7c 365-16 C3 - 2016-04-21 17.57.40_row_014860_col_048548.xcf',
    'C3H-H 339.7c 365-16 C3 - 2016-04-21 17.57.40_row_016932_col_012732.xcf',
    'C3H-H 339.7c 365-16 C3 - 2016-04-21 17.57.40_row_024604_col_016556.xcf',
    'C3H-H339.7d 366-16 B1 - 2016-04-21 11.09.04_row_021540_col_007652.xcf',
    'C3H-H339.7d 366-16 B1 - 2016-04-21 11.09.04_row_022644_col_051860.xcf',
    'C3H-H339.7d 366-16 B1 - 2016-04-21 11.09.04_row_035572_col_010188.xcf',
    'C3H-H339.7d 366-16 B2 - 2016-04-21 11.17.11_row_009380_col_027924.xcf',
    'C3H-H339.7d 366-16 B3 - 2016-04-21 11.25.18_row_011148_col_077412.xcf',
    'C3H-H339.7d 366-16 B3 - 2016-04-21 11.25.18_row_026260_col_029268.xcf',
    'C3H-H339.7d 366-16 B3 - 2016-04-21 11.25.18_row_028116_col_013316.xcf',
    'C3H-H339.7d 366-16 B3 - 2016-04-21 11.25.18_row_034604_col_064028.xcf',
    'C3H-H 339.7d 366-16 C2 - 2016-04-21 18.15.22_row_010740_col_009972.xcf',
    'C3H-H 339.7d 366-16 C2 - 2016-04-21 18.15.22_row_014052_col_063252.xcf',
    'C3H-H 339.7d 366-16 C2 - 2016-04-21 18.15.22_row_014284_col_061604.xcf',
    'C3H-H 339.7d 366-16 C2 - 2016-04-21 18.15.22_row_017572_col_047988.xcf',
    'C3H-H 339.7d 366-16 C2 - 2016-04-21 18.15.22_row_021684_col_047524.xcf',
    'C3H-H 339.7d 366-16 C3 - 2016-04-21 18.19.45_row_004140_col_007804.xcf',
    'C3H-H 339.7d 366-16 C3 - 2016-04-21 18.19.45_row_005940_col_005492.xcf',
    'C3H-H 339.7d 366-16 C3 - 2016-04-21 18.19.45_row_029612_col_066564.xcf',
    'C3H-H 340.6e 367-16 C2 - 2016-04-21 18.29.12_row_027340_col_019228.xcf',
    'C3H-H 340.6e 367-16 C3 - 2016-04-21 18.34.13_row_005404_col_001388.xcf',
    'C3H-H 340.6e 367-16 C3 - 2016-04-21 18.34.13_row_009540_col_009516.xcf',
    'C3H-H 340.6e 367-16 C3 - 2016-04-21 18.34.13_row_021356_col_009164.xcf',
    'C3H-H 340.6e 367-16 C3 - 2016-04-21 18.34.13_row_021580_col_058948.xcf',
    'C3HHM 3084.7a 358-16 B1 - 2016-04-19 11.28.31_row_008644_col_036500.xcf',
    'C3HHM 3084.7a 358-16 B1 - 2016-04-19 11.28.31_row_014444_col_014292.xcf',
    'C3HHM 3084.7a 358-16 B1 - 2016-04-19 11.28.31_row_028564_col_010852.xcf',
    'C3HHM 3084.7a 358-16 B1 - 2016-04-19 11.28.31_row_033116_col_012628.xcf',
    'C3HHM 3084.7a 358-16 B1 - 2016-04-19 11.28.31_row_035044_col_019812.xcf',
    'C3HHM 3084.7a 358-16 B2 - 2016-04-19 10.58.03_row_014364_col_056084.xcf',
    'C3HHM 3084.7a 358-16 B2 - 2016-04-19 10.58.03_row_016300_col_047612.xcf',
    'C3HHM 3084.7a 358-16 B2 - 2016-04-19 10.58.03_row_035788_col_033052.xcf',
    'C3HHM 3084.7a 358-16 B2 - 2016-04-19 10.58.03_row_036332_col_033508.xcf',
    'C3HHM 3084.7a 358-16 C1 - 2016-04-19 10.48.21_row_015956_col_032916.xcf',
    'C3HHM 3084.7a 358-16 C1 - 2016-04-19 10.48.21_row_018196_col_026468.xcf',
    'C3HHM 3084.7a 358-16 C1 - 2016-04-19 10.48.21_row_022444_col_038412.xcf',
    'C3HHM 3084.7a 358-16 C1 - 2016-04-19 10.48.21_row_023604_col_025348.xcf',
    'C3HHM 3084.7a 358-16 C1 - 2016-04-19 10.48.21_row_026044_col_035932.xcf',
    'C3HHM 3084.7a 358-16 C2 - 2016-04-19 10.44.06_row_017588_col_003572.xcf',
    'C3HHM 3084.7a 358-16 C2 - 2016-04-19 10.44.06_row_024316_col_026660.xcf',
    'C3HHM 3084.7a 358-16 C2 - 2016-04-19 10.44.06_row_028804_col_041340.xcf',
    'C3HHM 3084.7a 358-16 C2 - 2016-04-19 10.44.06_row_033596_col_031708.xcf',
    'C3HHM 3084.7a 358-16 C3 - 2016-04-19 10.39.32_row_011092_col_022532.xcf',
    'C3HHM 3084.7a 358-16 C3 - 2016-04-19 10.39.32_row_029796_col_047716.xcf',
    'C3HHM 3095.5a 353-16 A1 - 2016-04-15 11.03.26_row_013684_col_047428.xcf',
    'C3HHM 3095.5a 353-16 A1 - 2016-04-15 11.03.26_row_014356_col_029260.xcf',
    'C3HHM 3095.5a 353-16 A1 - 2016-04-15 11.03.26_row_021628_col_037836.xcf',
    'C3HHM 3095.5a 353-16 A1 - 2016-04-15 11.03.26_row_035972_col_029676.xcf',
    'C3HHM 3095.5a 353-16 A1 - 2016-04-15 11.03.26_row_036444_col_019524.xcf',
    'C3HHM 3095.5a 353-16 B1 - 2016-04-15 11.21.15_row_009132_col_018684.xcf',
    'C3HHM 3095.5a 353-16 B1 - 2016-04-15 11.21.15_row_023924_col_028804.xcf',
    'C3HHM 3095.5a 353-16 B1 - 2016-04-15 11.21.15_row_024772_col_052188.xcf',
    'C3HHM 3095.5a 353-16 B1 - 2016-04-15 11.21.15_row_028644_col_049540.xcf',
    'C3HHM 3095.5a 353-16 B1 - 2016-04-15 11.21.15_row_031284_col_038044.xcf',
    'C3HHM 3095.5a 353-16 B2 - 2016-04-15 11.26.26_row_013484_col_026364.xcf',
    'C3HHM 3095.5a 353-16 B2 - 2016-04-15 11.26.26_row_029452_col_053140.xcf',
    'C3HHM 3095.5a 353-16 B3 - 2016-04-15 11.31.32_row_005828_col_013876.xcf',
    'C3HHM 3095.5a 353-16 B3 - 2016-04-15 11.31.32_row_014828_col_044756.xcf',
    'C3HHM 3095.5a 353-16 B3 - 2016-04-15 11.31.32_row_022228_col_016740.xcf',
    'C3HHM 3095.5a 353-16 B3 - 2016-04-15 11.31.32_row_031780_col_039012.xcf',
    'C3HHM 3095.5a 353-16 C1 - 2016-04-15 11.54.24_row_007100_col_039564.xcf',
    'C3HHM 3095.5a 353-16 C1 - 2016-04-15 11.54.24_row_017404_col_030100.xcf',
    'C3HHM 3095.5a 353-16 C1 - 2016-04-15 11.54.24_row_020172_col_037148.xcf',
    'C3HHM 3095.5a 353-16 C1 - 2016-04-15 11.54.24_row_027028_col_021860.xcf',
    'C3HHM 3095.5a 353-16 C1 - 2016-04-15 11.54.24_row_031684_col_061236.xcf',
    'C3HHM 3095.5a 353-16 C2 - 2016-04-15 12.00.33_row_008380_col_052844.xcf',
    'C3HHM 3095.5a 353-16 C2 - 2016-04-15 12.00.33_row_021612_col_021964.xcf',
    'C3HHM 3095.5a 353-16 C2 - 2016-04-15 12.00.33_row_022308_col_057756.xcf',
    'C3HHM 3095.5a 353-16 C2 - 2016-04-15 12.00.33_row_028412_col_018900.xcf',
    'C3HHM 3095.5a 353-16 C2 - 2016-04-15 12.00.33_row_032956_col_012028.xcf',
    'C3HHM 3095.5a 353-16 C3 - 2016-04-15 12.06.20_row_000508_col_034156.xcf',
    'C3HHM 3095.5a 353-16 C3 - 2016-04-15 12.06.20_row_004268_col_031212.xcf',
    'C3HHM 3095.5a 353-16 C3 - 2016-04-15 12.06.20_row_012260_col_021172.xcf',
    'C3HHM 3095.5a 353-16 C3 - 2016-04-15 12.06.20_row_030460_col_040620.xcf',
    'C3HHM 3095.5a 353-16 C3 - 2016-04-15 12.06.20_row_030972_col_061748.xcf',
    'C3HHM 3095.5b 354-16 B1 - 2016-04-15 14.24.56_row_004556_col_047628.xcf',
    'C3HHM 3095.5b 354-16 B1 - 2016-04-15 14.24.56_row_016780_col_012476.xcf',
    'C3HHM 3095.5b 354-16 B1 - 2016-04-15 14.24.56_row_028484_col_032276.xcf',
    'C3HHM 3095.5b 354-16 B1 - 2016-04-15 14.24.56_row_029212_col_038540.xcf',
    'C3HHM 3095.5b 354-16 B1 - 2016-04-15 14.24.56_row_030708_col_016772.xcf',
    'C3HHM 3095.5b 354-16 B2 - 2016-04-15 14.30.35_row_002292_col_020708.xcf',
    'C3HHM 3095.5b 354-16 B2 - 2016-04-15 14.30.35_row_008060_col_012516.xcf',
    'C3HHM 3095.5b 354-16 B2 - 2016-04-15 14.30.35_row_027220_col_034980.xcf',
    'C3HHM 3095.5b 354-16 B2 - 2016-04-15 14.30.35_row_031964_col_036444.xcf',
    'C3HHM 3095.5b 354-16 B3 - 2016-04-15 14.36.08_row_003636_col_012244.xcf',
    'C3HHM 3095.5b 354-16 B3 - 2016-04-15 14.36.08_row_006948_col_044884.xcf',
    'C3HHM 3095.5b 354-16 B3 - 2016-04-15 14.36.08_row_007276_col_031900.xcf',
    'C3HHM 3095.5b 354-16 B3 - 2016-04-15 14.36.08_row_017396_col_043172.xcf',
    'C3HHM 3095.5b 354-16 C1 - 2016-04-15 14.41.54_row_022588_col_021500.xcf',
    'C3HHM 3095.5b 354-16 C1 - 2016-04-15 14.41.54_row_028684_col_054436.xcf',
    'C3HHM 3095.5b 354-16 C1 - 2016-04-15 14.41.54_row_030460_col_054316.xcf',
    'C3HHM 3095.5b 354-16 C1 - 2016-04-15 14.41.54_row_034044_col_021052.xcf',
    'C3HHM 3095.5b 354-16 C1 - 2016-04-15 14.41.54_row_035996_col_027164.xcf',
    'C3HHM 3095.5b 354-16 C2 - 2016-04-15 14.46.46_row_004756_col_033076.xcf',
    'C3HHM 3095.5b 354-16 C2 - 2016-04-15 14.46.46_row_006316_col_006132.xcf',
    'C3HHM 3095.5b 354-16 C2 - 2016-04-15 14.46.46_row_007932_col_032692.xcf',
    'C3HHM 3095.5b 354-16 C2 - 2016-04-15 14.46.46_row_022676_col_018524.xcf',
    'C3HHM 3095.5b 354-16 C3 - 2016-04-15 14.51.31_row_029956_col_049852.xcf',
    'C3HHM 3095.5b 354-16 C3 - 2016-04-15 14.51.31_row_033540_col_023252.xcf',
    'C3HHM 3095.5b 354-16 C3 - 2016-04-15 14.51.31_row_034516_col_023964.xcf',
    'C3HHM 3095.5b 354-16 C3 - 2016-04-15 14.51.31_row_038220_col_012436.xcf',
    'C3HHM 3095.5b 354-16 C3 - 2016-04-15 14.51.31_row_038956_col_012700.xcf',
    'C3HHM 3095.5c 368-16 B1 - 2016-04-21 15.50.56_row_012876_col_009404.xcf',
    'C3HHM 3095.5c 368-16 B1 - 2016-04-21 15.50.56_row_013020_col_008628.xcf',
    'C3HHM 3095.5c 368-16 B1 - 2016-04-21 15.50.56_row_024788_col_035484.xcf',
    'C3HHM 3095.5c 368-16 B1 - 2016-04-21 15.50.56_row_028612_col_029780.xcf',
    'C3HHM 3095.5c 368-16 B1 - 2016-04-21 15.50.56_row_029548_col_038324.xcf',
    'C3HHM 3095.5c 368-16 B2 - 2016-04-21 15.55.46_row_019460_col_003460.xcf',
    'C3HHM 3095.5c 368-16 B2 - 2016-04-21 15.55.46_row_021932_col_019788.xcf',
    'C3HHM 3095.5c 368-16 B2 - 2016-04-21 15.55.46_row_024284_col_030364.xcf',
    'C3HHM 3095.5c 368-16 B2 - 2016-04-21 15.55.46_row_025532_col_013108.xcf',
    'C3HHM 3095.5c 368-16 B2 - 2016-04-21 15.55.46_row_028716_col_032092.xcf',
    'C3HHM 3095.5c 368-16 B3 - 2016-04-21 15.59.57_row_007540_col_019156.xcf',
    'C3HHM 3095.5c 368-16 B3 - 2016-04-21 15.59.57_row_009532_col_036428.xcf',
    'C3HHM 3095.5c 368-16 B3 - 2016-04-21 15.59.57_row_013116_col_045692.xcf',
    'C3HHM 3095.5c 368-16 B3 - 2016-04-21 15.59.57_row_014788_col_012564.xcf',
    'C3HHM 3095.5c 368-16 B3 - 2016-04-21 15.59.57_row_031212_col_036564.xcf',
    'C3HHM 3095.5c 368-16 C1 - 2016-04-21 16.04.38_row_012780_col_018892.xcf',
    'C3HHM 3095.5c 368-16 C1 - 2016-04-21 16.04.38_row_013484_col_013276.xcf',
    'C3HHM 3095.5c 368-16 C1 - 2016-04-21 16.04.38_row_019676_col_087140.xcf',
    'C3HHM 3095.5c 368-16 C1 - 2016-04-21 16.04.38_row_023924_col_007060.xcf',
    'C3HHM 3095.5c 368-16 C1 - 2016-04-21 16.04.38_row_024828_col_071556.xcf',
    'C3HHM 3095.5c 368-16 C2 - 2016-04-21 16.12.43_row_031852_col_005380.xcf',
    'C3HHM 3095.5c 368-16 C3 - 2016-04-21 16.20.34_row_014420_col_081004.xcf',
    'C3HHM 3095.5c 368-16 C3 - 2016-04-21 16.20.34_row_015812_col_022220.xcf',
    'C3HHM 3095.5c 368-16 C3 - 2016-04-21 16.20.34_row_024556_col_003260.xcf',
    'C3HHM 3095.5c 368-16 C3 - 2016-04-21 16.20.34_row_034916_col_052596.xcf',
    'C3HHM-3095.5d-369-16 B1 - 2016-04-27 08.15.10_row_006948_col_025460.xcf',
    'C3HHM-3095.5d-369-16 B1 - 2016-04-27 08.15.10_row_017476_col_032724.xcf',
    'C3HHM-3095.5d-369-16 B1 - 2016-04-27 08.15.10_row_018620_col_035740.xcf',
    'C3HHM-3095.5d-369-16 B1 - 2016-04-27 08.15.10_row_022100_col_039364.xcf',
    'C3HHM-3095.5d-369-16 B2 - 2016-04-27 08.21.39_row_004204_col_025276.xcf',
    'C3HHM-3095.5d-369-16 B2 - 2016-04-27 08.21.39_row_009596_col_023716.xcf',
    'C3HHM-3095.5d-369-16 B2 - 2016-04-27 08.21.39_row_010244_col_010732.xcf',
    'C3HHM-3095.5d-369-16 B3 - 2016-04-27 08.26.18_row_008868_col_032140.xcf',
    'C3HHM-3095.5d-369-16 B3 - 2016-04-27 08.26.18_row_010676_col_027532.xcf',
    'C3HHM-3095.5d-369-16 B3 - 2016-04-27 08.26.18_row_019212_col_021820.xcf',
    'C3HHM-3095.5d-369-16 B3 - 2016-04-27 08.26.18_row_025948_col_003356.xcf',
    'C3HHM-3095.5d-369-16 C1 - 2016-04-27 08.31.07_row_018500_col_024212.xcf',
    'C3HHM-3095.5d-369-16 C1 - 2016-04-27 08.31.07_row_026372_col_047620.xcf',
    'C3HHM-3095.5d-369-16 C1 - 2016-04-27 08.31.07_row_037508_col_004836.xcf',
    'C3HHM-3095.5d-369-16 C1 - 2016-04-27 08.31.07_row_037620_col_007612.xcf',
    'C3HHM-3095.5d-369-16 C2 - 2016-04-27 08.36.01_row_009580_col_025028.xcf',
    'C3HHM-3095.5d-369-16 C2 - 2016-04-27 08.36.01_row_032180_col_039500.xcf',
    'C3HHM-3095.5d-369-16 C2 - 2016-04-27 08.36.01_row_035788_col_039972.xcf',
    'C3HHM-3095.5d-369-16 C3 - 2016-04-27 08.40.53_row_019436_col_003772.xcf',
    'C3HHM-3095.5d-369-16 C3 - 2016-04-27 08.40.53_row_022764_col_046628.xcf',
    'C3HHM-3095.5d-369-16 C3 - 2016-04-27 08.40.53_row_037196_col_004844.xcf',
    'C3HHM 3098.6a 359-16 B1 - 2016-04-19 11.59.59_row_012884_col_060332.xcf',
    'C3HHM 3104.1a 355-16 B1 - 2016-04-15 15.42.41_row_020276_col_026724.xcf',
    'C3HHM 3104.1a 355-16 B1 - 2016-04-15 15.42.41_row_021980_col_030108.xcf',
    'C3HHM 3104.1a 355-16 B1 - 2016-04-15 15.42.41_row_029812_col_022908.xcf',
    'C3HHM 3104.1a 355-16 B2 - 2016-04-15 15.49.06_row_013876_col_014524.xcf',
    'C3HHM 3104.1a 355-16 B2 - 2016-04-15 15.49.06_row_014060_col_029060.xcf',
    'C3HHM 3104.1a 355-16 B2 - 2016-04-15 15.49.06_row_030652_col_039884.xcf',
    'C3HHM 3104.1a 355-16 B2 - 2016-04-15 15.49.06_row_030916_col_042732.xcf',
    'C3HHM 3104.1a 355-16 B3 - 2016-04-15 15.55.42_row_009708_col_046956.xcf',
    'C3HHM 3104.1a 355-16 B3 - 2016-04-15 15.55.42_row_032836_col_029100.xcf',
    'C3HHM 3104.1a 355-16 B3 - 2016-04-15 15.55.42_row_033428_col_016780.xcf',
    'C3HHM 3104.1a 355-16 B3 - 2016-04-15 15.55.42_row_039932_col_057900.xcf',
    'C3HHM 3104.1a 355-16 B3 - 2016-04-15 15.55.42_row_040468_col_056244.xcf',
    'C3HHM 3104.1a 355-16 C1 - 2016-04-18 14.03.07_row_008660_col_029164.xcf',
    'C3HHM 3104.1a 355-16 C1 - 2016-04-18 14.03.07_row_014852_col_032620.xcf',
    'C3HHM 3104.1a 355-16 C1 - 2016-04-18 14.03.07_row_030220_col_044316.xcf',
    'C3HHM 3104.1a 355-16 C1 - 2016-04-18 14.03.07_row_034644_col_025308.xcf',
    'C3HHM 3104.1a 355-16 C1 - 2016-04-18 14.03.07_row_039500_col_015980.xcf',
    'C3HHM 3104.1a 355-16 C2 - 2016-04-18 14.09.36_row_006876_col_019276.xcf',
    'C3HHM 3104.1a 355-16 C2 - 2016-04-18 14.09.36_row_040724_col_022484.xcf',
    'C3HHM 3104.1a 355-16 C3 - 2016-04-18 14.15.49_row_034460_col_032876.xcf',
    'C3HHM 3104.1d 370-16 B1 - 2016-05-04 17.07.36_row_006636_col_026420.xcf',
    'C3HHM 3104.1d 370-16 B1 - 2016-05-04 17.07.36_row_015020_col_014316.xcf',
    'C3HHM 3104.1d 370-16 B1 - 2016-05-04 17.07.36_row_018316_col_051612.xcf',
    'C3HHM 3104.1d 370-16 B1 - 2016-05-04 17.07.36_row_020716_col_013676.xcf',
    'C3HHM 3104.1d 370-16 B2 - 2016-05-04 17.02.09_row_010332_col_032108.xcf',
    'C3HHM 3104.1d 370-16 B2 - 2016-05-04 17.02.09_row_011676_col_010556.xcf',
    'C3HHM 3104.1e 371-16 B2 - 2016-05-04 16.23.36_row_014556_col_072484.xcf',
    'C3HHM 3104.1e 371-16 B2 - 2016-05-04 16.23.36_row_022812_col_074060.xcf',
    'C3HHM 3104.1e 371-16 B2 - 2016-05-04 16.23.36_row_038916_col_013148.xcf',
    'C3HHM 3104.1e 371-16 B2 - 2016-05-04 16.23.36_row_044756_col_057900.xcf',
    'C3HHM 3104.1e 371-16 B2 - 2016-05-04 16.23.36_row_050452_col_066588.xcf',
    'C3HHM 3104.1e 371-16 B3 - 2016-05-04 16.20.08_row_006644_col_029676.xcf',
    'C3HHM 3104.1e 371-16 B3 - 2016-05-04 16.20.08_row_008100_col_045204.xcf',
    'C3HHM 3104.1e 371-16 B3 - 2016-05-04 16.20.08_row_010060_col_008028.xcf'
]
ndpi_files_list = [x.split('_row')[0] + '.ndpi' for x in hand_segmented_files_list]
ndpi_files_list = np.unique(ndpi_files_list)
ndpi_files_list = [os.path.join(ndpi_dir, x) for x in ndpi_files_list]

# Note: if you want to read the full list of C3H*.ndpi
# ndpi_files_list = glob.glob(os.path.join(ndpi_dir, 'C3H*.ndpi'))

for i_file, ndpi_file in enumerate(ndpi_files_list):

    print('File ' + str(i_file) + '/' + str(len(ndpi_files_list)) + ': ' + ndpi_file)

    if OLD_CODE:  # this is the code that was used for the KLF14 experiments

        # load file
        im = openslide.OpenSlide(os.path.join(ndpi_dir, ndpi_file))

        # level for a x8 downsample factor
        level_8 = im.get_best_level_for_downsample(downsample_factor)

        assert(im.level_downsamples[level_8] == downsample_factor)

        # Note: Now we have function cytometer.utils.rough_foreground_mask() to do the following, but the function

        # get downsampled image
        im_8 = im.read_region(location=(0, 0), level=level_8, size=im.level_dimensions[level_8])
        im_8 = np.array(im_8)
        im_8 = im_8[:, :, 0:3]

        if DEBUG:
            plt.clf()
            plt.imshow(im_8)
            plt.pause(.1)

        # reshape image to matrix with one column per colour channel
        im_8_mat = im_8.copy()
        im_8_mat = im_8_mat.reshape((im_8_mat.shape[0] * im_8_mat.shape[1], im_8_mat.shape[2]))

        # background colour
        background_colour = []
        for i in range(3):
            background_colour += [mode(im_8_mat[:, i]), ]
        background_colour_std = np.std(im_8_mat, axis=0)

        # threshold segmentation
        seg = np.ones(im_8.shape[0:2], dtype=bool)
        for i in range(3):
            seg = np.logical_and(seg, im_8[:, :, i] < background_colour[i] - background_colour_std[i])
        seg = seg.astype(dtype=np.uint8)
        seg[seg == 1] = 255

        # dilate the segmentation to fill gaps within tissue
        kernel = np.ones((25, 25), np.uint8)
        seg = cv2.dilate(seg, kernel, iterations=1)
        seg = cv2.erode(seg, kernel, iterations=1)

        # find connected components
        labels = seg.copy()
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(seg)
        lblareas = stats[:, cv2.CC_STAT_AREA]

        # labels of large components, that we assume correspond to tissue areas
        labels_large = np.where(lblareas > 5e5)[0]
        labels_large = list(labels_large)

        # label=0 is the background, so we remove it
        labels_large.remove(0)

        # only set pixels that belong to the large components
        seg = np.zeros(im_8.shape[0:2], dtype=np.uint8)
        for i in labels_large:
            seg[labels == i] = 255

    else:  # not OLD_CODE

        seg, im_8 = rough_foreground_mask(ndpi_file, downsample_factor=8.0, dilation_size=25,
                                          component_size_threshold=1e5, hole_size_treshold=0,
                                          return_im=True)
        seg *= 255

    # save segmentation as a tiff file (with ZLIB compression)
    outfilename = os.path.basename(ndpi_file)
    outfilename = os.path.splitext(outfilename)[0] + '_seg'
    outfilename = os.path.join(seg_dir, outfilename + '.tif')
    tifffile.imsave(outfilename, seg,
                    compress=9,
                    resolution=(int(im.properties["tiff.XResolution"]) / downsample_factor,
                                int(im.properties["tiff.YResolution"]) / downsample_factor,
                                im.properties["tiff.ResolutionUnit"].upper()))

    # plot the segmentation
    if DEBUG:
        plt.clf()
        plt.imshow(seg)
        plt.pause(.1)

    # these are the low-res centroids within the segmentation masks that were randomly selected in the old script
    if OLD_CODE:

        # get high resolution centroids for the current image according to the filenames
        ndpi_file_base = os.path.basename(ndpi_file).split('.ndpi')[0]
        idx = [ndpi_file_base in x for x in hand_segmented_files_list]
        hand_segmented_files_current = np.array(hand_segmented_files_list)[idx]

        # compute the centroid in high-res
        sample_centroid_upsampled = []
        for file in hand_segmented_files_current:
            row = int(file.split('_row_')[1].split('_col_')[0])
            col = int(file.split('_row_')[1].split('_col_')[1].split('.xcf')[0])
            sample_centroid_upsampled.append(
                (row, col)
            )

    else:  # not OLD_CODE

        np.random.seed(i_file)
        sample_centroid = []
        while len(sample_centroid) < n_samples:
            row = randint(0, seg.shape[0] - 1)
            col = randint(0, seg.shape[1] - 1)
            # if the centroid is a pixel that belongs to tissue...

            if seg[row, col] != 0:
                # ... add it to the list of random samples
                sample_centroid.append((row, col))

        # compute the centroid in high-res
        sample_centroid_upsampled = []
        for row, col in sample_centroid:
            sample_centroid_upsampled.append(
                (int(row * downsample_factor + np.round((downsample_factor - 1) / 2)),
                 int(col * downsample_factor + np.round((downsample_factor - 1) / 2)))
            )

    # create the training dataset by sampling the full resolution image with boxes around the centroids
    for j in range(len(sample_centroid_upsampled)):

        # high resolution centroid
        row, col = sample_centroid_upsampled[j]

        # compute top-left corner of the box from the centroid
        box_corner_row = row - box_half_size
        box_corner_col = col - box_half_size

        # extract tile from full resolution image
        tile = im.read_region(location=(box_corner_col, box_corner_row), level=0, size=(box_size, box_size))
        tile = np.array(tile)
        tile = tile[:, :, 0:3]

        # output filename
        outfilename = os.path.basename(ndpi_file)
        if OLD_CODE:
            outfilename = os.path.splitext(outfilename)[0] + '_row_' + str(row).zfill(6) \
                          + '_col_' + str(col).zfill(6)
        else:
            outfilename = os.path.splitext(outfilename)[0] + '_row_' + str(box_corner_row).zfill(6) \
                          + '_col_' + str(box_corner_col).zfill(6)
        outfilename = os.path.join(training_dir, outfilename + '.tif')

        # plot tile
        if DEBUG:
            plt.clf()
            plt.subplot(121)
            plt.imshow(tile)
            plt.subplot(122)
            foo = tifffile.imread(outfilename)
            plt.imshow(foo)
            plt.pause(.1)

        # save tile as a tiff file with ZLIB compression (LZMA or ZSTD can't be opened by QuPath)
        tifffile.imsave(outfilename, tile,
                        compress=9,
                        resolution=(int(im.properties["tiff.XResolution"]),
                                    int(im.properties["tiff.YResolution"]),
                                    im.properties["tiff.ResolutionUnit"].upper()))
