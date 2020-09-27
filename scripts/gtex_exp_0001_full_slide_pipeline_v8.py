"""
Processing full slides of GTEx WAT histology with pipeline v8:

 * data generation
   * training images (*0076*)
   * non-overlap training images (*0077*)
   * augmented training images (*0078*)
   * k-folds + extra "other" for classifier (*0094*)
 * segmentation
   * dmap (*0086*)
   * contour from dmap (*0091*)
 * classifier (*0095*)
 * segmentation correction (*0089*) networks"
 * validation (*0096*)

Difference with pipeline v7:
  * Contrast enhancement to compute rough tissue mask
  * Colour correction to match the median colour of the training data for segmentation
  * All segmented objects are saved, together with the white adipocyte probability score. That way, we can decide later
    which ones we want to keep, and which ones we want to reject.
  * If a processing window overlaps the previous one by 90% or more, we wipe it out.

Difference with rreb1_tm1b_exp_0003_full_slide_pipeline_v8.py:
  * Only the images it is applied to.

Difference with fus_delta_exp_0001_full_slide_pipeline_v8.py:
  * Change to GTEx data.

You can run this script limiting it to one GPU with:

    export CUDA_VISIBLE_DEVICES=0 && python gtex_exp_0001_full_slide_pipeline_v8.py

 Requirements for this script to work:

 1) Upload the cytometer project directory to ~/Software in the server where you are going to process the data.

 2) Run ./install_dependencies.sh in cytometer.

 3) Mount the network share//jesse.mrch.har.mrc.ac.uk/mousedata on ~/jesse_mousedata with CIFS so that we have access to
    GTEx .svs (TIFF) files. You can do it by creating an empty directory

    mkdir ~/jesse_mousedata

    and adding a line like this to /etc/fstab in the server (adjusting the uid and gid to your own).

    //jesse.mrch.har.mrc.ac.uk/mousedata          /home/rcasero/jesse_mousedata          cifs    credentials=/home/rcasero/.smbcredentials,vers=3.0,domain=mrch,sec=ntlmv2,uid=1005,gid=1005,user 0 0

    Then

    mount ~/jesse_mousedata

 4) Convert the .svs files to AIDA .dzi files, so that we can see the results of the segmentation.
    You need to go to the server that's going to process the slides, add a list of the files you want to process to
    ~/Software/cytometer/tools/TODO

    and run

    cd ~/Software/cytometer/tools
    ./TODO

 5) You need to have the models for the 10-folds of the pipeline that were trained on the KLF14 data in
    ~/Data/cytometer_data/klf14/saved_models.

 6) To monitor the segmentation as it's being processed, you need to have AIDA running

    cd ~/Software/AIDA/dist/
    node aidaLocal.js &

    You also need to create a soft link per .dzi file to the annotations you want to visualise for that file, whether
    the non-overlapping ones, or the corrected ones. E.g.

    ln -s 'FUS-DELTA-DBA-B6-IC-13.3a  1276-18 1 - 2018-12-03 12.18.01.ndpi_exp_0001_corrected.json' 'FUS-DELTA-DBA-B6-IC-13.3a  1276-18 1 - 2018-12-03 12.18.01.ndpi'

    Then you can use a browser to open the AIDA web interface by visiting the URL (note that you need to be on the MRC
    VPN, or connected from inside the office to get access to the titanrtx server)

    http://titanrtx:3000/dashboard

    You can use the interface to open a .dzi file that corresponds to an .ndpi file being segmented, and see the
    annotations (segmentation) being created for it.

"""

# script name to identify this experiment
experiment_id = 'gtex_exp_0001_full_slide_pipeline_v8.py'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
from pathlib import Path
import sys
if os.path.join(home, 'Software/cytometer') not in sys.path:
    sys.path.extend([os.path.join(home, 'Software/cytometer')])
import cytometer.utils
import cytometer.data

# Filter out INFO & WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# limit number of GPUs
if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
    print('Limiting visible CUDA devices to: ' + os.environ['CUDA_VISIBLE_DEVICES'])

# force tensorflow environment
os.environ['KERAS_BACKEND'] = 'tensorflow'

import warnings
import time
import openslide
import numpy as np
import matplotlib.pyplot as plt
from cytometer.utils import rough_foreground_mask, bspline_resample
import PIL
from keras import backend as K
import scipy.stats
from shapely.geometry import Polygon

import tensorflow as tf
if tf.test.is_gpu_available():
    print('GPU available')
else:
    raise SystemError('GPU is not available')

# # limit GPU memory used
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# set_session(tf.Session(config=config))

DEBUG = False
SAVE_FIGS = False

pipeline_root_data_dir = os.path.join(home, 'Data/cytometer_data/klf14')  # CNN models
histology_dir = os.path.join(home, 'jesse_mousedata/GTEx')
area2quantile_dir = os.path.join(home, 'GoogleDrive/Research/20190727_cytometer_paper/figures')
saved_models_dir = os.path.join(pipeline_root_data_dir, 'saved_models')
annotations_dir = os.path.join(home, 'Data/cytometer_data/aida_data_GTEx/annotations')

# file with area->quantile map precomputed from all automatically segmented slides in klf14_b6ntac_exp_0098_full_slide_size_analysis_v7.py
filename_area2quantile = os.path.join(area2quantile_dir, 'klf14_b6ntac_exp_0098_filename_area2quantile.npz')

# file with RGB modes from all training data
klf14_training_colour_histogram_file = os.path.join(saved_models_dir, 'klf14_training_colour_histogram.npz')

# model names
dmap_model_basename = 'klf14_b6ntac_exp_0086_cnn_dmap'
contour_model_basename = 'klf14_b6ntac_exp_0091_cnn_contour_after_dmap'
classifier_model_basename = 'klf14_b6ntac_exp_0095_cnn_tissue_classifier_fcn'
correction_model_basename = 'klf14_b6ntac_exp_0089_cnn_segmentation_correction_overlapping_scaled_contours'

# full resolution image window and network expected receptive field parameters
fullres_box_size = np.array([2751, 2751])
receptive_field = np.array([131, 131])

# rough_foreground_mask() parameters
downsample_factor_goal = 16  # approximate value, that may vary a bit in each histology file
dilation_size = 25
component_size_threshold = 50e3
hole_size_treshold = 8000
std_k = 1.00
enhance_contrast = 4.0
ignore_white_threshold = 253

# contour parameters
contour_downsample_factor = 0.1
bspline_k = 1

# block_split() parameters in downsampled image
block_len = np.ceil((fullres_box_size - receptive_field) / downsample_factor_goal)
block_overlap = np.ceil((receptive_field - 1) / 2 / downsample_factor_goal).astype(np.int)
window_overlap_fraction_max = 0.9

# segmentation parameters
min_cell_area = 200  # pixel
max_cell_area = 200e3  # pixel
min_mask_overlap = 0.8
phagocytosis = True
min_class_prop = 0.5
correction_window_len = 401
correction_smoothing = 11
batch_size = 16

# list of NDPI files to process
histo_files_list = [
"GTEX-1117F-0225.svs",
"GTEX-111CU-1825.svs",
"GTEX-111FC-0225.svs",
"GTEX-111FC-1425.svs",
"GTEX-111VG-2325.svs",
"GTEX-111YS-2425.svs",
"GTEX-1122O-2025.svs",
"GTEX-1128S-2125.svs",
"GTEX-113IC-0225.svs",
"GTEX-113JC-2525.svs",
"GTEX-117XS-2425.svs",
"GTEX-117YW-2325.svs",
"GTEX-117YX-2225.svs",
"GTEX-1192W-2425.svs",
"GTEX-1192X-0125.svs",
"GTEX-11DXW-0325.svs",
"GTEX-11DXX-2325.svs",
"GTEX-11DXY-2525.svs",
"GTEX-11DXZ-2225.svs",
"GTEX-11DYG-0225.svs",
"GTEX-11DZ1-0225.svs",
"GTEX-11EI6-0225.svs",
"GTEX-11EM3-2325.svs",
"GTEX-11EMC-2825.svs",
"GTEX-11EQ8-0225.svs",
"GTEX-11EQ9-2525.svs",
"GTEX-11GS4-2625.svs",
"GTEX-11GSO-2325.svs",
"GTEX-11GSP-2625.svs",
"GTEX-11H98-0225.svs",
"GTEX-11I78-2625.svs",
"GTEX-11ILO-0225.svs",
"GTEX-11LCK-1125.svs",
"GTEX-11NSD-2125.svs",
"GTEX-11NUK-0325.svs",
"GTEX-11NV4-0225.svs",
"GTEX-11O72-0225.svs",
"GTEX-11OC5-0225.svs",
"GTEX-11OF3-2425.svs",
"GTEX-11ONC-2525.svs",
"GTEX-11P7K-2025.svs",
"GTEX-11P81-2425.svs",
"GTEX-11P82-1725.svs",
"GTEX-11PRG-0625.svs",
"GTEX-11TT1-2425.svs",
"GTEX-11TTK-0225.svs",
"GTEX-11TUW-2625.svs",
"GTEX-11UD1-0225.svs",
"GTEX-11UD2-0125.svs",
"GTEX-11VI4-1825.svs",
"GTEX-11WQC-2425.svs",
"GTEX-11WQK-0625.svs",
"GTEX-11XUK-2125.svs",
"GTEX-11ZTS-0225.svs",
"GTEX-11ZTT-2525.svs",
"GTEX-11ZU8-2425.svs",
"GTEX-11ZUS-0725.svs",
"GTEX-11ZVC-2625.svs",
"GTEX-1211K-2225.svs",
"GTEX-12126-0425.svs",
"GTEX-1212Z-2425.svs",
"GTEX-12584-0225.svs",
"GTEX-12696-2525.svs",
"GTEX-1269C-2725.svs",
"GTEX-12BJ1-2325.svs",
"GTEX-12C56-1625.svs",
"GTEX-12KS4-0425.svs",
"GTEX-12WS9-0225.svs",
"GTEX-12WSA-0325.svs",
"GTEX-12WSB-0225.svs",
"GTEX-12WSC-0225.svs",
"GTEX-12WSD-0225.svs",
"GTEX-12WSE-0625.svs",
"GTEX-12WSF-0225.svs",
"GTEX-12WSG-2925.svs",
"GTEX-12WSH-2625.svs",
"GTEX-12WSI-2425.svs",
"GTEX-12WSJ-2125.svs",
"GTEX-12WSK-2525.svs",
"GTEX-12WSL-2425.svs",
"GTEX-12WSM-0225.svs",
"GTEX-12WSN-2425.svs",
"GTEX-12ZZW-0225.svs",
"GTEX-12ZZX-0225.svs",
"GTEX-12ZZY-0325.svs",
"GTEX-12ZZZ-0525.svs",
"GTEX-13111-1725.svs",
"GTEX-13112-2725.svs",
"GTEX-13113-1825.svs",
"GTEX-1313W-0225.svs",
"GTEX-1314G-1625.svs",
"GTEX-131XE-2425.svs",
"GTEX-131XF-2225.svs",
"GTEX-131XG-2425.svs",
"GTEX-131XH-2325.svs",
"GTEX-131XW-0525.svs",
"GTEX-131YS-0225.svs",
"GTEX-132AR-0225.svs",
"GTEX-132NY-0425.svs",
"GTEX-132Q8-0325.svs",
"GTEX-132QS-2525.svs",
"GTEX-1339X-2525.svs",
"GTEX-133LE-1925.svs",
"GTEX-1399Q-0625.svs",
"GTEX-1399R-2625.svs",
"GTEX-1399S-2525.svs",
"GTEX-1399T-2725.svs",
"GTEX-1399U-2425.svs",
"GTEX-139D8-0325.svs",
"GTEX-139T4-0125.svs",
"GTEX-139T6-2125.svs",
"GTEX-139T8-0225.svs",
"GTEX-139TS-2725.svs",
"GTEX-139TT-0225.svs",
"GTEX-139TU-0225.svs",
"GTEX-139UC-2625.svs",
"GTEX-139UW-2725.svs",
"GTEX-139YR-2425.svs",
"GTEX-13CF2-2625.svs",
"GTEX-13CF3-2325.svs",
"GTEX-13CIG-0125.svs",
"GTEX-13CZU-1825.svs",
"GTEX-13CZV-0125.svs",
"GTEX-13D11-2325.svs",
"GTEX-13FH7-2225.svs",
"GTEX-13FHO-0325.svs",
"GTEX-13FHP-0225.svs",
"GTEX-13FLV-2725.svs",
"GTEX-13FLW-2525.svs",
"GTEX-13FTW-2525.svs",
"GTEX-13FTX-1925.svs",
"GTEX-13FTY-0625.svs",
"GTEX-13FTZ-1925.svs",
"GTEX-13FXS-0125.svs",
"GTEX-13G51-2625.svs",
"GTEX-13IVO-0225.svs",
"GTEX-13JUV-2825.svs",
"GTEX-13JVG-0225.svs",
"GTEX-13N11-2625.svs",
"GTEX-13N1W-0225.svs",
"GTEX-13N2G-2725.svs",
"GTEX-13NYB-2625.svs",
"GTEX-13NYC-0425.svs",
"GTEX-13NYS-0225.svs",
"GTEX-13NZ8-0925.svs",
"GTEX-13NZ9-0525.svs",
"GTEX-13NZA-0125.svs",
"GTEX-13NZB-2425.svs",
"GTEX-13O1R-0225.svs",
"GTEX-13O21-1525.svs",
"GTEX-13O3O-0225.svs",
"GTEX-13O3P-0225.svs",
"GTEX-13O3Q-2325.svs",
"GTEX-13O61-2425.svs",
"GTEX-13OVG-2025.svs",
"GTEX-13OVH-0225.svs",
"GTEX-13OVI-1325.svs",
"GTEX-13OVJ-0325.svs",
"GTEX-13OVK-2125.svs",
"GTEX-13OVL-0325.svs",
"GTEX-13OW5-0225.svs",
"GTEX-13OW6-0525.svs",
"GTEX-13OW7-0225.svs",
"GTEX-13OW8-1225.svs",
"GTEX-13PDP-0225.svs",
"GTEX-13PL6-0225.svs",
"GTEX-13PL7-0925.svs",
"GTEX-13PLJ-0125.svs",
"GTEX-13PVQ-0225.svs",
"GTEX-13PVR-2425.svs",
"GTEX-13QBU-2525.svs",
"GTEX-13QIC-2425.svs",
"GTEX-13QJ3-0425.svs",
"GTEX-13QJC-2825.svs",
"GTEX-13RTJ-2525.svs",
"GTEX-13RTK-1525.svs",
"GTEX-13RTL-0225.svs",
"GTEX-13S7M-0225.svs",
"GTEX-13S86-2125.svs",
"GTEX-13SLW-0225.svs",
"GTEX-13SLX-0425.svs",
"GTEX-13U4I-0225.svs",
"GTEX-13VXT-0225.svs",
"GTEX-13VXU-0225.svs",
"GTEX-13W3W-2525.svs",
"GTEX-13W46-0325.svs",
"GTEX-13X6H-2225.svs",
"GTEX-13X6I-0725.svs",
"GTEX-13X6J-0225.svs",
"GTEX-13X6K-0225.svs",
"GTEX-13YAN-0425.svs",
"GTEX-1445S-0425.svs",
"GTEX-144FL-0225.svs",
"GTEX-144GL-0225.svs",
"GTEX-144GM-1925.svs",
"GTEX-144GN-2225.svs",
"GTEX-144GO-1625.svs",
"GTEX-145LS-0225.svs",
"GTEX-145LT-1825.svs",
"GTEX-145LU-0225.svs",
"GTEX-145LV-2225.svs",
"GTEX-145ME-1925.svs",
"GTEX-145MF-0525.svs",
"GTEX-145MG-0125.svs",
"GTEX-145MH-0225.svs",
"GTEX-145MI-0425.svs",
"GTEX-145MN-2625.svs",
"GTEX-145MO-0425.svs",
"GTEX-146FH-0225.svs",
"GTEX-146FQ-0225.svs",
"GTEX-146FR-1825.svs",
"GTEX-14753-0325.svs",
"GTEX-1477Z-1925.svs",
"GTEX-147F3-0625.svs",
"GTEX-147F4-0525.svs",
"GTEX-147GR-0425.svs",
"GTEX-147JS-0425.svs",
"GTEX-148VI-1725.svs",
"GTEX-148VJ-0225.svs",
"GTEX-1497J-2325.svs",
"GTEX-14A5H-0525.svs",
"GTEX-14A5I-0225.svs",
"GTEX-14A6H-2525.svs",
"GTEX-14ABY-0325.svs",
"GTEX-14AS3-2025.svs",
"GTEX-14ASI-0225.svs",
"GTEX-14B4R-1625.svs",
"GTEX-14BIL-0225.svs",
"GTEX-14BIM-0125.svs",
"GTEX-14BIN-0725.svs",
"GTEX-14BMU-2125.svs",
"GTEX-14BMV-0225.svs",
"GTEX-14C38-0325.svs",
"GTEX-14C39-2525.svs",
"GTEX-14C5O-0225.svs",
"GTEX-14DAQ-0225.svs",
"GTEX-14DAR-1925.svs",
"GTEX-14E1K-2125.svs",
"GTEX-14E6C-0225.svs",
"GTEX-14E6D-0225.svs",
"GTEX-14E6E-1925.svs",
"GTEX-14E7W-0225.svs",
"GTEX-14H4A-1025.svs",
"GTEX-14ICK-0325.svs",
"GTEX-14ICL-2025.svs",
"GTEX-14JFF-0225.svs",
"GTEX-14JG1-0525.svs",
"GTEX-14JG6-2325.svs",
"GTEX-14JIY-0525.svs",
"GTEX-14LLW-0225.svs",
"GTEX-14LZ3-0325.svs",
"GTEX-14PHW-0225.svs",
"GTEX-14PHX-2425.svs",
"GTEX-14PHY-2425.svs",
"GTEX-14PII-0325.svs",
"GTEX-14PJ2-0225.svs",
"GTEX-14PJ3-1925.svs",
"GTEX-14PJ4-2225.svs",
"GTEX-14PJ5-1725.svs",
"GTEX-14PJ6-2225.svs",
"GTEX-14PJM-0325.svs",
"GTEX-14PJN-2025.svs",
"GTEX-14PJO-0225.svs",
"GTEX-14PJP-0225.svs",
"GTEX-14PK6-2025.svs",
"GTEX-14PKU-2125.svs",
"GTEX-14PKV-0425.svs",
"GTEX-14PN3-2525.svs",
"GTEX-14PN4-0225.svs",
"GTEX-14PQA-0125.svs",
"GTEX-14XAO-2125.svs",
"GTEX-15CHC-2225.svs",
"GTEX-15CHQ-0225.svs",
"GTEX-15CHR-2325.svs",
"GTEX-15CHS-0225.svs",
"GTEX-15D1Q-0225.svs",
"GTEX-15D79-0225.svs",
"GTEX-15DCD-0225.svs",
"GTEX-15DCE-0325.svs",
"GTEX-15DCZ-0225.svs",
"GTEX-15DDE-0225.svs",
"GTEX-15DYW-0225.svs",
"GTEX-15DZA-1925.svs",
"GTEX-15EO6-0625.svs",
"GTEX-15EOM-0225.svs",
"GTEX-15ER7-0425.svs",
"GTEX-15ETS-0425.svs",
"GTEX-15EU6-0825.svs",
"GTEX-15F5U-1725.svs",
"GTEX-15FZZ-2225.svs",
"GTEX-15G19-0225.svs",
"GTEX-15G1A-2225.svs",
"GTEX-15RIE-2325.svs",
"GTEX-15RIF-2025.svs",
"GTEX-15RIG-0325.svs",
"GTEX-15RJ7-2325.svs",
"GTEX-15RJE-0225.svs",
"GTEX-15SB6-1625.svs",
"GTEX-15SDE-2425.svs",
"GTEX-15SHU-0225.svs",
"GTEX-15SHV-2325.svs",
"GTEX-15SHW-0425.svs",
"GTEX-15SKB-0225.svs",
"GTEX-15SZO-0225.svs",
"GTEX-15TU5-0325.svs",
"GTEX-15UF6-0925.svs",
"GTEX-15UF7-2125.svs",
"GTEX-15UKP-0525.svs",
"GTEX-169BO-1425.svs",
"GTEX-16A39-1925.svs",
"GTEX-16AAH-2025.svs",
"GTEX-16BQI-0525.svs",
"GTEX-16GPK-0225.svs",
"GTEX-16MT8-0525.svs",
"GTEX-16MT9-0525.svs",
"GTEX-16MTA-2125.svs",
"GTEX-16NFA-2225.svs",
"GTEX-16NGA-1825.svs",
"GTEX-16NPV-0225.svs",
"GTEX-16NPX-0325.svs",
"GTEX-16XZY-2525.svs",
"GTEX-16XZZ-0225.svs",
"GTEX-16YQH-0225.svs",
"GTEX-16Z82-0325.svs",
"GTEX-178AV-2125.svs",
"GTEX-17EUY-0525.svs",
"GTEX-17EVP-0425.svs",
"GTEX-17EVQ-0325.svs",
"GTEX-17F96-1725.svs",
"GTEX-17F97-0425.svs",
"GTEX-17F98-1125.svs",
"GTEX-17F9E-0125.svs",
"GTEX-17F9Y-0325.svs",
"GTEX-17GQL-2425.svs",
"GTEX-17HG3-2325.svs",
"GTEX-17HGU-0325.svs",
"GTEX-17HHE-1925.svs",
"GTEX-17HHY-0225.svs",
"GTEX-17HII-0325.svs",
"GTEX-17JCI-0225.svs",
"GTEX-17KNJ-2225.svs",
"GTEX-17MF6-0225.svs",
"GTEX-17MFQ-1825.svs",
"GTEX-183FY-0325.svs",
"GTEX-183WM-0225.svs",
"GTEX-18464-0225.svs",
"GTEX-18465-0225.svs",
"GTEX-18A66-0325.svs",
"GTEX-18A67-0225.svs",
"GTEX-18A6Q-0225.svs",
"GTEX-18A7A-0325.svs",
"GTEX-18A7B-0225.svs",
"GTEX-18D9A-2025.svs",
"GTEX-18D9B-0625.svs",
"GTEX-18D9U-0725.svs",
"GTEX-18QFQ-0225.svs",
"GTEX-19HZE-0225.svs",
"GTEX-1A32A-0225.svs",
"GTEX-1A3MV-2125.svs",
"GTEX-1A3MW-0225.svs",
"GTEX-1A3MX-0225.svs",
"GTEX-1A8FM-0125.svs",
"GTEX-1A8G6-0525.svs",
"GTEX-1A8G7-0225.svs",
"GTEX-1AMEY-1425.svs",
"GTEX-1AMFI-2125.svs",
"GTEX-1AX8Y-2125.svs",
"GTEX-1AX8Z-0225.svs",
"GTEX-1AX9I-0225.svs",
"GTEX-1AX9J-0325.svs",
"GTEX-1AX9K-2425.svs",
"GTEX-1AYCT-2225.svs",
"GTEX-1AYD5-2325.svs",
"GTEX-1B8KE-2525.svs",
"GTEX-1B8KZ-1925.svs",
"GTEX-1B8L1-2625.svs",
"GTEX-1B8SF-0525.svs",
"GTEX-1B8SG-0225.svs",
"GTEX-1B932-0225.svs",
"GTEX-1B933-0525.svs",
"GTEX-1B97I-1725.svs",
"GTEX-1B97J-1625.svs",
"GTEX-1B98T-0225.svs",
"GTEX-1B996-0125.svs",
"GTEX-1BAJH-0425.svs",
"GTEX-1C2JI-2225.svs",
"GTEX-1C475-1925.svs",
"GTEX-1C4CL-0425.svs",
"GTEX-1C64N-0225.svs",
"GTEX-1C64O-0225.svs",
"GTEX-1C6VQ-0225.svs",
"GTEX-1C6VR-2625.svs",
"GTEX-1C6VS-0425.svs",
"GTEX-1C6WA-0225.svs",
"GTEX-1CAMQ-0125.svs",
"GTEX-1CAMR-1825.svs",
"GTEX-1CAMS-0325.svs",
"GTEX-1CAV2-2325.svs",
"GTEX-1CB4E-1825.svs",
"GTEX-1CB4F-0525.svs",
"GTEX-1CB4G-0325.svs",
"GTEX-1CB4H-0325.svs",
"GTEX-1CB4I-0425.svs",
"GTEX-1CB4J-1125.svs",
"GTEX-1E1VI-0225.svs",
"GTEX-1E2YA-0225.svs",
"GTEX-1EH9U-0225.svs",
"GTEX-1EKGG-0125.svs",
"GTEX-1EKGG-0126.svs",
"GTEX-1EMGI-0225.svs",
"GTEX-1EN7A-0225.svs",
"GTEX-1EU9M-0325.svs",
"GTEX-1EWIQ-0225.svs",
"GTEX-1EX96-1425.svs",
"GTEX-1F48J-0225.svs",
"GTEX-1F52S-0225.svs",
"GTEX-1F5PK-2525.svs",
"GTEX-1F5PL-0125.svs",
"GTEX-1F6I4-0225.svs",
"GTEX-1F6IF-0625.svs",
"GTEX-1F6RS-0225.svs",
"GTEX-1F75A-0225.svs",
"GTEX-1F75B-0225.svs",
"GTEX-1F75I-0425.svs",
"GTEX-1F75W-0225.svs",
"GTEX-1F7RK-0225.svs",
"GTEX-1F88E-0525.svs",
"GTEX-1F88F-2625.svs",
"GTEX-1FIGZ-1625.svs",
"GTEX-1GF9U-2225.svs",
"GTEX-1GF9V-0225.svs",
"GTEX-1GF9W-0425.svs",
"GTEX-1GF9X-1925.svs",
"GTEX-1GMR2-2525.svs",
"GTEX-1GMR3-2025.svs",
"GTEX-1GMR8-0225.svs",
"GTEX-1GMRU-0225.svs",
"GTEX-1GN1U-0225.svs",
"GTEX-1GN1V-0225.svs",
"GTEX-1GN1W-0625.svs",
"GTEX-1GN2E-0325.svs",
"GTEX-1GN73-0325.svs",
"GTEX-1GPI6-0225.svs",
"GTEX-1GPI7-0225.svs",
"GTEX-1GTWX-0225.svs",
"GTEX-1GZ2Q-0325.svs",
"GTEX-1GZ4H-1825.svs",
"GTEX-1GZ4I-0225.svs",
"GTEX-1GZHY-0425.svs",
"GTEX-1H11D-0125.svs",
"GTEX-1H1CY-2625.svs",
"GTEX-1H1DE-2025.svs",
"GTEX-1H1DF-0225.svs",
"GTEX-1H1DG-0225.svs",
"GTEX-1H1E6-2025.svs",
"GTEX-1H1ZS-0525.svs",
"GTEX-1H23P-0225.svs",
"GTEX-1H2FU-0225.svs",
"GTEX-1H3NZ-0225.svs",
"GTEX-1H3O1-0525.svs",
"GTEX-1H3VE-0225.svs",
"GTEX-1H3VY-0225.svs",
"GTEX-1H4P4-0225.svs",
"GTEX-1HB9E-0425.svs",
"GTEX-1HBPH-0525.svs",
"GTEX-1HBPI-1425.svs",
"GTEX-1HBPM-0325.svs",
"GTEX-1HBPN-0725.svs",
"GTEX-1HC8U-2525.svs",
"GTEX-1HCU6-0225.svs",
"GTEX-1HCU7-0325.svs",
"GTEX-1HCU8-0725.svs",
"GTEX-1HCU9-1625.svs",
"GTEX-1HCUA-2225.svs",
"GTEX-1HCVE-0225.svs",
"GTEX-1HFI6-0325.svs",
"GTEX-1HFI7-2025.svs",
"GTEX-1HGF4-2525.svs",
"GTEX-1HKZK-0425.svs",
"GTEX-1HR98-0225.svs",
"GTEX-1HR9M-0225.svs",
"GTEX-1HSEH-0225.svs",
"GTEX-1HSGN-0125.svs",
"GTEX-1HSKV-0325.svs",
"GTEX-1HSMO-0225.svs",
"GTEX-1HSMP-0225.svs",
"GTEX-1HSMQ-1425.svs",
"GTEX-1HT8W-0225.svs",
"GTEX-1HUB1-0325.svs",
"GTEX-1I19N-0425.svs",
"GTEX-1I1CD-0225.svs",
"GTEX-1I1GP-2325.svs",
"GTEX-1I1GQ-0225.svs",
"GTEX-1I1GR-0225.svs",
"GTEX-1I1GS-0225.svs",
"GTEX-1I1GT-0525.svs",
"GTEX-1I1GU-2325.svs",
"GTEX-1I1GV-0425.svs",
"GTEX-1I1HK-0225.svs",
"GTEX-1I4MK-2025.svs",
"GTEX-1I6K6-0225.svs",
"GTEX-1I6K7-0225.svs",
"GTEX-1ICG6-2625.svs",
"GTEX-1ICLY-0225.svs",
"GTEX-1ICLZ-0225.svs",
"GTEX-1IDFM-0225.svs",
"GTEX-1IDJC-2125.svs",
"GTEX-1IDJD-1225.svs",
"GTEX-1IDJE-2525.svs",
"GTEX-1IDJF-0225.svs",
"GTEX-1IDJH-0425.svs",
"GTEX-1IDJI-0325.svs",
"GTEX-1IDJU-2525.svs",
"GTEX-1IDJV-0325.svs",
"GTEX-1IE54-0225.svs",
"GTEX-1IGQW-0225.svs",
"GTEX-1IKJJ-0225.svs",
"GTEX-1IKK5-0525.svs",
"GTEX-1IKOE-0225.svs",
"GTEX-1IKOH-0325.svs",
"GTEX-1IL2U-0325.svs",
"GTEX-1IL2V-0225.svs",
"GTEX-1IOXB-0225.svs",
"GTEX-1IY9M-0225.svs",
"GTEX-1J1OQ-0325.svs",
"GTEX-1J1R8-2225.svs",
"GTEX-1J8EW-0225.svs",
"GTEX-1J8JJ-0325.svs",
"GTEX-1J8Q2-0425.svs",
"GTEX-1J8Q3-1525.svs",
"GTEX-1J8QM-2325.svs",
"GTEX-1JJ6O-0225.svs",
"GTEX-1JJE9-1225.svs",
"GTEX-1JJEA-0225.svs",
"GTEX-1JK1U-2125.svs",
"GTEX-1JKYN-1225.svs",
"GTEX-1JKYR-0225.svs",
"GTEX-1JMLX-0125.svs",
"GTEX-1JMOU-1225.svs",
"GTEX-1JMPY-1425.svs",
"GTEX-1JMPZ-0325.svs",
"GTEX-1JMQI-2025.svs",
"GTEX-1JMQJ-0425.svs",
"GTEX-1JMQK-0125.svs",
"GTEX-1JMQL-0325.svs",
"GTEX-1JN1M-0125.svs",
"GTEX-1JN6P-0225.svs",
"GTEX-1JN76-0225.svs",
"GTEX-1K2DA-2125.svs",
"GTEX-1K2DU-2425.svs",
"GTEX-1K9T9-0225.svs",
"GTEX-1KAFJ-2425.svs",
"GTEX-1KANA-0825.svs",
"GTEX-1KANB-0425.svs",
"GTEX-1KANC-1925.svs",
"GTEX-1KD4Q-0125.svs",
"GTEX-1KD5A-0325.svs",
"GTEX-1KWVE-1325.svs",
"GTEX-1KXAM-2325.svs",
"GTEX-1L5NE-2625.svs",
"GTEX-1LB8K-0325.svs",
"GTEX-1LBAC-0225.svs",
"GTEX-1LC46-2125.svs",
"GTEX-1LC47-0225.svs",
"GTEX-1LG7Y-0425.svs",
"GTEX-1LG7Z-2025.svs",
"GTEX-1LGRB-2425.svs",
"GTEX-1LH75-2525.svs",
"GTEX-1LKK1-0125.svs",
"GTEX-1LNCM-1325.svs",
"GTEX-1LSNL-0525.svs",
"GTEX-1LSNM-2025.svs",
"GTEX-1LSVX-0125.svs",
"GTEX-1LVA9-0225.svs",
"GTEX-1LVAM-0225.svs",
"GTEX-1LVAN-0125.svs",
"GTEX-1LVAO-0225.svs",
"GTEX-1M4P7-0325.svs",
"GTEX-1M5QR-2125.svs",
"GTEX-1MA7W-1925.svs",
"GTEX-1MA7X-2125.svs",
"GTEX-1MCC2-2325.svs",
"GTEX-1MCQQ-0325.svs",
"GTEX-1MCYP-1825.svs",
"GTEX-1MGNQ-0225.svs",
"GTEX-1MJIX-2325.svs",
"GTEX-1MJK2-0525.svs",
"GTEX-1MJK3-0225.svs",
"GTEX-1MUQO-0525.svs",
"GTEX-1N2DV-0125.svs",
"GTEX-1N2DW-0325.svs",
"GTEX-1N2EE-0525.svs",
"GTEX-1N2EF-0625.svs",
"GTEX-1N5O9-0225.svs",
"GTEX-1N7R6-0325.svs",
"GTEX-1NHNU-1125.svs",
"GTEX-1NSGN-0225.svs",
"GTEX-1NUQO-0125.svs",
"GTEX-1NV5F-0125.svs",
"GTEX-1NV8Z-0225.svs",
"GTEX-1O97I-0225.svs",
"GTEX-1O9I2-0625.svs",
"GTEX-1OFPY-2325.svs",
"GTEX-1OJC3-0225.svs",
"GTEX-1OJC4-1725.svs",
"GTEX-1OKEX-2225.svs",
"GTEX-1OZHM-0325.svs",
"GTEX-1P4AB-0225.svs",
"GTEX-1PBJI-0425.svs",
"GTEX-1PBJJ-0225.svs",
"GTEX-1PDJ9-0425.svs",
"GTEX-1PFEY-1625.svs",
"GTEX-1PIEJ-0225.svs",
"GTEX-1PIGE-0525.svs",
"GTEX-1PIIG-0325.svs",
"GTEX-1POEN-0225.svs",
"GTEX-1PPGY-0325.svs",
"GTEX-1PPH6-0325.svs",
"GTEX-1PPH7-2425.svs",
"GTEX-1PPH8-0625.svs",
"GTEX-1PWST-0325.svs",
"GTEX-1QAET-1125.svs",
"GTEX-1QCLY-0525.svs",
"GTEX-1QCLZ-0225.svs",
"GTEX-1QEPI-0225.svs",
"GTEX-1QL29-0225.svs",
"GTEX-1QMI2-0125.svs",
"GTEX-1QP28-0225.svs",
"GTEX-1QP29-0725.svs",
"GTEX-1QP2A-0225.svs",
"GTEX-1QP66-1325.svs",
"GTEX-1QP67-1725.svs",
"GTEX-1QP6S-0125.svs",
"GTEX-1QP9N-1525.svs",
"GTEX-1QPFJ-0125.svs",
"GTEX-1QW4Y-1625.svs",
"GTEX-1R46S-0525.svs",
"GTEX-1R7EU-0325.svs",
"GTEX-1R7EV-1925.svs",
"GTEX-1R9JW-2525.svs",
"GTEX-1R9K4-0225.svs",
"GTEX-1R9K5-0225.svs",
"GTEX-1R9PM-2125.svs",
"GTEX-1R9PN-1725.svs",
"GTEX-1R9PO-2225.svs",
"GTEX-1RAZA-0225.svs",
"GTEX-1RAZQ-0625.svs",
"GTEX-1RAZR-0225.svs",
"GTEX-1RAZS-0225.svs",
"GTEX-1RB15-2025.svs",
"GTEX-1RDX4-0225.svs",
"GTEX-1RMOY-2225.svs",
"GTEX-1RNSC-0225.svs",
"GTEX-1RNTQ-0225.svs",
"GTEX-1RQED-0325.svs",
"GTEX-1S3DN-2325.svs",
"GTEX-1S5VW-0225.svs",
"GTEX-1S5ZA-0225.svs",
"GTEX-1S5ZU-0225.svs",
"GTEX-1S82P-0225.svs",
"GTEX-1S82U-1625.svs",
"GTEX-1S82Z-1025.svs",
"GTEX-1S831-0325.svs",
"GTEX-1S83E-0225.svs",
"GTEX-N7MS-0325.svs",
"GTEX-NFK9-0325.svs",
"GTEX-NL3G-0225.svs",
"GTEX-NL3H-0225.svs",
"GTEX-NL4W-0225.svs",
"GTEX-NPJ7-0525.svs",
"GTEX-NPJ8-0225.svs",
"GTEX-O5YT-0225.svs",
"GTEX-O5YU-0225.svs",
"GTEX-O5YV-0225.svs",
"GTEX-OHPJ-0325.svs",
"GTEX-OHPK-0225.svs",
"GTEX-OHPL-0225.svs",
"GTEX-OHPM-0225.svs",
"GTEX-OHPN-0225.svs",
"GTEX-OIZF-0225.svs",
"GTEX-OIZG-0825.svs",
"GTEX-OIZH-0225.svs",
"GTEX-OIZI-0225.svs",
"GTEX-OOBJ-0225.svs",
"GTEX-OOBK-0225.svs",
"GTEX-OXRK-0325.svs",
"GTEX-OXRL-0225.svs",
"GTEX-OXRN-0225.svs",
"GTEX-OXRO-0225.svs",
"GTEX-OXRP-0225.svs",
"GTEX-P44G-2325.svs",
"GTEX-P44H-0325.svs",
"GTEX-P4PP-0225.svs",
"GTEX-P4PQ-0225.svs",
"GTEX-P4QR-0425.svs",
"GTEX-P4QS-0225.svs",
"GTEX-P4QT-0225.svs",
"GTEX-P78B-0225.svs",
"GTEX-PLZ4-0225.svs",
"GTEX-PLZ5-1825.svs",
"GTEX-PLZ6-1325.svs",
"GTEX-POMQ-2325.svs",
"GTEX-POYW-0725.svs",
"GTEX-PSDG-0325.svs",
"GTEX-PVOW-0225.svs",
"GTEX-PW2O-1625.svs",
"GTEX-PWCY-1925.svs",
"GTEX-PWN1-0225.svs",
"GTEX-PWO3-1625.svs",
"GTEX-PWOO-2225.svs",
"GTEX-PX3G-0225.svs",
"GTEX-Q2AG-0225.svs",
"GTEX-Q2AH-1725.svs",
"GTEX-Q2AI-1425.svs",
"GTEX-Q734-1825.svs",
"GTEX-QCQG-1825.svs",
"GTEX-QDT8-0225.svs",
"GTEX-QDVJ-1825.svs",
"GTEX-QDVN-2125.svs",
"GTEX-QEG4-0325.svs",
"GTEX-QEG5-0325.svs",
"GTEX-QEL4-0325.svs",
"GTEX-QESD-1525.svs",
"GTEX-QLQ7-1525.svs",
"GTEX-QLQW-1225.svs",
"GTEX-QMR6-0125.svs",
"GTEX-QMRM-1725.svs",
"GTEX-QV31-1325.svs",
"GTEX-QV44-1826.svs",
"GTEX-QVJO-0225.svs",
"GTEX-QVUS-0125.svs",
"GTEX-QXCU-2425.svs",
"GTEX-R3RS-0225.svs",
"GTEX-R45C-0225.svs",
"GTEX-R53T-1625.svs",
"GTEX-R55C-1625.svs",
"GTEX-R55D-0325.svs",
"GTEX-R55E-0225.svs",
"GTEX-R55F-1425.svs",
"GTEX-R55G-2425.svs",
"GTEX-REY6-0325.svs",
"GTEX-RM2N-1725.svs",
"GTEX-RN5K-0125.svs",
"GTEX-RN64-0225.svs",
"GTEX-RNOR-0225.svs",
"GTEX-RTLS-0225.svs",
"GTEX-RU1J-1625.svs",
"GTEX-RU72-1025.svs",
"GTEX-RUSQ-1625.svs",
"GTEX-RVPU-2325.svs",
"GTEX-RVPV-1525.svs",
"GTEX-RWS6-2225.svs",
"GTEX-RWSA-0225.svs",
"GTEX-S32W-2225.svs",
"GTEX-S33H-1125.svs",
"GTEX-S341-1625.svs",
"GTEX-S3LF-0225.svs",
"GTEX-S3XE-1625.svs",
"GTEX-S4P3-1525.svs",
"GTEX-S4Q7-1625.svs",
"GTEX-S4UY-0225.svs",
"GTEX-S4Z8-1725.svs",
"GTEX-S7PM-0225.svs",
"GTEX-S7SE-0225.svs",
"GTEX-S7SF-1825.svs",
"GTEX-S95S-1325.svs",
"GTEX-SE5C-1725.svs",
"GTEX-SIU7-1925.svs",
"GTEX-SIU8-0225.svs",
"GTEX-SJXC-0225.svs",
"GTEX-SN8G-0225.svs",
"GTEX-SNMC-1325.svs",
"GTEX-SNOS-1425.svs",
"GTEX-SSA3-0225.svs",
"GTEX-SUCS-1825.svs",
"GTEX-T2IS-0225.svs",
"GTEX-T2YK-0625.svs",
"GTEX-T5JC-0525.svs",
"GTEX-T5JW-1725.svs",
"GTEX-T6MN-0225.svs",
"GTEX-T6MO-1725.svs",
"GTEX-TKQ1-1125.svs",
"GTEX-TKQ2-0725.svs",
"GTEX-TMKS-0225.svs",
"GTEX-TML8-2025.svs",
"GTEX-TMMY-0325.svs",
"GTEX-TMZS-0225.svs",
"GTEX-TSE9-0225.svs",
"GTEX-U3ZG-0225.svs",
"GTEX-U3ZH-1825.svs",
"GTEX-U3ZM-1025.svs",
"GTEX-U3ZN-2625.svs",
"GTEX-U412-0525.svs",
"GTEX-U4B1-1725.svs",
"GTEX-U8T8-0225.svs",
"GTEX-U8XE-0425.svs",
"GTEX-UJHI-1625.svs",
"GTEX-UJMC-1725.svs",
"GTEX-UPIC-1325.svs",
"GTEX-UPJH-0425.svs",
"GTEX-UPK5-0825.svs",
"GTEX-UTHO-0425.svs",
"GTEX-V1D1-2325.svs",
"GTEX-V955-2325.svs",
"GTEX-VJWN-0225.svs",
"GTEX-VJYA-1325.svs",
"GTEX-VUSG-2425.svs",
"GTEX-VUSH-0225.svs",
"GTEX-W5WG-2525.svs",
"GTEX-W5X1-2625.svs",
"GTEX-WCDI-0225.svs",
"GTEX-WEY5-1925.svs",
"GTEX-WFG7-2525.svs",
"GTEX-WFG8-2325.svs",
"GTEX-WFJO-1925.svs",
"GTEX-WFON-2225.svs",
"GTEX-WH7G-2225.svs",
"GTEX-WHPG-2125.svs",
"GTEX-WHSB-1725.svs",
"GTEX-WHSE-0325.svs",
"GTEX-WHWD-2125.svs",
"GTEX-WI4N-1125.svs",
"GTEX-WK11-0225.svs",
"GTEX-WL46-0325.svs",
"GTEX-WOFL-0225.svs",
"GTEX-WOFM-1525.svs",
"GTEX-WQUQ-0525.svs",
"GTEX-WRHK-1525.svs",
"GTEX-WRHU-2625.svs",
"GTEX-WVJS-0225.svs",
"GTEX-WVLH-0225.svs",
"GTEX-WWTW-0225.svs",
"GTEX-WWYW-0225.svs",
"GTEX-WXYG-2425.svs",
"GTEX-WY7C-2425.svs",
"GTEX-WYBS-0225.svs",
"GTEX-WYJK-0225.svs",
"GTEX-WYVS-2225.svs",
"GTEX-WZTO-0225.svs",
"GTEX-X15G-2325.svs",
"GTEX-X261-0225.svs",
"GTEX-X3Y1-2225.svs",
"GTEX-X4EO-0425.svs",
"GTEX-X4EP-0225.svs",
"GTEX-X4LF-1725.svs",
"GTEX-X4XX-0225.svs",
"GTEX-X4XY-0325.svs",
"GTEX-X585-0325.svs",
"GTEX-X5EB-2425.svs",
"GTEX-X62O-0225.svs",
"GTEX-X638-0225.svs",
"GTEX-X88G-0225.svs",
"GTEX-X8HC-0225.svs",
"GTEX-XAJ8-0925.svs",
"GTEX-XBEC-0325.svs",
"GTEX-XBED-2325.svs",
"GTEX-XBEW-0725.svs",
"GTEX-XGQ4-2225.svs",
"GTEX-XK95-0425.svs",
"GTEX-XLM4-0225.svs",
"GTEX-XMD1-0325.svs",
"GTEX-XMD2-0225.svs",
"GTEX-XMD3-0225.svs",
"GTEX-XMK1-2225.svs",
"GTEX-XOT4-0225.svs",
"GTEX-XOTO-0225.svs",
"GTEX-XPT6-1925.svs",
"GTEX-XPVG-2725.svs",
"GTEX-XQ3S-1625.svs",
"GTEX-XQ8I-0525.svs",
"GTEX-XUJ4-2525.svs",
"GTEX-XUW1-0525.svs",
"GTEX-XUYS-0225.svs",
"GTEX-XUZC-1825.svs",
"GTEX-XV7Q-2625.svs",
"GTEX-XXEK-2425.svs",
"GTEX-XYKS-2725.svs",
"GTEX-Y111-0225.svs",
"GTEX-Y114-2325.svs",
"GTEX-Y3I4-2225.svs",
"GTEX-Y3IK-2525.svs",
"GTEX-Y5LM-2025.svs",
"GTEX-Y5V5-2425.svs",
"GTEX-Y5V6-2525.svs",
"GTEX-Y8DK-0225.svs",
"GTEX-Y8E4-0725.svs",
"GTEX-Y8E5-0225.svs",
"GTEX-Y8LW-1925.svs",
"GTEX-Y9LG-2025.svs",
"GTEX-YB5E-2125.svs",
"GTEX-YB5K-2225.svs",
"GTEX-YBZK-0525.svs",
"GTEX-YEC3-1225.svs",
"GTEX-YEC4-2425.svs",
"GTEX-YECK-0225.svs",
"GTEX-YF7O-2425.svs",
"GTEX-YFC4-0225.svs",
"GTEX-YFCO-2025.svs",
"GTEX-YJ89-0225.svs",
"GTEX-YJ8A-0525.svs",
"GTEX-YJ8O-2425.svs",
"GTEX-Z93S-0225.svs",
"GTEX-Z93T-0225.svs",
"GTEX-Z9EW-1825.svs",
"GTEX-ZA64-1825.svs",
"GTEX-ZAB4-0325.svs",
"GTEX-ZAB5-1725.svs",
"GTEX-ZAJG-0225.svs",
"GTEX-ZAK1-0226.svs",
"GTEX-ZAKK-0225.svs",
"GTEX-ZC5H-0125.svs",
"GTEX-ZDTS-0225.svs",
"GTEX-ZDTT-2625.svs",
"GTEX-ZDXO-2425.svs",
"GTEX-ZDYS-2125.svs",
"GTEX-ZE7O-0225.svs",
"GTEX-ZE9C-2825.svs",
"GTEX-ZEX8-2325.svs",
"GTEX-ZF28-0225.svs",
"GTEX-ZF29-2225.svs",
"GTEX-ZF2S-2225.svs",
"GTEX-ZF3C-0225.svs",
"GTEX-ZG7Y-0425.svs",
"GTEX-ZGAY-0625.svs",
"GTEX-ZLFU-2625.svs",
"GTEX-ZLV1-1825.svs",
"GTEX-ZLWG-2225.svs",
"GTEX-ZP4G-2325.svs",
"GTEX-ZPCL-2125.svs",
"GTEX-ZPIC-1825.svs",
"GTEX-ZPU1-2425.svs",
"GTEX-ZQG8-1525.svs",
"GTEX-ZQUD-1125.svs",
"GTEX-ZT9W-2325.svs",
"GTEX-ZT9X-1925.svs",
"GTEX-ZTPG-0225.svs",
"GTEX-ZTSS-2025.svs",
"GTEX-ZTTD-0225.svs",
"GTEX-ZTX8-1525.svs",
"GTEX-ZU9S-0325.svs",
"GTEX-ZUA1-0225.svs",
"GTEX-ZV68-0225.svs",
"GTEX-ZV6S-2225.svs",
"GTEX-ZV7C-2125.svs",
"GTEX-ZVE1-0225.svs",
"GTEX-ZVE2-0325.svs",
"GTEX-ZVP2-1825.svs",
"GTEX-ZVT2-2225.svs",
"GTEX-ZVT3-0225.svs",
"GTEX-ZVT4-0225.svs",
"GTEX-ZVTK-0525.svs",
"GTEX-ZVZO-0225.svs",
"GTEX-ZVZP-2325.svs",
"GTEX-ZVZQ-0425.svs",
"GTEX-ZWKS-0225.svs",
"GTEX-ZXES-2025.svs",
"GTEX-ZXG5-0225.svs",
"GTEX-ZY6K-2225.svs",
"GTEX-ZYFC-0325.svs",
"GTEX-ZYFD-0225.svs",
"GTEX-ZYFG-2225.svs",
"GTEX-ZYT6-0325.svs",
"GTEX-ZYVF-0225.svs",
"GTEX-ZYW4-0225.svs",
"GTEX-ZYWO-2525.svs",
"GTEX-ZYY3-0225.svs",
"GTEX-ZZ64-1625.svs",
"GTEX-ZZPT-0325.svs",
"GTEX-ZZPU-2725.svs"
]

# load colour modes of the KLF14 training dataset
with np.load(klf14_training_colour_histogram_file) as data:
    mode_r_klf14 = data['mode_r']
    mode_g_klf14 = data['mode_g']
    mode_b_klf14 = data['mode_b']

########################################################################################################################
## Colourmap for AIDA, based on KLF14 automatically segmented data
########################################################################################################################

if os.path.isfile(filename_area2quantile):
    with np.load(filename_area2quantile, allow_pickle=True) as aux:
        f_area2quantile_f = aux['f_area2quantile_f']
        f_area2quantile_m = aux['f_area2quantile_m']
else:
    raise FileNotFoundError('Cannot find file with area->quantile map precomputed from all automatically segmented' +
                            ' slides in klf14_b6ntac_exp_0098_full_slide_size_analysis_v7.py')

# # load AIDA's colourmap
# cm = cytometer.data.aida_colourmap()

########################################################################################################################
## Segmentation loop
########################################################################################################################

for i_file, histo_file in enumerate(histo_files_list):

    print('File ' + str(i_file) + '/' + str(len(histo_files_list) - 1) + ': ' + histo_file)

    # make full path to ndpi file
    histo_file = os.path.join(histology_dir, histo_file)

    # check whether there's a lock on this file
    lock_file = os.path.basename(histo_file).replace('.svs', '.lock')
    lock_file = os.path.join(annotations_dir, lock_file)
    if os.path.isfile(lock_file):
        print('Lock on file, skipping')
        continue
    else:
        # create an empty lock file to prevent other other instances of the script to process the same .ndpi file
        Path(lock_file).touch()

    # choose a random fold for this image
    np.random.seed(i_file)
    i_fold = np.random.randint(0, 10)

    contour_model_file = os.path.join(saved_models_dir, contour_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    dmap_model_file = os.path.join(saved_models_dir, dmap_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    classifier_model_file = os.path.join(saved_models_dir,
                                         classifier_model_basename + '_model_fold_' + str(i_fold) + '.h5')
    correction_model_file = os.path.join(saved_models_dir,
                                         correction_model_basename + '_model_fold_' + str(i_fold) + '.h5')

    # name of file to save annotations to
    annotations_file = os.path.basename(histo_file)
    annotations_file = os.path.splitext(annotations_file)[0]
    annotations_file = os.path.join(annotations_dir, annotations_file + '_exp_0001_auto.json')

    annotations_corrected_file = os.path.basename(histo_file)
    annotations_corrected_file = os.path.splitext(annotations_corrected_file)[0]
    annotations_corrected_file = os.path.join(annotations_dir, annotations_corrected_file + '_exp_0001_corrected.json')

    # name of file to save rough mask, current mask, and time steps
    rough_mask_file = os.path.basename(histo_file)
    rough_mask_file = rough_mask_file.replace('.svs', '_rough_mask.npz')
    rough_mask_file = os.path.join(annotations_dir, rough_mask_file)

    # open full resolution histology slide
    im = openslide.OpenSlide(histo_file)

    # pixel size
    xres = float(im.properties['aperio.MPP']) * 1e-6 # m/pixel
    yres = float(im.properties['aperio.MPP']) * 1e-6 # m/pixel

    # check whether we continue previous execution, or we start a new one
    continue_previous = os.path.isfile(rough_mask_file)

    # true downsampled factor as reported by histology file
    level_actual = np.abs(np.array(im.level_downsamples) - downsample_factor_goal).argmin()
    downsample_factor_actual = im.level_downsamples[level_actual]
    if np.abs(downsample_factor_actual - downsample_factor_goal) > 1:
        warnings.warn('The histology file has no downsample level close enough to the target downsample level')
        continue

    # if the rough mask has been pre-computed, just load it
    if continue_previous:

        with np.load(rough_mask_file) as aux:
            lores_istissue = aux['lores_istissue']
            lores_istissue0 = aux['lores_istissue0']
            im_downsampled = aux['im_downsampled']
            step = aux['step'].item()
            perc_completed_all = list(aux['perc_completed_all'])
            time_step_all = list(aux['time_step_all'])
            prev_first_row = aux['prev_first_row'].item()
            prev_last_row = aux['prev_last_row'].item()
            prev_first_col = aux['prev_first_col'].item()
            prev_last_col = aux['prev_last_col'].item()

    else:

        time_prev = time.time()

        # compute the rough foreground mask of tissue vs. background
        lores_istissue0, im_downsampled = \
            rough_foreground_mask(histo_file, downsample_factor=downsample_factor_actual,
                                  dilation_size=dilation_size,
                                  component_size_threshold=component_size_threshold,
                                  hole_size_treshold=hole_size_treshold, std_k=std_k,
                                  return_im=True, enhance_contrast=enhance_contrast,
                                  ignore_white_threshold=ignore_white_threshold)

        if DEBUG:
            enhancer = PIL.ImageEnhance.Contrast(PIL.Image.fromarray(im_downsampled))
            im_downsampled_enhanced = np.array(enhancer.enhance(enhance_contrast))
            plt.clf()
            plt.subplot(211)
            plt.imshow(im_downsampled_enhanced)
            plt.axis('off')
            plt.subplot(212)
            plt.imshow(im_downsampled_enhanced)
            plt.contour(lores_istissue0)
            plt.axis('off')

        # segmentation copy, to keep track of what's left to do
        lores_istissue = lores_istissue0.copy()

        # initialize block algorithm variables
        step = 0
        perc_completed_all = [float(0.0),]
        time_step = time.time() - time_prev
        time_step_all = [time_step,]
        (prev_first_row, prev_last_row, prev_first_col, prev_last_col) = (0, 0, 0, 0)

        # save to the rough mask file
        np.savez_compressed(rough_mask_file, lores_istissue=lores_istissue, lores_istissue0=lores_istissue0,
                            im_downsampled=im_downsampled, step=step, perc_completed_all=perc_completed_all,
                            prev_first_row=prev_first_row, prev_last_row=prev_last_row,
                            prev_first_col=prev_first_col, prev_last_col=prev_last_col,
                            time_step_all=time_step_all)

        # end "computing the rough foreground mask"

    # checkpoint: here the rough tissue mask has either been loaded or computed
    time_step = time_step_all[-1]
    time_total = np.sum(time_step_all)
    print('File ' + str(i_file) + '/' + str(len(histo_files_list) - 1) + ': step ' +
          str(step) + ': ' +
          str(np.count_nonzero(lores_istissue)) + '/' + str(np.count_nonzero(lores_istissue0)) + ': ' +
          "{0:.1f}".format(100.0 - np.count_nonzero(lores_istissue) / np.count_nonzero(lores_istissue0) * 100) +
          '% completed: ' +
          'time step ' + "{0:.2f}".format(time_step) + ' s' +
          ', total time ' + "{0:.2f}".format(time_total) + ' s')

    if DEBUG:
            plt.clf()
            plt.subplot(211)
            plt.imshow(im_downsampled)
            plt.contour(lores_istissue0, colors='k')
            plt.subplot(212)
            plt.imshow(lores_istissue0)

            plt.cla()
            plt.imshow(lores_istissue)

    # estimate the colour mode of the downsampled image, so that we can correct the image tint to match the KLF14
    # training dataset. We apply the same correction to each tile, to avoid that a tile with e.g. only muscle gets
    # overcorrected
    mode_r_rrbe1 = scipy.stats.mode(im_downsampled[:, :, 0], axis=None).mode[0]
    mode_g_rrbe1 = scipy.stats.mode(im_downsampled[:, :, 1], axis=None).mode[0]
    mode_b_rrbe1 = scipy.stats.mode(im_downsampled[:, :, 2], axis=None).mode[0]

    # keep extracting histology windows until we have finished
    while np.count_nonzero(lores_istissue) > 0:

        time_prev = time.time()

        # next step (it starts from 1 here, because step 0 is the rough mask computation)
        step += 1

        # get indices for the next histology window to process
        (first_row, last_row, first_col, last_col), \
        (lores_first_row, lores_last_row, lores_first_col, lores_last_col) = \
            cytometer.utils.get_next_roi_to_process(lores_istissue, downsample_factor=downsample_factor_actual,
                                                    max_window_size=fullres_box_size,
                                                    border=np.round((receptive_field-1)/2))

        # overlap between current and previous window, as a fraction of current window area
        current_window = Polygon([(first_col, first_row), (last_col, first_row),
                                  (last_col, last_row), (first_col, last_row)])
        prev_window = Polygon([(prev_first_col, prev_first_row), (prev_last_col, prev_first_row),
                               (prev_last_col, prev_last_row), (prev_first_col, prev_last_row)])
        window_overlap_fraction = current_window.intersection(prev_window).area / current_window.area

        # check that we are not trying to process almost the same window
        if window_overlap_fraction > window_overlap_fraction_max:
            # if we are trying to process almost the same window as in the previous step, what's probably happening is
            # that we have some big labels on the edges that are not white adipocytes, and the segmentation algorithm is
            # also finding one or more spurious labels within the window. That prevents the whole lores_istissue window
            # from being wiped out, and the big edge labels keep the window selection being almost the same. Thus, we
            # wipe it out and move to another tissue area
            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0
            continue

        else:
            # remember processed window for next step
            (prev_first_row, prev_last_row, prev_first_col, prev_last_col) = (first_row, last_row, first_col, last_col)

        # load window from full resolution slide
        tile = im.read_region(location=(first_col, first_row), level=0,
                              size=(last_col - first_col, last_row - first_row))
        tile = np.array(tile)
        tile = tile[:, :, 0:3]

        # correct tint of the tile to match KLF14 training data
        tile[:, :, 0] = tile[:, :, 0] + (mode_r_klf14 - mode_r_rrbe1)
        tile[:, :, 1] = tile[:, :, 1] + (mode_g_klf14 - mode_g_rrbe1)
        tile[:, :, 2] = tile[:, :, 2] + (mode_b_klf14 - mode_b_rrbe1)

        # interpolate coarse tissue segmentation to full resolution
        istissue_tile = lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col]
        istissue_tile = cytometer.utils.resize(istissue_tile, size=(last_col - first_col, last_row - first_row),
                                               resample=PIL.Image.NEAREST)

        if DEBUG:
            plt.clf()
            plt.imshow(tile)
            plt.imshow(istissue_tile, alpha=0.5)
            plt.contour(istissue_tile, colors='k')
            plt.title('Yellow: Tissue mask. Purple: Background')
            plt.axis('off')

        # segment histology, split into individual objects, and apply segmentation correction
        labels, labels_class, todo_edge, \
        window_im, window_labels, window_labels_corrected, window_labels_class, index_list, scaling_factor_list \
            = cytometer.utils.segmentation_pipeline6(im=tile,
                                                     dmap_model=dmap_model_file,
                                                     contour_model=contour_model_file,
                                                     correction_model=correction_model_file,
                                                     classifier_model=classifier_model_file,
                                                     min_cell_area=0,
                                                     max_cell_area=np.inf,
                                                     mask=istissue_tile,
                                                     min_mask_overlap=min_mask_overlap,
                                                     phagocytosis=phagocytosis,
                                                     min_class_prop=0.0,
                                                     correction_window_len=correction_window_len,
                                                     correction_smoothing=correction_smoothing,
                                                     return_bbox=True, return_bbox_coordinates='xy',
                                                     batch_size=batch_size)


        # compute the "white adipocyte" probability for each object
        if len(window_labels) > 0:
            window_white_adipocyte_prob = np.sum(window_labels * window_labels_class, axis=(1, 2)) \
                                          / np.sum(window_labels, axis=(1, 2))
            window_white_adipocyte_prob_corrected = np.sum(window_labels_corrected * window_labels_class, axis=(1, 2)) \
                                                    / np.sum(window_labels_corrected, axis=(1, 2))
        else:
            window_white_adipocyte_prob = np.array([])
            window_white_adipocyte_prob_corrected = np.array([])

        # if no cells found, wipe out current window from tissue segmentation, and go to next iteration. Otherwise we'd
        # enter an infinite loop
        if len(index_list) == 0:  # empty segmentation

            lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0

        else:  # there's at least one object in the segmentation

            if DEBUG:
                j = 0
                plt.clf()
                plt.subplot(221)
                plt.imshow(tile[:, :, :])
                plt.title('Histology', fontsize=16)
                plt.axis('off')
                plt.subplot(222)
                plt.imshow(tile[:, :, :])
                plt.contour(labels, levels=np.unique(labels), colors='C0')
                plt.contourf(todo_edge, colors='C2', levels=[0.5, 1])
                plt.title('Full segmentation', fontsize=16)
                plt.axis('off')
                plt.subplot(212)
                plt.imshow(window_im[j, :, :, :])
                plt.contour(window_labels[j, :, :], colors='C0')
                plt.contour(window_labels_corrected[j, :, :], colors='C1')
                plt.title('Crop around object and corrected segmentation', fontsize=16)
                plt.axis('off')
                plt.tight_layout()

            # downsample "to do" mask so that the rough tissue segmentation can be updated
            lores_todo_edge = PIL.Image.fromarray(todo_edge.astype(np.uint8))
            lores_todo_edge = lores_todo_edge.resize((lores_last_col - lores_first_col,
                                                      lores_last_row - lores_first_row),
                                                     resample=PIL.Image.NEAREST)
            lores_todo_edge = np.array(lores_todo_edge)

            if DEBUG:
                plt.clf()
                plt.subplot(221)
                plt.imshow(lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col])
                plt.title('Low res tissue mask', fontsize=16)
                plt.axis('off')
                plt.subplot(222)
                plt.imshow(istissue_tile)
                plt.title('Full res tissue mask', fontsize=16)
                plt.axis('off')
                plt.subplot(223)
                plt.imshow(todo_edge.astype(np.uint8))
                plt.title('Full res left over tissue', fontsize=16)
                plt.axis('off')
                plt.subplot(224)
                plt.imshow(lores_todo_edge.astype(np.uint8))
                plt.title('Low res left over tissue', fontsize=16)
                plt.axis('off')
                plt.tight_layout()

            # convert labels in cropped images to contours (points), and add cropping window offset so that the
            # contours are in the whole slide coordinates
            offset_xy = index_list[:, [2, 3]]  # index_list: [i, lab, x0, y0, xend, yend]
            contours = cytometer.utils.labels2contours(window_labels, offset_xy=offset_xy,
                                                       scaling_factor_xy=scaling_factor_list)
            contours_corrected = cytometer.utils.labels2contours(window_labels_corrected, offset_xy=offset_xy,
                                                                 scaling_factor_xy=scaling_factor_list)

            if DEBUG:
                # no overlap
                plt.clf()
                plt.imshow(tile)
                for j in range(len(contours)):
                    plt.fill(contours[j][:, 0], contours[j][:, 1], edgecolor='C0', fill=False)
                    plt.text(contours[j][0, 0], contours[j][0, 1], str(j))

                # overlap
                plt.clf()
                plt.imshow(tile)
                for j in range(len(contours_corrected)):
                    plt.fill(contours_corrected[j][:, 0], contours_corrected[j][:, 1], edgecolor='C0', fill=False)
                    # plt.text(contours_corrected[j][0, 0], contours_corrected[j][0, 1], str(j))

            # downsample contours for AIDA annotations file
            lores_contours = []
            for c in contours:
                lores_c = bspline_resample(c, factor=contour_downsample_factor, min_n=10, k=bspline_k, is_closed=True)
                lores_contours.append(lores_c)

            lores_contours_corrected = []
            for c in contours_corrected:
                lores_c = bspline_resample(c, factor=contour_downsample_factor, min_n=10, k=bspline_k, is_closed=True)
                lores_contours_corrected.append(lores_c)

            if DEBUG:
                # no overlap
                plt.clf()
                plt.imshow(tile)
                for j in range(len(contours)):
                    plt.fill(lores_contours[j][:, 0], lores_contours[j][:, 1], edgecolor='C1', fill=False)

                # overlap
                plt.clf()
                plt.imshow(tile)
                for j in range(len(contours_corrected)):
                    plt.fill(lores_contours_corrected[j][:, 0], lores_contours_corrected[j][:, 1], edgecolor='C1', fill=False)

            # add tile offset, so that contours are in full slide coordinates
            for j in range(len(contours)):
                lores_contours[j][:, 0] += first_col
                lores_contours[j][:, 1] += first_row

            for j in range(len(contours_corrected)):
                lores_contours_corrected[j][:, 0] += first_col
                lores_contours_corrected[j][:, 1] += first_row

            # convert non-overlap contours to AIDA items
            # TODO: check whether the mouse is male or female, and use corresponding f_area2quantile
            contour_items = cytometer.data.aida_contour_items(lores_contours, f_area2quantile_m.item(),
                                                              cell_prob=window_white_adipocyte_prob,
                                                              xres=xres*1e6, yres=yres*1e6)
            rectangle = (first_col, first_row, last_col - first_col, last_row - first_row)  # (x0, y0, width, height)
            rectangle_item = cytometer.data.aida_rectangle_items([rectangle,])

            if step == 1:
                # in the first step, overwrite previous annotations file, or create new one
                cytometer.data.aida_write_new_items(annotations_file, rectangle_item, mode='w')
                cytometer.data.aida_write_new_items(annotations_file, contour_items, mode='append_new_layer')
            else:
                # in next steps, add contours to previous layer
                cytometer.data.aida_write_new_items(annotations_file, rectangle_item, mode='append_to_last_layer')
                cytometer.data.aida_write_new_items(annotations_file, contour_items, mode='append_new_layer')

            # convert corrected contours to AIDA items
            contour_items_corrected = cytometer.data.aida_contour_items(lores_contours_corrected, f_area2quantile_m.item(),
                                                                        cell_prob=window_white_adipocyte_prob_corrected,
                                                                        xres=xres*1e6, yres=yres*1e6)

            if step == 1:
                # in the first step, overwrite previous annotations file, or create new one
                cytometer.data.aida_write_new_items(annotations_corrected_file, rectangle_item, mode='w')
                cytometer.data.aida_write_new_items(annotations_corrected_file, contour_items_corrected, mode='append_new_layer')
            else:
                # in next steps, add contours to previous layer
                cytometer.data.aida_write_new_items(annotations_corrected_file, rectangle_item, mode='append_to_last_layer')
                cytometer.data.aida_write_new_items(annotations_corrected_file, contour_items_corrected, mode='append_new_layer')

            # update the tissue segmentation mask with the current window
            if np.all(lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] == lores_todo_edge):
                # if the mask remains identical, wipe out the whole window, as otherwise we'd have an
                # infinite loop
                lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = 0
            else:
                # if the mask has been updated, use it to update the total tissue segmentation
                lores_istissue[lores_first_row:lores_last_row, lores_first_col:lores_last_col] = lores_todo_edge

        # end of "if len(index_list) == 0:"
        # Thus, regardless of whether there were any objects in the segmentation or not, here we continue the execution
        # of the program

        perc_completed = 100.0 - np.count_nonzero(lores_istissue) / np.count_nonzero(lores_istissue0) * 100
        perc_completed_all.append(perc_completed)
        time_step = time.time() - time_prev
        time_step_all.append(time_step)
        time_total = np.sum(time_step_all)

        print('File ' + str(i_file) + '/' + str(len(histo_files_list) - 1) + ': step ' +
              str(step) + ': ' +
              str(np.count_nonzero(lores_istissue)) + '/' + str(np.count_nonzero(lores_istissue0)) + ': ' +
              "{0:.1f}".format(perc_completed) +
              '% completed: ' +
              'time step ' + "{0:.2f}".format(time_step) + ' s' +
              ', total time ' + "{0:.2f}".format(time_total) + ' s')

        # save to the rough mask file
        np.savez_compressed(rough_mask_file, lores_istissue=lores_istissue, lores_istissue0=lores_istissue0,
                            im_downsampled=im_downsampled, step=step, perc_completed_all=   perc_completed_all,
                            time_step_all=time_step_all,
                            prev_first_row=prev_first_row, prev_last_row=prev_last_row,
                            prev_first_col=prev_first_col, prev_last_col=prev_last_col)

        # clear keras session to prevent each segmentation iteration from getting slower. Note that this forces us to
        # reload the models every time, but that's not too slow
        K.clear_session()

    # end of "keep extracting histology windows until we have finished"
