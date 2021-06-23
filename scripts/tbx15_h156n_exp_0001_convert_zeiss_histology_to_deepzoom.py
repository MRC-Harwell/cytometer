"""
Convert list of Louisa Zolkiewski's Zeiss .czi histology files to DeepZoom format that can be read by AIDA.

Environment: cytometer_tensorflow_v2
"""

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

# script name to identify this experiment
experiment_id = 'tbx15_h156n_exp_0001_convert_zeiss_histology_to_deepzoom'

# cross-platform home directory
from pathlib import Path
home = str(Path.home())

import os
import sys
if os.path.join(home, 'Software/cytometer') not in sys.path:
    sys.path.extend([os.path.join(home, 'Software/cytometer')])
import cytometer.data

DEBUG = False

# data directories
histo_dir = os.path.join(home, 'coxgroup_zeiss_test')
dzi_dir = os.path.join(home, 'Data/cytometer_data/aida_data_Tbx15/images')

# list of files to convert
histo_list = [
    "TBX15-H156N-IC-0001-15012021.czi",
    "TBX15-H156N-IC-0002-15012021.czi",
    "TBX15-H156N-IC-0003-15012021.czi",
    "TBX15-H156N-IC-0004-15012021.czi",
    "TBX15-H156N-IC-0005-15012021.czi",
    "TBX15-H156N-IC-0006-15012021.czi",
    "TBX15-H156N-IC-0007-15012021.czi"
]

# add path to histology filenames
histo_list = [os.path.join(histo_dir, x) for x in histo_list]

# convert histology files to DeepZoom
cytometer.data.zeiss_to_deepzoom(histo_list, dzi_dir=dzi_dir, overwrite=True)
