# coding: utf-8
#
# This plugin saves all paths in the open image to the same path with
# extension .svg. For example, if input image is
# "~/Downloads/foo.tif", the paths will be saved to file
# "~/Downloads/foo.svg".
#
# The .svg file looks like this:
#
#==============================================================================
# <?xml version="1.0" encoding="UTF-8" standalone="no"?>
# <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 20010904//EN"
#               "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">
#
# <svg xmlns="http://www.w3.org/2000/svg"
#      width="0.454277mm" height="0.454236mm"
#      viewBox="0 0 1001 1001">
#   <path id="Other #5"
#         fill="none" stroke="black" stroke-width="1"
#         d="M 151.00,999.00
#            C 151.00,999.00 129.00,959.00 129.00,959.00
#              129.00,959.00 98.00,929.00 98.00,929.00
#              98.00,929.00 79.00,924.00 79.00,924.00
#              79.00,924.00 43.00,968.00 43.00,968.00
#              43.00,968.00 22.00,978.00 22.00,978.00
#              22.00,978.00 18.00,999.00 18.00,999.00
#              18.00,999.00 151.00,999.00 151.00,999.00 Z" />
# ...
# </svg>
#==============================================================================
#
#
# To enable this plugin, save it to or link to it from ~/.gimp-2.8/plug-ins

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

#!/usr/bin/env python

import os
from gimpfu import *


DEBUG=False

def export_Paths(img):
    outfilename, file_extension = os.path.splitext(img.filename)
    outfilename = outfilename + '.svg'
    if DEBUG:
        print("Saving paths to: " + outfilename)

    pdb.gimp_vectors_export_to_file(img, outfilename, None)

register(
    "export_Paths",    
    "Export all paths",   
    "Export all paths from current image to .svg file with same name",
    "Ramón Casero", 
    "MRC Harwell Institute, Oxfordshire, UK", 
    "2018",
    "Export all paths",
    "*", 
    [
        (PF_IMAGE, "img", "Input image", None),
    ], 
    [],
    export_Paths,
    menu="<Image>/Filters/Paths"
)

main()
