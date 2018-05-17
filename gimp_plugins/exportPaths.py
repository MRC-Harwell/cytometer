# coding: utf-8

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
