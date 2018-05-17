# coding: utf-8

#!/usr/bin/env python

from gimpfu import *

def exportPaths(img, layer) :
    pdb.gimp_vectors_export_to_file(img,"/tmp/output.svg",None)

register(
    "ExportPaths",    
    "Export all paths",   
    "Export all paths from current image",
    "Ramón Casero", 
    "MRC Harwell, UK", 
    "2018",
    "<Image>/Filters/Paths/Export paths", 
    "*", 
    [], 
    [],
    exportPaths)

main()
