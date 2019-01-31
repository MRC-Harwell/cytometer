#!/usr/bin/env python3

import os
import sys
import getopt

sys.path.extend([os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')])  # access to cytometer python modules
from cytometer.data import read_paths_from_svg_file


import glob
argv = glob.glob('/home/rcasero/Data/cytometer_data/klf14/klf14_b6ntac_training/KLF14-B6NTAC-*.svg')
def main(argv):

    # default output directory
    outdir = '.'

    print("Number of total arguments: " + str(len(argv)))

    try:
        opts, svg_files = getopt.getopt(argv, 'ho:', ['outdir='])
    except getopt.GetoptError:
        print(sys.argv[0] + ' -o <outdir> file1.svg [file2.svg, ...]')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(sys.argv[0] + ' -o <outdir> file1.svg [file2.svg, ...]')
            sys.exit()
        elif opt in ("-o", "--outdir"):
            outdir = arg

    print("Number of SVG files: " + str(len(svg_files)))

    # collapse the list of multiple windows per image to list of original images
    im_files = []
    for file in

    # loop SVG files
    contour = []
    for file in svg_files:

        # read the contours
        contour.append(read_paths_from_svg_file(file, tag='Cell', add_offset_from_filename=True))




if __name__ == "__main__":

    # call main method
    main(sys.argv[1:])
