#!/usr/bin/env python3

"""
gimp_contours_to_aida.py.

   Convert contours from Gimp .svg files to AIDA annotations .json format.

usage: gimp_contours_to_aida.py [-h] [-o OUTDIR]
                                filename_row_Y_col_X.svg
                                [filename_row_Y_col_X.svg ...]

Convert contours from Gimp .svg files to AIDA annotations .json format

positional arguments:
  filename_row_Y_col_X.svg
                        .svg files. All inputs with the same "filename" will
                        be grouped into a single .json output. The X,Y values
                        will be added as offsets to the contour coordinates.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTDIR, --outdir OUTDIR
                        directory .json files will be saved to
"""

"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

import os
import sys
import argparse
import numpy as np

sys.path.extend([os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')])  # access to cytometer python modules
from cytometer.data import read_paths_from_svg_file, write_paths_to_aida_json_file


# helper to check that input argument is a valid directory
# by brantfaircloth (https://gist.github.com/brantfaircloth/1443543/5b22c31649e83152925e7c8ad00d58c7f4d49818)
class FullPaths(argparse.Action):
    """Expand user- and relative-paths"""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))


def is_dir(dirname):
    """Checks if a path is an actual directory"""
    if not os.path.isdir(dirname):
        msg = "{0} is not a directory".format(dirname)
        raise argparse.ArgumentTypeError(msg)
    else:
        return dirname


# main method
def main():

    # input argument parser
    parser = argparse.ArgumentParser(description='Convert contours from Gimp .svg files to AIDA annotations .json format')
    parser.add_argument('-o', '--outdir', action=FullPaths, default='.', type=is_dir,
                        help='directory .json files will be saved to')
    parser.add_argument('svg_files', type=argparse.FileType('r'), nargs='+',
                        metavar='filename_row_Y_col_X.svg',
                        help='.svg files. All inputs with the same "filename" will be grouped into a single ' +
                        '.json output. The X,Y values will be added as offsets to the contour coordinates.')

    args = parser.parse_args()

    # remove the '_row_008032_col_009304.svg' end of the file string, and then remove duplicates, so that
    # we get a list of images
    im_files = []
    for file in args.svg_files:
        im_files.append(file.name.split('_row_')[0])
    im_files = list(set(im_files))

    # loop images
    for im_file in im_files:

        # list all svg files that correspond to this image
        svg_files_for_im = np.array(args.svg_files)[[im_file in s.name for s in args.svg_files]]

        # read the contours
        contours = []
        for svg_file in svg_files_for_im:
            print('--> Reading ' + svg_file.name)
            contours += read_paths_from_svg_file(svg_file.name, tag='Cell', add_offset_from_filename=True)

        # write the contours to AIDA annotations file format
        outfile = os.path.join(args.outdir, os.path.basename(im_file) + '.json')
        print('xxx Writing ' + outfile + '\n')
        fp = open(outfile, 'w')
        write_paths_to_aida_json_file(fp, contours)
        fp.close()


if __name__ == "__main__":

    # call main method
    main()
