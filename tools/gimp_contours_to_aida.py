#!/usr/bin/env python3

import sys
import getopt


def main(argv):

    # default output directory
    outdir = '.'

    print("Number of total arguments: " + str(len(argv)))

    try:
        opts, svg_files = getopt.getopt(argv, "ho:", ["outdir="])
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


if __name__ == "__main__":

    # call main method
    main(sys.argv[1:])
