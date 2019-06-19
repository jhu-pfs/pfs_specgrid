#!python

import os
import pickle
import argparse
import numpy as np

from pfsspec.stellarmod.kuruczgrid import KuruczGrid
from pfsspec.stellarmod.io.kuruczspectrumreader import KuruczSpectrumReader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Model data directory base path\n")
    parser.add_argument("--out", type=str, help="Output file, must be .npz\n")
    parser.add_argument("--grid", type=str, choices=['kurucz', 'nover', 'anover', 'odfnew', 'aodfnew'],
                        default = 'kurucz', help="Model subtype\n")

    return parser.parse_args()

def __main__(args):
    grid = KuruczSpectrumReader.read_grid(args.path, args.grid)
    print("Grid loaded with flux grid shape ", grid.flux.shape)
    grid.save(args.out)

__main__(parse_args())