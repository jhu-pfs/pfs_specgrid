import os
import pickle
import argparse
import numpy as np

from pfsspec.stellarmod.kuruczgrid import KuruczGrid
from pfsspec.stellarmod.io.kuruczspectrumreader import KuruczSpectrumReader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Model data directory base path\n")
    parser.add_argument("--grid", type=str, choices=['kurucz', 'nover', 'anover', 'odfnew', 'aodfnew'],
                        default = 'kurucz', help="Survey data used\n")

    return parser.parse_args()

def __main__(args):
    grid = KuruczSpectrumReader.read_all(args.path, args.grid)
    print(grid.shape)

__main__(parse_args())