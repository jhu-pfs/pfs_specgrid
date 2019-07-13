#!python

from pfsspec.scripts.utils.util import *
from pfsspec.stellarmod.kuruczspectrumreader import KuruczSpectrumReader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Model data directory base path\n")
    parser.add_argument("--out", type=str, help="Output file, must be .npz\n")
    parser.add_argument("--grid", type=str, choices=['kurucz', 'nover', 'anover', 'odfnew', 'aodfnew'],
                        default = 'kurucz', help="Model subtype\n")

    return parser.parse_args()

def export_spectra(args):
    create_output_dir(args.out)

    dump_json(args, os.path.join(args.out, 'args.json'))

    grid = KuruczSpectrumReader.read_grid(args.path, args.grid)
    grid.save(os.path.join(args.out, 'spectra.npz'))

def __main__(args):
    export_spectra(args)

__main__(parse_args())