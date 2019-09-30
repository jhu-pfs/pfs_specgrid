#!python

import os

from pfsspec.scripts.import_ import Import
from pfsspec.stellarmod.boszspectrumreader import BoszSpectrumReader

class ImportBosz(Import):
    def __init__(self):
        super(ImportBosz, self).__init__()

    def add_args(self, parser):
        super(ImportBosz, self).add_args(parser)
        parser.add_argument("--wave", type=float, nargs=2, default=None, help="Wavelength limits.\n")
        parser.add_argument("--stop", type=int, default=None, help="Stop after this many items.\n")

    def run(self):
        super(ImportBosz, self).run()

        r = BoszSpectrumReader()
        r.wave_lim = self.args['wave']
        grid = r.read_grid(self.args['path'], stop=self.args['stop'])
        grid.save(os.path.join(self.args['out'], 'spectra.h5'), 'h5')

def main():
    script = ImportBosz()
    script.execute()

if __name__ == "__main__":
    main()