#!python

import os

from pfsspec.scripts.import_ import Import
from pfsspec.stellarmod.boszspectrumreader import BoszSpectrumReader
from pfsspec.stellarmod.boszgrid import BoszGrid

class ImportBosz(Import):
    def __init__(self):
        super(ImportBosz, self).__init__()

    def add_args(self, parser):
        super(ImportBosz, self).add_args(parser)
        parser.add_argument("--wave", type=float, nargs=2, default=None, help="Wavelength limits.\n")
        parser.add_argument("--max", type=int, default=None, help="Stop after this many items.\n")

    def run(self):
        super(ImportBosz, self).run()

        grid = BoszGrid()

        r = BoszSpectrumReader(grid)
        r.path = self.args['path']
        r.wave_lim = self.args['wave']
        r.max = self.args['max']

        # Load the first spectrum to get wavelength grid
        fn = BoszSpectrumReader.get_filename(Fe_H=0.0, T_eff=5000.0, log_g=1.0, a_Fe=0.0, C_M=0.0)
        fn = os.path.join(self.args['path'], fn)
        spec = r.read(fn)

        r.grid.init_storage(spec.wave)
        r.read_grid()
        r.grid.save(os.path.join(self.args['out'], 'spectra.h5'), 'h5')

def main():
    script = ImportBosz()
    script.execute()

if __name__ == "__main__":
    main()