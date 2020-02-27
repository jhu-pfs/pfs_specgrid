#!python

import os
import glob
import logging

from pfsspec.scripts.import_ import Import
from pfsspec.stellarmod.boszspectrumreader import BoszSpectrumReader
from pfsspec.stellarmod.boszgridreader import BoszGridReader
from pfsspec.stellarmod.boszgrid import BoszGrid

class ImportBosz(Import):
    def __init__(self):
        super(ImportBosz, self).__init__()

    def add_args(self, parser):
        super(ImportBosz, self).add_args(parser)
        parser.add_argument("--wave", type=float, nargs=2, default=None, help="Wavelength limits.\n")
        parser.add_argument("--res", type=int, default=None, help="Resolution.\n")
        parser.add_argument("--max", type=int, default=None, help="Stop after this many items.\n")
        parser.add_argument('--preload-arrays', action='store_true', help='Do not preload flux arrays to save memory\n')

    def run(self):
        super(ImportBosz, self).run()

        if 'res' in self.args and self.args['res'] is not None:
            res = self.args['res']
        else:
            res = 5000

        grid = BoszGrid()
        r = BoszSpectrumReader(self.args['path'], self.args['wave'], res)
        gr = BoszGridReader(grid, r, self.args.max)
        gr.parallel = not self.debug

        if 'preload_arrays' in self.args and self.args['preload_arrays'] is not None:
            grid.preload_arrays = self.args['preload_arrays']

        if os.path.isdir(self.args['path']):
            logging.info('Running in grid mode')

            # Load the first spectrum to get wavelength grid
            fn = BoszSpectrumReader.get_filename(Fe_H=0.0, T_eff=5000.0, log_g=1.0, O_M=0.0, C_M=0.0, R=res)
            fn = os.path.join(self.args['path'], fn)
            spec = r.read(fn)
        else:
            logging.info('Running in file list mode')
            files = glob.glob(os.path.expandvars(self.args['path']))
            logging.info('Found {} files.'.format(len(files)))

            # Load the first spectrum to get wavelength grid
            spec = r.read(files[0])

        logging.info('Found spectrum with {} wavelength elements.'.format(spec.wave.shape))

        grid.wave = spec.wave
        grid.init_data()
        grid.build_params_index()
        grid.save(os.path.join(self.args['out'], 'spectra.h5'), 'h5')

        if os.path.isdir(self.args['path']):
            r.path = self.args['path']
            r.read_grid()
        else:
            r.read_files(files)

        #r.grid.build_flux_index(rebuild=True)
        r.grid.save(os.path.join(self.args['out'], 'spectra.h5'), 'h5')

def main():
    script = ImportBosz()
    script.execute()

if __name__ == "__main__":
    main()