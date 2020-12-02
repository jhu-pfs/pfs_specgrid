#!python

import os
import glob
import logging

from pfsspec.scripts.import_ import Import
from pfsspec.stellarmod.boszspectrumreader import BoszSpectrumReader
from pfsspec.stellarmod.boszgridreader import BoszGridReader
from pfsspec.stellarmod.boszgrid import BoszGrid

class ImportBosz(Import):
    # TODO: Consider rewriting import logic in the same way as prepare logic works,
    #       where datasetbuilders initialize their own parsers and parse their own
    #       arguments. This way importers could be consolidated into a single script.

    def __init__(self):
        super(ImportBosz, self).__init__()

        self.wave = None
        self.resolution = 5000
        self.max = None
        self.preload_arrays = False

    def add_args(self, parser):
        super(ImportBosz, self).add_args(parser)
        parser.add_argument("--wave", type=float, nargs=2, default=None, help="Wavelength limits.\n")
        parser.add_argument("--resolution", type=int, default=None, help="Resolution.\n")
        parser.add_argument("--max", type=int, default=None, help="Stop after this many items.\n")
        parser.add_argument('--preload-arrays', action='store_true', help='Do not preload flux arrays to save memory\n')

    def parse_args(self):
        super(ImportBosz, self).parse_args()
        
        self.wave = self.get_arg('wave', self.wave)
        self.resolution = self.get_arg('resolution', self.resolution)
        self.max = self.get_arg('max', self.max)
        self.preload_arrays = self.get_arg('preload_arrays', self.preload_arrays)

    def run(self):
        super(ImportBosz, self).run()

        filename =os.path.join(self.args['out'], 'spectra.h5')

        grid = BoszGrid()
        grid.preload_arrays = self.preload_arrays
        if self.resume:
            if grid.preload_arrays:
                raise NotImplementedError("Can only resume import when preload_arrays is False.")
            grid.load(filename, format='h5')
        
        reader = BoszSpectrumReader(self.path, self.wave, self.resolution)
        gridreader = BoszGridReader(grid, reader, parallel=self.thread != 1, threads=self.threads, max=self.max)

        if os.path.isdir(self.path):
            self.logger.info('Running in grid mode')

            # Load the first spectrum to get wavelength grid. Here we use constants
            # because this particular model must exist in every grid.
            fn = BoszSpectrumReader.get_filename(Fe_H=0.0, T_eff=5000.0, log_g=1.0, O_M=0.0, C_M=0.0, R=self.resolution)
            fn = os.path.join(self.path, fn)
            spec = reader.read(fn)
        else:
            self.logger.info('Running in file list mode')
            files = glob.glob(os.path.expandvars(self.path))
            files.sort()
            self.logger.info('Found {} files.'.format(len(files)))

            # Load the first spectrum to get wavelength grid
            spec = reader.read(files[0])

        self.logger.info('Found spectrum with {} wavelength elements.'.format(spec.wave.shape))

        # Initialize the wavelength grid based on the first spectrum read
        if not self.resume:
            grid.wave = spec.wave
            grid.allocate_data()
            grid.build_params_index()
            grid.save(filename, format='h5')

        if os.path.isdir(self.path):
            gridreader.read_grid(resume=self.resume)
        else:
            gridreader.read_files(files, resume=self.resume)

        #r.grid.build_flux_index(rebuild=True)
        grid.save(filename, format='h5')

def main():
    script = ImportBosz()
    script.execute()

if __name__ == "__main__":
    main()