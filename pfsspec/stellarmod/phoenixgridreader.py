import os
import glob
import logging
import multiprocessing
import time

from pfsspec.data.arraygrid import ArrayGrid
from pfsspec.stellarmod.modelgridreader import ModelGridReader
from pfsspec.stellarmod.phoenixspectrumreader import PhoenixSpectrumReader
from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.stellarmod.phoenix import Phoenix



class PhoenixGridReader(ModelGridReader):
    def __init__(self, grid=None, orig=None):
        super(PhoenixGridReader, self).__init__(grid=grid, orig=orig)

        if isinstance(orig, PhoenixGridReader):
            self.path = orig.path
            self.files = None
        else:
            self.path = None
            self.files = None

    def process_item(self, i):
        logger = multiprocessing.get_logger()

        index, params = i
        fn = self.reader.get_filename(**params)
        #self.reader.get_filename(R=self.reader.resolution, **params)
        fn = os.path.join(self.reader.path, fn)

        if os.path.isfile(fn):
            tries = 3
            while True:
                try:
                    spec = self.reader.read(fn)
                    return index, params, spec
                except Exception as e:
                    logger.error('Error parsing {}'.format(fn))
                    time.sleep(0.01)    # ugly hack
                    tries -= 1
                    if tries == 0:
                        raise e

        else:
            logger.debug('Cannot find file {}'.format(fn))
            return None

    def process_file(self, file):
        logger = multiprocessing.get_logger()

        params = PhoenixSpectrumReader.parse_filename(file)
        index = self.grid.get_index(**params)
        spec = self.reader.read(file)

        return index, params, spec

    def store_item(self, res):
        if res is not None:
            index, params, spec = res

            if self.grid.get_wave() is None:
                self.grid.set_wave(spec.wave)

            self.grid.set_flux_at(index, spec.flux, spec.cont) #added abck in cont...

    def create_grid(self):
        grid = ModelGrid(Phoenix(), ArrayGrid)
        grid.preload_arrays = self.preload_arrays
        return grid

    def open_data(self, input_path, output_path):
        # Initialize input

        if self.reader is None:
            self.reader = PhoenixSpectrumReader(input_path, self.wave)#, self.resolution)

        if os.path.isdir(input_path):
            self.logger.info('Running in grid mode')
            self.path = input_path

            # Load the first spectrum to get wavelength grid. Here we use constants
            # because this particular model must exist in every grid.
            fn = PhoenixSpectrumReader.get_filename(Fe_H=0.0, T_eff=5000.0, log_g=1.0)
            #PhoenixSpectrumReader.get_filename(Fe_H=0.0, T_eff=5000.0, log_g=1.0, O_M=0.0, C_M=0.0, R=self.resolution)
            fn = os.path.join(self.path, fn)
            spec = self.reader.read(fn)
        else:
            self.logger.info('Running in file list mode')
            self.files = glob.glob(os.path.expandvars(input_path))
            self.files.sort()
            self.logger.info('Found {} files.'.format(len(self.files)))

            # Load the first spectrum to get wavelength grid
            spec = self.reader.read(files[0])

        self.logger.info('Found spectrum with {} wavelength elements.'.format(spec.wave.shape))
        
        # Initialize output

        fn = os.path.join(output_path, 'spectra.h5')

        if self.grid is None:
            self.grid = self.create_grid()

        if self.resume:
            if self.grid.preload_arrays:
                raise NotImplementedError("Can only resume import when preload_arrays is False.")
            self.grid.load(fn, format='h5')
        else:
            # Initialize the wavelength grid based on the first spectrum read
            self.grid.set_wave(spec.wave)
            self.grid.build_axis_indexes()
           
            # Force creating output file for direct hdf5 writing
            self.grid.save(fn, format='h5')

    def save_data(self):
        self.grid.save(self.grid.filename, self.grid.fileformat)

    def run(self):
        if os.path.isdir(self.path):
            self.read_grid(resume=self.resume)
        else:
            self.read_files(files, resume=self.resume)