import os
import logging
import multiprocessing
import time

from pfsspec.stellarmod.modelgridreader import ModelGridReader
from pfsspec.stellarmod.boszspectrumreader import BoszSpectrumReader

class BoszGridReader(ModelGridReader):
    def __init__(self, grid, reader, max=None, parallel=True):
        super(BoszGridReader, self).__init__(grid=grid, max=max, parallel=parallel)
        self.reader = reader

    def process_item(self, i):
        logger = multiprocessing.get_logger()

        index, params = i
        fn = self.reader.get_filename(**params)
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

        params = BoszSpectrumReader.parse_filename(file)
        index = self.grid.get_index(**params)
        spec = self.reader.read(file)

        return index, params, spec

    def store_item(self, res):
        if res is not None:
            index, params, spec = res

            if self.grid.wave is None:
                self.grid.wave = spec.wave

            self.grid.set_flux_idx(index, spec.flux, spec.cont)