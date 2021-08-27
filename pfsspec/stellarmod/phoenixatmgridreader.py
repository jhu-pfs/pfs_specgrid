import os
import logging
import multiprocessing
import time

from pfsspec.stellarmod.atmgridreader import AtmGridReader
from pfsspec.stellarmod.phoenixspectrumreader import PhoenixSpectrumReader
from pfsspec.stellarmod.phoenixatmreader import PhoenixAtmReader

class PhoenixAtmGridReader(AtmGridReader):
    def __init__(self, grid, reader, max=None, parallel=True):
        super(PhoenixAtmGridReader, self).__init__(grid, max=max)
        self.reader = reader

    def process_item(self, i):
        raise NotImplementedError()

    def process_file(self, file):
        logger = multiprocessing.get_logger()

        # Use parser function from spectrum reader but otherwise
        # self.reader should be a KuruczAtmReader
        params = PhoenixSpectrumReader.parse_filename(file)
        index = self.grid.get_index(**params)
        with open(file, mode='r') as f:
            atm = self.reader.read(f)

        return index, params, atm

    def store_item(self, res):
        if res is not None:
            index, params, atm = res

            self.grid.set_value_at('ABUNDANCE', index, atm.ABUNDANCE)
            self.grid.set_value_at('RHOX', index, atm.RHOX)
            self.grid.set_value_at('T', index, atm.T)
            self.grid.set_value_at('P', index, atm.P)
            self.grid.set_value_at('XNE', index, atm.XNE)
            self.grid.set_value_at('ABROSS', index, atm.ABROSS)
            self.grid.set_value_at('ACCRAD', index, atm.ACCRAD)
            self.grid.set_value_at('VTURB', index, atm.VTURB)
            self.grid.set_value_at('FLXCNV', index, atm.FLXCNV)
            self.grid.set_value_at('VCONV', index, atm.VCONV)
            self.grid.set_value_at('VELSND', index, atm.VELSND)

        