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
            pass
        else:
            pass

    def create_grid(self):
        return ModelGrid(Phoenix(), ArrayGrid)

    def create_reader(self, input_path, output_path, wave=None, resolution=None):
        return PhoenixSpectrumReader(input_path, wave, resolution)

    def get_example_filename(self):
        # Here we use constants because this particular model must exist in every grid.
        return self.reader.get_filename(Fe_H=0.0, T_eff=5000.0, log_g=1.0)