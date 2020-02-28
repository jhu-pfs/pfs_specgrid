import os
import logging
import multiprocessing
import time

from pfsspec.stellarmod.atmgridreader import AtmGridReader
from pfsspec.stellarmod.kuruczatmreader import KuruczAtmReader

class KuruczAtmGridReader(AtmGridReader):
    def __init__(self, grid, reader, max=None, parallel=True):
        super(KuruczAtmRGridReader, self).__init__()
        self.reader = reader