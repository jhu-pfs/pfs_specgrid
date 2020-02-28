import os
import logging
import numpy as numpy

from pfsspec.stellarmod.kuruczatmreader import KuruczAtmReader
from pfsspec.stellarmod.kuruczatm import KuruczAtm

class BoszAtmReader(KuruczAtmReader):
    def __init__(self, path=None):
        super(BoszAtmReader, self).__init__()
        self.path = path

