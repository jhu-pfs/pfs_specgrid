import numpy as np

from pfsspec.constants import Constants
from pfsspec.filter import Filter

class SubaruOptics(Filter):
    def read(self, file):
        data = np.loadtxt(file)
        self.wave = data[:, 0]
        self.thru = data[:, 1]