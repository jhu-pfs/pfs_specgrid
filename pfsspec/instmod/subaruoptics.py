import numpy as np

from pfsspec.obsmod.filter import Filter

class SubaruOptics(Filter):
    def read(self, file):
        data = np.loadtxt(file)
        self.wave = data[:, 0]
        self.thru = data[:, 1]