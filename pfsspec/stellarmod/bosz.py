import numpy as np

from pfsspec.data.gridaxis import GridAxis
from pfsspec.stellarmod.boszspectrum import BoszSpectrum

class Bosz():
    # Mixin for BOSZ grid

    def __init__(self, orig=None):
        pass

    def init_axes(self):
        self.init_axis('Fe_H', np.arange(-2.5, 1.0, 0.25))
        self.init_axis('T_eff', np.hstack((np.arange(3500.0, 12500.0, 250.0),
                                             np.arange(12500.0, 20000.0, 500.0),
                                             np.arange(20000.0, 36000.0, 1000.0))))
        self.init_axis('log_g', np.arange(0, 5.5, 0.5))
        self.init_axis('C_M', np.arange(-0.75, 0.75, 0.25))
        self.init_axis('O_M', np.arange(-0.25, 0.75, 0.25))

    def init_values(self):
        pass

    def create_spectrum(self):
        spec = BoszSpectrum()
        return spec
