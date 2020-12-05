import numpy as np

from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.data.gridaxis import GridAxis
from pfsspec.stellarmod.kuruczspectrum import KuruczSpectrum

class BoszGrid(ModelGrid):
    def __init__(self):
        super(BoszGrid, self).__init__()

    def init_axes(self):
        self.init_axis('Fe_H', np.arange(-2.5, 1.0, 0.25))
        self.init_axis('T_eff', np.hstack((np.arange(3500.0, 12500.0, 250.0),
                                             np.arange(12500.0, 20000.0, 500.0),
                                             np.arange(20000.0, 36000.0, 1000.0))))
        self.init_axis('log_g', np.arange(0, 5.5, 0.5))
        self.init_axis('C_M', np.arange(-0.75, 0.75, 0.25))
        self.init_axis('O_M', np.arange(-0.25, 0.75, 0.25))

    def create_spectrum(self):
        spec = KuruczSpectrum()
        return spec