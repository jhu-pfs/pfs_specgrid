import numpy as np

from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.data.gridparam import GridParam
from pfsspec.stellarmod.kuruczspectrum import KuruczSpectrum

class BoszGrid(ModelGrid):
    def __init__(self):
        super(BoszGrid, self).__init__()

    def init_params(self):
        self.init_param('Fe_H', np.arange(-2.5, 1.0, 0.25))
        self.init_param('T_eff', np.hstack((np.arange(3500, 12500, 250),
                                             np.arange(12500, 20000, 500),
                                             np.arange(20000, 36000, 1000))))
        self.init_param('log_g', np.arange(0, 5.5, 0.5))
        self.init_param('C_M', np.arange(-0.75, 0.75, 0.25))
        self.init_param('O_M', np.arange(-0.25, 0.75, 0.25))

    def create_spectrum(self):
        spec = KuruczSpectrum()
        return spec