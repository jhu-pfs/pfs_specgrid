import numpy as np

from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.stellarmod.modelparam import ModelParam
from pfsspec.stellarmod.kuruczspectrum import KuruczSpectrum

class BoszGrid(ModelGrid):
    def __init__(self):
        super(BoszGrid, self).__init__(use_cont=True)

        self.params['Fe_H'] = ModelParam('Fe_H', np.arange(-2.5, 1.0, 0.25))
        self.params['T_eff'] = ModelParam('T_eff', np.array([3500, 3750, 4000, 4250, 4500, 4750, 5000]))
        self.params['log_g'] = ModelParam('log_g', np.arange(2.5, 5.5, 0.5))
        self.params['C_M'] = ModelParam('C_M', np.arange(-0.75, 0.75, 0.25))
        self.params['a_Fe'] = ModelParam('a_Fe', np.arange(-0.25, 0.75, 0.25))

    def create_spectrum(self):
        spec = KuruczSpectrum()
        return spec