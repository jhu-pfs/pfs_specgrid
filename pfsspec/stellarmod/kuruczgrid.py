import numpy as np

from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.data.gridparam import GridParam
from pfsspec.stellarmod.kuruczspectrum import KuruczSpectrum

class KuruczGrid(ModelGrid):
    def __init__(self, model='kurucz'):
        self.model = model
        super(KuruczGrid, self).__init__()
        
    def init_params(self):
        if self.model == 'test':
            self.init_param('Fe_H', np.array([0.0, 0.1]))
            self.init_param('T_eff',
                 np.array([3500., 3750., 4000., 4250., 4500., 4750., 5000., 5250., 5500.,
                           5750., 6000., 6250., 6500., 6750., 7000., 7250., 7500., 7750.,
                           8000., 8250., 8500., 8750., 9000., 9250., 9500., 9750., 10000.,
                           10500., 11000., 11500., 12000., 12500., 13000., 14000., 15000.,
                           16000., 17000., 18000., 19000., 20000., 21000., 22000., 23000.,
                           24000., 25000., 26000., 27000., 28000., 29000., 30000., 31000.,
                           32000., 33000., 34000., 35000., 37500., 40000., 42500., 45000.,
                           47500., 50000.]))
            self.init_param('log_g', np.arange(0, 5.1, 0.5))
        elif self.model == 'kurucz':
            self.init_param('Fe_H',
                np.array(
                    [-5.0, -4.5, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.3, -0.2, -0.1,
                     0.0, 0.1, 0.2, 0.3,
                     # 0.4,
                     0.5, 1.0]))
            self.init_param('T_eff',
                 np.array([3500., 3750., 4000., 4250., 4500., 4750., 5000., 5250., 5500.,
                           5750., 6000., 6250., 6500., 6750., 7000., 7250., 7500., 7750.,
                           8000., 8250., 8500., 8750., 9000., 9250., 9500., 9750., 10000.,
                           10500., 11000., 11500., 12000., 12500., 13000., 14000., 15000.,
                           16000., 17000., 18000., 19000., 20000., 21000., 22000., 23000.,
                           24000., 25000., 26000., 27000., 28000., 29000., 30000., 31000.,
                           32000., 33000., 34000., 35000., 37500., 40000., 42500., 45000.,
                           47500., 50000.]))
            self.init_param('log_g', np.arange(0, 5.1, 0.5))

    def create_spectrum(self):
        spec = KuruczSpectrum()
        return spec