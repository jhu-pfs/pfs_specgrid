import numpy as np

from pfsspec.data.grid import Grid
from pfsspec.data.gridparam import GridParam
from pfsspec.stellarmod.kuruczatm import KuruczAtm

class KuruczAtmGrid(Grid):
    def __init__(self):
        super(KuruczAtmGrid, self).__init__()

    def init_params(self):
        self.init_param('Fe_H', np.arange(-2.5, 1.0, 0.25))
        self.init_param('T_eff', np.hstack((np.arange(3500, 12500, 250),
                                             np.arange(12500, 20000, 500),
                                             np.arange(20000, 36000, 1000))))
        self.init_param('log_g', np.arange(0, 5.5, 0.5))
        self.init_param('C_M', np.arange(-0.75, 0.75, 0.25))
        self.init_param('O_M', np.arange(-0.25, 0.75, 0.25))

    def init_data(self):
        self.init_data_item('ABUNDANCE', (99,))
        self.init_data_item('RHOX', (72,))
        self.init_data_item('T', (72,))
        self.init_data_item('P', (72,))
        self.init_data_item('XNE', (72,))
        self.init_data_item('ABROSS', (72,))
        self.init_data_item('ACCRAD', (72,))
        self.init_data_item('VTURB', (72,))
        self.init_data_item('FLXCNV', (72,))
        self.init_data_item('VCONV', (72,))
        self.init_data_item('VELSND', (72,))