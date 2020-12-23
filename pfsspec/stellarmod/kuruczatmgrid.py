import numpy as np

from pfsspec.data.arraygrid import ArrayGrid
from pfsspec.data.gridaxis import GridAxis
from pfsspec.stellarmod.kuruczatm import KuruczAtm

class KuruczAtmGrid(ArrayGrid):
    def __init__(self):
        super(KuruczAtmGrid, self).__init__()

    def init_axes(self):
        self.init_axis('Fe_H', np.arange(-2.5, 1.0, 0.25))
        self.init_axis('T_eff', np.hstack((np.arange(3500, 12500, 250),
                                             np.arange(12500, 20000, 500),
                                             np.arange(20000, 36000, 1000))))
        self.init_axis('log_g', np.arange(0, 5.5, 0.5))
        self.init_axis('C_M', np.arange(-0.75, 0.75, 0.25))
        self.init_axis('O_M', np.arange(-0.25, 0.75, 0.25))

    def init_values(self):
        self.init_value('ABUNDANCE', (99,))
        self.init_value('RHOX', (72,))
        self.init_value('T', (72,))
        self.init_value('P', (72,))
        self.init_value('XNE', (72,))
        self.init_value('ABROSS', (72,))
        self.init_value('ACCRAD', (72,))
        self.init_value('VTURB', (72,))
        self.init_value('FLXCNV', (72,))
        self.init_value('VCONV', (72,))
        self.init_value('VELSND', (72,))