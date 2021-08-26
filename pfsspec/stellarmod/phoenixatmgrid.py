import numpy as np

from pfsspec.stellarmod.kuruczatmgrid import KuruczAtmGrid


class PhoenixAtmGrid(KuruczAtmGrid):

    # TODO: merge this into bosz.py
    def init_axes(self):
        self.init_axis('Fe_H', np.array([-4.0, -3.0, -2.,
                                          -1.5, -1.0 , -0.5, 0.0  , 0.5, 1.0]))
        self.init_axis('T_eff', np.hstack((np.arange(2300.0, 7000.0, 100.0),
                                             np.arange(7000.0, 12200.0, 200.0))))
        self.init_axis('log_g', np.arange(0, 6.5, 0.5))
        #self.init_axis('C_M', np.arange(-0.75, 0.75, 0.25))
        #self.init_axis('O_M', np.arange(-0.25, 0.75, 0.25))
