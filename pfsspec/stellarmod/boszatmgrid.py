import numpy as np

from pfsspec.stellarmod.kuruczatmgrid import KuruczAtmGrid

class BoszAtmGrid(KuruczAtmGrid):

    # TODO: merge this into bosz.py
    def init_axes(self):
        self.init_axis('Fe_H', np.array([-5.0, -4.5, -4.0, -3.5, -3.0, -2.75, -2.5 , -2.25, -2.,
                                          -1.75, -1.5 , -1.25, -1.  , -0.75, -0.5 ,
                                          -0.25,  0.  ,  0.25,  0.5 ,  0.75, 1.0, 1.5]))
        self.init_axis('T_eff', np.hstack((np.arange(3500, 12500, 250),
                                             np.arange(12500, 20000, 500),
                                             np.arange(20000, 36000, 1000))))
        self.init_axis('log_g', np.arange(0, 5.5, 0.5))
        self.init_axis('C_M', np.arange(-0.75, 0.75, 0.25))
        self.init_axis('O_M', np.arange(-0.25, 0.75, 0.25))