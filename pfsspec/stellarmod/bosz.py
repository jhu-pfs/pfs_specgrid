import numpy as np

from pfsspec.stellarmod.modelgridconfig import ModelGridConfig
from pfsspec.stellarmod.boszspectrum import BoszSpectrum

class Bosz(ModelGridConfig):
    def __init__(self, orig=None, normalized=False, pca=None):
        super(Bosz, self).__init__(orig=orig, normalized=normalized, pca=pca)

        if isinstance(orig, Bosz):
            pass
        else:
            pass

    def add_args(self, parser):
        super(Bosz, self).add_args(parser)

    def init_from_args(self, args):
        super(Bosz, self).init_from_args(args)

    def init_axes(self, grid):
        grid.init_axis('Fe_H', np.arange(-2.5, 1.0, 0.25))
        grid.init_axis('T_eff', np.hstack((np.arange(3500.0, 12250.0, 250.0),
                                             np.arange(12500.0, 20000.0, 500.0),
                                             np.arange(20000.0, 36000.0, 1000.0))))
        grid.init_axis('log_g', np.arange(0, 5.5, 0.5))
        grid.init_axis('C_M', np.arange(-0.75, 0.75, 0.25))
        grid.init_axis('O_M', np.arange(-0.25, 0.75, 0.25))

        grid.build_axis_indexes()

    def init_values(self, grid):
        super(Bosz, self).init_values(grid)
      
    def allocate_values(self, grid, wave):
        super(Bosz, self).allocate_values(grid, wave)

    def create_spectrum(self):
        spec = BoszSpectrum()
        return spec
