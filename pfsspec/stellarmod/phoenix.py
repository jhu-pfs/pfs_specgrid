import numpy as np

from pfsspec.stellarmod.continuummodels.chebyshev import Chebyshev #LogChebyshevContinuumModel
from pfsspec.stellarmod.modelgridconfig import ModelGridConfig
from pfsspec.stellarmod.phoenixspectrum import PhoenixSpectrum

    
class Phoenix(ModelGridConfig):
    def __init__(self, orig=None, normalized=False, pca=None):
        super(Phoenix, self).__init__(orig=orig,normalized=normalized, pca=pca) #added normalised
        if isinstance(orig, Phoenix):
            pass
        else:
            pass

    def add_args(self, parser):
        super(Phoenix, self).add_args(parser)

    def init_from_args(self, args):
        super(Phoenix, self).init_from_args(args)
        
    def init_axes(self, grid):
        grid.init_axis('Fe_H', np.hstack((np.arange(-4.0, -2.0, 1),
                                             np.arange(-2.0, 1.5, 0.50))))
        grid.init_axis('T_eff', np.hstack((np.arange(2300.0, 7000.0, 100.0),
                                             np.arange(7000.0, 12200.0, 200.0))))
        grid.init_axis('log_g', np.arange(0, 6.5, 0.5))
        #grid.init_axis('C_M', np.arange(-0.75, 0.75, 0.25))
        #grid.init_axis('O_M', np.arange(-0.25, 0.75, 0.25))

        grid.build_axis_indexes()

    def init_values(self, grid):
        super(Phoenix, self).init_values(grid)
      
    def allocate_values(self, grid, wave):
        super(Phoenix, self).allocate_values(grid, wave)

    def create_spectrum(self):
        spec = PhoenixSpectrum()
        return spec
