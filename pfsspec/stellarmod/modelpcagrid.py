import numpy as np

from pfsspec.data.arraygrid import ArrayGrid
from pfsspec.data.rbfgrid import RbfGrid
from pfsspec.data.pcagrid import PcaGrid
from pfsspec.stellarmod.modelgrid import ModelGrid

class ModelPcaGrid(PcaGrid):
    def __init__(self, orig=None):
        if isinstance(orig, ModelPcaGrid):
            self.grid = orig.grid
            self.continuum_model = orig.continuum_model
            self.wave = orig.wave
        else:
            self.grid = self.create_pca_grid()
            self.continuum_model = self.create_continuum_model()
            self.wave = None
            
        super(ModelPcaGrid, self).__init__(self.grid, orig=orig)

    def add_args(self, parser):
        self.continuum_model.add_args(parser)

    def init_from_args(self, args):
        self.continuum_model.init_from_args(args)
    
    def init_axes(self):
        super(ModelPcaGrid, self).init_axes()
        self.init_axis('Fe_H')
        self.init_axis('T_eff')
        self.init_axis('log_g')

    def init_values(self):
        self.init_value('params')           # Continuum fit parameters
        self.init_value('flux', pca=True)   # Normalized flux

    def allocate_values(self):
        super(ModelPcaGrid, self).allocate_values()
        self.allocate_value('params')

    def save_items(self):
        super(ModelPcaGrid, self).save_items()
        self.save_item('wave', self.wave)
       
    def load_items(self, s=None):
        super(ModelPcaGrid, self).load_items(s=s)
        self.wave = self.load_item('wave', np.ndarray)

    def get_parameterized_spectrum(self, s=None, **kwargs):
        spec = self.create_spectrum()
        self.set_object_params(spec, **kwargs)
        spec.wave = self.wave[s or slice(None)]
        return spec

    def interpolate_model_rbf(self, **kwargs):
        # TODO: sort this out with existing but different function names

        xi = self.ip_to_index(**kwargs)
        params = self.params_rbf(*xi)
        coeffs = self.coeffs_rbf(*xi)

        # TODO: add support for slicing
        # s = self.slice[-1] if self.slice is not None else None
        spec = self.get_parameterized_spectrum(s=None, **kwargs)
        
        # Evaluate continuum model
        spec.flux = np.dot(coeffs, self.eigv)
        self.continuum_model.denormalize(spec, params)

        return spec

    def fit_rbf(self, value, axes, mask=None, function='multiquadric', epsilon=None, smooth=0.0):
        return self.grid.fit_rbf(value, axes, mask=mask, function=function, epsilon=epsilon, smooth=smooth)