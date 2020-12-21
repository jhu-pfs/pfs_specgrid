import numpy as np

from pfsspec.data.pcagrid import PCAGrid

class ModelPCAGrid(PCAGrid):
    def __init__(self, orig=None):
        super(ModelPCAGrid, self).__init__(orig=orig)

        if isinstance(orig, ModelPCAGrid):
            self.wave = orig.wave
            self.params_rbf = orig.params_rbf
            self.coeffs_rbf = orig.coeffs_rbf
            self.continuum_model = orig.continuum_model
        else:
            self.wave = None
            self.params_rbf = None
            self.coeffs_rbf = None
            self.continuum_model = self.create_continuum_model()

    # TODO: figure out how to make slicing common to this grid
    #       and the "standard" ModelGrid implementation.

    def add_args(self, parser):
        self.continuum_model.add_args(parser)

    def init_from_args(self, args):
        self.continuum_model.init_from_args(args)
    
    def init_axes(self):
        super(ModelPCAGrid, self).init_axes()
        self.init_axis('Fe_H')
        self.init_axis('T_eff')
        self.init_axis('log_g')

    def init_values(self):
        super(ModelPCAGrid, self).init_values()
        self.init_value('params')

    def allocate_values(self):
        super(ModelPCAGrid, self).allocate_values()
        self.allocate_value('params')

    def save_items(self):
        self.save_item('wave', self.wave)
        self.save_item('params_rbf_xi', self.params_rbf.xi)
        self.save_item('params_rbf_nodes', self.params_rbf.nodes)
        self.save_item('coeffs_rbf_xi', self.coeffs_rbf.xi)
        self.save_item('coeffs_rbf_nodes', self.coeffs_rbf.nodes)

        super(ModelPCAGrid, self).save_items()

    def load_items(self, s=None):
        super(ModelPCAGrid, self).load_items(s=s)
        
        self.wave = self.load_item('wave', np.ndarray)

        # Now this is trickier, we need the list of points
        self.params_rbf = self.load_rbf(
            self.load_item('params_rbf_xi', np.ndarray),
            self.load_item('params_rbf_nodes', np.ndarray))
        self.coeffs_rbf = self.load_rbf(
            self.load_item('coeffs_rbf_xi', np.ndarray),
            self.load_item('coeffs_rbf_nodes', np.ndarray))

    def get_parameterized_spectrum(self, s=None, **kwargs):
        spec = self.create_spectrum()
        self.set_object_params(spec, **kwargs)
        spec.wave = self.wave[s or slice(None)]
        return spec

    def ip_to_index(self, **kwargs):
        xi = []
        for k in self.axes:
            if k not in kwargs:
                raise Exception('Interpolation requires all parameters specificed. Missing: {}'.format(k))
            xi.append(self.axes[k].ip_to_index(kwargs[k]))
        return xi

    def interpolate_model_rbf(self, **kwargs):
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
