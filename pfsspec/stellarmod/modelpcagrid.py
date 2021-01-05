import numpy as np

from pfsspec.data.pcagrid import PcaGrid

class ModelPcaGrid(PcaGrid):
    
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