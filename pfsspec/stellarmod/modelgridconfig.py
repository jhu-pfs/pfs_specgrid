import numpy as np

from pfsspec.stellarmod.modelspectrum import ModelSpectrum
from pfsspec.stellarmod.logchebyshevcontinuummodel import LogChebyshevContinuumModel

class ModelGridConfig():
    def __init__(self, pca=False, orig=None):
        if isinstance(orig, ModelGridConfig):
            self.pca = pca if pca is not None else orig.pca
        else:
            self.pca = pca

    def init_axes(self, grid):
        grid.init_axis('Fe_H')
        grid.init_axis('T_eff')
        grid.init_axis('log_g')

    def init_constants(self, grid):
        grid.init_constant('constants')

    def init_values(self, grid):
        grid.init_value('flux', pca=self.pca)
        grid.init_value('cont', pca=self.pca)
        grid.init_value('params')

    def allocate_values(self, grid, wave):
        if self.pca is not None and self.pca:
            raise NotImplementedError()
        else:
            grid.allocate_value('flux', wave.shape)
            grid.allocate_value('cont', wave.shape)
        grid.allocate_value('params')

    def get_is_value_valid_method(self):
        return ModelGridConfig.is_value_valid
    
    def get_get_chunks_method(self):
        return ModelGridConfig.get_chunks

    def create_spectrum(self):
        return ModelSpectrum()

    def create_continuum_model(self):
        return LogChebyshevContinuumModel()

    @staticmethod
    def is_value_valid(self, name, value):
        return np.logical_not(np.any(np.isnan(value), axis=-1)) & ((value.max(axis=-1) != 0) | (value.min(axis=-1) != 0))

    @staticmethod
    def get_chunks(self, name, shape, s=None):
        # The chunking strategy for spectrum grids should observe the following
        # - we often need only parts of the wavelength coverage
        # - interpolation algorithms iterate over the wavelengths in the outer loop
        # - interpolation algorithms need nearby models, cubic splines require models
        #   in memory along the entire interpolation axis

        # The shape of the spectrum grid is (param1, param2, wave)
        if name in self.values and name in ['flux', 'cont']:
            newshape = []
            # Keep neighboring 3 models together in every direction
            for i, k in enumerate(self.axes.keys()):
                if k in ['log_g', 'Fe_H', 'T_eff']:
                    newshape.append(min(shape[i], 3))
                else:
                    newshape.append(1)
            # Use small chunks along the wavelength direction
            newshape.append(min(256, shape[-1]))
            return tuple(newshape)
        else:
            return None