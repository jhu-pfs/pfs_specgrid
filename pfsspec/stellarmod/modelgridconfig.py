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
        grid.init_value('flux')
        grid.init_value('cont')
        grid.init_value('params')

    def allocate_values(self, grid, wave):
        grid.allocate_value('flux', wave.shape)
        grid.allocate_value('cont', wave.shape)
        grid.allocate_value('params')

    def is_value_valid(self, name, value):
        return np.logical_not(np.any(np.isnan(value), axis=-1)) & ((value.max(axis=-1) != 0) | (value.min(axis=-1) != 0))

    def create_spectrum(self):
        return ModelSpectrum()

    def create_continuum_model(self):
        return LogChebyshevContinuumModel()