import numpy as np

from pfsspec.stellarmod.modelspectrum import ModelSpectrum
from pfsspec.stellarmod.logchebyshevcontinuummodel import LogChebyshevContinuumModel
from pfsspec.stellarmod.planckcontinuummodel import PlanckContinuumModel
from pfsspec.stellarmod.alexcontinuummodel import AlexContinuumModel

class ModelGridConfig():
    # Implements functions to initialize a model grid. Inherited classes should
    # implement grid-specific functionality in overridden functions.

    CONTINUUM_MODEL_TYPES = {
        'planck': PlanckContinuumModel,
        'alex': AlexContinuumModel,
        'logchebyshev': LogChebyshevContinuumModel,
    }

    def __init__(self, normalized=False, pca=False, orig=None):
        if isinstance(orig, ModelGridConfig):
            self.continuum_model_type = orig.continuum_model_type
            self.normalized = normalized if normalized is not None else orig.normalized
            self.pca = pca if pca is not None else orig.pca
        else:
            self.continuum_model_type = None
            self.normalized = normalized
            self.pca = pca

    def add_args(self, parser):
        choices = [k for k in ModelGridConfig.CONTINUUM_MODEL_TYPES.keys()]
        parser.add_argument('--continuum-model', type=str, choices=choices, help='Continuum model.\n')

    def init_from_args(self, args):
        if 'continuum_model' in args and args['continuum_model'] is not None:
            self.continuum_model_type = ModelGridConfig.CONTINUUM_MODEL_TYPES[args['continuum_model']]

    def init_axes(self, grid):
        grid.init_axis('Fe_H')
        grid.init_axis('T_eff')
        grid.init_axis('log_g')

    def init_values(self, grid):
        grid.init_value('flux', pca=self.pca)
        grid.init_value('cont', pca=self.pca)

    def allocate_values(self, grid, wave):
        if self.pca is not None and self.pca:
            raise NotImplementedError()
        else:
            grid.allocate_value('flux', wave.shape)
            grid.allocate_value('cont', wave.shape)

    def create_spectrum(self):
        return ModelSpectrum()

    def create_continuum_model(self):
        if self.continuum_model_type is not None:
            model = self.continuum_model_type()
            return model
        else:
            return None

    def is_value_valid(self, grid, name, value):
        return np.logical_not(np.any(np.isnan(value), axis=-1)) & ((value.max(axis=-1) != 0) | (value.min(axis=-1) != 0))

    def get_chunks(self, grid, name, shape, s=None):
        # The chunking strategy for spectrum grids should observe the following
        # - we often need only parts of the wavelength coverage
        # - interpolation algorithms iterate over the wavelengths in the outer loop
        # - interpolation algorithms need nearby models, cubic splines require models
        #   in memory along the entire interpolation axis

        # The shape of the spectrum grid is (param1, param2, wave)
        if name in grid.values and name in ['flux', 'cont']:
            newshape = []
            # Keep neighboring 3 models together in every direction
            for i, k in enumerate(grid.axes.keys()):
                if k in ['log_g', 'Fe_H', 'T_eff']:
                    newshape.append(min(shape[i], 3))
                else:
                    newshape.append(1)
            # Use small chunks along the wavelength direction
            newshape.append(min(256, shape[-1]))
            return tuple(newshape)
        else:
            return None
