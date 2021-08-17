import copy
import numpy as np
import scipy as sp
import logging

from pfsspec.util.physics import Physics
from pfsspec.stellarmod.continuummodels.continuummodel import ContinuumModel
from pfsspec.stellarmod.continuummodels.modelparameter import ModelParameter
from pfsspec.util.array_filters import *

class Planck(ContinuumModel):
    # Normalize stellar models with the Planck curve associated with
    # the effective temperature.
    
    # We do not fit the Planck curve but simply use the effective temperature
    # of the spectrum model. For compatibility, however, functions are implemented
    # to return T_eff as a parameters. This later can be extended if fitting of
    # the luminosity is necessary.

    def __init__(self, orig=None):
        super(Planck, self).__init__(orig=orig)

        if isinstance(orig, Planck):
            pass
        else:
            pass

    @property
    def name(self):
        return "planck"

    def get_model_parameters(self):
        params = super(Planck, self).get_model_parameters()
        params.append(ModelParameter(name='planck'))
        return params

    def init_wave(self, wave):
        self.wave = wave
        self.wave_mask = np.full(wave.shape, True)

    def init_values(self, grid):
        grid.init_value('planck')

    def allocate_values(self, grid):
        grid.allocate_value('planck', (1,))

    def safe_log(self, x):
        return np.log(np.where(x <= 1, 1, x))

    def fit(self, spec):
        return { 'planck': np.array([spec.T_eff])}

    def eval(self, params):
        n = 1e-7 * Physics.planck(self.wave * 1e-10, params['planck'][0])
        return self.wave, self.safe_log(n)

    def normalize(self, spec, params):
        _, n = self.eval(params)
        spec.flux = self.safe_log(spec.flux) - n
        if spec.cont is not None:
            spec.cont = self.safe_log(spec.cont) - n

    def denormalize(self, spec, params):
        params['planck'] = np.array([spec.T_eff])
        _, n = self.eval(params)
        spec.flux = np.exp(spec.flux + n)
        if spec.cont is not None:
            spec.cont = np.exp(spec.cont + n)
        else:
            spec.cont = np.exp(n)

