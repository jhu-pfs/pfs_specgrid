import copy
import numpy as np
import scipy as sp
import logging

from pfsspec.stellarmod.continuummodels.continuummodel import ContinuumModel

class Log(ContinuumModel):
    # Simply take the logarithm of the flux

    def __init__(self, orig=None):
        super(Log, self).__init__(orig=orig)

        if isinstance(orig, Log):
            pass
        else:
            pass

    @property
    def name(self):
        return "log"

    def get_params_names(self):
        names = super(Log, self).get_params_names()
        names.append('log')
        return names

    def init_wave(self, wave):
        self.wave = wave
        self.wave_mask = np.full(wave.shape, True)

    def init_values(self, grid):
        grid.init_value('log')

    def allocate_values(self, grid):
        grid.allocate_value('log', (1,))

    def safe_log(self, x):
        return np.log(np.where(x <= 1, 1, x))

    def fit(self, spec):
        n = np.array([np.max(self.safe_log(spec.flux))])
        return { 'log': n}

    def eval(self, params):
        n = params['log']       # scalar
        return self.wave, np.full(self.wave.shape, n)

    def normalize(self, spec, params):
        _, n = self.eval(params)
        spec.flux = self.safe_log(spec.flux) - n
        if spec.cont is not None:
            spec.cont = self.safe_log(spec.cont) - n

    def denormalize(self, spec, params):
        _, n = self.eval(params)
        spec.flux = np.exp(spec.flux + n)
        if spec.cont is not None:
            spec.cont = np.exp(spec.cont + n)
