import numpy as np

from pfsspec.common.spectrum import Spectrum
from pfsspec.util.physics import Physics

class StellarSpectrum(Spectrum):
    # TODO: make it a mixin instead of an inherited class
    def __init__(self, orig=None):
        super(StellarSpectrum, self).__init__(orig=orig)
        
        if isinstance(orig, StellarSpectrum):
            self.T_eff = orig.T_eff
            self.T_eff_err = orig.T_eff_err
            self.log_g = orig.log_g
            self.log_g_err = orig.log_g_err
            self.Fe_H = orig.Fe_H
            self.Fe_H_err = orig.Fe_H_err
            self.a_Fe = orig.a_Fe
            self.a_Fe_err = orig.a_Fe_err
        else:
            self.T_eff = np.nan
            self.T_eff_err = np.nan
            self.log_g = np.nan
            self.log_g_err = np.nan
            self.Fe_H = np.nan
            self.Fe_H_err = np.nan
            self.a_Fe = np.nan
            self.a_Fe_err = np.nan

    def get_param_names(self):
        params = super(StellarSpectrum, self).get_param_names()
        params = params + ['T_eff', 'T_eff_err',
                           'log_g', 'log_g_err',
                           'Fe_H', 'Fe_H_err',
                           'a_Fe', 'a_Fe_err']
        return params

    def normalize_by_T_eff(self, T_eff=None):
        T_eff = T_eff or self.T_eff
        self.logger.debug('Normalizing spectrum with black-body of T_eff={}'.format(T_eff))
        n = 1e-7 * Physics.planck(self.wave*1e-10, T_eff)
        self.multiply(1 / n)

    def denormalize_by_T_eff(self, T_eff=None):
        T_eff = T_eff or self.T_eff
        self.logger.debug('Denormalizing spectrum with black-body of T_eff={}'.format(T_eff))
        n = 1e-7 * Physics.planck(self.wave*1e-10, T_eff)
        self.multiply(n)

    def print_info(self):
        super(StellarSpectrum, self).print_info()

        print('T_eff=', self.T_eff)
        print('log g=', self.log_g)
        print('[Fe/H]=', self.Fe_H)
        print('[a/Fe]=', self.a_Fe)