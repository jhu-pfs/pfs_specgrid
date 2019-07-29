import numpy as np

from pfsspec.obsmod.spectrum import Spectrum

class ModelSpectrum(Spectrum):
    def __init__(self):
        super(ModelSpectrum, self).__init__()
        self.T_eff = np.nan
        self.log_g = np.nan
        self.Fe_H = np.nan
        self.a_Fe = np.nan
        self.N_He = np.nan
        self.v_turb = np.nan
        self.L_H = np.nan

    def get_param_names(self):
        params = super(ModelSpectrum, self).get_param_names()
        params = params + ['T_eff',
                           'log_g',
                           'Fe_H',
                           'a_Fe',
                           'N_He',
                           'v_turb',
                           'L_H']
        return params

    def print_info(self):
        # TODO: call super
        print('T_eff=', self.T_eff)
        print('log g=', self.log_g)
        print('[M/H]=', self.Fe_H)
        print('[a/Fe]=', self.a_Fe)
        print('N(He)=', self.N_He)
        print('v_turb=', self.v_turb)
        print('L/H=', self.L_H)