import numpy as np

from pfsspec.obsmod.stellarspectrum import StellarSpectrum

class ModelSpectrum(StellarSpectrum):
    def __init__(self, orig=None):
        super(ModelSpectrum, self).__init__()
        if isinstance(orig, ModelSpectrum):
            self.N_He = orig.N_He
            self.v_turb = orig.v_turb
            self.L_H = orig.L_H
            self.C_M = orig.C_M
            self.O_M = orig.O_M
            self.interp_param = orig.interp_param
        else:
            self.N_He = np.nan
            self.v_turb = np.nan
            self.L_H = np.nan
            self.C_M = np.nan
            self.O_M = np.nan
            self.interp_param = ''

    def get_param_names(self):
        params = super(ModelSpectrum, self).get_param_names()
        params = params + ['N_He',
                           'v_turb',
                           'L_H',
                           'C_M',
                           'O_M',
                           'interp_param']
        return params

    def print_info(self):
        super(ModelSpectrum, self).print_info()

        print('N(He)=', self.N_He)
        print('v_turb=', self.v_turb)
        print('L/H=', self.L_H)
        print('[C/M]=', self.C_M)
        print('[O/M]=', self.O_M)