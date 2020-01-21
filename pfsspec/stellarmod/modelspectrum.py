import numpy as np

from pfsspec.obsmod.spectrum import Spectrum

class ModelSpectrum(Spectrum):
    def __init__(self, orig=None):
        super(ModelSpectrum, self).__init__()
        if isinstance(orig, ModelSpectrum):
            self.T_eff = orig.T_eff
            self.log_g = orig.log_g
            self.Fe_H = orig.Fe_H
            self.a_Fe = orig.a_Fe
            self.N_He = orig.N_He
            self.v_turb = orig.v_turb
            self.L_H = orig.L_H
            self.C_M = orig.C_M
            self.O_M = orig.O_M
            self.interp_param = orig.interp_param
        else:
            self.T_eff = np.nan
            self.log_g = np.nan
            self.Fe_H = np.nan
            self.a_Fe = np.nan
            self.N_He = np.nan
            self.v_turb = np.nan
            self.L_H = np.nan
            self.C_M = np.nan
            self.O_M = np.nan
            self.interp_param = ''

    def get_param_names(self):
        params = super(ModelSpectrum, self).get_param_names()
        params = params + ['T_eff',
                           'log_g',
                           'Fe_H',
                           'a_Fe',
                           'N_He',
                           'v_turb',
                           'L_H',
                           'C_M',
                           'O_M',
                           'interp_param']
        return params

    def normalize_by_T_eff(self, T_eff=None):
        T_eff = T_eff or self.T_eff
        logging.debug('Normalizing spectrum with black-body of T_eff={}'.format(T_eff))
        n = 1e-7 * Physics.planck(self.wave*1e-10, T_eff)
        self.multiply(1 / n)

    def print_info(self):
        # TODO: call super
        print('T_eff=', self.T_eff)
        print('log g=', self.log_g)
        print('[M/H]=', self.Fe_H)
        print('[a/Fe]=', self.a_Fe)
        print('N(He)=', self.N_He)
        print('v_turb=', self.v_turb)
        print('L/H=', self.L_H)
        print('[C/M]=', self.C_M)
        print('[O/M]=', self.O_M)