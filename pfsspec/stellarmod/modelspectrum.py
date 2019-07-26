from pfsspec.obsmod.spectrum import Spectrum

class ModelSpectrum(Spectrum):
    def __init__(self):
        super(ModelSpectrum, self).__init__()
        self.T_eff = None
        self.log_g = None
        self.Fe_H = None
        self.a_Fe = None
        self.N_He = None
        self.v_turb = None
        self.L_H = None

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