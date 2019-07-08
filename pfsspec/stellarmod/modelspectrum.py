from pfsspec.spectrum import Spectrum

class ModelSpectrum(Spectrum):
    def __init__(self):
        super(ModelSpectrum, self).__init__()
        self.T_eff = None
        self.log_g = None
        self.M_H = None
        self.alpha = None
        self.N_He = None
        self.v_turb = None
        self.L_H = None

    def print_info(self):
        print('T_eff=', self.T_eff)
        print('log g=', self.log_g)
        print('[M/H]=', self.M_H)
        print('alpha=', self.alpha)
        print('N(He)=', self.N_He)
        print('v_turb=', self.v_turb)
        print('L/H=', self.L_H)