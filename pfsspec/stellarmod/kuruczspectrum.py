from pfsspec.stellarmod.modelspectrum import ModelSpectrum

class KuruczSpectrum(ModelSpectrum):
    def __init__(self):
        super(ModelSpectrum, self).__init__()

    def print_info(self):
        print('T_eff=', self.T_eff)
        print('log g=', self.log_g)
        print('[M/H]=', self.M_H)
        print('alpha=', self.alpha)
        print('N(He)=', self.N_He)
        print('v_turb=', self.v_turb)
        print('L/H=', self.L_H)