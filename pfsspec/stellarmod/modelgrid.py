import numpy as np

class ModelGrid():
    def __init__(self):
        self.M_H = None
        self.T_eff = None
        self.log_g = None
        self.M_H_idx = None
        self.T_eff_idx = None
        self.log_g_idx = None
        self.wave = None
        self.flux = None

    def init_storage(self, wave):
        shape = (self.M_H.shape[0], self.T_eff.shape[0], self.log_g.shape[0], wave.shape[0])
        self.wave = wave
        self.flux = np.empty(shape)

    def build_index(self):
        self.M_H_idx = dict((v, i) for i, v in np.ndenumerate(self.M_H))
        self.T_eff_idx = dict((v, i) for i, v in np.ndenumerate(self.T_eff))
        self.log_g_idx = dict((v, i) for i, v in np.ndenumerate(self.log_g))

    def set_flux(self, M_H, T_eff, log_g, flux):
        i = self.M_H_idx[M_H]
        j = self.T_eff_idx[T_eff]
        k = self.log_g_idx[log_g]
        self.flux[i, j, k, :] = flux