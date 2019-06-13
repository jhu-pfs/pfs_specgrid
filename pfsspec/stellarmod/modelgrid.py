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
        """
        Sets the flux at a given point of the grid

        Parameters must exactly match grid coordinates.

        Parameters
        ----------
        :param M_H: float
            Metallicity [M/H]
        :param T_eff: float
            Effective temperature
        :param log_g: float
            surface gravity
        """

        i = self.M_H_idx[M_H]
        j = self.T_eff_idx[T_eff]
        k = self.log_g_idx[log_g]
        self.flux[i, j, k, :] = flux

    def save(self, filename):
        np.savez(filename,
                 M_H=self.M_H, T_eff=self.T_eff, log_g=self.log_g,
                 wave=self.wave, flux=self.flux)

    def load(self, filename):
        data = np.load(filename)
        self.M_H = data['M_H']
        self.T_eff = data['T_eff']
        self.log_g = data['log_g']
        self.wave = data['wave']
        self.flux = data['flux']
        self.build_index()

    def create_spectrum(self):
        raise NotImplementedError()

    def get_nearest_model(self, M_H, T_eff, log_g):
        """
        Finds grid point closest to the parameters specified

        Parameters
        ----------
        :param M_H: float
            Metallicity [M/H]
        :param T_eff: float
            Effective temperature
        :param log_g: float
            surface gravity
        :return:
            Flux density of model
        """
        i = np.abs(self.M_H - M_H).argmin()
        j = np.abs(self.T_eff - T_eff).argmin()
        k = np.abs(self.log_g - log_g).argmin()

        spec = self.create_spectrum()
        spec.wave = self.wave
        spec.flux = self.flux[i, j, k, :]
        spec.M_H = self.M_H[i]
        spec.T_eff = self.T_eff[j]
        spec.log_g = self.log_g[k]

        return spec