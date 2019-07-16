import numpy as np
from scipy.interpolate import RegularGridInterpolator

class ModelGrid():
    def __init__(self):
        # TODO: figure out how to have an arbitrary number of grid dimensions
        self.Fe_H = None
        self.T_eff = None
        self.log_g = None
        self.Fe_H_idx = None
        self.T_eff_idx = None
        self.log_g_idx = None
        self.Fe_H_min = None
        self.Fe_H_max = None
        self.T_eff_min = None
        self.T_eff_max = None
        self.log_g_min = None
        self.log_g_max = None
        self.wave = None
        self.flux = None
        self.flux_idx = None

    def init_storage(self, wave):
        shape = (self.Fe_H.shape[0], self.T_eff.shape[0], self.log_g.shape[0], wave.shape[0])
        self.wave = wave
        self.flux = np.empty(shape)

    def build_index(self):
        self.Fe_H_idx = dict((v, i) for i, v in np.ndenumerate(self.Fe_H))
        self.Fe_H_min, self.Fe_H_max = np.min(self.Fe_H), np.max(self.Fe_H)
        self.T_eff_idx = dict((v, i) for i, v in np.ndenumerate(self.T_eff))
        self.T_eff_min, self.T_eff_max = np.min(self.T_eff), np.max(self.T_eff)
        self.log_g_idx = dict((v, i) for i, v in np.ndenumerate(self.log_g))
        self.log_g_min, self.log_g_max = np.min(self.log_g), np.max(self.log_g)

        if self.flux is not None:
            self.flux_idx = (self.flux.max(axis=3) != 0) | (self.flux.min(axis=3) != 0)

    def set_flux(self, Fe_H, T_eff, log_g, flux):
        """
        Sets the flux at a given point of the grid

        Parameters must exactly match grid coordinates.

        Parameters
        ----------
        :param Fe_H: float
            Metallicity [M/H]
        :param T_eff: float
            Effective temperature
        :param log_g: float
            surface gravity
        """

        i = self.Fe_H_idx[Fe_H]
        j = self.T_eff_idx[T_eff]
        k = self.log_g_idx[log_g]
        self.flux[i, j, k, :] = flux

    def save(self, filename):
        np.savez(filename,
                 Fe_H=self.Fe_H, T_eff=self.T_eff, log_g=self.log_g,
                 wave=self.wave, flux=self.flux)

    def load(self, filename):
        data = np.load(filename)
        self.Fe_H = data['Fe_H']
        self.T_eff = data['T_eff']
        self.log_g = data['log_g']
        self.wave = data['wave']
        self.flux = data['flux']
        self.build_index()

    def create_spectrum(self):
        raise NotImplementedError()

    def get_nearest_index(self, Fe_H, T_eff, log_g):
        i = np.abs(self.Fe_H - Fe_H).argmin()
        j = np.abs(self.T_eff - T_eff).argmin()
        k = np.abs(self.log_g - log_g).argmin()
        return i, j, k

    def get_nearby_indexes(self, Fe_H, T_eff, log_g):
        i1, j1, k1 = self.get_nearest_index(Fe_H, T_eff, log_g)

        if Fe_H < self.Fe_H[i1]:
            i1, i2 = i1 - 1, i1
        else:
            i1, i2 = i1, i1 + 1

        if T_eff < self.T_eff[j1]:
            j1, j2 = j1 - 1, j1
        else:
            j1, j2 = j1, j1 + 1

        if log_g < self.log_g[k1]:
            k1, k2 = k1 - 1, k1
        else:
            k1, k2 = k1, k1 + 1

        # Verify if indexes inside bounds
        if i1 < 0 or j1 < 0 or k1 < 0 or \
                i2 >= self.Fe_H.shape[0] or \
                j2 >= self.T_eff.shape[0] or \
                k2 >= self.log_g.shape[0]:
            return None

        # Verify if model exists
        # Here we don't assume that there are holes in the grid
        # but check if we're outside of covered ranges
        if not (self.flux_idx[i1, j1, k1] and self.flux_idx[i2, j1, k1]):
            return None

        return i1, j1, k1, i2, j2, k2

    def get_model(self, i, j, k):
        spec = self.create_spectrum()
        spec.Fe_H = self.Fe_H[i]
        spec.T_eff = self.T_eff[j]
        spec.log_g = self.log_g[k]

        spec.wave = np.array(self.wave, copy=True)
        spec.flux = np.array(self.flux[i, j, k, :], copy=True)

        return spec

    def get_nearest_model(self, Fe_H, T_eff, log_g):
        """
        Finds grid point closest to the parameters specified

        Parameters
        ----------
        :param Fe_H: float
            Metallicity [M/H]
        :param T_eff: float
            Effective temperature
        :param log_g: float
            surface gravity
        :return:
            Flux density of model
        """
        i, j, k = self.get_nearest_index(Fe_H, T_eff, log_g)
        spec = self.get_model(i, j, k)
        return spec

    def interpolate_model(self, Fe_H, T_eff, log_g):
        idx = self.get_nearby_indexes(Fe_H, T_eff, log_g)
        if idx is None:
            return None
        else:
            i1, j1, k1, i2, j2, k2 = idx

        x = [self.Fe_H[i1], self.Fe_H[i2]]
        y = [self.T_eff[j1], self.T_eff[j2]]
        z = [self.log_g[k1], self.log_g[k2]]
        V = np.empty((2, 2, 2, self.wave.shape[0]))

        i = 0
        for ii in (i1, i2):
            j = 0
            for jj in (j1, j2):
                k = 0
                for kk in (k1, k2):
                    V[i, j, k] = self.flux[ii, jj, kk, :]
                    k += 1
                j += 1
            i += 1

        fn = RegularGridInterpolator((x, y, z), V)

        spec = self.create_spectrum()
        spec.Fe_H = Fe_H
        spec.T_eff = T_eff
        spec.log_g = log_g

        spec.wave = self.wave
        spec.flux = fn((Fe_H, T_eff, log_g))

        return spec
