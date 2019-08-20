import numpy as np
from scipy.interpolate import RegularGridInterpolator
import itertools

from pfsspec.stellarmod.modelparam import ModelParam

class ModelGrid():
    def __init__(self):
        self.params = {
            'Fe_H': None,
            'T_eff': None,
            'log_g': None
        }
        self.wave = None
        self.flux = None
        self.flux_idx = None

    def init_storage(self, wave):
        shape = [self.params[p].values.shape[0] for p in self.params]
        shape.append(wave.shape[0])
        self.wave = wave
        self.flux = np.empty(shape)

    def build_index(self):
        for p in self.params:
            self.params[p].build_index()
        axis = len(self.params)
        if self.flux is not None:
            self.flux_idx = (self.flux.max(axis=axis) != 0) | (self.flux.min(axis=axis) != 0)

    def set_flux(self, flux, **kwargs):
        """
        Sets the flux at a given point of the grid

        Parameters must exactly match grid coordinates.
        """
        idx = [self.params[p].index[kwargs[p]] for p in self.params]
        idx.append(slice(None, None, 1))
        idx = tuple(idx)
        self.flux[idx] = flux

    def save(self, filename):
        params = {p: self.params[p].values for p in self.params}
        np.savez(filename,
                 **params,
                 wave=self.wave, flux=self.flux)

    def load(self, filename):
        data = np.load(filename)
        for p in self.params:
            self.params[p] = ModelParam(p, data[p])
        self.wave = data['wave']
        self.flux = data['flux']
        self.build_index()

    def create_spectrum(self):
        raise NotImplementedError()

    def get_nearest_index(self, **kwargs):
        idx = tuple(self.params[p].get_nearest_index(kwargs[p]) for p in self.params)
        return idx

    def get_nearby_indexes(self, **kwargs):
        idx1 = list(self.get_nearest_index(**kwargs))
        idx2 = list((0, ) * len(idx1))

        i = 0
        for p in self.params:
            if kwargs[p] < self.params[p].values[idx1[i]]:
                idx1[i], idx2[i] = idx1[i] - 1, idx1[i]
            else:
                idx1[i], idx2[i] = idx1[i], idx1[i] + 1

            # Verify if indexes are inside bounds
            if idx1[i] < 0 or idx2[i] < 0 or \
               idx1[i] >= self.params[p].values.shape[0] or \
               idx2[i] >= self.params[p].values.shape[0]:
                return None

            i += 1

        idx1 = tuple(idx1)
        idx2 = tuple(idx2)

        # Verify if model exists
        # Here we don't assume that there are holes in the grid
        # but check if we're outside of covered ranges
        if not (self.flux_idx[idx1] and self.flux_idx[idx2]):
            return None

        return idx1, idx2

    def get_model(self, idx):
        spec = self.create_spectrum()
        i = 0
        for i, p in enumerate(self.params):
            setattr(spec, p, self.params[p].values[idx[i]])
        idx = list(idx)
        idx.append(slice(None, None, 1))
        idx = tuple(idx)
        spec.wave = np.array(self.wave, copy=True)
        spec.flux = np.array(self.flux[idx], copy=True)

        return spec

    def get_nearest_model(self, **kwargs):
        """
        Finds grid point closest to the parameters specified
        """
        idx = self.get_nearest_index(**kwargs)
        spec = self.get_model(idx)
        return spec

    def interpolate_model(self, **kwargs):
        idx = self.get_nearby_indexes(**kwargs)
        if idx is None:
            return None
        else:
            idx1, idx2 = idx

        # Parameter values to interpolate between
        x = tuple([[self.params[p].values[idx1[i]], self.params[p].values[idx2[i]]] for i, p in enumerate(self.params)])

        # Will hold flux values
        s = [2, ] * len(x)
        s.append(self.wave.shape[0])
        V = np.empty(s)

        ii = tuple(np.array(tuple(itertools.product(*([[0, 1],] * len(x))))).transpose())
        kk = tuple(np.array(tuple(itertools.product(*[[idx1[i], idx2[i]] for i in range(3)]))).transpose())

        V[ii] = self.flux[kk]

        #V = np.empty((2, 2, 2, self.wave.shape[0]))

        #ii = list(itertools.product(*([[0, 1],] * len(x))))
        #ii = np.array(ii)
        #ii.append(slice(None, None, 1))

        #V[ii] = self.flux_idx[ii]

        #xx = list(itertools.product(*x))

        #i = 0
        #for ii in (i1, i2):
        #    j = 0
        #    for jj in (j1, j2):
        #        k = 0
        #        for kk in (k1, k2):
        #            V[i, j, k] = self.flux[ii, jj, kk, :]
        #            k += 1
        #        j += 1
        #    i += 1

        fn = RegularGridInterpolator(x, V)

        #fn = RegularGridInterpolator((x, y, z), V)

        spec = self.create_spectrum()
        for p in self.params:
            setattr(spec, p, kwargs[p])

        spec.wave = self.wave
        spec.flux = fn(tuple([kwargs[p] for p in self.params]))

        return spec
