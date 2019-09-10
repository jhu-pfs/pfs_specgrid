import logging
import itertools
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pfsspec.stellarmod.modelparam import ModelParam

class ModelGrid():
    def __init__(self, use_cont=False):
        self.use_cont = use_cont
        self.params = {
            'Fe_H': None,
            'T_eff': None,
            'log_g': None
        }
        self.wave = None
        self.cont = None
        self.flux = None
        self.flux_idx = None

    def init_storage(self, wave):
        shape = [self.params[p].values.shape[0] for p in self.params]
        shape.append(wave.shape[0])
        self.wave = wave
        self.flux = np.empty(shape)
        if self.use_cont:
            self.cont = np.empty(shape)

    def build_index(self):
        for p in self.params:
            self.params[p].build_index()
        axis = len(self.params)
        if self.flux is not None:
            self.flux_idx = (self.flux.max(axis=axis) != 0) | (self.flux.min(axis=axis) != 0)

    def set_flux(self, flux, cont=None, **kwargs):
        """
        Sets the flux at a given point of the grid

        Parameters must exactly match grid coordinates.
        """
        idx = [self.params[p].index[kwargs[p]] for p in self.params]
        idx.append(slice(None, None, 1))
        idx = tuple(idx)
        self.flux[idx] = flux
        if self.cont is not None and cont is not None:
            self.cont[idx] = cont

    def save(self, filename):
        params = {p: self.params[p].values for p in self.params}
        np.savez(filename,
                 **params,
                 wave=self.wave, flux=self.flux, cont=self.cont)

    def load(self, filename):
        data = np.load(filename)
        for p in self.params:
            self.params[p] = ModelParam(p, data[p])
        self.wave = data['wave']
        self.flux = data['flux']
        self.cont = data['cont']
        self.build_index()

        logging.info('Loaded model grid with shape {} containing {} valid spectra.'.format(self.flux.shape, np.sum(self.flux_idx)))

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
        if self.flux_idx[idx]:
            spec = self.create_spectrum()
            i = 0
            for i, p in enumerate(self.params):
                setattr(spec, p, self.params[p].values[idx[i]])
            idx = list(idx)
            idx.append(slice(None, None, 1))
            idx = tuple(idx)
            spec.wave = np.array(self.wave, copy=True)
            spec.flux = np.array(self.flux[idx], copy=True)
            if self.cont is not None:
                spec.cont = np.array(self.cont[idx], copy=True)
            return spec
        else:
            return None

    def get_nearest_model(self, **kwargs):
        """
        Finds grid point closest to the parameters specified
        """
        idx = self.get_nearest_index(**kwargs)
        spec = self.get_model(idx)
        return spec


    def float_spec(self, **kwargs, free_param_name):
        list_para = list(self.params.keys())
        free_idx = list_para.index(free_param_name)
        spec = list(self.get_nearest_index(**kwargs))
        spec[free_idx] = slice(None)
        spec = tuple(spec)
        #spectp: 5-dims tuple slice of 5-dims parameter space
        valid_flux_bool = self.flux_idx[spec]
        para_wave = self.flux[spec][valid_flux_bool]
        free_para_val = self.params[free_param_name].values[valid_flux_bool]
        return para_wave, free_para_val

    def interpolate_model_spline(self,para_wave, free_para_val):
        x,y = para_wave, free_para_val
        cs = CubicSpline(x,y)
        return cs

    def interpolate_model(self, **kwargs):
        # TODO: interpolate continuum, if available

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

        fn = RegularGridInterpolator(x, V)

        spec = self.create_spectrum()
        for p in self.params:
            setattr(spec, p, kwargs[p])

        spec.wave = self.wave
        spec.flux = fn(tuple([kwargs[p] for p in self.params]))

        return spec
