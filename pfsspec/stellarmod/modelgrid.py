import logging
import itertools
import numpy as np
from random import choice
from scipy.interpolate import RegularGridInterpolator, CubicSpline

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
        if 'cont' in data:
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

    def get_parameterized_spec(self, fn, **kwargs):
        spec = self.create_spectrum()
        for p in self.params:
            setattr(spec, p, kwargs[p])
        spec.wave = self.wave
        return spec

    def interpolate_model_spline(self, free_param_name, **kwargs):
        list_para = list(self.params.keys())
        if free_param_name not in list_para:
            free_param_name = choice(list_para)
        else:
            free_idx = list_para.index(free_param_name)

        # Find nearest model to requested parameters
        idx = list(self.get_nearest_index(**kwargs))
        if idx is None:
            return None

        # Set all params to nearest value except the one in which we interpolate
        for i, p in enumerate(self.params):
            if p != free_param_name:
                kwargs[p] = self.params[p].values[idx[i]]

        # Determine index of models
        idx[free_idx] = slice(None)
        idx = tuple(idx)

        # Find index of models that actually exists
        valid_flux = self.flux_idx[idx]
        pars = self.params[free_param_name].values[valid_flux]
        flux = self.flux[idx][valid_flux]

        # If we are at the edge of the grid, it might happen that we try to
        # interpolate over zero valid parameters, in this case return None and
        # the calling code will generate another set of random parameters
        if pars.shape[0] < 1:
            return None

        # Do as many parallel cubic spline interpolations as many wavelength bins we have
        x, y = pars, flux
        fn = CubicSpline(x, y)
        spec = self.get_parameterized_spec(fn, **kwargs)
        spec.flux = fn(kwargs[free_param_name])

        return spec

    def interpolate_model_linear(self, **kwargs):
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
        spec = self.get_parameterized_spec(fn, **kwargs)
        spec.flux = fn(tuple([kwargs[p] for p in self.params]))

        return spec
