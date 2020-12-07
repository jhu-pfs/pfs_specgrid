import logging
import itertools
import numpy as np
from random import choice
from scipy.interpolate import RegularGridInterpolator, CubicSpline

from pfsspec.data.grid import Grid
from pfsspec.data.gridaxis import GridAxis

class ModelGrid(Grid):
    def __init__(self):
        super(ModelGrid, self).__init__()
        self.wave = None
        self.slice = None

    def add_args(self, parser):
        for k in self.axes:
            parser.add_argument('--' + k, type=float, nargs='*', default=None, help='Limit on ' + k)

    def init_from_args(self, args):
        # If a limit is specified on any of the parameters on the command-line,
        # try to slice the grid while loading from HDF5
        s = []
        for k in self.axes:
            if args[k] is not None:
                idx = np.digitize([args[k][0], args[k][1]], self.axes[k].values)
                s.append(slice(idx[0], idx[1] + 1, None))
            else:
                s.append(slice(None))
        s.append(slice(None))  # wave axis
        self.slice = tuple(s)

        # Override physical parameters grid ranges, if specified
        # TODO: extend this to sample physically meaningful models only
        for k in self.axes:
            if args[k] is not None and len(args[k]) >= 2:
                self.axes[k].min = args[k][0]
                self.axes[k].max = args[k][1]

    def get_model_count(self, use_limits=False):
        return self.get_valid_value_count('flux', use_limits=use_limits)

    def get_flux_shape(self):
        return self.get_shape() + self.wave.shape

    def init_axes(self):
        self.init_axis('Fe_H')
        self.init_axis('T_eff')
        self.init_axis('log_g')

    def init_values(self):
        self.init_value('flux')
        self.init_value('cont')

    def allocate_values(self):
        self.allocate_value('flux', self.wave.shape)
        self.allocate_value('cont', self.wave.shape)

    def is_value_valid(self, name, value):
        return np.logical_not(np.any(np.isnan(value), axis=-1)) & ((value.max(axis=-1) != 0) | (value.min(axis=-1) != 0))

    def save_items(self):
        self.save_item('wave', self.wave)
        super(ModelGrid, self).save_items()

    def load(self, filename, s=None, format=None):
        s = s or self.slice
        super(ModelGrid, self).load(filename, s=s, format=format)

    def load_items(self, s=None):
        self.wave = self.load_item('wave', np.ndarray)
        self.init_values()
        super(ModelGrid, self).load_items(s=s)

    def get_chunks(self, name, shape, s=None):
        # The chunking strategy for spectrum grids should observe the following
        # - we often need only parts of the wavelength coverage
        # - interpolation algorithms iterate over the wavelengths in the outer loop
        # - interpolation algorithms need nearby models, cubic splines require models
        #   in memory along the entire interpolation axis

        # The shape of the spectrum grid is (param1, param2, wave)
        if name in self.values:
            newshape = []
            # Keep neighboring 3 models together in every direction
            for i, k in enumerate(self.axes.keys()):
                if k in ['log_g', 'Fe_H', 'T_eff']:
                    newshape.append(min(shape[i], 3))
                else:
                    newshape.append(1)
            # Use small chunks along the wavelength direction
            newshape.append(min(128, shape[-1]))
            return tuple(newshape)
        else:
            return None

###

    def set_flux(self, flux, cont=None, **kwargs):
        """
        Sets the flux at a given point of the grid

        Parameters must exactly match grid coordinates.
        """
        self.set_value('flux', flux, **kwargs)
        if cont is not None:
            self.set_value('cont', cont, **kwargs)

    def set_flux_at(self, index, flux, cont=None):
        self.set_value_at('flux', index, flux)
        if cont is not None:
            self.set_value_at('cont', index, cont)

    def create_spectrum(self):
        raise NotImplementedError()

    def get_parameterized_spectrum(self, idx=None, s=None, **kwargs):
        spec = self.create_spectrum()
        self.set_object_params(spec, idx=idx, **kwargs)
        spec.wave = self.wave[s or slice(None)]
        return spec

    def get_model(self, idx):
        if self.has_value_at('flux', idx):
            spec = self.get_parameterized_spectrum(idx)
            spec.flux = np.array(self.get_value_at('flux', idx), copy=True)
            if self.has_value('cont'):
                spec.cont = np.array(self.get_value_at('cont', idx), copy=True)

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

    def interpolate_model_linear(self, **kwargs):
        r = self.interpolate_value_linear('flux', **kwargs)
        if r is None:
            return None
        flux, kwargs = r

        if flux is not None:
            spec = self.get_parameterized_spectrum(**kwargs)
            spec.flux = flux
            if self.has_value('cont'):
                spec.cont = self.interpolate_value_linear('cont', **kwargs)
            return spec
        else:
            return None

    def interpolate_model_spline(self, free_param, **kwargs):
        r = self.interpolate_value_spline('flux', free_param, **kwargs)
        if r is None:
            return None
        flux, bestargs = r

        if flux is not None:
            spec = self.get_parameterized_spectrum(**bestargs)
            spec.interp_param = free_param
            spec.flux = flux
            if self.has_value('cont'):
                spec.cont, _ = self.interpolate_value_spline('cont', free_param, **kwargs)
            return spec
        else:
            return None

    def get_slice_rbf(self, s=None, extrapolation='xyz', **kwargs):
        # Interpolate the continuum and flux in a wavelength slice `s` and parameter
        # slices defined by kwargs using RBF. The input RBF is padded with linearly extrapolated
        # values to make the interpolation smooth

        flux, axes = self.get_value_padded('flux', s=s, extrapolation=extrapolation, **kwargs)
        cont, axes = self.get_value_padded('cont', s=s, extrapolation=extrapolation, **kwargs)

        # Max nans and where the continuum is zero
        mask = ~np.isnan(cont) & (cont != 0)
        if mask.ndim > len(axes):
            mask = np.all(mask, axis=-(mask.ndim - len(axes)))

        # Rbf must be generated on a uniform grid
        aa = {p: GridAxis(p, np.arange(axes[p].values.shape[0]) - 1.0) for p in axes}

        rbf_flux = self.interpolate_value_rbf(flux, aa, mask=mask)
        rbf_cont = self.interpolate_value_rbf(cont, aa, mask=mask)

        return rbf_flux, rbf_cont, axes