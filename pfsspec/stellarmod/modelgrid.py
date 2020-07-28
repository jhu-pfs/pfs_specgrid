import logging
import itertools
import numpy as np
from random import choice
from scipy.interpolate import RegularGridInterpolator, CubicSpline

from pfsspec.data.grid import Grid

class ModelGrid(Grid):
    def __init__(self):
        super(ModelGrid, self).__init__()
        self.wave = None
        self.slice = None

    def add_args(self, parser):
        for k in self.params:
            parser.add_argument('--' + k, type=float, nargs='*', default=None, help='Limit on ' + k)

    def init_from_args(self, args):
        # If a limit is specified on any of the parameters on the command-line,
        # try to slice the grid while loading from HDF5
        s = []
        for k in self.params:
            if args[k] is not None:
                idx = np.digitize([args[k][0], args[k][1]], self.params[k].values)
                s.append(slice(idx[0], idx[1] + 1, None))
            else:
                s.append(slice(None))
        s.append(slice(None))  # wave axis
        self.slice = tuple(s)

        # Override physical parameters grid ranges, if specified
        # TODO: extend this to sample physically meaningful models only
        for k in self.params:
            if args[k] is not None and len(args[k]) >= 2:
                self.params[k].min = args[k][0]
                self.params[k].max = args[k][1]

    def get_model_count(self, use_limits=False):
        return self.get_valid_data_item_count('flux', use_limits=use_limits)

    def get_flux_shape(self):
        return self.get_shape() + self.wave.shape

    def init_params(self):
        self.init_param('Fe_H')
        self.init_param('T_eff')
        self.init_param('log_g')

    def init_data(self):
        self.init_data_item('flux')
        self.init_data_item('cont')

    def allocate_data(self):
        self.allocate_data_item('flux', self.wave.shape)
        self.allocate_data_item('cont', self.wave.shape)

    def is_data_valid(self, name, data):
        return np.logical_not(np.any(np.isnan(data), axis=-1)) & ((data.max(axis=-1) != 0) | (data.min(axis=-1) != 0))

    def save_items(self):
        self.save_item('wave', self.wave)
        super(ModelGrid, self).save_items()

    def load(self, filename, s=None, format=None):
        s = s or self.slice
        super(ModelGrid, self).load(filename, s=s, format=format)

    def load_items(self, s=None):
        self.wave = self.load_item('wave', np.ndarray)
        self.init_data()
        super(ModelGrid, self).load_items(s=s)

###

    def set_flux(self, flux, cont=None, **kwargs):
        """
        Sets the flux at a given point of the grid

        Parameters must exactly match grid coordinates.
        """
        self.set_data_item('flux', flux, **kwargs)
        if cont is not None:
            self.set_data_item('cont', cont, **kwargs)

    def set_flux_idx(self, index, flux, cont=None):
        self.set_data_item_idx('flux', index, flux)
        if cont is not None:
            self.set_data_item_idx('cont', index, cont)

    def create_spectrum(self):
        raise NotImplementedError()

    def get_parameterized_spectrum(self, idx=None, s=None, **kwargs):
        spec = self.create_spectrum()
        self.set_object_params(spec, idx=idx, **kwargs)
        spec.wave = self.wave[s or slice(None)]
        return spec

    def get_model(self, idx):
        if self.is_data_item_idx('flux', idx):
            spec = self.get_parameterized_spectrum(idx)
            spec.flux = np.array(self.get_data_item_idx('flux', idx), copy=True)
            if self.is_data_item('cont'):
                spec.cont = np.array(self.get_data_item_idx('cont', idx), copy=True)

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
        r = self.interpolate_data_item_linear('flux', **kwargs)
        if r is None:
            return None
        flux, kwargs = r

        if flux is not None:
            spec = self.get_parameterized_spectrum(**kwargs)
            spec.flux = flux
            if self.is_data_item('cont'):
                spec.cont = self.interpolate_data_item_linear('cont', **kwargs)
            return spec
        else:
            return None

    def interpolate_model_spline(self, free_param, **kwargs):
        r = self.interpolate_data_item_spline('flux', free_param, **kwargs)
        if r is None:
            return None
        flux, kwargs = r

        if flux is not None:
            spec = self.get_parameterized_spectrum(**kwargs)
            spec.interp_param = free_param
            spec.flux = flux
            if self.is_data_item('cont'):
                spec.cont = self.interpolate_data_item_spline('cont', free_param, **kwargs)
            return spec
        else:
            return None

