import logging
import itertools
import numpy as np
from random import choice
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.interpolate import interp1d, interpn

from pfsspec.data.arraygrid import ArrayGrid
from pfsspec.data.gridaxis import GridAxis

class ModelArrayGrid(ArrayGrid):
    def __init__(self, config, orig=None):
        if isinstance(orig, ModelArrayGrid):
            self.config = config if config is not None else orig.config
            self.wave = orig.wave
            self.slice = orig.slice
        else:
            self.config = config
            self.wave = None
            self.slice = None

        super(ModelArrayGrid, self).__init__(orig=orig)

    def init_axes(self):
        self.config.init_axes(self)

    def init_constants(self):
        self.config.init_constants(self)

    def init_values(self):
        self.config.init_values(self)

    def allocate_values(self):
        self.config.allocate_values(self, self.wave)

    def is_value_valid(self, name, value):
        self.config.is_value_valid(name, value)

    def add_args(self, parser):
        parser.add_argument('--lambda', type=float, nargs='*', default=None, help='Limit on lambda.')
        for k in self.axes:
            parser.add_argument('--' + k, type=float, nargs='*', default=None, help='Limit on ' + k)

    def init_from_args(self, args):
        self.slice = self.get_slice_from_args(args)

        # Override physical parameters grid ranges, if specified
        # TODO: extend this to sample physically meaningful models only
        for k in self.axes:
            if k in args and args[k] is not None:
                if len(args[k]) >= 2:
                    self.axes[k].min = args[k][0]
                    self.axes[k].max = args[k][1]
                else:
                    self.axes[k].min = args[k][0]
                    self.axes[k].max = args[k][0]

    def get_slice_from_args(self, args):
        # If a limit is specified on any of the parameters on the command-line,
        # try to slice the grid while loading from HDF5
        s = []
        for k in self.axes:
            if k in args and args[k] is not None:
                if len(args[k]) == 2:
                    idx = np.digitize([args[k][0], args[k][1]], self.axes[k].values)
                    s.append(slice(max(0, idx[0] - 1), idx[1], None))
                elif len(args[k]) == 1:
                    idx = np.digitize([args[k][0]], self.axes[k].values)
                    s.append(max(0, idx[0] - 1))
                else:
                    raise Exception('Only two or one values are allowed for parameter {}'.format(k))
            else:
                s.append(slice(None))

        # Wave axis
        if 'lambda' in args and args['lambda'] is not None:
            if len(args['lambda']) == 2:
                idx = np.digitize([args['lambda'][0], args['lambda'][1]], self.wave)
                s.append(slice(max(0, idx[0] - 1), idx[1], None))
            elif len(args['lambda']) == 1:
                idx = np.digitize([args['lambda'][0]], self.wave)
                s.append(max(0, idx[0] - 1))
            else:
                raise Exception('Only two or one values are allowed for parameter lambda.')
        else:
            s.append(slice(None))

        return tuple(s)

    def get_axes(self):
        # Return axes that are limited by the slices
        if self.slice is not None:
            axes = {}
            for i, k in enumerate(self.axes):
                if type(self.slice[i]) is slice:
                    axes[k] = GridAxis(k, self.axes[k].values[self.slice[i]])
            return axes
        else:
            return self.axes

    def get_shape(self):
        if self.slice is not None:
            ss = []
            for i, k in enumerate(self.axes):
                if type(self.slice[i]) is slice:
                    s = self.axes[k].values[self.slice[i]].shape[0]
                    ss.append(s)
            return tuple(ss)
        else:
            return super(ModelArrayGrid, self).get_shape()

    def get_value_sliced(self, name):
        if self.slice is not None:
            if name in ['flux', 'cont']:
                # Slice in the wavelength direction as well
                return self.get_value_at(name, idx=self.slice[:-1], s=self.slice[-1])
            else:
                # Slice only in the stellar parameter directions
                return self.get_value_at(name, idx=self.slice[:-1])
        else:
            return self.get_value_at(name, idx=None)

    def get_value_padded(self, name, interpolation='ijk'):
        """Returns a slice of the grid and pads with a single item in every direction using linearNd extrapolation.

        Extrapolation is done either in grid coordinates or in axis coordinates

        Args:
            name (str): Name of value array
            s (slice, optional): Slice to apply to value array. Defaults to None.
            interpolation: Whether to extrapolate based on array indices ('ijk', default)
                or axis coordinates ('xyz').
            **kwargs: Values of axis coordinates. Only exact values are supported. For
                missing direction, full, padded slices will be returned.
        """

        # If slicing is turned on, these functions will automatically return the sliced
        # value array and the sliced (and squeezed) axes.
        orig_axes = self.get_axes()
        orig_value = self.get_value_sliced(name)

        return ArrayGrid.pad_array(orig_axes, orig_value)
        
    def get_sliced_value_index(self, name):
        # Return a boolean index that is limited by the axis bound overrides.
        # The shape will be the same as the original array. Use this index to
        # load a limited subset of the data directly from the disk.
        if self.slice is not None:
            index = np.full(self.value_indexes[name].shape, False)
            index[self.slice[:-1]] = self.value_indexes[name][self.slice[:-1]]
            return index
        else:
            return self.value_indexes[name]

    def get_valid_value_count(self, name):
        if self.slice is not None:
            return np.sum(self.get_sliced_value_index(name))
        else:
            return super(ModelArrayGrid, self).get_valid_value_count(name)

    def get_model_count(self, use_limits=False):
        return self.get_valid_value_count('flux')

    def get_wave(self):
        if self.slice is not None:
            return self.wave[self.slice[-1]]
        else:
            return self.wave

    def set_wave(self, wave):
        self.wave = wave

    def get_flux_shape(self):
        return self.get_shape() + self.wave.shape

    def get_value_index(self, name):
        if self.has_value_index(name):
            index = self.value_indexes[name]
            if self.slice is not None:
                return index[self.slice[:-1]]
            else:
                return index
        else:
            return None

    def is_value_valid(self, name, value):
        return np.logical_not(np.any(np.isnan(value), axis=-1)) & ((value.max(axis=-1) != 0) | (value.min(axis=-1) != 0))

    def save_items(self):
        self.save_item('wave', self.wave)
        super(ModelArrayGrid, self).save_items()

    def load(self, filename, s=None, format=None):
        s = s or self.slice
        super(ModelArrayGrid, self).load(filename, s=s, format=format)

    def load_items(self, s=None):
        self.wave = self.load_item('wave', np.ndarray)
        self.init_values()
        super(ModelArrayGrid, self).load_items(s=s)

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

    def get_parameterized_spectrum(self, idx=None, s=None, **kwargs):
        spec = self.config.create_spectrum()
        self.set_object_params(spec, idx=idx, **kwargs)
        spec.wave = self.wave[s or slice(None)]
        return spec

    def get_model_at(self, idx):
        # Slice along lambda if requested
        s = self.slice[-1] if self.slice is not None else None
        if self.has_value_at('flux', idx):
            spec = self.get_parameterized_spectrum(idx, s=s)
            spec.flux = np.array(self.get_value_at('flux', idx, s=s), copy=True)
            if self.has_value('cont'):
                spec.cont = np.array(self.get_value_at('cont', idx, s=s), copy=True)
            return spec
        else:
            return None

    def get_nearest_model(self, **kwargs):
        """
        Finds grid point closest to the parameters specified
        """
        idx = self.get_nearest_index(**kwargs)
        spec = self.get_model_at(idx)
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

#region RBF interpolation

    # TODO: is it used for anything?
    def get_slice_rbf(self, s=None, interpolation='xyz', padding=True, **kwargs):
        # Interpolate the continuum and flux in a wavelength slice `s` and parameter
        # slices defined by kwargs using RBF. The input RBF is padded with linearly extrapolated
        # values to make the interpolation smooth

        if padding:
            flux, axes = self.get_value_padded('flux', s=s, interpolation=interpolation, **kwargs)
            cont, axes = self.get_value_padded('cont', s=s, interpolation=interpolation, **kwargs)
        else:
            flux = self.get_value('flux', s=s, **kwargs)
            cont = self.get_value('cont', s=s, **kwargs)

            axes = {}
            for p in self.axes.keys():
                if p not in kwargs:            
                    if interpolation == 'ijk':
                        axes[p] = GridAxis(p, np.arange(self.axes[p].values.shape[0], dtype=np.float64))
                    elif interpolation == 'xyz':
                        axes[p] = self.axes[p]

        # Max nans and where the continuum is zero
        mask = ~np.isnan(cont) & (cont != 0)
        if mask.ndim > len(axes):
            mask = np.all(mask, axis=-(mask.ndim - len(axes)))

        # Rbf must be generated on a uniform grid
        if padding:
            aa = {p: GridAxis(p, np.arange(axes[p].values.shape[0]) - 1.0) for p in axes}
        else:
            aa = {p: GridAxis(p, np.arange(axes[p].values.shape[0])) for p in axes}

        rbf_flux = self.fit_rbf(flux, aa, mask=mask)
        rbf_cont = self.fit_rbf(cont, aa, mask=mask)

        return rbf_flux, rbf_cont, axes

#endregion