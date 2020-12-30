import logging
import numpy as np
import itertools
from collections import Iterable
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.interpolate import interp1d, interpn

from pfsspec.rbf import Rbf
from pfsspec.pfsobject import PfsObject
from pfsspec.data.grid import Grid
from pfsspec.data.gridaxis import GridAxis

class ArrayGrid(Grid):
    """Implements a generic grid class to store and interpolate data.

    This class implements a generic grid class with an arbitrary number of axes
    and multiple value arrays in each grid point. Value arrays must have the same
    leading dimensions as the size of the axes. An index is build on every
    value array which indicated valid/invalid data in a particular grid point.

    Args:
        PfsObject ([type]): [description]
    """
    def __init__(self, orig=None):
        super(ArrayGrid, self).__init__(orig=orig)

        if isinstance(orig, ArrayGrid):
            self.values = orig.values
            self.value_shapes = orig.value_shapes
            self.value_indexes = orig.value_indexes
        else:
            self.values = {}
            self.value_shapes = {}
            self.value_indexes = {}

            self.init_values()

    def get_value_shape(self, name):
        # Gets the full shape of the grid. It assumes different last
        # dimensions for each data array.
        shape = self.get_shape() + self.value_shapes[name]
        return shape

    def init_value(self, name, shape=None):
        if shape is None:
            self.values[name] = None
            self.value_shapes[name] = None
            self.value_indexes[name] = None
        else:
            if len(shape) != 1:
                raise NotImplementedError()

            gridshape = self.get_shape()
            valueshape = gridshape + tuple(shape)

            self.value_shapes[name] = shape

            if self.preload_arrays:
                self.logger.info('Initializing memory for grid "{}" of size {}...'.format(name, valueshape))
                self.values[name] = np.full(valueshape, np.nan)
                self.value_indexes[name] = np.full(gridshape, False, dtype=np.bool)
                self.logger.info('Initialized memory for grid "{}" of size {}.'.format(name, valueshape))
            else:
                self.values[name] = None
                self.logger.info('Initializing data file for grid "{}" of size {}...'.format(name, valueshape))
                if not self.has_item(name):
                    self.allocate_item(name, valueshape, dtype=np.float)
                    self.allocate_item(name + '_idx', gridshape, dtype=np.bool)
                self.logger.info('Skipped memory initialization for grid "{}". Will read random slices from storage.'.format(name))

            self.value_indexes[name] = None

    def allocate_value(self, name, shape=None):
        if shape is not None:
            self.value_shapes[name] = shape
        self.init_value(name, self.value_shapes[name])

    def is_value_valid(self, name, value):
        return np.logical_not(np.any(np.isnan(value), axis=-1))

    def build_value_indexes(self, rebuild=False):
        for name in self.values:
            self.build_value_index(name, rebuild=rebuild)

    def build_value_index(self, name, rebuild=False):
        if rebuild and not self.preload_arrays:
            raise Exception('Cannot build index on lazy-loaded grid.')
        elif rebuild or name not in self.value_indexes or self.value_indexes[name] is None:
            self.logger.debug('Building indexes on grid "{}" of size {}'.format(name, self.value_shapes[name]))
            self.value_indexes[name] = self.is_value_valid(name, self.values[name])
        else:
            self.logger.debug('Skipped building indexes on grid "{}" of size {}'.format(name, self.value_shapes[name]))
        self.logger.debug('{} valid vectors in grid "{}" found'.format(np.sum(self.value_indexes[name]), name))

    def get_valid_value_count(self, name):
        return np.sum(self.value_indexes[name])
           
    def get_index(self, **kwargs):
        """Returns the indexes along all axes corresponding to the values specified.

        If an axis name is missing, a whole slice is returned.

        Returns:
            [type]: [description]
        """
        idx = ()
        for p in self.axes:
            if p in kwargs:
                idx += (self.axes[p].index[kwargs[p]],)
            else:
                idx += (slice(None),)
        return idx

    def get_nearest_index(self, **kwargs):
        idx = ()
        for p in self.axes:
            if p in kwargs:
                idx += (self.axes[p].get_nearest_index(kwargs[p]),)
            else:
                idx += (slice(None),)
        return idx

    def get_nearby_indexes(self, **kwargs):
        """Returns the indices bracketing the values specified. Used for linear interpolation.

        Returns:
            [type]: [description]
        """
        idx1 = list(self.get_nearest_index(**kwargs))
        idx2 = list((0, ) * len(idx1))

        i = 0
        for p in self.axes:
            if kwargs[p] < self.axes[p].values[idx1[i]]:
                idx1[i], idx2[i] = idx1[i] - 1, idx1[i]
            else:
                idx1[i], idx2[i] = idx1[i], idx1[i] + 1

            # Verify if indexes are inside bounds
            if idx1[i] < 0 or self.axes[p].values.shape[0] <= idx1[i] or \
               idx2[i] < 0 or self.axes[p].values.shape[0] <= idx2[i]:
                return None

            i += 1

        return tuple(idx1), tuple(idx2)

    def has_value_index(self, name):
        if self.preload_arrays:
            return self.value_indexes is not None and \
                   name in self.value_indexes and self.value_indexes[name] is not None
        else:
            return name in self.value_indexes and self.has_item(name + '_idx')

    def get_value_index(self, name):
        if self.has_value_index(name):
            return self.value_indexes[name]
        else:
            return None

    def has_value(self, name):
        if self.preload_arrays:
            return name in self.values and self.values[name] is not None and \
                   name in self.value_indexes and self.value_indexes[name] is not None
        else:
            return name in self.values and self.has_item(name) and \
                   name in self.value_indexes and self.value_indexes[name] is not None

    def has_value_at(self, name, idx, mode='any'):
        # Returns true if the specified grid point or points are filled, i.e. the
        # corresponding index values are all set to True
        if self.has_value_index(name):
            if mode == 'any':
                return np.any(self.value_indexes[name][tuple(idx)])
            elif mode == 'all':
                return np.all(self.value_indexes[name][tuple(idx)])
            else:
                raise NotImplementedError()
        else:
            return True

    def set_values(self, values, s=None, **kwargs):
        idx = self.get_index(**kwargs)
        self.set_values_at(idx, values, s)

    def set_values_at(self, idx, values, s=None):
        for name in values:
            self.set_value_at(name, idx, values[name], s)

    def set_value(self, name, value, s=None, **kwargs):
        idx = self.get_index(**kwargs)
        self.set_value_at(name, idx, value, s)

    def set_value_at(self, name, idx, value, s=None):
        self.ensure_lazy_load()
        
        idx = Grid.rectify_index(idx)
        if self.has_value_index(name):
            valid = self.is_value_valid(name, value)
            if self.preload_arrays:
                self.value_indexes[name][idx] = valid
            else:
                self.save_item(name + '_idx', np.array(valid), s=idx)

        idx = Grid.rectify_index(idx, s)
        if self.preload_arrays:
            self.values[name][idx] = value
        else:
            self.save_item(name, value, idx)

    def get_values(self, s=None, **kwargs):
        idx = self.get_index(**kwargs)
        return self.get_values_at(idx, s)

    def get_nearest_values(self, s=None, **kwargs):
        idx = self.get_nearest_index(**kwargs)
        return self.get_values_at(idx, s)

    def get_values_at(self, idx, s=None):
        return {name: self.get_value_at(name, idx, s) for name in self.values}

    def get_value(self, name, s=None, **kwargs):
        idx = self.get_index(**kwargs)
        return self.get_value_at(name, idx, s)

    def get_nearest_value(self, name, s=None, **kwargs):
        idx = self.get_nearest_index(**kwargs)
        return self.get_value_at(name, idx, s)

    def get_value_at(self, name, idx, s=None):
        # TODO: consider adding a squeeze=False option to keep exactly indexed dimensions
        idx = Grid.rectify_index(idx)
        if self.has_value_at(name, idx):
            idx = Grid.rectify_index(idx, s)
            if self.preload_arrays:
                return self.values[name][idx]
            else:
                self.ensure_lazy_load()
                return self.load_item(name, np.ndarray, idx)
        else:
            return None

    def save_values(self):
        for name in self.values:
            if self.values[name] is not None:
                if self.preload_arrays:
                    self.logger.info('Saving grid "{}" of size {}'.format(name, self.values[name].shape))
                    self.save_item(name, self.values[name])
                    self.logger.info('Saved grid "{}" of size {}'.format(name, self.values[name].shape))
                else:
                    shape = self.get_value_shape(name)
                    self.logger.info('Allocating grid "{}" with size {}...'.format(name, shape))
                    self.allocate_item(name, shape, np.float)
                    self.logger.info('Allocated grid "{}" with size {}. Will write directly to storage.'.format(name, shape))

    def load_values(self, s=None):
        gridshape = self.get_shape()
        for name in self.values:
            # If not running in memory saver mode, load entire array
            if self.preload_arrays:
                if s is not None:
                    self.logger.info('Loading grid "{}" of size {}'.format(name, s))
                    self.values[name][s] = self.load_item(name, np.ndarray, s=s)
                    self.logger.info('Loaded grid "{}" of size {}'.format(name, s))
                else:
                    self.logger.info('Loading grid "{}" of size {}'.format(name, self.value_shapes[name]))
                    self.values[name] = self.load_item(name, np.ndarray)
                    self.logger.info('Loaded grid "{}" of size {}'.format(name, self.value_shapes[name]))
                self.value_shapes[name] = self.values[name].shape[len(gridshape):]
            else:
                # When lazy-loading, we simply ignore the slice
                shape = self.get_item_shape(name)
                if shape is not None:
                    self.value_shapes[name] = shape[len(gridshape):]
                else:
                    self.value_shapes[name] = None
                
                self.logger.info('Skipped loading grid "{}". Will read directly from storage.'.format(name))

    def save_value_indexes(self):
        for name in self.values:
            self.save_item(name + '_idx', self.value_indexes[name])

    def load_value_indexes(self):
        for name in self.value_indexes:
            self.value_indexes[name] = self.load_item(name + '_idx', np.ndarray, s=None)

    def save_items(self):
        super(ArrayGrid, self).save_items()
        self.save_value_indexes()
        self.save_values()

    def load_items(self, s=None):
        super(ArrayGrid, self).load_items(s=s)
        self.load_value_indexes()
        self.load_values(s)

    def set_object_params(self, obj, idx=None, **kwargs):
        if idx is not None:
            for i, p in enumerate(self.axes):
                setattr(obj, p, float(self.axes[p].values[idx[i]]))
        if kwargs is not None:
            for p in kwargs:
                setattr(obj, p, float(kwargs[p]))

    def interpolate_value_linear(self, name, **kwargs):
        if len(self.axes) == 1:
            return self.interpolate_value_linear1d(name, **kwargs)
        else:
            return self.interpolate_value_linearNd(name, **kwargs)

    def interpolate_value_linear1d(self, name, **kwargs):
        idx = self.get_nearby_indexes(**kwargs)
        if idx is None:
            return None
        else:
            idx1, idx2 = idx
            if not self.has_value_at(name, idx1) or not self.has_value_at(name, idx2):
                return None

        # Parameter values to interpolate between
        p = list(kwargs.keys())[0]
        x = kwargs[p]
        xa = self.axes[p].values[idx1[0]]
        xb = self.axes[p].values[idx2[0]]
        a = self.get_value_at(name, idx1)
        b = self.get_value_at(name, idx2)
        m = (b - a) / (xb - xa)
        value = a + (x - xa) * m
        return value, kwargs

    def interpolate_value_linearNd(self, name, **kwargs):
        idx = self.get_nearby_indexes(**kwargs)
        if idx is None:
            return None
        else:
            idx1, idx2 = idx
            if not self.has_value_at(name, idx1) or not self.has_value_at(name, idx2):
                return None

        # Parameter values to interpolate between
        x = tuple([[self.axes[p].values[idx1[i]], self.axes[p].values[idx2[i]]] for i, p in enumerate(self.axes)])

        # Will hold data values
        s = [2, ] * len(x)
        value = self.get_value_at(name, idx1)
        s.append(value.shape[0])
        V = np.empty(s)

        ii = tuple(np.array(tuple(itertools.product(*([[0, 1],] * len(x))))).transpose())
        kk = tuple(np.array(tuple(itertools.product(*[[idx1[i], idx2[i]] for i in range(len(idx1))]))).transpose())

        V[ii] = self.get_value_at(name, kk)

        fn = RegularGridInterpolator(x, V)
        pp = tuple([kwargs[p] for p in self.axes])
        value = fn(pp)
        return value, kwargs

    def interpolate_value_spline(self, name, free_param, **kwargs):
        axis_list = list(self.axes.keys())
        free_param_idx = axis_list.index(free_param)

        # Find nearest model to requested parameters
        idx = list(self.get_nearest_index(**kwargs))
        if idx is None:
            self.logger.debug('No nearest model found.')
            return None

        # Set all params to nearest value except the one in which we interpolate
        for i, p in enumerate(self.axes):
            if p != free_param:
                kwargs[p] = self.axes[p].values[idx[i]]

        # Determine index of models
        idx[free_param_idx] = slice(None)
        idx = tuple(idx)

        # Find index of models that actually exists
        valid_value = self.value_indexes[name][idx]
        pars = self.axes[free_param].values[valid_value]

        # If we are at the edge of the grid, it might happen that we try to
        # interpolate over zero valid parameters, in this case return None and
        # the calling code will generate another set of random parameters
        if pars.shape[0] < 2 or kwargs[free_param] < pars.min() or pars.max() < kwargs[free_param]:
            self.logger.debug('Parameters are at the edge of grid, no interpolation possible.')
            return None

        if self.preload_arrays:
            value = self.values[name][idx][valid_value]
        else:
            self.ensure_lazy_load()
            value = self.load_item(name, np.ndarray, idx)
            value = value[valid_value]

        self.logger.debug('Interpolating values to {} using cubic splines along {}.'.format(kwargs, free_param))

        # Do as many parallel cubic spline interpolations as many wavelength bins we have
        x, y = pars, value
        fn = CubicSpline(x, y)

        return fn(kwargs[free_param]), kwargs

    @staticmethod
    def pad_array(orig_axes, orig_value):
        # Depending on the interpolation method, the original axes are converted from
        # actual values to index values. The padded axes will have the original values
        # extrapolated linearly.
        orig_xi = {}
        padded_xi = {}
        padded_axes = {}
        for p in orig_axes:
            # Padded axis with linear extrapolation from the original edge values
            paxis = np.empty(orig_axes[p].values.shape[0] + 2)
            paxis[1:-1] = orig_axes[p].values
            paxis[0] = paxis[1] - (paxis[2] - paxis[1])
            paxis[-1] = paxis[-2] + (paxis[-2] - paxis[-3])
            padded_axes[p] = GridAxis(p, paxis)

            if interpolation == 'ijk':
                orig_xi[p] = np.arange(orig_axes[p].values.shape[0], dtype=np.float64)
                padded_xi[p] = np.arange(-1, orig_axes[p].values.shape[0] + 1, dtype=np.float64)
            elif interpolation == 'xyz':
                orig_xi[p] = orig_axes[p].values
                padded_xi[p] = padded_axes[p].values
            else:
                raise NotImplementedError()

        # Pad original slice with phantom cells
        # We a do a bit of extra work here because we interpolated the entire new slice, not just
        # the edges. The advantage is that we can fill in some of the holes this way.
        oijk = [orig_xi[p] for p in orig_xi]
        pijk = np.stack(np.meshgrid(*[padded_xi[p] for p in padded_xi], indexing='ij'), axis=-1)
        padded_value = interpn(oijk, orig_value, pijk, method='linear', bounds_error=False, fill_value=None)

        return padded_value, padded_axes