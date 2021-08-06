import logging
import numpy as np
import itertools
from collections import Iterable
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator, CubicSpline
from scipy.interpolate import interp1d, interpn

from pfsspec.util.interp.rbf import Rbf
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
    def __init__(self, config=None, orig=None):
        super(ArrayGrid, self).__init__(orig=orig)

        if isinstance(orig, ArrayGrid):
            self.config = config if config is not None else orig.config
            self.values = orig.values
            self.value_shapes = orig.value_shapes
            self.value_indexes = orig.value_indexes
            self.slice = orig.slice
        else:
            self.config = config
            self.values = {}
            self.value_shapes = {}
            self.value_indexes = {}
            self.slice = None

            self.init_values()

    @property
    def array_grid(self):
        return self

    @property
    def rbf_grid(self):
        return None

#region Support slicing via command-line arguments

    def init_from_args(self, args):
        super(ArrayGrid, self).init_from_args(args)
        self.slice = self.get_slice_from_args(args)

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

        return tuple(s)

    def get_axes(self, squeeze=False):
        # Return axes that are limited by the slices
        if self.slice is not None:
            axes = {}
            for i, k in enumerate(self.axes):
                if type(self.slice[i]) is slice:
                    v = self.axes[k].values[self.slice[i]]
                    if not squeeze or v.shape[0] > 1:
                        axes[k] = GridAxis(k, v)
                        axes[k].build_index()
            return axes
        else:
            return super(ArrayGrid, self).get_axes()

    def get_shape(self):
        if self.slice is not None:
            ss = []
            for i, k in enumerate(self.axes):
                if type(self.slice[i]) is slice:
                    s = self.axes[k].values[self.slice[i]].shape[0]
                    ss.append(s)
            return tuple(ss)
        else:
            return super(ArrayGrid, self).get_shape()

    def get_shape_unsliced(self):
        return super(ArrayGrid, self).get_shape()

#endregion

    def get_value_shape(self, name):
        # Gets the full shape of the grid. It assumes different last
        # dimensions for each data array.
        shape = self.get_shape() + self.value_shapes[name]
        return shape

    def init_value(self, name, shape=None, **kwargs):
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
                self.value_indexes[name] = None
                self.logger.info('Initializing data file for grid "{}" of size {}...'.format(name, valueshape))
                if not self.has_item(name):
                    self.allocate_item(name, valueshape, dtype=np.float)
                    self.allocate_item(name + '_idx', gridshape, dtype=np.bool)
                self.logger.info('Skipped memory initialization for grid "{}". Will read random slices from storage.'.format(name))

    def allocate_value(self, name, shape=None):
        if shape is not None:
            self.value_shapes[name] = shape
        self.init_value(name, self.value_shapes[name])

    def is_value_valid(self, name, value):
        if self.config is not None:
            return self.config.is_value_valid(self, name, value)
        else:
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
        return np.sum(self.get_value_index(name))
           
    def get_index(self, **kwargs):
        """Returns the indexes along all axes corresponding to the values specified.

        If an axis name is missing, a whole slice is returned.

        Returns:
            [type]: [description]
        """
        idx = ()
        for i, p in enumerate(self.axes):
            if p in kwargs:
                idx += (self.axes[p].index[kwargs[p]],)
            else:
                if self.slice is None:
                    idx += (slice(None),)
                else:
                    idx += (self.slice[i],)
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
            index = self.value_indexes[name]
            if self.slice is not None:
                return index[self.slice]
            else:
                return index
        else:
            return None

    def get_value_index_unsliced(self, name):
        # Return a boolean index that is limited by the axis bound overrides.
        # The shape will be the same as the original array. Use this index to
        # load a limited subset of the data directly from the disk.
        if self.slice is not None:
            index = np.full(self.value_indexes[name].shape, False)
            index[self.slice] = self.value_indexes[name][self.slice]
            return index
        else:
            return self.value_indexes[name]

    def get_mask_unsliced(self):
        if self.slice is not None:
            mask = np.full(self.get_shape_unsliced(), False)
            mask[self.slice] = True
        else:
            mask = np.full(self.get_shape(), True)
        return mask

    def get_chunks(self, name, shape, s=None):
        if self.config is not None:
            self.config.get_chunks(self, name, shape, s=s)
        else:
            super(ArrayGrid, self).get_chunks(name, shape, s=s)

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

    def set_value(self, name, value, valid=None, s=None, **kwargs):
        idx = self.get_index(**kwargs)
        self.set_value_at(name, idx, value=value, valid=valid, s=s)

    def set_value_at(self, name, idx, value, valid=None, s=None):
        self.ensure_lazy_load()
        
        idx = Grid.rectify_index(idx)
        if self.has_value_index(name):
            if valid is None:
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

    def get_value(self, name, s=None, squeeze=False, **kwargs):
        idx = self.get_index(**kwargs)
        v = self.get_value_at(name, idx, s)
        if squeeze:
            v = np.squeeze(v)
        return v

    def get_nearest_value(self, name, s=None, **kwargs):
        idx = self.get_nearest_index(**kwargs)
        return self.get_value_at(name, idx, s)

    def get_value_at(self, name, idx, s=None, raw=None):
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

    def load(self, filename, s=None, format=None):
        s = s or self.slice
        super(ArrayGrid, self).load(filename, s=s, format=format)

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
    def get_grid_points(axes, padding=False, squeeze=False, interpolation='ijk'):
        # Return a dictionary of the grid points, either by ijk index or xyz coordinates.
        # When sqeeze=True, only axis with more than 1 grid point will be included.
        # The result can be used to call np.meshgrid.

        xi = {}
        for p in axes:
            if interpolation == 'ijk':
                if axes[p].values.shape[0] == 1:
                    if not squeeze:
                        xi[p] = np.array([0.0])
                else:
                    xi[p] = np.arange(axes[p].values.shape[0], dtype=np.float64)
                    if padding:
                        xi[p] -= 1.0
            elif interpolation == 'xyz':
                if axes[p].values.shape[0] > 1 or not squeeze:
                    xi[p] = axes[p].values
            else:
                raise NotImplementedError()
        return xi

    @staticmethod
    def pad_axes(orig_axes, size=1):
        padded_axes = {}
        for p in orig_axes:
            # Padded axis with linear extrapolation from the original edge values
            if orig_axes[p].values.shape[0] > 1:
                paxis = np.empty(orig_axes[p].values.shape[0] + 2 * size)
                paxis[size:-size] = orig_axes[p].values
                for i in range(size - 1, -1, -1):
                    paxis[i] = paxis[i + 1] - (paxis[i + 2] - paxis[i + 1])
                    paxis[-1 - i] = paxis[-2 - i] + (paxis[-2 - i] - paxis[-3 - i])
                padded_axes[p] = GridAxis(p, paxis)
            else:
                padded_axes[p] = orig_axes[p]
        return padded_axes