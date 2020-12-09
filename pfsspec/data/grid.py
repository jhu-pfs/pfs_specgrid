import logging
import numpy as np
import itertools
from collections import Iterable
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.interpolate import Rbf, interp1d, interpn

from pfsspec.pfsobject import PfsObject
from pfsspec.data.gridaxis import GridAxis

class Grid(PfsObject):
    """Implements a generic grid class to store and interpolate data.

    This class implements a generic grid class with an arbitrary number of axes
    and multiple value arrays in each grid point. Value arrays must have the same
    leading dimensions as the size of the axes. An index is build on every
    value array which indicated valid/invalid data in a particular grid point.

    Args:
        PfsObject ([type]): [description]
    """
    def __init__(self, orig=None):
        super(Grid, self).__init__(orig=orig)

        self.preload_arrays = False
        self.axes = {}
        self.values = {}
        self.value_shapes = {}
        self.value_indexes = {}

        self.init_axes()
        self.init_values()

    def get_shape(self):
        shape = tuple(self.axes[p].values.shape[0] for p in self.axes)
        return shape

    def get_value_shape(self, name):
        # Gets the full shape of the grid. It assumes different last
        # dimensions for each data array.
        shape = self.get_shape() + self.value_shapes[name]
        return shape

    def ensure_lazy_load(self):
        # This works with HDF5 format only!
        if self.fileformat != 'h5':
            raise NotImplementedError()

    def init_axes(self):
        pass

    def init_axis(self, name, values=None):
        self.axes[name] = GridAxis(name, values)

    def init_values(self):
        pass

    def allocate_values(self):
        pass

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
                self.logger.info('Initialized memory for grid "{}" of size {}.'.format(name, valueshape))
            else:
                self.values[name] = None
                self.logger.info('Skipped memory initialization for grid "{}". Will read random slices from storage.'.format(name))

            self.value_indexes[name] = np.full(gridshape, False, dtype=np.bool)

    def allocate_value(self, name, shape):
        self.init_value(name, shape)

    def build_axis_indexes(self):
        for p in self.axes:
            self.axes[p].build_index()

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

    @staticmethod
    def rectify_index(idx, s=None):
        idx = tuple(idx)
        if isinstance(s, Iterable):
            idx = idx + tuple(s)
        elif s is not None:
            idx = idx + (s,)

        return tuple(idx)

    def get_limited_value_index(self, name):
        # TODO: test this with lazy-loading, we might just skip this entirely in that case
        idx = ()
        for p in self.axes:
            start = np.where(self.axes[p].values >= self.axes[p].min)[0][0]
            stop = np.where(self.axes[p].values <= self.axes[p].max)[0][-1]
            idx = idx + (slice(start, stop + 1), )
        value_indexes = np.full(self.value_indexes[name].shape, False)
        value_indexes[idx] = self.value_indexes[name][idx]
        return value_indexes

    def get_valid_value_count(self, name, use_limits=False):
        if use_limits:
            return np.sum(self.get_limited_value_index(name))
        else:
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
        return self.value_indexes is not None and name in self.value_indexes and self.value_indexes[name] is not None

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
        idx = Grid.rectify_index(idx)
        if self.has_value_index(name):
            valid = self.is_value_valid(name, value)
            self.value_indexes[name][idx] = valid
            if not self.preload_arrays:
                self.save_item(name + '_idx', np.array(valid), s=idx)

        idx = Grid.rectify_index(idx, s)
        if self.preload_arrays:
            self.values[name][idx] = value
        else:
            self.ensure_lazy_load()
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

    def save_axes(self):
        for p in self.axes:
            self.save_item(p, self.axes[p].values)

    def load_axes(self):
        for p in self.axes:
            self.axes[p].values = self.load_item(p, np.ndarray)

    def save_values(self):
        for name in self.values:
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
                self.value_shapes[name] = self.get_item_shape(name)[len(gridshape):]
                self.logger.info('Skipped loading grid "{}". Will read directly from storage.'.format(name))

    def save_value_indexes(self):
        for name in self.values:
            self.save_item(name + '_idx', self.value_indexes[name])

    def load_value_indexes(self):
        for name in self.value_indexes:
            self.value_indexes[name] = self.load_item(name + '_idx', np.ndarray, s=None)

    def save_items(self):
        self.save_axes()
        self.save_value_indexes()
        self.save_values()

    def load(self, filename, s=None, format=None):
        super(Grid, self).load(filename, s=s, format=format)
        self.build_axis_indexes()

    def load_items(self, s=None):
        self.load_axes()
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

    def get_value_padded(self, name, s=None, interpolation='ijk', **kwargs):
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

        # Slice before padding. This array is squeezed in the directions specified
        # in kwargs. All other directions are full slices.
        orig = self.get_value(name, s=s, **kwargs)

        # Create the mesh grid of the original slice. Do it only in the directions
        # not specified in kwargs because those directions are squeezed.
        oaxes = {}
        paxes = {}
        for p in self.axes.keys():
            if p not in kwargs:
                if interpolation == 'ijk':
                    oaxes[p] = GridAxis(p, np.arange(self.axes[p].values.shape[0], dtype=np.float64))
                    paxes[p] = GridAxis(p, np.arange(-1, self.axes[p].values.shape[0] + 1, dtype=np.float64))
                elif interpolation == 'xyz':
                    axis = self.axes[p].values
                    paxis = np.empty(axis.shape[0] + 2)
                    paxis[1:-1] = axis
                    paxis[0] = paxis[1] - (paxis[2] - paxis[1])
                    paxis[-1] = paxis[-2] + (paxis[-2] - paxis[-3])

                    oaxes[p] = GridAxis(p, axis)
                    paxes[p] = GridAxis(p, paxis)
                else:
                    raise NotImplementedError()

        pijk = np.meshgrid(*[paxes[p].values for p in paxes], indexing='ij')
        pijk = np.stack(pijk, axis=-1)

        # Pad original slice with phantom cells
        # We a do a bit of extra work here because we interpolated the entire new slice, not just
        # the edges. The advantage is that we can fill in some of the holes this way.
        padded = interpn([oaxes[p].values for p in oaxes], orig, pijk, method='linear', bounds_error=False, fill_value=None)

        return padded, paxes

    def interpolate_value_rbf(self, value, axes, mask=None, function='multiquadric', epsilon=None, smooth=0.0):
        """Returns the Radial Base Function interpolation of a grid slice.

        Args:
            value
            axes
            mask (array): Mask, must be the same shape as the grid.
            function (str): Basis function, see RBF documentation.
            epsilon (number): See RBF documentation.
            smooth (number): See RBF documentation.
        """

        # Since we must have the same number of grid points, we need to contract the
        # mask along all value array dimensions that are not along the axes. Since `value`
        # is already squeezed, only use axes that do not match axes in kwargs.
        m = ~np.isnan(value)
        if len(m.shape) > len(axes):
            m = np.all(m, axis=-(len(m.shape) - len(axes)))

        # We assume that the provided mask has the same shape
        if mask is not None:
            m &= mask
            
        m = m.flatten()

        # Flatten slice along axis dimensions
        sh = 1
        for i in range(len(axes)):
            sh *= value.shape[i]
        value = value.reshape((sh,) + value.shape[len(axes):])
        value = value[m]

        points = np.meshgrid(*[axes[p].values for p in axes], indexing='ij')
        points = [p.flatten() for p in points]
        points = [p[m] for p in points]

        if len(value.shape) == 1:
            mode = '1-D'
        else:
            mode = 'N-D'

        rbf = Rbf(*points, value, function=function, epsilon=epsilon, smooth=smooth, mode=mode)

        return rbf

