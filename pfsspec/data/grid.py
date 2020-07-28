import logging
import numpy as np
import itertools
from collections import Iterable
from scipy.interpolate import RegularGridInterpolator, CubicSpline

from pfsspec.pfsobject import PfsObject
from pfsspec.data.gridparam import GridParam

class Grid(PfsObject):
    def __init__(self, orig=None):
        super(Grid, self).__init__(orig=orig)

        self.preload_arrays = False
        self.params = {}
        self.data = {}
        self.data_shape = {}
        self.data_index = {}

        self.init_params()
        self.init_data()

    def get_shape(self):
        shape = tuple(self.params[p].values.shape[0] for p in self.params)
        return shape

    def get_data_item_shape(self, name):
        shape = self.get_shape() + self.data_shape[name]
        return shape

    def ensure_lazy_load(self):
        # This works with HDF5 format only!
        if self.fileformat != 'h5':
            raise NotImplementedError()

    def init_params(self):
        pass

    def init_param(self, name, values=None):
        self.params[name] = GridParam(name, values)

    def init_data(self):
        pass

    def allocate_data(self):
        pass

    def init_data_item(self, name, shape=None):
        if shape is None:
            self.data[name] = None
            self.data_shape[name] = None
            self.data_index[name] = None
        else:
            if len(shape) != 1:
                raise NotImplementedError()

            gridshape = self.get_shape()
            datashape = gridshape + tuple(shape)

            self.data_shape[name] = shape

            if self.preload_arrays:
                logging.info('Initializing memory for grid "{}" of size {}...'.format(name, datashape))
                self.data[name] = np.full(datashape, np.nan)
                logging.info('Initialized memory for grid "{}" of size {}.'.format(name, datashape))
            else:
                self.data[name] = None
                logging.info('Skipped memory initialization for grid "{}". Will read random slices from storage.'.format(name))

            self.data_index[name] = np.full(gridshape, False, dtype=np.bool)

    def allocate_data_item(self, name, shape):
        self.init_data_item(name, shape)

    def build_params_index(self):
        for p in self.params:
            self.params[p].build_index()

    def is_data_valid(self, name, data):
        return np.logical_not(np.any(np.isnan(data), axis=-1))

    def build_data_index(self, rebuild=False):
        for name in self.data:
            self.build_data_item_index(name, rebuild=rebuild)

    def build_data_item_index(self, name, rebuild=False):
        if rebuild and not self.preload_arrays:
            raise Exception('Cannot build index on lazy-loaded grid.')
        elif rebuild or name not in self.data_index or self.data_index[name] is None:
            logging.debug('Building indexes on grid "{}" of size {}'.format(name, self.data_shape[name]))
            self.data_index[name] = self.is_data_valid(name, self.data[name])
        else:
            logging.debug('Skipped building indexes on grid "{}" of size {}'.format(name, self.data_shape[name]))
        logging.debug('{} valid vectors in grid "{}" found'.format(np.sum(self.data_index[name]), name))

    @staticmethod
    def rectify_index(idx, s=None):
        idx = tuple(idx)
        if isinstance(s, Iterable):
            idx = idx + tuple(s)
        elif s is not None:
            idx = idx + (s,)

        return tuple(idx)

    def get_limited_data_index(self, name):
        idx = ()
        for p in self.params:
            start = np.where(self.params[p].values >= self.params[p].min)[0][0]
            stop = np.where(self.params[p].values <= self.params[p].max)[0][-1]
            idx = idx + (slice(start, stop + 1), )
        data_index = np.full(self.data_index[name].shape, False)
        data_index[idx] = self.data_index[name][idx]
        return data_index

    def get_valid_data_item_count(self, name, use_limits=False):
        if use_limits:
            return np.sum(self.get_limited_data_index(name))
        else:
            return np.sum(self.data_index[name])

    def get_index(self, **kwargs):
        idx = tuple(self.params[p].index[kwargs[p]] for p in self.params)
        return idx

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
            if idx1[i] < 0 or self.params[p].values.shape[0] <= idx1[i] or \
               idx2[i] < 0 or self.params[p].values.shape[0] <= idx2[i]:
                return None

            i += 1

        return tuple(idx1), tuple(idx2)

    def is_data_index(self, name):
        return self.data_index is not None and name in self.data_index and self.data_index[name] is not None

    def is_data_item(self, name):
        return name in self.data and self.data[name] is not None and \
               name in self.data_index and self.data_index[name] is not None

    def is_data_item_idx(self, name, idx):
        if self.is_data_index(name):
            return np.all(self.data_index[name][idx])
        else:
            return True

    def set_data(self, data, s=None, **kwargs):
        idx = self.get_index(**kwargs)
        self.set_data_idx(idx, data, s)

    def set_data_idx(self, idx, data, s=None):
        for name in data:
            self.set_data_item_idx(name, idx, data[name], s)

    def set_data_item(self, name, data, s=None, **kwargs):
        idx = self.get_index(**kwargs)
        self.set_data_item_idx(name, idx, data, s)

    def set_data_item_idx(self, name, idx, data, s=None):
        idx = Grid.rectify_index(idx)
        if self.is_data_index(name):
            self.data_index[name][idx] = self.is_data_valid(name, data)

        idx = Grid.rectify_index(idx, s)
        if self.preload_arrays:
            self.data[name][idx] = data
        else:
            self.ensure_lazy_load()
            self.save_item(name, data, idx)

    def get_data(self, s=None, **kwargs):
        idx = self.get_index(**kwargs)
        return self.get_data_idx(idx, s)

    def get_nearest_data(self, s=None, **kwargs):
        idx = self.get_nearest_index(**kwargs)
        return self.get_data_idx(idx, s)

    def get_data_idx(self, idx, s=None):
        return {name: self.get_data_item_idx(name, idx, s) for name in self.data}

    def get_data_item(self, name, s=None, **kwargs):
        idx = self.get_index(**kwargs)
        return self.get_data_item_idx(name, idx, s)

    def get_nearest_data_item(self, name, s=None, **kwargs):
        idx = self.get_nearest_index(**kwargs)
        return self.get_data_item_idx(name, idx, s)

    def get_data_item_idx(self, name, idx, s=None):
        idx = Grid.rectify_index(idx)
        if self.is_data_item_idx(name, idx):
            idx = Grid.rectify_index(idx, s)
            if self.preload_arrays:
                return self.data[name][idx]
            else:
                self.ensure_lazy_load()
                return self.load_item(name, np.ndarray, idx)
        else:
            return None

    def save_params(self):
        for p in self.params:
            self.save_item(p, self.params[p].values)

    def load_params(self):
        for p in self.params:
            self.params[p].values = self.load_item(p, np.ndarray)

    def save_data(self):
        for name in self.data:
            if self.preload_arrays:
                logging.info('Saving grid "{}" of size {}'.format(name, self.data[name].shape))
                self.save_item(name, self.data[name])
                logging.info('Saved grid "{}" of size {}'.format(name, self.data[name].shape))
            else:
                shape = self.get_data_item_shape(name)
                logging.info('Allocating grid "{}" with size {}...'.format(name, shape))
                self.allocate_item(name, shape, np.float)
                logging.info('Allocated grid "{}" with size {}. Will write directly to storage.'.format(name, shape))

    def load_data(self, s=None):
        for name in self.data:
            # If not running in memory saver mode, load entire array
            if self.preload_arrays:
                if s is not None:
                    logging.info('Loading grid "{}" of size {}'.format(name, s))
                    self.data[name][s] = self.load_item(name, np.ndarray, s=s)
                    logging.info('Loaded grid "{}" of size {}'.format(name, s))
                else:
                    logging.info('Loading grid "{}" of size {}'.format(name, self.data_shape[name]))
                    self.data[name] = self.load_item(name, np.ndarray)
                    logging.info('Loaded grid "{}" of size {}'.format(name, self.data_shape[name]))
            else:
                logging.info('Skipped loading grid "{}". Will read directly from storage.'.format(name))

    def save_data_index(self):
        for name in self.data:
            self.save_item(name + '_idx', self.data_index[name])

    def load_data_index(self):
        for name in self.data_index:
            self.data_index[name] = self.load_item(name + '_idx', np.ndarray, s=None)

    def save_items(self):
        self.save_params()
        self.save_data_index()
        self.save_data()

    def load(self, filename, s=None, format=None):
        super(Grid, self).load(filename, s=s, format=format)
        self.build_params_index()

    def load_items(self, s=None):
        self.load_params()
        self.load_data_index()
        self.load_data(s)

    def set_object_params(self, obj, idx=None, **kwargs):
        if idx is not None:
            for i, p in enumerate(self.params):
                setattr(obj, p, float(self.params[p].values[idx[i]]))
        if kwargs is not None:
            for p in kwargs:
                setattr(obj, p, float(kwargs[p]))

    def interpolate_data_item_linear(self, name, **kwargs):
        if len(self.params) == 1:
            return self.interpolate_data_item_linear_1D(name, **kwargs)
        else:
            return self.interpolate_data_item_linear_nD(name, **kwargs)

    def interpolate_data_item_linear_1D(self, name, **kwargs):
        idx = self.get_nearby_indexes(**kwargs)
        if idx is None:
            return None
        else:
            idx1, idx2 = idx
            if not self.is_data_item_idx(name, idx1) or not self.is_data_item_idx(name, idx2):
                return None

        # Parameter values to interpolate between
        p = list(kwargs.keys())[0]
        x = kwargs[p]
        xa = self.params[p].values[idx1[0]]
        xb = self.params[p].values[idx2[0]]
        a = self.get_data_item_idx(name, idx1)
        b = self.get_data_item_idx(name, idx2)
        m = (b - a) / (xb - xa)
        data = a + (x - xa) * m
        return data, kwargs

    def interpolate_data_item_linear_nD(self, name, **kwargs):
        idx = self.get_nearby_indexes(**kwargs)
        if idx is None:
            return None
        else:
            idx1, idx2 = idx
            if not self.is_data_item_idx(name, idx1) or not self.is_data_item_idx(name, idx2):
                return None

        # Parameter values to interpolate between
        x = tuple([[self.params[p].values[idx1[i]], self.params[p].values[idx2[i]]] for i, p in enumerate(self.params)])

        # Will hold data values
        s = [2, ] * len(x)
        data = self.get_data_item_idx(name, idx1)
        s.append(data.shape[0])
        V = np.empty(s)

        ii = tuple(np.array(tuple(itertools.product(*([[0, 1],] * len(x))))).transpose())
        kk = tuple(np.array(tuple(itertools.product(*[[idx1[i], idx2[i]] for i in range(len(idx1))]))).transpose())

        V[ii] = self.get_data_item_idx(name, kk)

        fn = RegularGridInterpolator(x, V)
        pp = tuple([kwargs[p] for p in self.params])
        data = fn(pp)
        return data, kwargs

    def interpolate_data_item_spline(self, name, free_param, **kwargs):
        params_list = list(self.params.keys())
        free_param_idx = params_list.index(free_param)

        # Find nearest model to requested parameters
        idx = list(self.get_nearest_index(**kwargs))
        if idx is None:
            logging.debug('No nearest model found.')
            return None

        # Set all params to nearest value except the one in which we interpolate
        for i, p in enumerate(self.params):
            if p != free_param:
                kwargs[p] = self.params[p].values[idx[i]]

        # Determine index of models
        idx[free_param_idx] = slice(None)
        idx = tuple(idx)

        # Find index of models that actually exists
        valid_data = self.data_index[name][idx]
        pars = self.params[free_param].values[valid_data]

        # If we are at the edge of the grid, it might happen that we try to
        # interpolate over zero valid parameters, in this case return None and
        # the calling code will generate another set of random parameters
        if pars.shape[0] < 2 or kwargs[free_param] < pars.min() or pars.max() < kwargs[free_param]:
            logging.debug('Parameters are at the edge of grid, no interpolation possible.')
            return None

        if self.preload_arrays:
            data = self.data[name][idx][valid_data]
        else:
            self.ensure_lazy_load()
            data = self.load_item(name, np.ndarray, idx)
            data = data[valid_data]

        logging.debug('Interpolating data to {} using cubic splines along {}.'.format(kwargs, free_param))

        # Do as many parallel cubic spline interpolations as many wavelength bins we have
        x, y = pars, data
        fn = CubicSpline(x, y)

        return fn(kwargs[free_param]), kwargs