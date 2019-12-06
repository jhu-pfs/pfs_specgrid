import logging
import numpy as np
import itertools
from scipy.interpolate import RegularGridInterpolator, CubicSpline

from pfsspec.pfsobject import PfsObject
from pfsspec.data.gridparam import GridParam

class Grid(PfsObject):
    def __init__(self):
        self.preload_arrays = False
        self.params = {}
        self.data = {}
        self.data_shape = {}
        self.data_index = {}

    def get_shape(self):
        shape = tuple(self.params[p].values.shape[0] for p in self.params)
        return shape

    def ensure_lazy_load(self):
        # This works with HDF5 format only!
        if self.fileformat != 'h5':
            raise NotImplementedError()

    def init_param(self, name, values):
        self.params[name] = GridParam(name, values)

    def init_data(self):
        raise NotImplementedError()

    def init_data_item(self, name, shape):
        if len(shape) != 1:
            raise NotImplementedError()

        gridshape = self.get_shape()
        datashape = gridshape + tuple(shape)

        if self.preload_arrays:
            logging.info('Initializing memory for grid "{}" of size {}...'.format(name, datashape))
            self.data[name] = np.full(datashape, np.nan)
            logging.info('Initialized memory for grid "{}" of size {}.'.format(name, datashape))
        else:
            logging.info('Skipped memory initialization for grid "{}". Will read random slices from storage.'.format(name))

        self.data_shape[name] = shape
        self.data_index[name] = np.full(gridshape, False, dtype=np.bool)

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
            if idx1[i] < 0 or idx2[i] < 0 or \
               idx1[i] >= self.params[p].values.shape[0] or \
               idx2[i] >= self.params[p].values.shape[0]:
                return None

            i += 1

        idx1 = tuple(idx1)
        idx2 = tuple(idx2)

        return idx1, idx2

    def is_data_index(self, name):
        return self.data_index is not None and name in self.data_index and self.data_index[name] is not None

    def is_data_item_idx(self, name, idx):
        if self.is_data_index(name):
            return self.data_index[name][idx]
        else:
            return True

    def set_data(self, data, **kwargs):
        idx = self.get_index(**kwargs)
        self.set_data_idx(idx, data)

    def set_data_idx(self, idx, data):
        for name in data:
            self.set_data_item_idx(name, idx, data[name])

    def set_data_item(self, name, data, **kwargs):
        idx = self.get_index(**kwargs)
        self.set_data_item_idx(name, idx, data)

    def set_data_item_idx(self, name, idx, data):
        if self.is_data_index(name):
            self.data_index[name][idx] = self.is_data_valid(name, data)
        idx = tuple(idx) + (slice(None),)
        if self.preload_arrays:
            self.data[name][idx] = data
        else:
            self.ensure_lazy_load()
            self.save_item(name, data, idx)

    def get_data(self, **kwargs):
        idx = self.get_index(**kwargs)
        return self.get_data_idx(idx)

    def get_nearest_data(self, **kwargs):
        idx = self.get_nearest_index(**kwargs)
        return self.get_data_idx(idx)

    def get_data_idx(self, idx):
        return {name: self.get_data_item_idx(name, idx) for name in self.data}

    def get_data_item(self, name, **kwargs):
        idx = self.get_index(**kwargs)
        return self.get_data_item_idx(name, idx)

    def get_nearest_data_item(self, name, **kwargs):
        idx = self.get_nearest_index(**kwargs)
        return self.get_data_item_idx(name, idx)

    def get_data_item_idx(self, name, idx):
        if self.is_data_item_idx(name, idx):
            idx = tuple(idx) + (slice(None),)
            if self.preload_arrays:
                return self.data[name][idx]
            else:
                self.ensure_lazy_load()
                return self.load_item(name, np.ndarray, idx)
        else:
            return None

    def set_object_params(self, obj, **kwargs):
        for p in self.params:
            setattr(obj, p, kwargs[p])

    def interpolate_data_item_linear(self, name, **kwargs):
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
        s.append(self.data[name][idx1].shape[0])
        V = np.empty(s)

        ii = tuple(np.array(tuple(itertools.product(*([[0, 1],] * len(x))))).transpose())
        kk = tuple(np.array(tuple(itertools.product(*[[idx1[i], idx2[i]] for i in range(len(idx1))]))).transpose())

        V[ii] = self.data[name][kk]

        fn = RegularGridInterpolator(x, V)
        data = fn(tuple([kwargs[p] for p in self.params]))
        return data

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

        if self.preload_arrays:
            data = self.data[name][idx][valid_data]
        else:
            self.ensure_lazy_load()
            data = self.load_item(name, np.ndarray, idx)
            data = data[valid_data]

        # If we are at the edge of the grid, it might happen that we try to
        # interpolate over zero valid parameters, in this case return None and
        # the calling code will generate another set of random parameters
        if pars.shape[0] < 2:
            logging.debug('Parameters are at the edge of grid, no interpolation possible.')
            return None

        logging.debug('Interpolating data to {} using cubic splines along {}.'.format(kwargs, free_param))

        # Do as many parallel cubic spline interpolations as many wavelength bins we have
        x, y = pars, data
        fn = CubicSpline(x, y)

        return fn(kwargs[free_param]), kwargs