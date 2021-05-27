import logging
import numpy as np

from pfsspec.pfsobject import PfsObject

class PcaGrid(PfsObject):
    # Wraps an ArrayGrid or an RbfGrid and adds PCA decompression support

    def __init__(self, grid, orig=None):
        super(PcaGrid, self).__init__(orig=orig)

        if isinstance(orig, PcaGrid):
            self.grid = grid if grid is not None else orig.grid

            self.eigs = orig.eigs
            self.eigv = orig.eigv
            self.k = orig.k
        else:
            self.grid = grid

            self.eigs = {}
            self.eigv = {}
            self.k = None

    @property
    def preload_arrays(self):
        return self.grid.preload_arrays

    @preload_arrays.setter
    def preload_arrays(self, value):
        self.grid.preload_arrays = value

    @property
    def array_grid(self):
        return self.grid.array_grid

    @property
    def rbf_grid(self):
        return self.grid.rbf_grid

    def init_from_args(self, args):
        self.grid.init_from_args(args)

    def get_shape(self):
        return self.grid.get_shape()

    def get_constants(self):
        return self.grid.get_constants()

    def set_constants(self, constants):
        self.grid.set_constants(constants)

    def init_axes(self):
        raise NotImplementedError()

    def init_axis(self, name, values=None):
        self.grid.init_axis(name, values=values)

    def set_axes(self, axes):
        self.grid.set_axes(axes)

    def get_axes(self, squeeze=False):
        return self.grid.get_axes(squeeze=squeeze)

    def build_axis_indexes(self):
        self.grid.build_axis_indexes()

    def init_constant(self, name):
        self.grid.init_constant(name)

    def get_constant(self, name):
        return self.grid.get_constant(name)

    def set_constant(self, name, value):
        self.grid.set_constant(name, value)

    def init_value(self, name, shape=None, pca=False, **kwargs):
        if pca:
            if shape is not None:
                # Last dimension is number of coefficients
                self.grid.init_value(name, shape=shape[-1])
            else:
                self.grid.init_value(name)
            self.eigs[name] = None
            self.eigv[name] = None
        else:
            self.grid.init_value(name, shape=shape)

    def allocate_values(self):
        raise NotImplementedError()

    def allocate_value(self, name, shape=None, pca=False):
        if pca:
            if shape is not None:
                self.grid.allocate_value(name, shape=(shape[-1],))
                # TODO: there should be some trickery here regarding
                #       the size of these arrays
                self.eigs[name] = np.full(shape[:-1], np.nan)
                self.eigv[name] = np.full(shape[:-1], np.nan)
            else:
                self.grid.allocate_value(name)
                self.eigs[name] = None
                self.eigv[name] = None
        else:
            self.grid.allocate_value(name, shape=shape)

    def add_args(self, parser):
        self.grid.add_args(parser)

    def init_from_args(self, args):
        self.grid.init_from_args(args)

    def get_index(self, **kwargs):
        return self.grid.get_index(**kwargs)

    def get_nearest_index(self, **kwargs):
        return self.grid.get_nearest_index(**kwargs)

    def has_value_index(self, name):
        return self.grid.has_value_index(name)

    def get_valid_value_count(self, name):
        return self.grid.get_valid_value_count(name)

    def has_value(self, name):
        return self.grid.has_value(name)

    def has_value_at(self, name, idx, mode='any'):
        return self.grid.has_value_at(name, idx, mode=mode)

    def set_value(self, name, value, s=None, pca=False, **kwargs):
        if not pca:
            self.grid.set_value(name, value, s=s, **kwargs)
        else:
            # TODO: slicing?
            self.grid.set_value(name, value[0], s=s, **kwargs)
            self.eigs[name] = value[1]
            self.eigv[name] = value[2]

    def get_value_at(self, name, idx, s=None, raw=False):
        if not raw and name in self.eigs:
            pc = self.grid.get_value_at(name, idx)
            if self.k is None:
                v = np.dot(self.eigv[name], pc)
            else:
                v = np.dot(self.eigv[name][:, :self.k], pc[:self.k])
            return v[s or slice(None)]
        else:
            return self.grid.get_value_at(name, idx, s=s)

    def get_value(self, name, s=None, **kwargs):
        idx = self.get_index(**kwargs)
        return self.get_value_at(name, idx, s=s)

    def get_values_at(self, idx, s=None):
        return {name: self.get_value_at(name, idx, s) for name in self.values}

    def get_values(self, s=None, **kwargs):
        idx = self.get_index(**kwargs)
        return self.get_values_at(idx, s)

    def save_items(self):
        self.grid.filename = self.filename
        self.grid.fileformat = self.fileformat
        self.grid.save_items()

        for name in self.eigs:
            if self.eigs[name] is not None:
                self.save_item('{}_eigs'.format(name), self.eigs[name])
                self.save_item('{}_eigv'.format(name), self.eigv[name])

    def load_items(self, s=None):
        self.grid.filename = self.filename
        self.grid.fileformat = self.fileformat
        self.grid.load_items(s=s)

        for name in self.eigs:
            self.eigs[name] = self.load_item('{}_eigs'.format(name), np.ndarray)
            self.eigv[name] = self.load_item('{}_eigv'.format(name), np.ndarray)

    def set_object_params(self, obj, idx=None, **kwargs):
        self.grid.set_object_params(obj, idx=idx, **kwargs)