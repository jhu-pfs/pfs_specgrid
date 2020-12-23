import logging
import numpy as np

from pfsspec.pfsobject import PfsObject
from pfsspec.data.gridaxis import GridAxis

class PcaGrid(PfsObject):
    # Wraps an ArrayGrid or an RbfGrid and adds PCA decompression support

    def __init__(self, grid, orig=None):
        super(PcaGrid, self).__init__(orig=orig)

        if isinstance(orig, PcaGrid):
            self.grid = grid if grid is not None else orig.grid

            self.eigs = orig.eigs
            self.eigv = orig.eigv
        else:
            self.grid = grid

            self.eigs = {}
            self.eigv = {}

            self.init_values()

    def get_shape(self):
        return self.grid.get_shape()

    def init_axes(self):
        raise NotImplementedError()

    def init_axis(self, name, values=None):
        self.grid.init_axis(name, values=values)

    def set_axes(self, axes):
        self.grid.set_axes(axes)

    def get_axes(self):
        return self.grid.get_axes()

    def build_axis_indexes(self):
        self.grid.build_axis_indexes()

    def save_axes(self):
        self.grid.save_axes()

    def load_axes(self):
        self.grid.load_axes()

    def get_constant(self, name):
        return self.grid.get_constant(name)

    def set_constant(self, name, value):
        self.grid.set_constant(name, value)

    def init_values(self):
        raise NotImplementedError()

    def init_value(self, name, shape=None, pca=False):
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

    def get_index(self, **kwargs):
        return self.grid.get_index(**kwargs)

    def get_params(self, idx):
        return self.grid.get_params(idx)

    def set_value(self, name, value, s=None, pca=False, **kwargs):
        if not pca:
            self.grid.set_value(name, value, s=s, **kwargs)
        else:
            # TODO: slicing?
            self.grid.set_value(name, value[0], s=s, **kwargs)
            self.eigs[name] = value[1]
            self.eigv[name] = value[2]

    def get_value_at(self, name, idx, s=None):
        if name in self.eigs:
            raise NotImplementedError()
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