import logging
import numpy as np
from collections import Iterable

from pfsspec.pfsobject import PfsObject
from pfsspec.data.gridaxis import GridAxis

class Grid(PfsObject):
    def __init__(self, orig=None):
        super(Grid, self).__init__(orig=orig)

        if isinstance(orig, Grid):
            self.preload_arrays = orig.preload_arrays
            self.axes = orig.axes
        else:
            self.preload_arrays = False
            self.axes = {}

            self.init_axes()

    def get_shape(self):
        shape = tuple(self.axes[p].values.shape[0] for p in self.axes)
        return shape

    def ensure_lazy_load(self):
        # This works with HDF5 format only!
        if not self.preload_arrays and self.fileformat != 'h5':
            raise NotImplementedError()

    def init_axes(self):
        pass

    def init_axis(self, name, values=None):
        self.axes[name] = GridAxis(name, values)

    def set_axes(self, axes):
        self.axes = axes

    def get_axes(self):
        return self.axes

    def build_axis_indexes(self):
        for p in self.axes:
            self.axes[p].build_index()

    def init_values(self):
        pass

    def allocate_values(self):
        raise NotImplementedError()

    @staticmethod
    def rectify_index(idx, s=None):
        idx = tuple(idx)
        if isinstance(s, Iterable):
            idx = idx + tuple(s)
        elif s is not None:
            idx = idx + (s,)

        return tuple(idx)

    def save_axes(self):
        for p in self.axes:
            self.save_item(p, self.axes[p].values)

    def load_axes(self):
        # TODO: This might not be the best solution
        # Throw away axes that are not in the data grid, a reason might be
        # that the grid was squeezed during transformation
        axes = {}
        for p in self.axes:
            if self.has_item(p):
                self.axes[p].values = self.load_item(p, np.ndarray)
                axes[p] = self.axes[p]
        self.axes = axes

    def save_items(self):
        self.save_axes()

    def load(self, filename, s=None, format=None):
        super(Grid, self).load(filename, s=s, format=format)
        self.build_axis_indexes()

    def load_items(self, s=None):
        self.load_axes()