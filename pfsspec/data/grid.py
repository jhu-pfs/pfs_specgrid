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
            self.constants = orig.constants
        else:
            self.preload_arrays = False
            self.axes = {}
            self.constants = {}

            self.init_axes()
            self.init_constants()

    def get_shape(self):
        shape = tuple(self.axes[p].values.shape[0] for p in self.axes)
        return shape

    def ensure_lazy_load(self):
        # This works with HDF5 format only!
        if not self.preload_arrays and self.fileformat != 'h5':
            raise NotImplementedError()

    def get_constants(self):
        return self.constants

    def set_constants(self, constants):
        self.constants = constants

    def init_axes(self):
        pass

    def init_axis(self, name, values=None):
        self.axes[name] = GridAxis(name, values)

    def build_axis_indexes(self):
        for p in self.axes:
            self.axes[p].build_index()

    def set_axes(self, axes):
        self.axes = {}
        for k in axes:
            self.axes[k] = type(axes[k])(k, orig=axes[k])
            self.axes[k].build_index()

    def get_axes(self):
        return self.axes

    def init_constants(self):
        pass

    def init_constant(self, name):
        self.constants[name] = None

    def get_constant(self, name):
        return self.constants[name]

    def set_constant(self, name, value):
        self.constants[name] = value

    def save_constants(self):
        for p in self.constants:
            self.save_item(p, self.constants[p])

    def load_constants(self):
        constants = {}
        for p in self.constants:
            if self.has_item(p):
                constants[p] = self.load_item(p, np.ndarray)
        self.constants = constants

    def init_values(self):
        pass

    def allocate_values(self):
        raise NotImplementedError()

    def add_args(self, parser):
        for k in self.axes:
            parser.add_argument('--' + k, type=float, nargs='*', default=None, help='Limit on ' + k)

    def init_from_args(self, args):
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
        self.save_constants()

    def load(self, filename, s=None, format=None):
        super(Grid, self).load(filename, s=s, format=format)
        self.build_axis_indexes()

    def load_items(self, s=None):
        self.load_axes()
        self.load_constants()

#region Indexing utility functions

    @staticmethod
    def rectify_index(idx, s=None):
        idx = tuple(idx)
        if isinstance(s, Iterable):
            idx = idx + tuple(s)
        elif s is not None:
            idx = idx + (s,)

        return tuple(idx)

    def set_object_params(self, obj, idx=None, **kwargs):
        if idx is not None:
            for i, p in enumerate(self.axes):
                setattr(obj, p, float(self.axes[p].values[idx[i]]))
        if kwargs is not None:
            for p in kwargs:
                setattr(obj, p, float(kwargs[p]))

#endregion