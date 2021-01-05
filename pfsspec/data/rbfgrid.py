import logging
import numpy as np

from pfsspec.rbf import Rbf
from pfsspec.data.grid import Grid
from pfsspec.data.gridaxis import GridAxis

class RbfGrid(Grid):
    # Implements a grid that supports interpolation based on RBF. This is not
    # necessarily a grid, as RBF interpolation extends function to any point
    # but nodes are defined based on the predefined values along the axes.

    def __init__(self, orig=None):
        super(RbfGrid, self).__init__(orig=orig)

        if isinstance(orig, RbfGrid):
            self.values = orig.values
            self.value_shapes = orig.value_shapes
        else:
            self.values = {}
            self.value_shapes = {}

            self.init_values()

    def init_value(self, name, shape=None, **kwargs):
        if shape is None:
            self.values[name] = None
            self.value_shapes[name] = shape
        else:
            if len(shape) != 1:
                raise NotImplementedError()

            self.value_shapes[name] = shape

            valueshape = self.xi.shape[:1] + tuple(shape)

            if self.preload_arrays:
                self.logger.info('Initializing memory for RBF "{}" of size {}...'.format(name, valueshape))
                self.values[name] = np.full(valueshape, np.nan)
                self.logger.info('Initialized memory for RBF "{}" of size {}.'.format(name, valueshape))
            else:
                self.values[name] = None
                self.logger.info('Initializing data file for RBF "{}" of size {}...'.format(name, valueshape))
                if not self.has_item(name):
                    self.allocate_item(name, valueshape, dtype=np.float)
                self.logger.info('Skipped memory initialization for RBF "{}". Will read random slices from storage.'.format(name))

    def allocate_value(self, name, shape=None):
        if shape is not None:
            self.value_shapes[name] = shape
        self.init_value(name, self.value_shapes[name])

    def get_nearest_index(self, **kwargs):
        return self.get_index(**kwargs)

    def get_index(self, **kwargs):
        # Convert values to axis index, the input to RBF
        idx = ()
        for k in self.axes:
            if k in kwargs:
                idx += (self.axes[k].ip_to_index(kwargs[k]),)
            else:
                idx += (None,)
        return idx

    def get_params(self, idx):
        # Convert axis index to axis values
        params = {}
        for i, k in enumerate(self.axes):
            if idx[i] is not None:
                params[k] = self.axes[k].ip_to_value(idx[i])
            else:
                params[k] = None
        return params

    def has_value(self, name):
        if self.preload_arrays:
            return name in self.values and self.values[name] is not None
        else:
            return name in self.values and self.has_item(name)

    def set_values(self, values, s=None, **kwargs):
        for k in values:
            self.set_value(k, values[k], s=s, **kwargs)

    def set_value(self, name, value, s=None, **kwargs):
        if s is not None or len(kwargs) > 0:
            raise Exception('RbfGrid does not support slicing and indexing.')
        if not isinstance(value, Rbf):
            raise Exception('Value must be an Rbf object.')
        self.values[name] = value

    def get_value_at(self, name, idx, s=None):
        # TODO: implement slicing along the value dimensions if necessary
        if s is not None:
            raise NotImplementedError()

        idx = Grid.rectify_index(idx)
        return self.values[name](idx)

    def get_value(self, name, s=None, **kwargs):
        idx = self.get_index(**kwargs)
        return self.get_value_at(name, idx, s=s)

    def get_values_at(self, idx, s=None):
        return {name: self.get_value_at(name, idx, s) for name in self.values}

    def get_values(self, s=None, **kwargs):
        idx = self.get_index(**kwargs)
        return self.get_values_at(idx, s)

    def save_items(self):
        super(RbfGrid, self).save_items()
        self.save_values()

    def save_values(self):
        # TODO: implement chunking along the value dimensions if necessary
        # TODO: this is a little bit redundant here because we store the xi values
        #       for each RBF. The coordinates are supposed to be the same for
        #       all data values.
        for name in self.values:
            if self.values[name] is not None:
                self.logger.info('Saving RBF "{}" of size {}'.format(name, self.values[name].nodes.shape))
                self.save_item('{}_rbf_xi'.format(name), self.values[name].xi)
                self.save_item('{}_rbf_nodes'.format(name), self.values[name].nodes)
                self.logger.info('Saved RBF "{}" of size {}'.format(name, self.values[name].nodes.shape))

    def load_items(self, s=None):
        super(RbfGrid, self).load_items(s=s)
        self.load_values(s=s)

    def load_rbf(self, xi, nodes, function='multiquadric', epsilon=None, smooth=0.0):
        rbf = Rbf()
        rbf.load(nodes, xi, mode='N-D')
        return rbf

    def load_values(self, s=None):
        # TODO: implement chunked lazy loading along the value dimension, if necessary
        if s is not None:
            raise NotImplementedError()

        for name in self.values:
            self.logger.info('Loading RBF "{}" of size {}'.format(name, s))
            xi = self.load_item('{}_rbf_xi'.format(name), np.ndarray)
            nodes = self.load_item('{}_rbf_nodes'.format(name), np.ndarray)
            self.values[name] = self.load_rbf(xi, nodes)
            self.logger.info('Loaded RBF "{}" of size {}'.format(name, s))

    def set_object_params(self, obj, idx=None, **kwargs):
        if idx is not None:
            for i, p in enumerate(self.axes):
                raise NotImplementedError()
                # TODO: interpolate here
                # setattr(obj, p, float(self.axes[p].values[idx[i]]))
        if kwargs is not None:
            for p in kwargs:
                setattr(obj, p, float(kwargs[p]))

    def interpolate_value_rbf(self, name, **kwargs):
        return self.get_value(name, s=None, **kwargs)
    