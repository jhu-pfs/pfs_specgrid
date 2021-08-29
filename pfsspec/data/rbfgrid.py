import logging
import numpy as np

from pfsspec.util.interp.rbf import Rbf
from pfsspec.data.grid import Grid
from pfsspec.data.gridaxis import GridAxis

class RbfGrid(Grid):
    # Implements a grid that supports interpolation based on RBF. This is not
    # necessarily a grid, as RBF interpolation extends function to any point
    # but nodes are defined based on the predefined values along the axes.

    def __init__(self, config=None, orig=None):
        super(RbfGrid, self).__init__(orig=orig)

        if isinstance(orig, RbfGrid):
            self.config = config if config is not None else orig.config
            self.values = orig.values
            self.value_shapes = orig.value_shapes
        else:
            self.config = config
            self.values = {}
            self.value_shapes = {}

            self.init_values()

    @property
    def array_grid(self):
        return None

    @property
    def rbf_grid(self):
        return self

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

    def has_value_index(self, name):
        # RBF doesn't have an index, it's continuous
        return False

    def get_nearest_index(self, **kwargs):
        # RBF doesn't have an index per se, but we can return the interpolated
        # grid values from physical values
        return self.get_index(**kwargs)

    def get_index(self, **kwargs):
        # Convert values to axis index, the input to RBF
        
        # RBF works a little bit differently when a dimension is 1 length than
        # normal arrays because even though that dimension is not squeezed, the
        # RBF indexes are.
        
        idx = ()
        for k in self.axes:
            # Only include index along the axis if it's not a 1-length axis
            if self.axes[k].values.shape[0] > 1:
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
            return name in self.values and self.has_item(name + '/rbf/xi')

    def set_values(self, values, s=None, **kwargs):
        for k in values:
            self.set_value(k, values[k], s=s, **kwargs)

    def set_value(self, name, value, s=None, **kwargs):
        if s is not None or len(kwargs) > 0:
            raise Exception('RbfGrid does not support slicing and indexing.')
        if not isinstance(value, Rbf):
            raise Exception('Value must be an Rbf object.')
        self.values[name] = value

    def has_value_at(self, name, idx, mode='any'):
        return True

    def get_value_at(self, name, idx, s=None):
        idx = Grid.rectify_index(idx)
        value = self.values[name](*idx)
        return value[s or ()]

    def get_value(self, name, s=None, **kwargs):
        idx = self.get_index(**kwargs)
        return self.get_value_at(name, idx, s=s)

    def get_values_at(self, idx, s=None, names=None):
        # Return all values interpolated to idx. This function is optimized for
        # RBF grids where the distance matrix computation is expensive.
        idx = Grid.rectify_index(idx)
        names = names or self.values.keys()
        
        # We can potentially reuse the distance/kernel matrix if the grid points
        # and the kernels are the same
        values = {}
        A = None
        prev = None
        for name in names:
            if prev is None or not prev.is_compatible(self.values[name]):
                prev = self.values[name]
                y, A = self.values[name].eval(*idx)
            else:
                y, A = self.values[name].eval(*idx, A=A)
            values[name] = y

        return values

    def get_values(self, s=None, names=None, **kwargs):
        idx = self.get_index(**kwargs)
        return self.get_values_at(idx, s, names=names)

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
                self.save_item('{}/rbf/xi'.format(name), self.values[name].xi)
                self.save_item('{}/rbf/nodes'.format(name), self.values[name].nodes)
                self.save_item('{}/rbf/c'.format(name), self.values[name].c)
                self.save_item('{}/rbf/function'.format(name), self.values[name].function)
                self.save_item('{}/rbf/epsilon'.format(name), self.values[name].epsilon)
                self.logger.info('Saved RBF "{}" of size {}'.format(name, self.values[name].nodes.shape))

    def load_items(self, s=None):
        super(RbfGrid, self).load_items(s=s)
        self.load_values(s=s)

    def load_rbf(self, xi, nodes, c, function='multiquadric', epsilon=None):
        # TODO: bring out kernel function name as parameter or save into hdf5
        rbf = Rbf()
        rbf.load(nodes, xi, c, function=function, epsilon=epsilon, mode='N-D')
        return rbf

    def load_values(self, s=None):
        # TODO: implement chunked lazy loading along the value dimension, if necessary
        if s is not None:
            raise NotImplementedError()

        for name in self.values:
            self.logger.info('Loading RBF "{}" of size {}'.format(name, s))
            xi = self.load_item('{}/rbf/xi'.format(name), np.ndarray)
            nodes = self.load_item('{}/rbf/nodes'.format(name), np.ndarray)
            c = self.load_item('{}/rbf/c'.format(name), np.ndarray)
            function = self.load_item('{}/rbf/function'.format(name), str)
            epsilon = self.load_item('{}/rbf/epsilon'.format(name), float)
            if xi is not None and nodes is not None:
                # TODO: save function name to hdf5 and load back from there
                self.values[name] = self.load_rbf(xi, nodes, c, function=function, epsilon=epsilon)
                self.logger.info('Loaded RBF "{}" of size {}'.format(name, s))
            else:
                self.values[name] = None
                self.logger.info('Skipped loading RBF "{}" of size {}'.format(name, s))
            
    def set_object_params_idx(self, obj, idx):
        # idx can be squeezed
        i = 0
        for p in self.axes:
            if self.axes[p].values.shape[0] > 1:
                setattr(obj, p, float(self.axes[p].ip_to_value(idx[i])))
                i += 1
            else:
                setattr(obj, p, float(self.axes[p].values[0]))

    def interpolate_value_rbf(self, name, **kwargs):
        return self.get_value(name, s=None, **kwargs)
    