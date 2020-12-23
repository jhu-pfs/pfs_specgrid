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

    def init_value(self, name, shape=None):
        self.values[name] = None
        self.value_shapes[name] = shape

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
        for name in self.values:
            if self.values[name] is not None:
                self.logger.info('Saving RBF "{}" of size {}'.format(name, self.values.nodes.shape))
                self.save_item('{}_rbf_xi'.format(name), self.values[name].xi)
                self.save_item('{}_rbf_nodes'.format(name), self.values[name].nodes)
                self.logger.info('Saved RBF "{}" of size {}'.format(name, self.values[name].shape))

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
        return self.get_values(s=None, **kwargs)
    
    def fit_rbf(self, value, axes, mask=None, function='multiquadric', epsilon=None, smooth=0.0):
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

        rbf = Rbf()
        rbf.fit(*points, value, function=function, epsilon=epsilon, smooth=smooth, mode=mode)

        return rbf