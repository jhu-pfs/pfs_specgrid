import os
import numpy as np
import time
from tqdm import tqdm

from pfsspec.data.gridbuilder import GridBuilder
from pfsspec.rbf import Rbf
from pfsspec.data.arraygrid import ArrayGrid

class RbfGridBuilder(GridBuilder):
    def __init__(self, input_grid=None, output_grid=None, orig=None):
        super(RbfGridBuilder, self).__init__(input_grid=input_grid, output_grid=output_grid, orig=orig)

        if isinstance(orig, RbfGridBuilder):
            self.pca = orig.pca
        else:
            self.pca = None

    def add_args(self, parser):
        super(RbfGridBuilder, self).add_args(parser)
        parser.add_argument('--pca', action='store_true', help='Run on a PCA grid')

    def parse_args(self):
        super(RbfGridBuilder, self).parse_args()
        self.pca = self.get_arg('pca', self.pca)

    def fit_rbf(self, value, axes, mask=None, padding=False, interpolation='ijk', function='multiquadric', epsilon=None, smooth=0.0):
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

        points = ArrayGrid.get_grid_points(axes, padding=padding, interpolation=interpolation)
        # points = np.meshgrid(*[axes[p].values for p in axes], indexing='ij')
        points = np.meshgrid(*[points[p] for p in points], indexing='ij')
        points = [p.flatten() for p in points]
        points = [p[m] for p in points]

        if len(value.shape) == 1:
            mode = '1-D'
        else:
            mode = 'N-D'

        rbf = Rbf()
        rbf.fit(*points, value, function=function, epsilon=epsilon, smooth=smooth, mode=mode)

        return rbf