import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
import time
from tqdm import tqdm

from pfsspec.data.gridbuilder import GridBuilder

class PcaGridBuilder(GridBuilder):
    def __init__(self, input_grid=None, output_grid=None, orig=None):
        super(PcaGridBuilder, self).__init__(input_grid=input_grid, output_grid=output_grid, orig=orig)

        if isinstance(orig, PcaGridBuilder):
            self.svd_method = orig.svd_method
            self.svd_truncate = orig.svd_truncate

            self.X = orig.X
            self.C = orig.C
            self.S = orig.S
            self.V = orig.V
            self.PC = orig.PC
        else:
            self.svd_method = 'svd'
            self.svd_truncate = None

            self.X = None
            self.C = None
            self.S = None
            self.V = None
            self.PC = None

    def add_args(self, parser):
        super(PcaGridBuilder, self).add_args(parser)

        parser.add_argument('--svd-method', type=str, default='svd', choices=['svd', 'trsvd'], help='Truncate PCA')
        parser.add_argument('--svd-truncate', type=int, default=None, help='Truncate SVD')

    def parse_args(self):
        super(PcaGridBuilder, self).parse_args()

        self.svd_method = self.get_arg('svd_method', self.svd_method)
        self.svd_truncate = self.get_arg('svd_truncate', self.svd_truncate)

    def get_vector_shape(self):
        # Return the shape of data vectors, a one element tuple
        raise NotImplementedError()

    def get_vector(self, i):
        # Return the ith data vector
        raise NotImplementedError()

    def run(self):
        # TODO: clean it up

        # Copy all data vectors into a single matrix
        vector_count = self.get_input_count()
        vector_shape = self.get_vector_shape()
        data_shape = (vector_count,) + vector_shape

        # Build the data matrix and calculate the covariance
        self.X = np.empty(data_shape)
        for i in tqdm(range(vector_count)):
            v = self.get_vector(i)
            self.X[i, :] = v
        self.C = np.matmul(self.X.transpose(), self.X)

        # Compute the SVD of the covariance matrix
        start = time.time()

        if self.svd_method == 'svd' or self.svd_truncate is None:
            _, self.S, self.V = np.linalg.svd(self.C, full_matrices=False)
            self.V = self.V.transpose()
        elif self.svd_method == 'trsvd':
            svd = TruncatedSVD(n_components=self.svd_truncate)
            svd.fit(self.C)
            self.S = svd.singular_values_            # shape: (truncate,)
            self.V = svd.components_.transpose()     # shape: (dim, truncate)

        end = time.time()
        elapsed = end - start

        # Calculate principal components
        if self.svd_truncate is None:
            self.PC = np.dot(self.X, self.V)
        else:
            self.PC = np.dot(self.X, self.V[:, :self.svd_truncate])       # shape: (items, truncate)

        self.output_grid.svd_truncate = self.svd_truncate
