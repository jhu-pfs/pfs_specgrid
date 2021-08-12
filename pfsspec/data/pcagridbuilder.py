import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
import logging
from tqdm import tqdm

from pfsspec.util.timer import Timer
from pfsspec.data.gridbuilder import GridBuilder

class PcaGridBuilder(GridBuilder):
    def __init__(self, input_grid=None, output_grid=None, orig=None):
        super(PcaGridBuilder, self).__init__(input_grid=input_grid, output_grid=output_grid, orig=orig)

        if isinstance(orig, PcaGridBuilder):
            self.pca_method = orig.pca_method
            self.svd_method = orig.svd_method
            self.svd_truncate = orig.svd_truncate

            self.X = orig.X
            self.C = orig.C
            self.S = orig.S
            self.V = orig.V
            self.PC = orig.PC
        else:
            self.pca_method = 'cov'
            self.svd_method = 'svd'
            self.svd_truncate = None

            self.X = None
            self.C = None
            self.S = None
            self.V = None
            self.PC = None

    def add_args(self, parser):
        super(PcaGridBuilder, self).add_args(parser)

        parser.add_argument('--pca-method', type=str, default='cov', choices=['svd', 'cov', 'dual'], help='PCA method\n')
        parser.add_argument('--svd-method', type=str, default='svd', choices=['svd', 'trsvd'], help='Truncate PCA')
        parser.add_argument('--svd-truncate', type=int, default=None, help='Truncate SVD')

    def parse_args(self):
        super(PcaGridBuilder, self).parse_args()

        self.pca_method = self.get_arg('pca_method', self.pca_method)
        self.svd_method = self.get_arg('svd_method', self.svd_method)
        self.svd_truncate = self.get_arg('svd_truncate', self.svd_truncate)

    def get_vector_shape(self):
        # Return the shape of data vectors, a one element tuple
        raise NotImplementedError()

    def get_vector(self, i):
        # Return the ith data vector
        raise NotImplementedError()

    def run(self):
        # Copy all data vectors into a single matrix
        vector_count = self.get_input_count()
        vector_shape = self.get_vector_shape()
        data_shape = (vector_count,) + vector_shape

        # Build the data matrix
        self.X = np.empty(data_shape)
        self.logger.info('Allocated {} bytes for data matrix.'.format(self.X.size * self.X.itemsize))
        with Timer('Assembling data matrix...', logging.INFO):
            for i in tqdm(range(vector_count)):
                v = self.get_vector(i)
                self.X[i, :] = v

        if self.pca_method == 'svd':
            self.run_pca_svd()
            self.truncate_and_calculate_pc()
        elif self.pca_method == 'cov':
            self.run_pca_cov()
            self.truncate_and_calculate_pc()
        elif self.pca_method == 'dual':
            self.run_dual_pca()
            self.truncate_and_calculate_pc()
        else:
            raise NotImplementedError()

        # Save options to the grid
        self.output_grid.svd_truncate = self.svd_truncate

    def run_pca_svd(self):
        with Timer('Computing SVD with method `{}`, truncated at {}...'.format(self.svd_method, self.svd_truncate), logging.INFO):
            if self.svd_method == 'svd' or self.svd_truncate is None:
                _, self.S, Vh = np.linalg.svd(self.X, full_matrices=False)
                self.V = Vh.T
            elif self.svd_method == 'trsvd':
                svd = TruncatedSVD(n_components=self.svd_truncate)
                svd.fit(self.X)                          # shape: (items, dim)
                self.S = svd.singular_values_            # shape: (truncate,)
                self.V = svd.components_.T               # shape: (dim, truncate)

    def run_pca_cov(self):
        with Timer('Calculating covariance matrix...', logging.INFO):
            self.C = np.matmul(self.X.T, self.X)        # shape: (dim, dim)
        self.logger.info('Allocated {} bytes for covariance matrix.'.format(self.C.size * self.C.itemsize))

        # Compute the SVD of the covariance matrix
        with Timer('Computing SVD with method `{}`, truncated at {}...'.format(self.svd_method, self.svd_truncate), logging.INFO):
            if self.svd_method == 'svd' or self.svd_truncate is None:
                _, self.S, Vh = np.linalg.svd(self.C, full_matrices=False)
                self.V = Vh.T
            elif self.svd_method == 'trsvd':
                svd = TruncatedSVD(n_components=self.svd_truncate)
                svd.fit(self.C)
                self.S = svd.singular_values_            # shape: (truncate,)
                self.V = svd.components_.transpose()     # shape: (dim, truncate)

    def run_dual_pca(self):
        with Timer('Calculating X X^T matrix...', logging.INFO):
            self.D = np.matmul(self.X, self.X.T)        # shape: (items, items)
        self.logger.info('Allocated {} bytes for X X^T matrix.'.format(self.D.size * self.D.itemsize))

        with Timer('Computing SVD with method `{}`, truncated at {}...'.format(self.svd_method, self.svd_truncate), logging.INFO):
            if self.svd_method == 'svd' or self.svd_truncate is None:
                raise NotImplementedError()
            elif self.svd_method == 'trsvd':
                raise NotImplementedError()

    def truncate_and_calculate_pc(self):
        # Calculate principal components and truncated basis
        with Timer('Calculating principal components...', logging.INFO):
            if self.svd_truncate is None:
                self.PC = np.dot(self.X, self.V)
            else:
                self.S = self.S[:self.svd_truncate]
                self.V = self.V[:, :self.svd_truncate]
                self.PC = np.dot(self.X, self.V)       # shape: (items, truncate)
