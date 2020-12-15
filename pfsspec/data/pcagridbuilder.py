import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
import time
from tqdm import tqdm

from pfsspec.pfsobject import PfsObject
from pfsspec.physics import Physics
from pfsspec.data.pcagrid import PCAGrid

class PCAGridBuilder(PfsObject):
    def __init__(self, input_grid=None, output_grid=None, orig=None):
        super(PCAGridBuilder, self).__init__()

        if isinstance(orig, PCAGridBuilder):
            self.input_grid = input_grid if input_grid is not None else orig.input_grid
            self.output_grid = output_grid if output_grid is not None else orig.output_grid
            self.grid_index = None

            self.top = orig.top
            self.svd_method = orig.svd_method
            self.truncate = orig.truncate
        else:
            self.input_grid = input_grid
            self.output_grid = output_grid
            self.grid_index = None

            self.top = None
            self.svd_method = 'svd'
            self.truncate = None

    def add_args(self, parser):
        parser.add_argument('--top', type=int, default=None, help='Limit number of results')
        parser.add_argument('--svd-method', type=str, default='svd', choices=['svd', 'trsvd'], help='Truncate PCA')
        parser.add_argument('--truncate', type=int, default=None, help='Truncate PCA')

    def parse_args(self, args):
        if 'top' in args and args['top'] is not None:
            self.top = args['top']

        if 'svd_method' in args and args['svd_method'] is not None:
            self.svd_method = args['svd_method']

        if 'truncate' in args and args['truncate'] is not None:
            self.truncate = args['truncate']

    def create_grid(self):
        raise NotImplementedError()

    def create_pca_grid(self):
        raise NotImplementedError()

    def open_data(self, input_path, output_path):
        raise NotImplementedError()

    def save_data(self, output_path):
        raise NotImplementedError()

    def get_vector_count(self):
        # Return the number of data vectors
        return self.grid_index.shape[1]

    def get_vector_shape(self):
        # Return the shape of data vectors, a one element tuple
        raise NotImplementedError()

    def get_vector(self, i):
        # Return the ith data vector
        raise NotImplementedError()

    def init_process(self):
        pass

    def run(self):
        # TODO: clean it up

        # Copy all data vectors into a single matrix
        vector_count = self.get_vector_count()
        if self.top is not None:
            vector_count = min(self.top, vector_count)

        vector_shape = self.get_vector_shape()
        data_shape = (vector_count,) + vector_shape

        # Build the data matrix and calculate the covariance
        X = np.empty(data_shape)
        for i in tqdm(range(vector_count)):
            v = self.get_vector(i)
            X[i, :] = v
        C = np.matmul(X.transpose(), X)

        # Compute the SVD of the covariance matrix
        start = time.time()

        if self.svd_method == 'svd' or self.truncate is None:
            _, S, V = np.linalg.svd(C, full_matrices=False)
            V = V.transpose()
        elif self.svd_method == 'trsvd':
            svd = TruncatedSVD(n_components=self.truncate)
            svd.fit(C)
            S = svd.singular_values_            # shape: (truncate,)
            V = svd.components_.transpose()     # shape: (dim, truncate)

        end = time.time()
        elapsed = end - start

        # Calculate principal components
        if self.truncate is None:
            PC = V * X.transpose()
        else:
            PC = np.dot(X, V[:, :self.truncate])       # shape: (items, truncate)

        self.output_grid.truncate = self.truncate
        self.output_grid.value_shapes['params'] = self.input_grid.value_shapes['params']
        self.output_grid.value_shapes['coeffs'] = (PC.shape[1],)
        #self.output_grid.value_shapes['rbf'] = self.input_grid.wave
        self.output_grid.allocate_values()

        # These are not grid value, set them separately
        self.output_grid.eigs = S[:self.truncate]
        self.output_grid.eigv = V[:, :self.truncate]
        # These are grid values, use setter functions
        self.output_grid.set_value('params', self.input_grid.get_value('params'))

        # Save principal components to the grid
        coeffs = np.full(self.output_grid.get_shape() + (PC.shape[1],), np.nan)
        for i in range(vector_count):
            idx = tuple(self.grid_index[:, i])
            coeffs[idx] = PC[i, :]
        self.output_grid.set_value('coeffs', coeffs)
