import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
import time
from tqdm import tqdm

from pfsspec.pfsobject import PfsObject

class PcaGridBuilder(PfsObject):
    def __init__(self, input_grid=None, output_grid=None, orig=None):
        super(PcaGridBuilder, self).__init__()

        if isinstance(orig, PcaGridBuilder):
            self.input_grid = input_grid if input_grid is not None else orig.input_grid
            self.output_grid = output_grid if output_grid is not None else orig.output_grid
            self.input_grid_index = None
            self.output_grid_index = None
            self.grid_shape = None

            self.top = orig.top
            self.svd_method = orig.svd_method
            self.svd_truncate = orig.svd_truncate

            self.X = orig.X
            self.C = orig.C
            self.S = orig.S
            self.V = orig.V
            self.PC = orig.PC
        else:
            self.input_grid = input_grid
            self.output_grid = output_grid
            self.input_grid_index = None
            self.output_grid_index = None
            self.grid_shape = None

            self.top = None
            self.svd_method = 'svd'
            self.svd_truncate = None

            self.X = None
            self.C = None
            self.S = None
            self.V = None
            self.PC = None

    def add_args(self, parser):
        parser.add_argument('--top', type=int, default=None, help='Limit number of results')
        parser.add_argument('--svd-method', type=str, default='svd', choices=['svd', 'trsvd'], help='Truncate PCA')
        parser.add_argument('--svd-truncate', type=int, default=None, help='Truncate SVD')

    def parse_args(self):
        self.top = self.get_arg('top', self.top)
        self.svd_method = self.get_arg('svd_method', self.svd_method)
        self.svd_truncate = self.get_arg('svd_truncate', self.svd_truncate)

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
        vector_count = self.input_grid_index.shape[1]
        if self.top is not None:
            vector_count = min(self.top, vector_count)
        return vector_count

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
