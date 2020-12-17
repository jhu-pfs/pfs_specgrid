import os
import numpy as np
import time
from tqdm import tqdm

from pfsspec.physics import Physics
from pfsspec.data.pcagridbuilder import PCAGridBuilder

class ModelPCAGridBuilder(PCAGridBuilder):
    def __init__(self, grid=None, orig=None):
        super(ModelPCAGridBuilder, self).__init__(orig=orig)
    
    def add_args(self, parser):
        super(ModelPCAGridBuilder, self).add_args(parser)

        # Axes of input grid can be used as parameters to filter the range
        grid = self.create_grid()
        grid.add_args(parser)

    def parse_args(self):
        super(ModelPCAGridBuilder, self).parse_args()

    def open_data(self, input_path, output_path):
        # Open input
        fn = os.path.join(input_path, 'spectra') + '.h5'
        self.input_grid = self.create_grid()
        self.input_grid.load(fn)
        self.input_grid.init_from_args(self.args)
        self.input_grid.build_axis_indexes()

        # Source indexes
        index = self.input_grid.get_sliced_value_index('flux')
        self.input_grid_index = np.array(np.where(index))

        # Target indexes
        # TODO: rename members
        index = self.input_grid.get_value_index('flux')
        self.output_grid_index = np.array(np.where(index))

        self.grid_shape = self.input_grid.get_shape()

        # Open output
        fn = os.path.join(output_path, 'spectra') + '.h5'
        self.output_grid = self.create_pca_grid()
        # TODO: copy axes
        self.output_grid.axes = self.input_grid.get_axes()
        self.output_grid.build_axis_indexes()

        # DEBUG
        self.output_grid.preload_arrays = True
        # END DEBUG

        self.output_grid.filename = fn
        self.output_grid.fileformat = 'h5'

    def save_data(self, output_path):
        self.output_grid.save(self.output_grid.filename, format=self.output_grid.fileformat)

    def get_vector_shape(self):
        return self.input_grid.wave.shape

    def get_vector(self, i):
        idx = tuple(self.input_grid_index[:, i])
        spec = self.input_grid.get_model(idx)
        return spec.flux - spec.cont

    def run(self):
        super(ModelPCAGridBuilder, self).run()

        self.output_grid.wave = self.input_grid.wave

        # # Build RBF for model parameters
        # pad_params, pad_axes = self.output_grid.get_value_padded('params', extrapolation='ijk')
        # rbf = self.output_grid.interpolate_value_rbf(pad_params, pad_axes, mask=None, function='multiquadric', epsilon=None, smooth=0.0)
        # self.output_grid.params_rbf = rbf.nodes     # shape: (nodes, params)

        # # Build RBF on principal components
        # pad_coeffs, pad_axes = self.output_grid.get_value_padded('coeffs', extrapolation='ijk')
        # rbf = self.output_grid.interpolate_value_rbf(pad_coeffs, pad_axes, mask=None, function='multiquadric', epsilon=None, smooth=0.0)
        # self.output_grid.coeffs_rbf = rbf.nodes     # shape: (nodes, coeffs)
        
    