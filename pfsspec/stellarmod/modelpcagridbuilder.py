import os
import numpy as np
import time
from tqdm import tqdm

from pfsspec.physics import Physics
from pfsspec.data.pcagridbuilder import PcaGridBuilder

class ModelPcaGridBuilder(PcaGridBuilder):
    def __init__(self, grid=None, orig=None):
        super(ModelPcaGridBuilder, self).__init__(orig=orig)
    
    def add_args(self, parser):
        super(ModelPcaGridBuilder, self).add_args(parser)

        # Axes of input grid can be used as parameters to filter the range
        grid = self.create_grid()
        grid.add_args(parser)

    def parse_args(self):
        super(ModelPcaGridBuilder, self).parse_args()

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
        self.output_grid.set_axes(self.input_grid.get_axes())

        # DEBUG
        self.output_grid.preload_arrays = True
        self.output_grid.grid.preload_arrays = True
        # END DEBUG

        self.output_grid.filename = fn
        self.output_grid.fileformat = 'h5'

    def save_data(self, output_path):
        self.output_grid.save(self.output_grid.filename, format=self.output_grid.fileformat)

    def get_vector_shape(self):
        return self.input_grid.get_wave().shape

    def get_vector(self, i):
        # When fitting, the output fluxes will be already normalized, so
        # here we return the flux field only
        idx = tuple(self.input_grid_index[:, i])
        spec = self.input_grid.get_model(idx)
        return spec.flux

    def run(self):
        super(ModelPcaGridBuilder, self).run()

        # Copy data from the input grid
        self.output_grid.set_axes(self.input_grid.get_axes())
        self.output_grid.wave = self.input_grid.get_wave()

        # Copy continuum fit parameters
        params = self.input_grid.get_value_sliced('params')
        self.output_grid.allocate_value('params', shape=(params.shape[-1],))
        self.output_grid.set_value('params', params)
        self.output_grid.set_constant('constants', self.input_grid.get_constant('constants'))

        # Save principal components to a grid
        coeffs = np.full(self.input_grid.get_shape() + (self.PC.shape[1],), np.nan)
        vector_count = self.get_vector_count()
        for i in range(vector_count):
            idx = tuple(self.output_grid_index[:, i])
            coeffs[idx] = self.PC[i, :]

        self.output_grid.allocate_value('flux', shape=self.V.shape, pca=True)
        self.output_grid.set_value('flux', (coeffs, self.S, self.V), pca=True)

        """
        # Pad the array of continuum fit parameters, fit with RBF and save intou output array
        padded_params, padded_axes = self.input_grid.get_value_padded('params', interpolation='ijk')
        rbf = self.output_grid.fit_rbf(padded_params, padded_axes, mask=None, function='multiquadric', epsilon=None, smooth=0.0)
        self.output_grid.set_value('params', rbf)
        """
        
    