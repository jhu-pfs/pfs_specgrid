import os
import numpy as np
import time
from tqdm import tqdm

from pfsspec.physics import Physics
from pfsspec.data.arraygrid import ArrayGrid
from pfsspec.data.pcagridbuilder import PcaGridBuilder
from pfsspec.stellarmod.modelgrid import ModelGrid

class ModelPcaGridBuilder(PcaGridBuilder):
    def __init__(self, config, orig=None):
        super(ModelPcaGridBuilder, self).__init__(orig=orig)

        if isinstance(orig, ModelPcaGridBuilder):
            self.config = config if config is not None else orig.config
            self.normalization = orig.normalization
        else:
            self.config = config
            self.normalization = None

    def add_args(self, parser):
        super(ModelPcaGridBuilder, self).add_args(parser)

        parser.add_argument('--normalization', type=str, default='none', choices=['none', 'max', 'planck'], help='Normalization method.\n')

    def parse_args(self):
        super(ModelPcaGridBuilder, self).parse_args()

        self.normalization = self.get_arg('normalization', self.normalization)
    
    def create_input_grid(self):
        return ModelGrid(self.config, ArrayGrid)

    def create_output_grid(self):
        config = type(self.config)(orig=self.config, pca=True)
        return ModelGrid(config, ArrayGrid)

    def open_input_grid(self, input_path):
        fn = os.path.join(input_path, 'spectra') + '.h5'
        self.input_grid = self.create_input_grid()
        self.input_grid.load(fn)

    def open_output_grid(self, output_path):
        fn = os.path.join(output_path, 'spectra') + '.h5'
        self.output_grid = self.create_output_grid()

        # Copy data from the input grid
        self.output_grid.set_axes(self.input_grid.get_axes())
        self.output_grid.wave = self.input_grid.get_wave()
        self.output_grid.build_axis_indexes()

        # Force creating output file for direct hdf5 writing
        self.output_grid.save(fn, format='h5')

        # DEBUG
        # self.output_grid.preload_arrays = True
        # self.output_grid.grid.preload_arrays = True
        # END DEBUG

    def get_vector_shape(self):
        return self.input_grid.get_wave().shape

    def get_vector(self, i):
        # When fitting, the output fluxes will be already normalized, so
        # here we return the flux field only
        idx = tuple(self.input_grid_index[:, i])
        spec = self.input_grid.get_model_at(idx)

        if self.normalization is None or self.normalization == 'none':
            pass
        elif self.normalization == 'max':
            spec.multiply(1 / spec.flux.max())
        elif self.normalization == 'planck':
            spec.normalize_by_T_eff()
        else:
            raise NotImplementedError()

        return spec.flux

    def run(self):
        super(ModelPcaGridBuilder, self).run()

        # Copy continuum fit parameters
        if self.input_grid.grid.has_value('params'):
            params = self.input_grid.get_value_sliced('params')
            self.output_grid.grid.allocate_value('params', shape=(params.shape[-1],))
            self.output_grid.grid.set_value('params', params)
        if self.input_grid.grid.has_constant('constants'):
            self.output_grid.grid.set_constant('constants', self.input_grid.grid.get_constant('constants'))

        # Save principal components to a grid
        coeffs = np.full(self.input_grid.get_shape() + (self.PC.shape[1],), np.nan)
        input_count = self.get_input_count()
        for i in range(input_count):
            idx = tuple(self.output_grid_index[:, i])
            coeffs[idx] = self.PC[i, :]

        self.output_grid.grid.allocate_value('flux', shape=coeffs.shape, pca=True)
        self.output_grid.grid.set_value('flux', (coeffs, self.S, self.V), pca=True)

        """
        # Pad the array of continuum fit parameters, fit with RBF and save intou output array
        padded_params, padded_axes = self.input_grid.get_value_padded('params', interpolation='ijk')
        rbf = self.output_grid.fit_rbf(padded_params, padded_axes, mask=None, function='multiquadric', epsilon=None, smooth=0.0)
        self.output_grid.set_value('params', rbf)
        """
        
    