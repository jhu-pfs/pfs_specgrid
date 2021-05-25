import os
import numpy as np
import time
from tqdm import tqdm

from pfsspec.physics import Physics
from pfsspec.data.arraygrid import ArrayGrid
from pfsspec.data.pcagridbuilder import PcaGridBuilder
from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.stellarmod.modelgridbuilder import ModelGridBuilder

class ModelPcaGridBuilder(PcaGridBuilder, ModelGridBuilder):
    def __init__(self, config, orig=None):
        PcaGridBuilder.__init__(self, orig=orig)
        ModelGridBuilder.__init__(self, config, orig=orig)

        if isinstance(orig, ModelPcaGridBuilder):
            self.config = config if config is not None else orig.config
            self.normalization = orig.normalization
        else:
            self.config = config
            self.normalization = None

    def add_args(self, parser):
        PcaGridBuilder.add_args(self, parser)
        ModelGridBuilder.add_args(self, parser)

        parser.add_argument('--normalization', type=str, default='none', choices=['none', 'max', 'planck'], help='Normalization method.\n')

    def parse_args(self):
        PcaGridBuilder.parse_args(self)
        ModelGridBuilder.parse_args(self)

        self.normalization = self.get_arg('normalization', self.normalization)
    
    def create_input_grid(self):
        # Input should not be a PCA grid
        return ModelGridBuilder.create_input_grid(self)

    def create_output_grid(self):
        # Output is always a PCA grid
        config = type(self.config)(orig=self.config, pca=True)
        return ModelGrid(config, ArrayGrid)

    def open_input_grid(self, input_path):
        ModelGridBuilder.open_input_grid(self, input_path)

    def open_output_grid(self, output_path):
        ModelGridBuilder.open_output_grid(self, output_path)

    def open_data(self, input_path, output_path, params_path=None):
        return ModelGridBuilder.open_data(self, input_path, output_path, params_path=params_path)

    def build_data_index(self):
        return ModelGridBuilder.build_data_index(self)

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

        # Copy continuum fit parameters, if available
        if self.params_grid is not None:
            for name in self.params_grid.continuum_model.get_params_names():
                params = self.params_grid.get_value_sliced(name)
                index = self.params_grid.grid.get_value_index(name)
                self.output_grid.grid.grid.allocate_value(name, shape=(params.shape[-1],))
                self.output_grid.grid.grid.set_value(name, params)
                self.output_grid.grid.grid.value_indexes[name] = index
        
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

        
    