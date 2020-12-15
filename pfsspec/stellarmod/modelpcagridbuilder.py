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

    def parse_args(self, args):
        super(ModelPCAGridBuilder, self).parse_args(args)

    def open_data(self, input_path, output_path):
        # Open input
        fn = os.path.join(input_path, 'spectra') + '.h5'
        self.input_grid = self.create_grid()
        self.input_grid.load(fn)

        index = self.input_grid.value_indexes['flux']
        self.grid_index = np.array(np.where(index))

        # Open output
        fn = os.path.join(output_path, 'spectra') + '.h5'
        self.output_grid = self.create_pca_grid()

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
        # TODO: specialize this for every grid
        idx = tuple(self.grid_index[:, i])
        spec = self.input_grid.get_model(idx)
        return spec.flux - spec.cont

    def run(self):
        super(ModelPCAGridBuilder, self).run()

        self.output_grid.wave = self.input_grid.wave
    