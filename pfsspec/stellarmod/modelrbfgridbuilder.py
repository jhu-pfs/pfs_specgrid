import os
import numpy as np
import time
from tqdm import tqdm

from pfsspec.data.arraygrid import ArrayGrid
from pfsspec.data.rbfgrid import RbfGrid
from pfsspec.data.pcagrid import PcaGrid
from pfsspec.data.rbfgridbuilder import RbfGridBuilder
from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.util.array_filters import *

class ModelRbfGridBuilder(RbfGridBuilder):
    def __init__(self, config, grid=None, orig=None):
        super(ModelRbfGridBuilder, self).__init__(orig=orig)

        if isinstance(orig, ModelRbfGridBuilder):
            self.config = config if config is not None else orig.config
        else:
            self.config = config

    def create_input_grid(self):
        # It doesn't really matter if the input is already a PCA grid or just a direct
        # array because RBF interpolation is the same. On the other hand,
        # when we want to slice a PCA grid in wavelength, we have to load the
        # eigenvectors so this needs to be extended here.
        config = self.config
        if self.pca is not None and self.pca:
            config = type(config)(pca=True)
        grid = ModelGrid(config, ArrayGrid)
        return grid

    def create_output_grid(self):
        config = self.config
        if self.pca is not None and self.pca:
            config = type(config)(pca=True)
        grid = ModelGrid(config, RbfGrid)
        return grid

    def open_input_grid(self, input_path):
        fn = os.path.join(input_path, 'spectra') + '.h5'
        self.input_grid = self.create_input_grid()
        self.input_grid.load(fn, format='h5')

        # Source indexes
        if isinstance(self.input_grid.grid, PcaGrid):
            index = self.input_grid.grid.grid.get_value_index_unsliced('flux')
        else:
            index = self.input_grid.grid.get_value_index_unsliced('flux')
        self.input_grid_index = np.array(np.where(index))
        if self.top is not None:
            self.input_grid_index = self.input_grid_index[:, :min(self.top, self.input_grid_index.shape[1])]

    def open_output_grid(self, output_path):
        fn = os.path.join(output_path, 'spectra') + '.h5'
        self.output_grid = self.create_output_grid()
        
        # Pad the output axes. This automatically takes the parameter ranges into
        # account since the grid is sliced.
        orig_axes = self.input_grid.get_axes()
        if self.padding:
            padded_axes = ArrayGrid.pad_axes(orig_axes)
            self.output_grid.set_axes(padded_axes)
        else:
            self.output_grid.set_axes(orig_axes)

        # DEBUG
        self.output_grid.preload_arrays = True
        self.output_grid.grid.preload_arrays = True
        # END DEBUG

        self.output_grid.filename = fn
        self.output_grid.fileformat = 'h5'

    def run(self):
        self.output_grid.set_constants(self.input_grid.get_constants())
        self.output_grid.set_wave(self.input_grid.get_wave())

        # Copy eigv if PCA
        if isinstance(self.input_grid.grid, PcaGrid):
            for k in self.input_grid.grid.eigs:
                if self.input_grid.grid.eigs[k] is not None:
                    self.output_grid.grid.eigs[k] = self.input_grid.grid.eigs[k]
                    self.output_grid.grid.eigv[k] = self.input_grid.grid.eigv[k][self.input_grid.wave_slice or slice(None)]

            grid = self.input_grid.grid.grid
            value_slice = None                       # Do not slice PCs
        else:
            grid = self.input_grid.grid
            value_slice = self.input_grid.wave_slice

        # Fit RBF
        # TODO: can we do it in one run and use the same xi and distance matrix for all?
        for name in grid.values:
            if grid.has_value(name):              
                value = grid.get_value(name, s=value_slice)
                mask = grid.get_value_index(name)
                axes = grid.get_axes()

                if self.padding:
                    value, axes, mask = pad_array(axes, value, mask=mask)
                    self.logger.info('Array `{}` padded to shape {}'.format(name, value.shape))

                self.logger.info('Fitting RBF to array `{}`'.format(name))
                rbf = self.fit_rbf(value, axes, mask=mask)
                self.output_grid.grid.set_value(name, rbf)

#endregion