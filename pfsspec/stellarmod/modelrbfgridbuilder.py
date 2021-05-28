import os
import numpy as np
import time
from tqdm import tqdm

from pfsspec.data.arraygrid import ArrayGrid
from pfsspec.data.rbfgrid import RbfGrid
from pfsspec.data.pcagrid import PcaGrid
from pfsspec.data.rbfgridbuilder import RbfGridBuilder
from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.stellarmod.modelgridbuilder import ModelGridBuilder
from pfsspec.util.array_filters import *

class ModelRbfGridBuilder(RbfGridBuilder, ModelGridBuilder):

    STEPS = ['fit', 'pca']

    def __init__(self, config, grid=None, orig=None):
        RbfGridBuilder.__init__(self, orig=orig)
        ModelGridBuilder.__init__(self, config, orig=orig)

        if isinstance(orig, ModelRbfGridBuilder):
            self.step = orig.step
            self.flux = orig.flux
        else:
            self.step = None
            self.flux = False

    def add_args(self, parser):
        RbfGridBuilder.add_args(self, parser)
        ModelGridBuilder.add_args(self, parser)

        parser.add_argument('--step', type=str, choices=ModelRbfGridBuilder.STEPS, help='RBF step to perform.\n')
        parser.add_argument('--flux', action='store_true', help='Compute RBF of flux.\n')

    def parse_args(self):
        RbfGridBuilder.parse_args(self)
        ModelGridBuilder.parse_args(self)

        self.step = self.get_arg('step', self.step)
        self.flux = self.get_arg('flux', self.flux)

    def create_input_grid(self):
        # It doesn't really matter if the input is already a PCA grid or just a direct
        # array because RBF interpolation is the same. On the other hand,
        # when we want to slice a PCA grid in wavelength, we have to load the
        # eigenvectors so this needs to be extended here.
        self.pca = (self.step == 'pca')
        return ModelGridBuilder.create_input_grid(self)

    def open_input_grid(self, input_path):
        return ModelGridBuilder.open_input_grid(self, input_path)

    def create_output_grid(self):
        config = self.config
        if self.step == 'pca':
            config = type(config)(pca=True)
        grid = ModelGrid(config, RbfGrid)
        return grid

    def open_output_grid(self, output_path):
        return ModelGridBuilder.open_output_grid(self, output_path)

    def open_data(self, input_path, output_path, params_path=None):
        return ModelGridBuilder.open_data(self, input_path, output_path, params_path=params_path)

    def build_data_index(self):
        return ModelGridBuilder.build_data_index(self)

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

    def build_rbf(self, params_grid, output_grid, name, s=None):
        value = params_grid.get_value(name)[s or ()]
        mask = params_grid.get_value_index(name)
        axes = params_grid.get_axes()

        if self.padding:
            value, axes, mask = pad_array(axes, value, mask=mask)
            self.logger.info('Array `{}` padded to shape {}'.format(name, value.shape))

        self.logger.info('Fitting RBF to array `{}`'.format(name))
        rbf = self.fit_rbf(value, axes, mask=mask)
        output_grid.set_value(name, rbf)

    def fit_params(self, params_grid, output_grid):
        # Calculate RBF interpolation of continuum fit parameters
        # This is done parameter by parameter so continuum models which cannot
        # be fitted everywhere are still interpolated to as many grid positions
        # as possible

        for name in params_grid.continuum_model.get_params_names():
            # TODO: can we run this with a PcaGrid output?
            if params_grid.array_grid.has_value(name):
                self.build_rbf(params_grid.array_grid, output_grid.rbf_grid, name)

    def copy_params(self, params_grid, output_grid):
        # Copy RBF interpolation of continuum fit parameters from an existing grid

        output_grid.set_constants(params_grid.get_constants())
        output_grid.set_wave(params_grid.get_wave())

        for name in self.continuum_model.get_params_names():
            self.copy_rbf(params_grid.grid, output_grid.grid.grid, name)

    def fit_flux(self, input_grid, output_grid):
        # Calculate RBF interpolation in the flux vector directly

        # The wave slice should apply to the last dimension of the flux grid
        if input_grid.wave_slice is not None:
            s = (Ellipsis, input_grid.wave_slice)
        else:
            s = None

        output_grid.set_wave(input_grid.get_wave())

        for name in ['flux', 'cont']:
            if input_grid.grid.has_value(name):
                self.build_rbf(input_grid.array_grid, output_grid.rbf_grid, name, s=s)

    def run_step_fit(self):
        # Fit RBF to continuum parameters or/and the flux directly
    
        if self.params_grid is not None:
            params_grid = self.params_grid
        elif self.input_grid.continuum_model is not None:
            params_grid = self.input_grid
        else:
            params_grid = None

        # Copy continuum fit parameters
        if params_grid is not None:
            self.copy_wave(params_grid, self.output_grid)
            self.copy_constants(params_grid, self.output_grid)
            self.fit_params(params_grid, self.output_grid)
        
        if self.flux:
            self.copy_wave(self.input_grid, self.output_grid)
            self.fit_flux(self.input_grid, self.output_grid)

    def run_step_pca(self):
        # Calculate the RBF interpolation of the principal components. Optionally,
        # if a params grid is specified, copy the existing RBF interpolation of
        # those to the output grid.

        if self.params_grid is not None:
            self.copy_wave(self.input_grid, self.output_grid)
            self.copy_constants(self.params_grid, self.output_grid)

            if self.rbf:
                # Copy RBF interpolation of continuum parameters
                self.copy_params(self.params_grid, self.output_grid)
            else:
                # Run interpolation of continuum parameters
                self.fit_params(self.params_grid, self.output_grid)
        else:
            # Run interpolation of continuum parameters taken from the PCA grid
            self.copy_wave(self.input_grid, self.output_grid)
            self.copy_constants(self.input_grid, self.output_grid)

            self.fit_params(self.input_grid, self.output_grid)

        # Calculate RBF interpolation of principal components
        grid = self.input_grid.grid
        for name in ['flux', 'cont']:
            if self.input_grid.has_value(name):
                self.build_rbf(self.input_grid.grid.grid, self.output_grid.grid.grid, name)

        # Copy eigenvalues and eigenvectors
        for name in ['flux', 'cont']:
            self.output_grid.grid.eigs[name] = self.input_grid.grid.eigs[name]
            self.output_grid.grid.eigv[name] = self.input_grid.grid.eigv[name]

    def run(self):
        if self.step == 'fit':
            self.run_step_fit()
        elif self.step == 'pca':
            self.run_step_pca()
        else:
            raise NotImplementedError()

#endregion