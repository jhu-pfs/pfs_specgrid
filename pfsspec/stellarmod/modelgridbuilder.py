import os
import numpy as np

from pfsspec.data.arraygrid import ArrayGrid
from pfsspec.data.rbfgrid import RbfGrid
from pfsspec.data.gridbuilder import GridBuilder
from pfsspec.stellarmod.modelgrid import ModelGrid

class ModelGridBuilder():
    # Mixin class for spectrum model specific grid building operations
    #
    # - params_grid holds a reference to the grid from which model parameters
    #   are taken. These can be continuum model fit parameters or else
    # - params_grid_index keeps track of the model grid locations that hold
    #   valid data. This index is combined with the index on input_grid

    def __init__(self, config, orig=None):
        if isinstance(orig, ModelGridBuilder):
            self.params_grid = orig.params_grid
            self.params_grid_index = None

            self.pca = orig.pca
            self.rbf = orig.rbf
            self.config = config if config is not None else orig.config
            self.continuum_model = orig.continuum_model
        else:
            self.params_grid = None
            self.params_grid_index = None

            self.pca = None
            self.rbf = None
            self.config = config
            self.continuum_model = None

    def add_args(self, parser):
        self.config.add_args(parser)
        parser.add_argument('--pca', action='store_true', help='Run on a PCA input grid.')
        parser.add_argument('--rbf', action='store_true', help='Run on an RBF params grid.')

    def parse_args(self):
        self.pca = self.get_arg('pca', self.pca)
        self.rbf = self.get_arg('rbf', self.rbf)
        self.config.init_from_args(self.args)
        self.continuum_model = self.config.create_continuum_model()
        if self.continuum_model is not None:
            self.continuum_model.init_from_args(self.args)

    def create_params_grid(self):
        if self.rbf is not None and self.rbf:
            t = RbfGrid
        else:
            t = ArrayGrid
        grid = ModelGrid(type(self.config)(orig=self.config, normalized=True), t)
        return grid

    def create_input_grid(self):
        # TODO: add support for RBF grid

        config = self.config
        if self.pca is not None and self.pca:
            config = type(config)(pca=True)
        grid = ModelGrid(config, ArrayGrid)
        return grid

    def create_output_grid(self):
        return ModelGrid(self.config, ArrayGrid)

    def open_params_grid(self, params_path):
        fn = os.path.join(params_path, 'spectra') + '.h5'
        self.params_grid = self.create_params_grid()
        self.params_grid.load(fn)

    def open_input_grid(self, input_path):
        fn = os.path.join(input_path, 'spectra') + '.h5'
        self.input_grid = self.create_input_grid()
        self.input_grid.load(fn)

    def open_output_grid(self, output_path):
        self.output_grid = self.create_output_grid()

        # Copy data from the input grid
        g = self.input_grid or self.params_grid

        self.output_grid.set_axes(g.grid.get_axes())
        self.output_grid.set_wave(g.get_wave())
        self.output_grid.build_axis_indexes()
        
        # DEBUG
        # self.output_grid.preload_arrays = True
        # END DEGUB

    def open_data(self, input_path, output_path, params_path=None):
        if params_path is not None:
            self.open_params_grid(params_path)
            self.params_grid.init_from_args(self.args)
            self.params_grid.build_axis_indexes()
            self.grid_shape = self.params_grid.get_shape()

            # Initialize continuum model
            if self.continuum_model is None:
                self.continuum_model = self.params_grid.continuum_model

            if self.continuum_model.wave is None:
                self.continuum_model.init_wave(self.params_grid.wave)
            
            # TODO: do we need this here? This should come from the grid config
            #       when normalized=True
            # self.params_grid.set_continuum_model(self.continuum_model)

        GridBuilder.open_data(self, input_path, output_path)

        if self.continuum_model is None:
            self.continuum_model = self.input_grid.continuum_model
        
        if self.continuum_model.wave is None:
            self.continuum_model.init_wave(self.input_grid.wave)

        # This has to happen after loading the input grid because params_index
        # is combined with the input index with logical and
        if self.params_grid is not None:
            self.build_params_index()

        # Initialize continuum model, if it still isn't
        if self.continuum_model.wave is None:
            self.continuum_model.init_wave(self.input_grid.get_wave())
        self.output_grid.set_continuum_model(self.continuum_model)          
        
        # Force creating output file for direct hdf5 writing
        fn = os.path.join(output_path, 'spectra') + '.h5'
        self.output_grid.save(fn, format='h5')

    def build_data_index(self):
        # Source indexes
        index = self.input_grid.array_grid.get_value_index_unsliced('flux')
        self.input_grid_index = np.array(np.where(index))

        # Target indexes
        index = self.input_grid.array_grid.get_value_index('flux')
        self.output_grid_index = np.array(np.where(index))

    def build_params_index(self):
        # Build and index on the continuum fit parametes. This is a logical AND
        # combination

        if isinstance(self.params_grid.grid, ArrayGrid):
            params_index = None

            for name in self.continuum_model.get_params_names():
                pi, _ = self.get_params_index(name)
                params_index = pi if params_index is None else params_index & pi

            params_index = np.array(np.where(params_index))
            self.params_grid_index = params_index
        else:
            self.params_grid_index = None

    def get_params_index(self, params_name):
        # Source indexes, make sure that the params grid index and the input grid
        # index are combined to avoid all holes in the grid.
        # We have to do a bit of trickery here since params index and input index 
        # can have different shapes, although they must slice down to the same
        # shape.

        params_index = self.params_grid.array_grid.get_value_index_unsliced(params_name)

        if self.input_grid is not None and self.input_grid.array_grid.slice is not None:
            params_slice = self.params_grid.array_grid.slice
            input_slice = self.input_grid.array_grid.slice
            
            input_index = self.input_grid.array_grid.get_value_index_unsliced('flux')
            
            ii = input_index[input_slice or ()]
            pi = params_index[params_slice or ()]
            iis = ii.shape
            pis = pi.shape

            ii = ii.flatten() & pi.flatten()

            input_index[input_slice or ()] = ii.reshape(iis)
            params_index[params_slice or ()] = ii.reshape(pis)
        else:
            input_index = None

        return params_index, input_index

    def verify_data_index(self):
        # Make sure all data indices have the same shape
        # TODO: verify this here
        GridBuilder.verify_data_index(self)
        if self.params_grid is not None:
            assert(self.params_grid_index.shape[-1] == self.output_grid_index.shape[-1])

    def get_gridpoint_model(self, i):
        input_idx = tuple(self.input_grid_index[:, i])
        output_idx = tuple(self.output_grid_index[:, i])
        spec = self.input_grid.get_model_at(input_idx)
        return input_idx, output_idx, spec

    def get_gridpoint_params(self, i):
        # Get all parameters of the continuum model at a gridpoint

        # TODO: return all params as a dict
        raise NotImplementedError()

        params_idx = tuple(self.params_grid_index[:, i])
        output_idx = tuple(self.output_grid_index[:, i])

        # TODO: rewrite to get all params of the continuum model
        params = self.params_grid.grid.get_value_at('params', params_idx)
        
        return params_idx, output_idx, params

    def get_interpolated_params(self, **kwargs):
        # Interpolate the params grid to a location defined by kwargs

        params = {}
        for name in self.continuum_model.get_params_names():
            p = self.params_grid.grid.get_value(name, **kwargs)
            params[name] = p

        return params

    def copy_value(self, input_grid, output_grid, name):
        self.logger.info('Copying value array `{}`'.format(name))
        raise NotImplementedError()

    def copy_rbf(self, input_grid, output_grid, name):
        self.logger.info('Copying RBF array `{}`'.format(name))
        rbf = input_grid.values[name]
        output_grid.set_value(name, rbf)

    def copy_wave(self, params_grid, output_grid):
        output_grid.set_wave(params_grid.get_wave())

    def copy_constants(self, params_grid, output_grid):
        output_grid.set_constants(params_grid.get_constants())