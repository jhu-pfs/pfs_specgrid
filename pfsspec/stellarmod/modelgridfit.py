import os
import numpy as np
from tqdm import tqdm
import multiprocessing

from pfsspec.parallel import SmartParallel
from pfsspec.data.gridbuilder import GridBuilder
from pfsspec.data.arraygrid import ArrayGrid
from pfsspec.stellarmod.modelgrid import ModelGrid

class ModelGridFit(GridBuilder):

    STEPS = ['fit', 'smooth', 'norm']

    def __init__(self, config, orig=None):
        super(ModelGridFit, self).__init__(orig=orig)

        if isinstance(orig, ModelGridFit):
            self.params_grid = orig.params_grid
            self.params_grid_index = None

            self.config = config if config is not None else orig.config
            self.parallel = orig.parallel
            self.threads = orig.threads

            self.step = orig.step

            self.continuum_model = orig.continuum_model
        else:
            self.params_grid = None
            self.params_grid_index = None

            self.config = config
            self.parallel = True
            self.threads = multiprocessing.cpu_count() // 2

            self.step = None

            self.continuum_model = self.config.create_continuum_model()

    def add_args(self, parser):
        super(ModelGridFit, self).add_args(parser)

        parser.add_argument('--step', type=str, choices=ModelGridFit.STEPS, help='Fitting steps to perform.\n')

    def parse_args(self):
        super(ModelGridFit, self).parse_args()
        self.continuum_model.init_from_args(self.args)

        if 'step' in self.args and self.args['step'] is not None:
            self.step = self.args['step']

    def create_params_grid(self):
        return ModelGrid(self.config, ArrayGrid)

    def create_input_grid(self):
        return ModelGrid(self.config, ArrayGrid)

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
        fn = os.path.join(output_path, 'spectra') + '.h5'
        self.output_grid = self.create_output_grid()

        # Copy data from the input grid
        g = self.input_grid or self.params_grid
        self.output_grid.grid.set_axes(g.grid.get_axes())
        self.output_grid.set_wave(g.get_wave())
        self.output_grid.build_axis_indexes()

        # Force creating output file for direct hdf5 writing
        self.output_grid.save(fn, format='h5')
        
        # DEBUG
        # self.output_grid.preload_arrays = True
        # END DEGUB

    def open_data(self, input_path, output_path, params_path):
        if params_path is not None:
            self.open_params_grid(params_path)
            self.params_grid.init_from_args(self.args)
            self.params_grid.build_axis_indexes()
            self.grid_shape = self.params_grid.get_shape()

        super(ModelGridFit, self).open_data(input_path, output_path)

    def build_data_index(self):
        super(ModelGridFit, self).build_data_index()

        # Source indexes, make sure that the params grid index and the input grid
        # index are combined to avoid all holes in the grid.
        # We have to do a bit of trickery here since params index and input index 
        # can have different shapes, although they must slice down to the same
        # shape.

        params_index = self.params_grid.array_grid.get_value_index_unsliced('params')
        if self.input_grid is not None and self.input_grid.array_grid.slice is not None:
            params_slice = self.params_grid.array_grid.slice
            input_slice = self.input_grid.array_grid.slice
            
            input_index = self.input_grid.array_grid.get_value_index_unsliced('flux')
            input_index[input_slice or ()] &= params_index[params_slice or ()]
            params_index[params_slice or ()] &= input_index[input_slice or ()]
            
            self.input_grid_index = np.array(np.where(input_index))
        self.params_grid_index = np.array(np.where(params_index))

        # Target indexes - this is already sliced down
        index = self.params_grid.array_grid.get_value_index('params')
        self.output_grid_index = np.array(np.where(params_index))

    def verify_data_index(self):
        # Make sure all data indices have the same shape
        super(ModelGridFit, self).verify_data_index()
        assert(self.params_grid_index.shape[-1] == self.output_grid_index.shape[-1])

    def get_gridpoint_model(self, i):
        input_idx = tuple(self.input_grid_index[:, i])
        output_idx = tuple(self.output_grid_index[:, i])
        spec = self.input_grid.get_model_at(input_idx)
        return input_idx, output_idx, spec

    def get_gridpoint_params(self, i):
        params_idx = tuple(self.params_grid_index[:, i])
        output_idx = tuple(self.output_grid_index[:, i])
        params = self.params_grid.grid.get_value_at('params', params_idx)
        return params_idx, output_idx, params

    def store_item(self, idx, spec, params):
        self.output_grid.grid.set_value_at('params', idx, params, valid=True)

        if self.step in ['norm']:
            self.output_grid.grid.set_value_at('flux', idx, spec.flux)
            self.output_grid.grid.set_value_at('cont', idx, spec.cont)

    def process_item_fit(self, i):
        input_idx, output_idx, spec = self.get_gridpoint_model(i)

        if not self.model_initialized:
            self.continuum_model.init_wave(spec.wave)
            self.model_initialized = True

        params = self.continuum_model.fit(spec)
        return i, input_idx, output_idx, spec, params

    def run_step_fit(self):
        self.model_initialized = False
        output_initialized = False
        input_count = self.get_input_count()

        # Fit every model
        t = tqdm(total=input_count)
        with SmartParallel(initializer=self.init_process, verbose=False, parallel=self.parallel, threads=self.threads) as p:
            for i, input_idx, output_idx, spec, params in p.map(self.process_item_fit, range(input_count)):
                if not output_initialized:
                    self.output_grid.grid.value_shapes['params'] =  params.shape
                    self.output_grid.set_wave(np.array([0]))    # Dummy size
                    self.output_grid.allocate_values()
                    self.output_grid.build_axis_indexes()
                    output_initialized = True
                self.store_item(output_idx, spec, params)
                t.update(1)

        self.output_grid.grid.constants['constants'] = self.continuum_model.get_constants()

    def run_step_smooth(self):
        params = self.params_grid.grid.get_value('params')

        if self.params_grid.grid.has_value_index('params'):
            mask = self.params_grid.grid.get_value_index('params')
            params[~mask] = np.nan

        self.continuum_model.init_wave(self.input_grid.wave)
        smooth_params = self.continuum_model.smooth_params(params)

        # Allocate output grid
        self.output_grid.grid.value_shapes['params'] = (params.shape[-1],)
        self.output_grid.set_wave(np.array([0]))    # Dummy size
        self.output_grid.allocate_values()
        self.output_grid.build_axis_indexes()

        self.output_grid.grid.set_value('params', smooth_params)
        self.output_grid.grid.value_indexes['params'] = mask

        self.output_grid.grid.set_constants(self.params_grid.grid.get_constants())

    def process_item_normalize(self, i):
        input_idx, output_idx, spec = self.get_gridpoint_model(i)
        _, _, params = self.get_gridpoint_params(i)

        self.continuum_model.normalize(spec, params)
        return i, input_idx, output_idx, spec, params

    def run_step_normalize(self):
        output_initialized = False
        input_count = self.get_input_count()

        # Initialize model
        self.continuum_model.init_wave(self.input_grid.get_wave())
        self.output_grid.grid.constants['constants'] = self.continuum_model.get_constants()

        # Normalize every model
        t = tqdm(total=input_count)
        with SmartParallel(initializer=self.init_process, verbose=False, parallel=self.parallel, threads=self.threads) as p:
            for i, input_idx, output_idx, spec, params in p.map(self.process_item_normalize, range(input_count)):
                if not output_initialized:
                    self.output_grid.grid.value_shapes['params'] =  params.shape
                    self.output_grid.set_wave(self.continuum_model.wave)
                    self.output_grid.allocate_values()
                    self.output_grid.build_axis_indexes()
                    output_initialized = True
                self.store_item(output_idx, spec, params)
                t.update(1)

    def run(self):
        if self.step == 'fit':
            self.run_step_fit()
        elif self.step == 'smooth':
            self.run_step_smooth()
        elif self.step == 'norm':
            self.run_step_normalize()
        else:
            raise NotImplementedError()
