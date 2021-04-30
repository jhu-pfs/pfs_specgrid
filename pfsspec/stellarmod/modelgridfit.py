import os
import numpy as np
from tqdm import tqdm
import multiprocessing

from pfsspec.parallel import SmartParallel
from pfsspec.data.gridbuilder import GridBuilder
from pfsspec.data.arraygrid import ArrayGrid
from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.util.array_filters import *

class ModelGridFit(GridBuilder):

    STEPS = ['fit', 'smooth', 'norm']

    def __init__(self, config, orig=None):
        super(ModelGridFit, self).__init__(orig=orig)

        if isinstance(orig, ModelGridFit):
            self.config = config if config is not None else orig.config
            self.parallel = orig.parallel
            self.threads = orig.threads

            self.step = orig.step

            self.continuum_model = orig.continuum_model
        else:
            self.config = config
            self.parallel = True
            self.threads = multiprocessing.cpu_count() // 2

            self.step = None

            self.continuum_model = self.config.create_continuum_model()

    def add_args(self, parser):
        super(ModelGridFit, self).add_args(parser)
        self.continuum_model.add_args(parser)

        parser.add_argument('--step', type=str, choices=ModelGridFit.STEPS, help='Fitting steps to perform.\n')

    def parse_args(self):
        super(ModelGridFit, self).parse_args()
        self.continuum_model.init_from_args(self.args)

        if 'step' in self.args and self.args['step'] is not None:
            self.step = self.args['step']

    def create_input_grid(self):
        return ModelGrid(self.config, ArrayGrid)

    def create_output_grid(self):
        return ModelGrid(self.config, ArrayGrid)

    def open_input_grid(self, input_path):
        fn = os.path.join(input_path, 'spectra') + '.h5'
        self.input_grid = self.create_input_grid()
        self.input_grid.load(fn)

    def open_output_grid(self, output_path):
        fn = os.path.join(output_path, 'spectra') + '.h5'
        self.output_grid = self.create_output_grid()

        # Copy data from the input grid
        self.output_grid.grid.set_axes(self.input_grid.grid.get_axes())
        self.output_grid.set_wave(self.input_grid.get_wave())
        self.output_grid.build_axis_indexes()

        # Force creating output file for direct hdf5 writing
        self.output_grid.save(fn, format='h5')
        
        # DEBUG
        # self.output_grid.preload_arrays = True
        # END DEGUB

    def get_gridpoint_model(self, i):
        input_idx = tuple(self.input_grid_index[:, i])
        output_idx = tuple(self.output_grid_index[:, i])
        spec = self.input_grid.get_model_at(input_idx)
        return input_idx, output_idx, spec

    def process_item(self, i):
        input_idx, output_idx, spec = self.get_gridpoint_model(i)
        params = self.continuum_model.normalize(spec)
        return i, input_idx, output_idx, spec, params

    def store_item(self, idx, spec, params):
        self.output_grid.grid.set_value_at('params', idx, params, valid=True)

        if self.step in ['norm']:
            self.output_grid.grid.set_value_at('flux', idx, spec.flux)
            self.output_grid.grid.set_value_at('cont', idx, spec.cont)

    def run_step_fit(self):
        output_initialized = False
        input_count = self.get_input_count()

        # Fit every model
        t = tqdm(total=input_count)
        with SmartParallel(initializer=self.init_process, verbose=False, parallel=self.parallel, threads=self.threads) as p:
            for i, input_idx, output_idx, spec, params in p.map(self.process_item, range(input_count)):
                if not output_initialized:
                    self.output_grid.grid.value_shapes['params'] =  params.shape
                    self.output_grid.set_wave(spec.wave)
                    self.output_grid.allocate_values()
                    self.output_grid.build_axis_indexes()
                    output_initialized = True
                self.store_item(output_idx, spec, params)
                t.update(1)

        self.output_grid.grid.constants['constants'] = self.continuum_model.get_constants(self.output_grid.wave)

    def run_step_smooth(self):
        params = self.input_grid.grid.get_value('params')

        if self.input_grid.grid.has_value_index('params'):
            mask = self.input_grid.grid.get_value_index('params')
            params[~mask] = np.nan
        
        # Fill in holes of the grid
        filled_params = np.empty_like(params)
        for i in range(params.shape[-1]):
            filled_params[..., i] = fill_holes_filter(params[..., i], fill_filter=np.nanmean, value_filter=np.nanmin)

        # Smooth the parameters. This needs to be done parameter by parameter
        smooth_params = np.empty_like(params)
        for i in range(params.shape[-1]):
            fp = filled_params[..., i]
            shape = fp.shape
            fp = fp.squeeze()
            sp = anisotropic_diffusion(fp)
            smooth_params[..., i] = sp.reshape(shape)

        # Allocate output grid
        self.output_grid.grid.value_shapes['params'] = (params.shape[-1],)
        self.output_grid.set_wave(np.array([0]))    # Dummy size
        self.output_grid.allocate_values()
        self.output_grid.build_axis_indexes()

        self.output_grid.grid.set_value('params', smooth_params)
        self.output_grid.grid.value_indexes['params'] = np.full(smooth_params.shape, True)

        self.output_grid.grid.set_constants(self.input_grid.grid.get_constants())

    def run_step_normalize(self):
        pass

    def run(self):
        if self.step == 'fit':
            self.run_step_fit()
        elif self.step == 'smooth':
            self.run_step_smooth()
        elif self.step == 'norm':
            self.run_step_normalize()
        else:
            raise NotImplementedError()
