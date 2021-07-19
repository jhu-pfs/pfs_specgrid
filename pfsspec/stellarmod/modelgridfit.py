import os
import numpy as np
from tqdm import tqdm
import multiprocessing

from pfsspec.parallel import SmartParallel
from pfsspec.data.gridbuilder import GridBuilder
from pfsspec.data.arraygrid import ArrayGrid
from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.stellarmod.modelgridbuilder import ModelGridBuilder

class ModelGridFit(GridBuilder, ModelGridBuilder):
    # Fit continuum models to stellar model spectra

    STEPS = ['fit', 'fill', 'smooth', 'norm']

    def __init__(self, config, orig=None):
        GridBuilder.__init__(self, orig=orig)
        ModelGridBuilder.__init__(self, config, orig=orig)

        if isinstance(orig, ModelGridFit):
            self.parallel = orig.parallel
            self.threads = orig.threads
            self.step = orig.step
        else:
            self.parallel = True
            self.threads = multiprocessing.cpu_count() // 2
            self.step = None

    def add_args(self, parser):
        GridBuilder.add_args(self, parser)
        ModelGridBuilder.add_args(self, parser)

        parser.add_argument('--step', type=str, choices=ModelGridFit.STEPS, help='Fitting step to perform.\n')

    def parse_args(self):
        GridBuilder.parse_args(self)
        ModelGridBuilder.parse_args(self)

        if 'step' in self.args and self.args['step'] is not None:
            self.step = self.args['step']

    def create_input_grid(self):
        return ModelGridBuilder.create_input_grid(self)

    def open_input_grid(self, input_path):
        return ModelGridBuilder.open_input_grid(self, input_path)

    def create_output_grid(self):
        return ModelGridBuilder.create_output_grid(self)

    def open_output_grid(self, output_path):
        return ModelGridBuilder.open_output_grid(self, output_path)

    def open_data(self, input_path, output_path, params_path=None):
        return ModelGridBuilder.open_data(self, input_path, output_path, params_path=params_path)

    def build_data_index(self):
        return ModelGridBuilder.build_data_index(self)

    def store_item(self, idx, spec, params):
        for k in params:
            self.output_grid.grid.set_value_at(k, idx, params[k], valid=True)

        if self.step in ['norm']:
            self.output_grid.grid.set_value_at('flux', idx, spec.flux)
            self.output_grid.grid.set_value_at('cont', idx, spec.cont)

    def process_item_fit(self, i):
        input_idx, output_idx, spec = self.get_gridpoint_model(i)
        params = self.continuum_model.fit(spec)
        return i, input_idx, output_idx, spec, params

    def run_step_fit(self):
        output_initialized = False
        input_count = self.get_input_count()

        # Fit every model
        t = tqdm(total=input_count)
        with SmartParallel(initializer=self.init_process, verbose=False, parallel=self.parallel, threads=self.threads) as p:
            for i, input_idx, output_idx, spec, params in p.map(self.process_item_fit, range(input_count)):
                if not output_initialized:
                    # Determine the size of the output arrays and allocate them on
                    # the disk before starting the processing.

                    # We do not want to calculate and save the normalized spectra in
                    # this step (because the fitted parameters might be smoothed first),
                    # so set the output wave vector to a dummy value
                    self.output_grid.set_wave(np.array([0]))    # Dummy size

                    # Shapes of the model parameters
                    for k in params:
                        self.output_grid.grid.value_shapes[k] = params[k].shape
                    
                    self.output_grid.allocate_values()
                    self.output_grid.build_axis_indexes()

                    # Reset wave vector
                    self.output_grid.set_wave(self.continuum_model.wave)

                    output_initialized = True
                self.store_item(output_idx, spec, params)
                t.update(1)

        self.output_grid.grid.constants.update(self.continuum_model.get_constants())

    def run_step_fill_smooth(self, fill=True, smooth=False):
        # Initialize continuum model class and call smoothing. The class
        # can decide which parameters to smooth and which to keep intact.
        self.continuum_model.init_wave(self.input_grid.wave)

        # Allocate output grid
        for p in self.continuum_model.get_model_parameters():
            self.output_grid.grid.value_shapes[p.name] = self.params_grid.grid.value_shapes[p.name]
        
        # We do not save the continuum and flux at this step, so temporarily set
        # the wave vector to a dummy value to avoid allocating large arrays
        self.output_grid.set_wave(np.array([0]))
        self.output_grid.allocate_values()
        self.output_grid.build_axis_indexes()

        # Set wave vector back to real value
        self.output_grid.set_wave(self.params_grid.wave)
        self.output_grid.grid.set_constants(self.params_grid.grid.get_constants())

        for p in self.continuum_model.get_model_parameters():
            # Get original fit parameters from the input grid
            params = self.params_grid.grid.get_value(p.name)

            if self.params_grid.grid.has_value_index(p.name):
                mask = self.params_grid.grid.get_value_index(p.name)
            else:
                mask = None

            if fill:
                if mask is not None:
                    params[~mask] = np.nan
                params = self.continuum_model.fill_params(p.name, params)
            
            if smooth:
                params = self.continuum_model.smooth_params(p.name, params)

            # Save data into the output grid. 
            self.output_grid.grid.set_value(p.name, params, valid=mask)

            # The original mask is kept so that we know which spectrum model were
            # in the original data set and NaNs will mark the spectra that could not be fitted.
            self.output_grid.grid.value_indexes[p.name] = mask

    def process_item_normalize(self, i):
        input_idx, output_idx, spec = self.get_gridpoint_model(i)

        if self.params_grid is None:
            # No fitted parameters are available, fit continuum now
            _, _, _, _, params = self.process_item_fit(i)
        elif self.params_grid_index is not None:
            # Parameters come from an array grid
            _, _, params = self.get_gridpoint_params(i)
        else:
            # Parameters are interpolated from RBF
            params = self.get_interpolated_params(**spec.get_params())

        self.continuum_model.normalize(spec, params)
        return i, input_idx, output_idx, spec, params

    def run_step_normalize(self):
        output_initialized = False
        input_count = self.get_input_count()

        # By default, the parameters grid should have the same wave vector that
        # comes out of continuum_model, hence we cannot use the wave_mask of the
        # model to slice down the input grid. Update the continuum_model here to
        # have the correct mask.
        self.continuum_model.init_wave(self.input_grid.get_wave())

        # Normalize every spectrum
        t = tqdm(total=input_count)
        with SmartParallel(initializer=self.init_process, verbose=False, parallel=self.parallel, threads=self.threads) as p:
            for i, input_idx, output_idx, spec, params in p.map(self.process_item_normalize, range(input_count)):
                if not output_initialized:
                    for k in params:
                        self.output_grid.grid.value_shapes[k] = params[k].shape
                    self.output_grid.set_wave(self.continuum_model.wave)
                    self.output_grid.allocate_values()
                    self.output_grid.build_axis_indexes()
                    output_initialized = True
                self.store_item(output_idx, spec, params)
                t.update(1)

        self.output_grid.grid.constants.update(self.continuum_model.get_constants())

    def run(self):
        if self.step == 'fit':
            self.run_step_fit()
        elif self.step == 'fill':
            self.run_step_fill_smooth(fill=True, smooth=False)
        elif self.step == 'smooth':
            self.run_step_fill_smooth(fill=True, smooth=True)
        elif self.step == 'norm':
            self.run_step_normalize()
        else:
            raise NotImplementedError()
