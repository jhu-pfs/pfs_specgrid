import os
import numpy as np
from tqdm import tqdm
import multiprocessing

from pfsspec.parallel import SmartParallel
from pfsspec.pfsobject import PfsObject
from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.stellarmod.modelarraygrid import ModelArrayGrid

class ModelGridFit(PfsObject):
    def __init__(self, config, grid=None, orig=None):
        super(ModelGridFit, self).__init__(orig=orig)

        if isinstance(orig, ModelGridFit):
            self.config = config if config is not None else orig.config
            self.parallel = orig.parallel
            self.threads = orig.threads

            self.top = orig.top

            self.input_grid = grid if grid is not None else orig.input_grid
            self.output_grid = orig.output_grid
            self.grid_index = None

            self.continuum_model = orig.continuum_model
        else:
            self.config = config
            self.parallel = True
            self.threads = multiprocessing.cpu_count() // 2

            self.top = None

            self.input_grid = grid
            self.output_grid = None
            self.grid_index = None

            self.continuum_model = self.config.create_continuum_model()

    def add_args(self, parser):
        parser.add_argument('--top', type=int, default=None, help='Limit number of results')
        self.continuum_model.add_args(parser)

    def parse_args(self, args):
        if 'top' in args and args['top'] is not None:
            self.top = args['top']
        self.continuum_model.parse_args(args)

    def create_grid(self):
        return ModelGrid(self.config, ModelArrayGrid)
        
    def open_data(self, input_path, output_path):
        # Open input
        fn = os.path.join(input_path, 'spectra') + '.h5'
        self.input_grid = self.create_grid()
        self.input_grid.load(fn)

        index = self.input_grid.grid.value_indexes['flux']
        self.grid_index = np.array(np.where(index))

        # Open output
        fn = os.path.join(output_path, 'spectra') + '.h5'
        self.output_grid = self.create_grid()
        self.output_grid.set_wave(self.input_grid.get_wave())
        self.output_grid.build_axis_indexes()

        # Force creating output file for direct hdf5 writing
        self.output_grid.save(fn, format='h5')

        # DEBUG
        # self.output_grid.preload_arrays = True
        # END DEGUB

    def save_data(self):
        self.output_grid.save(self.output_grid.grid.filename, self.output_grid.grid.fileformat)

    def get_data_count(self):
        return self.grid_index.shape[1]        

    def get_gridpoint_model(self, i):
        idx = tuple(self.grid_index[:, i])
        spec = self.input_grid.get_model_at(idx)
        return idx, spec

    def init_process(self):
        pass

    def process_item(self, i):
        idx, spec = self.get_gridpoint_model(i)
        params = self.continuum_model.normalize(spec)
        return idx, spec, params

    def store_item(self, idx, spec, params):
        self.output_grid.grid.set_value_at('flux', idx, spec.flux)
        self.output_grid.grid.set_value_at('cont', idx, spec.cont)
        self.output_grid.grid.set_value_at('params', idx, params)

    def run(self):
        output_initialized = False
        data_count = self.get_data_count()
        if self.top is not None:
            data_count = min(self.top, data_count)

        # Fit every model
        t = tqdm(total=data_count)
        with SmartParallel(initializer=self.init_process, verbose=False, parallel=self.parallel, threads=self.threads) as p:
            for idx, spec, params in p.map(self.process_item, range(data_count)):
                if not output_initialized:
                    self.output_grid.set_wave(spec.wave)
                    self.output_grid.grid.value_shapes['params'] =  params.shape
                    self.output_grid.allocate_values()
                    self.output_grid.build_axis_indexes()
                    # self.output_grid.save(self.output_grid.filename, self.output_grid.fileformat)
                    output_initialized = True
                self.store_item(idx, spec, params)
                t.update(1)

        self.output_grid.grid.constants['constants'] = self.continuum_model.get_constants(self.output_grid.grid.wave)