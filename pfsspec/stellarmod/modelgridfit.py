import os
import numpy as np
from tqdm import tqdm
import multiprocessing

from pfsspec.parallel import SmartParallel
from pfsspec.pfsobject import PfsObject

class ModelGridFit(PfsObject):
    def __init__(self, grid=None, model=None, orig=None):
        super(ModelGridFit, self).__init__(orig=orig)

        if isinstance(orig, ModelGridFit):
            self.parallel = orig.parallel
            self.threads = orig.threads

            self.top = orig.top

            self.input_grid = grid if grid is not None else orig.input_grid
            self.output_grid = orig.output_grid
            self.grid_index = None

            self.model = model if model is not None else orig.model
        else:
            self.parallel = True
            self.threads = multiprocessing.cpu_count() // 2

            self.top = None

            self.input_grid = grid
            self.output_grid = None
            self.grid_index = None

            self.model = model

    def add_args(self, parser):
        parser.add_argument('--top', type=int, default=None, help='Limit number of results')

    def parse_args(self, args):
        if 'top' in args and args['top'] is not None:
            self.top = args['top']

    def create_grid(self):
        raise NotImplementedError()
        
    def open_data(self, input_path, output_path):
        # Open input
        fn = os.path.join(input_path, 'spectra') + '.h5'
        self.input_grid = self.create_grid()
        self.input_grid.load(fn)

        index = self.input_grid.value_indexes['flux']
        self.grid_index = np.array(np.where(index))

        # Open output
        fn = os.path.join(output_path, 'spectra') + '.h5'
        self.output_grid = self.create_grid()

        # DEBUG
        self.output_grid.preload_arrays = True
        # END DEGUB

        self.output_grid.filename = fn
        self.output_grid.fileformat = 'h5'

    def save_data(self):
        self.output_grid.save(self.output_grid.filename, self.output_grid.fileformat)

    def get_data_count(self):
        return self.grid_index.shape[1]        

    def get_gridpoint_model(self, i):
        idx = tuple(self.grid_index[:, i])
        spec = self.input_grid.get_model(idx)
        return idx, spec

    def init_process(self):
        pass

    def process_item(self, i):
        idx, spec = self.get_gridpoint_model(i)
        params = self.model.fit(spec)
        self.model.eval(spec, params)

        return idx, spec, params

    def store_item(self, idx, spec, params):
        self.output_grid.set_value_at('flux', idx, spec.flux)
        self.output_grid.set_value_at('cont', idx, spec.cont)
        self.output_grid.set_value_at('params', idx, params)

    def run(self):
        if self.model is None:
            self.model = self.create_model()

        output_initialized = False
        data_count = self.get_data_count()
        if self.top is not None:
            data_count = min(self.top, data_count)
        t = tqdm(total=data_count)

        # Fit every model

        with SmartParallel(initializer=self.init_process, verbose=False, parallel=self.parallel, threads=self.threads) as p:
            for idx, spec, params in p.map(self.process_item, range(data_count)):
                if not output_initialized:
                    self.output_grid.wave = spec.wave
                    self.output_grid.value_shapes['params'] =  params.shape
                    self.output_grid.allocate_values()
                    self.output_grid.build_axis_indexes()
                    # self.output_grid.save(self.output_grid.filename, self.output_grid.fileformat)
                    output_initialized = True
                self.store_item(idx, spec, params)
                t.update(1)
