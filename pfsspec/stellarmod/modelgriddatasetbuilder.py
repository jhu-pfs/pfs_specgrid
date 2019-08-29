import numpy as np
import pandas as pd

from pfsspec.data.datasetbuilder import DatasetBuilder
from pfsspec.stellarmod.modelspectrum import ModelSpectrum

class ModelGridDatasetBuilder(DatasetBuilder):
    def __init__(self, orig=None):
        super(ModelGridDatasetBuilder, self).__init__(orig)
        if orig is None:
            self.grid = None
            self.sample_mode = None
            self.sample_count = 0
        else:
            self.grid = orig.grid
            self.sample_mode = orig.sample_mode
            self.sample_count = orig.sample_count

    def add_args(self, parser):
        super(ModelGridDatasetBuilder, self).add_args(parser)

        parser.add_argument('--sample-mode', type=str, choices=['all', 'grid', 'interp'], default='all', help='Sampling mode\n')
        parser.add_argument('--sample-count', type=int, default=None, help='Number of interpolations between models\n')

        for k in self.grid.params:
            parser.add_argument('--' + k, type=float, nargs=2, default=None, help='Limit ' + k)

    def init_from_args(self, args):
        super(ModelGridDatasetBuilder, self).init_from_args(args)

        self.sample_mode = args['sample_mode']
        self.sample_count = args['sample_count']

        # Override grid range if specified
        # TODO: extend this to sample physically meaningful models only
        for k in self.grid.params:
            if args[k] is not None:
                self.grid.params[k].min = args[k][0]
                self.grid.params[k].max = args[k][1]

    def get_spectrum_count(self):
        if self.sample_mode in ('grid', 'interp'):
            return self.sample_count
        elif self.sample_mode == 'all':
            # TODO: how to deal with parameters limits?
            # return self.index[0].shape[0]
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def get_wave_count(self):
        return self.pipeline.rebin.shape[0]

    def create_dataset(self):
        super(ModelGridDatasetBuilder, self).create_dataset()

    def process_item(self, i):
        if self.sample_mode == 'grid':
            spec = self.process_sample_grid(i)
        elif self.sample_mode == 'interp':
            spec = self.process_sample_interp(i)
        elif self.sample_mode == 'all':
            spec = self.process_gridpoint(i)
        else:
            raise NotImplementedError()

        self.pipeline.run(spec)
        return spec

    def get_random_params(self):
        params = {}
        for p in self.grid.params:
            params[p] = np.random.uniform(self.grid.params[p].min, self.grid.params[p].max)
        return params

    def process_sample_grid(self, i):
        spec = None
        while spec is None:
            params = self.get_random_params()
            spec = self.grid.get_nearest_model(**params)
        return spec

    def process_sample_interp(self, i):
        spec = None
        while spec is None:
            params = self.get_random_params()
            spec = self.grid.interpolate_model(**params)
        return spec

    def process_gridpoint(self, i):
        # TODO: rewrite this to use index of grid
        #       and to observer parameter limits
        fi = self.index[0][i]
        fj = self.index[1][i]
        fk = self.index[2][i]

        spec = self.grid.get_model(fi, fj, fk)
        return spec

    def build(self):
        # non-existing models have 0 flux
        self.nonempty = self.grid.flux_idx
        self.index = np.where(self.nonempty)

        super(ModelGridDatasetBuilder, self).build()

        self.dataset.wave[:] = self.pipeline.rebin
        return self.dataset