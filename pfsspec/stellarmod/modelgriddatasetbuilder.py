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
            self.interp_mode = 'grid'
            self.interp_param = None
        else:
            self.grid = orig.grid
            self.sample_mode = orig.sample_mode
            self.sample_count = orig.sample_count
            self.interp_mode = orig.interp_mode
            self.interp_param = orig.interp_param

    def add_args(self, parser):
        super(ModelGridDatasetBuilder, self).add_args(parser)

        parser.add_argument('--sample-mode', type=str, choices=['all', 'random'], default='all', help='Sampling mode\n')
        parser.add_argument('--sample-count', type=int, default=None, help='Number of interpolations between models\n')
        parser.add_argument('--interp-mode', type=str, choices=['grid', 'linear', 'spline'], default='grid', help='Type of interpolation\n')
        parser.add_argument('--interp-param', type=str, default=None, help='Parameter direction of interpolation\n')

        for k in self.grid.params:
            parser.add_argument('--' + k, type=float, nargs=2, default=None, help='Limit ' + k)

    def init_from_args(self, args):
        super(ModelGridDatasetBuilder, self).init_from_args(args)

        self.sample_mode = args['sample_mode']
        self.sample_count = args['sample_count']
        self.interp_mode = args['interp_mode']
        self.interp_param = args['interp_param']

        # Override grid range if specified
        # TODO: extend this to sample physically meaningful models only
        for k in self.grid.params:
            if args[k] is not None:
                self.grid.params[k].min = args[k][0]
                self.grid.params[k].max = args[k][1]

    def get_spectrum_count(self):
        if self.sample_mode == 'all':
            # TODO: how to deal with parameters limits?
            # return self.index[0].shape[0]
            raise NotImplementedError()
        elif self.sample_mode == 'random':
            return self.sample_count
        else:
            raise NotImplementedError()

    def get_wave_count(self):
        return self.pipeline.rebin.shape[0]

    def create_dataset(self):
        super(ModelGridDatasetBuilder, self).create_dataset()

    def process_item(self, i):
        if self.sample_mode == 'all':
            spec = self.get_gridpoint(i)
        elif self.sample_mode == 'random':
            spec = self.draw_sample(i)
        else:
            raise NotImplementedError()

        self.pipeline.run(spec)
        return spec

    def get_random_params(self):
        params = {}
        for p in self.grid.params:
            params[p] = np.random.uniform(self.grid.params[p].min, self.grid.params[p].max)
        return params

    def draw_sample(self, i):
        spec = None
        while spec is None:
            params = self.get_random_params()
            if self.interp_mode == 'grid':
                spec = self.grid.get_nearest_model(**params)
            elif self.interp_mode == 'linear':
                spec = self.grid.interpolate_model_linear(**params)
            elif self.interp_mode == 'spline':
                spec = self.grid.interpolate_model_spline(self.interp_param, **params)
            else:
                raise NotImplementedError()

        return spec

    def get_gridpoint(self, i):
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