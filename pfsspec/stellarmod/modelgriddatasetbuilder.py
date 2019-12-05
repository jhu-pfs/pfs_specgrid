import numpy as np
import pandas as pd
import logging

from pfsspec.data.datasetbuilder import DatasetBuilder
from pfsspec.stellarmod.modelspectrum import ModelSpectrum

class ModelGridDatasetBuilder(DatasetBuilder):
    def __init__(self, orig=None, random_seed=None):
        super(ModelGridDatasetBuilder, self).__init__(orig=orig, random_seed=random_seed)
        if orig is None:
            self.grid = None
            self.sample_mode = None
            self.sample_dist = None
            self.sample_count = 0
            self.interp_mode = 'grid'
            self.interp_param = None
            self.use_cont = True    # Load model continuum
        else:
            self.grid = orig.grid
            self.sample_mode = orig.sample_mode
            self.sample_dist = orig.sample_dist
            self.sample_count = orig.sample_count
            self.interp_mode = orig.interp_mode
            self.interp_param = orig.interp_param
            self.use_cont = orig.use_cont

    def add_args(self, parser):
        super(ModelGridDatasetBuilder, self).add_args(parser)

        parser.add_argument('--preload-arrays', action='store_true', help='Do not preload flux arrays to save memory\n')
        parser.add_argument('--sample-mode', type=str, choices=['all', 'random'], default='all', help='Sampling mode\n')
        parser.add_argument('--sample-dist', type=str, choices=['uniform', 'beta'], default='uniform', help='Randomly sampled parameter distribution')
        parser.add_argument('--sample-count', type=int, default=None, help='Number of samples to be interpolated between models\n')
        parser.add_argument('--interp-mode', type=str, choices=['grid', 'linear', 'spline'], default='grid', help='Type of interpolation\n')
        parser.add_argument('--interp-param', type=str, default='random', help='Parameter direction of interpolation\n')

        for k in self.grid.params:
            parser.add_argument('--' + k, type=float, nargs=2, default=None, help='Limit ' + k)

    def init_from_args(self, args):
        super(ModelGridDatasetBuilder, self).init_from_args(args)

        if 'preload_arrays' in args and args['preload_arrays'] is not None:
            self.grid.preload_arrays = args['preload_arrays']

        if 'sample_mode' in args and args['sample_mode'] is not None:
            self.sample_mode = args['sample_mode']
        if 'sample_dist' in args and args['sample_dist'] is not None:
            self.sample_dist = args['sample_dist']
        if 'sample_count' in args and args['sample_count'] is not None:
            self.sample_count = args['sample_count']
        if 'interp_mode' in args and args['interp_mode'] is not None:
            self.interp_mode = args['interp_mode']
        if 'interp_param' in args and args['interp_param'] is not None:
            self.interp_param = args['interp_param']

        # Override grid range if specified
        # TODO: extend this to sample physically meaningful models only
        for k in self.grid.params:
            if args[k] is not None:
                self.grid.params[k].min = args[k][0]
                self.grid.params[k].max = args[k][1]

    def get_spectrum_count(self):
        if self.params is not None:
            return self.params.shape[0]
        elif self.sample_mode == 'all':
            # TODO: how to deal with parameters limits?
            # return self.index[0].shape[0]
            raise NotImplementedError()
        elif self.sample_mode == 'random':
            return self.sample_count
        else:
            raise NotImplementedError()

    def get_wave_count(self):
        return self.pipeline.wave.shape[0]

    def create_dataset(self, init_storage=True):
        return super(ModelGridDatasetBuilder, self).create_dataset(init_storage=init_storage)

    def process_item(self, i):
        self.init_random_state()

        spec = None
        while spec is None:
            if self.params is not None or self.sample_mode == 'random':
                spec, params = self.get_interpolated_model(i)
            elif self.sample_mode == 'all':
                # TODO: implement, see below
                spec, params = self.get_gridpoint_model(i)
            else:
                raise NotImplementedError()

            try:
                spec.id = i
                self.pipeline.run(spec, **params)
                return spec
            except Exception as e:
                logging.exception(e)
                spec = None

    def get_params(self, i):
        params = self.params[self.params['id'] == i].to_dict('records')[0]
        if self.interp_param == 'random':
            free_param = params['interp_param']
        elif self.interp_param is not None:
            free_param = self.interp_param
        else:
            free_param = params['interp_param']
        return params, free_param

    def draw_random_params(self):
        # Always draw random parameters from self.random_state
        params = {}
        for p in self.grid.params:
            if self.sample_dist == 'uniform':
                r = self.random_state.uniform(0, 1)
            elif self.sample_dist == 'beta':
                r = self.random_state.beta(0.7, 0.7)    # Add a bit more weight to the tails
            else:
                raise NotImplementedError()
            params[p] = self.grid.params[p].min + r * (self.grid.params[p].max - self.grid.params[p].min)

        if self.interp_param == 'random':
            free_param = self.random_state.choice(list(self.grid.params.keys()))
        else:
            free_param = self.interp_param

        return params, free_param

    def get_interpolated_model(self, i):
        spec = None
        while spec is None:
            # Use existing params or draw new ones
            if self.params is not None:
                params, free_param = self.get_params(i)
            else:
                params, free_param = self.draw_random_params()

            if self.interp_mode == 'grid':
                spec = self.grid.get_nearest_model(**params)
            elif self.interp_mode == 'linear':
                spec = self.grid.interpolate_model_linear(**params)
            elif self.interp_mode == 'spline':
                spec = self.grid.interpolate_model_spline(free_param, **params)
            else:
                raise NotImplementedError()

        return spec, params

    def get_gridpoint_model(self, i):
        # TODO: rewrite this to use index of grid
        #       and to observer parameter limits
        # TODO: how to index models
        #fi = self.index[0][i]
        #fj = self.index[1][i]
        #fk = self.index[2][i]

        #spec = self.grid.get_model(fi, fj, fk)
        #return spec
        raise NotImplementedError()

    def build(self):
        # non-existing models have 0 flux
        self.nonempty = self.grid.flux_idx
        self.index = np.where(self.nonempty)

        spectra = super(ModelGridDatasetBuilder, self).build()
        self.copy_params_from_spectra(spectra)

        self.dataset.wave[:] = self.pipeline.wave
        return self.dataset
