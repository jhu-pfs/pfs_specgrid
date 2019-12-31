import numpy as np
import pandas as pd
import logging

from pfsspec.data.gridparam import GridParam
from pfsspec.data.datasetbuilder import DatasetBuilder
from pfsspec.stellarmod.modelspectrum import ModelSpectrum

class ModelGridDatasetBuilder(DatasetBuilder):
    def __init__(self, orig=None, random_seed=None):
        super(ModelGridDatasetBuilder, self).__init__(orig=orig, random_seed=random_seed)
        if isinstance(orig, ModelGridDatasetBuilder):
            self.grid = orig.grid
            self.grid_index = None
            self.sample_mode = orig.sample_mode
            self.sample_dist = orig.sample_dist
            self.sample_count = orig.sample_count
            self.interp_mode = orig.interp_mode
            self.interp_param = orig.interp_param
            self.use_cont = orig.use_cont

            self.z_random = orig.z_random
            self.ext_random = orig.ext_random
            self.mag_random = orig.mag_random
            self.sky_level_random = orig.sky_level_random
        else:
            self.grid = None
            self.grid_index = None
            self.sample_mode = None
            self.sample_dist = None
            self.sample_count = 0
            self.interp_mode = 'grid'
            self.interp_param = None
            self.use_cont = True  # Load model continuum

            self.z_random = None
            self.ext_random = None
            self.mag_random = None
            self.sky_level_random = None

        self.params = {
            'redshift': None,
            'extinction': None,
            'mag' : None
        }

    def add_args(self, parser):
        super(ModelGridDatasetBuilder, self).add_args(parser)

        parser.add_argument('--preload-arrays', action='store_true', help='Do not preload flux arrays to save memory\n')
        parser.add_argument('--sample-mode', type=str, choices=['grid', 'random'], default='grid', help='Sampling mode\n')
        parser.add_argument('--sample-dist', type=str, choices=['uniform', 'beta'], default='uniform', help='Randomly sampled parameter distribution')
        parser.add_argument('--sample-count', type=int, default=None, help='Number of samples to be interpolated between models\n')
        parser.add_argument('--interp-mode', type=str, choices=['grid', 'linear', 'spline'], default='grid', help='Type of interpolation\n')
        parser.add_argument('--interp-param', type=str, default='random', help='Parameter direction of interpolation\n')

        parser.add_argument('--z-grid', type=float, nargs=3, default=None, help='Redshift grid')
        parser.add_argument('--z-random', type=float, nargs=2, default=None, help='Radial velocity mean and dispersion')
        parser.add_argument('--ext-grid', type=float, nargs=3, default=None, help='Extinction grid')
        parser.add_argument('--ext-random', type=float, nargs=2, default=None, help='Extinction mean and sigma')
        parser.add_argument('--mag-grid', type=float, nargs=3, default=None, help='Magnitude grid')
        parser.add_argument('--mag-random', type=float, nargs=2, default=None, help='Apparent magnitude mean and sigma.\n')

        # TODO: update this to new noise model
        #       draw observation parameters randomly
        parser.add_argument('--sky-level-random', type=float, nargs=2, default=None, help='Random sky level mean and sigma.\n')

        for k in self.grid.params:
            parser.add_argument('--' + k, type=float, nargs='*', default=None, help='Limit ' + k)

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

        if self.is_arg('z_grid', args):
            self.params['redshift'] = GridParam('redshift', np.linspace(*args['z_grid']))
        if self.is_arg('z_random', args):
             self.z_random = lambda: self.random_state.normal(args['z_random'][0], args['z_random'][1])
        if self.is_arg('ext_grid', args):
            self.params['extinction'] = GridParam('extinction', np.linspace(*args['ext_grid']))
        if self.is_arg('ext_random', args):
            self.ext_random = lambda: self.random_state.lognormal(args['ext_random'][0], args['ext_random'][1])
        if self.is_arg('mag_grid', args):
            self.params['mag'] = GridParam('mag', np.linspace(*args['mag_grid']))
        if self.is_arg('mag_random', args):
            self.mag_random = lambda: self.random_state.normal(args['mag_random'][0], args['mag_random'][1])
        if self.is_arg('sky_level_random', args):
            self.sky_level_random = lambda: np.abs(self.random_state.normal(args['sky_level_random'][0], args['sky_level_random'][1]))

        # Override grid range if specified
        # TODO: extend this to sample physically meaningful models only
        for k in self.grid.params:
            if args[k] is not None and len(args[k]) >= 2:
                self.grid.params[k].min = args[k][0]
                self.grid.params[k].max = args[k][1]

    def get_grid_param_count(self):
        count = 1
        for p in self.params:
            if self.params[p] is not None:
                count *= self.params[p].values.size
        return count

    def get_spectrum_count(self):
        if self.match_params is not None:
            return self.match_params.shape[0]
        elif self.sample_mode == 'grid':
            count = self.grid.get_model_count(use_limits=True)
            count *= self.get_grid_param_count()
            return count
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
            if self.sample_mode == 'grid':
                spec, params = self.get_gridpoint_model(i)
            elif self.match_params is not None or self.sample_mode == 'random':
                spec, params = self.get_interpolated_model(i)
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
        params = self.match_params[self.match_params['id'] == i].to_dict('records')[0]
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

        if self.z_random is not None:
            params['redshift'] = self.z_random()
        if self.ext_random is not None:
            params['extinction'] = self.ext_random()
        if self.mag_random is not None:
            params['mag'] = self.mag_random()
        if self.sky_level_random is not None:
            params['sky_level'] = self.sky_level_random()

        return params, free_param

    def get_interpolated_model(self, i):
        spec = None
        while spec is None:
            # Use existing params or draw new ones
            if self.match_params is not None:
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
        # Get grid model
        idx = tuple(self.grid_index[:, i % self.grid_index.shape[1]])
        spec = self.grid.get_model(idx)
        params = spec.get_params()

        # Append grid params
        i = i // self.grid_index.shape[1]
        j = 0
        for p in self.params:
            if self.params[p] is not None:
                params[p] = self.params[p].values[self.param_index[j, i]]
                j += 1

        return spec, params

    def build(self):
        if self.grid.is_data_index('flux'):
            # rows: parameters, columns: models
            index = self.grid.get_limited_data_index('flux')
            self.grid_index = np.array(np.where(index))

        count = 0
        size = 1
        shape = ()
        for p in self.params:
            if self.params[p] is not None:
                count += 1
                size *= self.params[p].values.size
                shape = shape + (self.params[p].values.size,)
        self.param_index = np.indices(shape).reshape(count, size)

        spectra = super(ModelGridDatasetBuilder, self).build()
        self.copy_params_from_spectra(spectra)

        self.dataset.wave[:] = self.pipeline.wave
        return self.dataset
