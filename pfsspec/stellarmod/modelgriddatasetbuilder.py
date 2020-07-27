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

            self.z = orig.z
            self.mag = orig.mag
            self.ext = orig.ext

            self.target_zenith_angle = orig.target_zenith_angle
            self.target_field_angle = orig.target_field_angle
            self.moon_zenith_angle = orig.moon_zenith_angle
            self.moon_target_angle = orig.moon_target_angle
            self.moon_phase = orig.moon_phase
        else:
            self.grid = None
            self.grid_index = None
            self.sample_mode = None
            self.sample_dist = None
            self.sample_count = 0
            self.interp_mode = 'grid'
            self.interp_param = None
            self.use_cont = True  # Load model continuum

            self.z = None
            self.mag = None
            self.ext = None

            self.target_zenith_angle = None
            self.target_field_angle = None
            self.moon_zenith_angle = None
            self.moon_target_angle = None
            self.moon_phase = None

        self.params = {
            'redshift': None,
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

        parser.add_argument('--z', type=float, nargs='*', default=None, help='Radial velocity or distribution parameters')
        parser.add_argument('--mag', type=float, nargs='*', default=None, help='Apparent magnitude or distribution parameters.\n')
        parser.add_argument('--ext', type=float, nargs='*', default=None, help='Extinction or distribution parameters.\n')

        parser.add_argument('--z-dist', type=str, default=None, help='Redshift distribution.')
        parser.add_argument('--mag-dist', type=str, default=None, help='Magnitude distribution.')
        parser.add_argument('--ext-dist', type=str, default=None, help='Extinction distribution.')

        parser.add_argument('--target-zenith-angle', type=float, nargs='*', default=[0, 45], help='Zenith angle\n')
        parser.add_argument('--target-field-angle', type=float, nargs='*', default=[0, 0.65], help='Field angle\n')
        parser.add_argument('--moon-zenith-angle', type=float, nargs='*', default=[30, 90], help='Moon zenith angle\n')
        parser.add_argument('--moon-target-angle', type=float, nargs='*', default=[60, 180], help='Moon target angle\n')
        parser.add_argument('--moon-phase', type=float, nargs='*', default=[0, 0.25], help='Moon phase\n')

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

        self.z = self.get_arg('z', self.z, args)
        self.mag = self.get_arg('mag', self.mag, args)
        self.ext = self.get_arg('ext', self.ext, args)

        self.z_dist = self.get_arg('z_dist', self.z, args)
        self.mag_dist = self.get_arg('mag_dist', self.z, args)
        self.ext_dist = self.get_arg('ext_dist', self.z, args)

        # Do not initialize random distributions here because they should
        # use the random state of the worker sub-process to get independent
        # numbers wher running in parallel!

        self.target_zenith_angle = self.get_arg('target_zenith_angle', self.target_zenith_angle, args)
        self.target_field_angle = self.get_arg('target_field_angle', self.target_field_angle, args)
        self.moon_zenith_angle = self.get_arg('moon_zenith_angle', self.moon_zenith_angle, args)
        self.moon_target_angle = self.get_arg('moon_target_angle', self.moon_target_angle, args)
        self.moon_phase = self.get_arg('moon_phase', self.moon_phase, args)

        # Observational parameter grid
        # TODO: add more grid parameters here
        if self.z_dist == 'grid':
            self.params['redshift'] = GridParam('redshift', np.linspace(*self.z))
        if self.mag_dist == 'grid':
            self.params['mag'] = GridParam('mag', np.linspace(*self.mag))

    def load_grid(self, filename, args):
        self.grid.use_cont = self.use_cont
        self.grid.load(filename, format='h5')

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
            # Pysical parameter grid of the models
            count = self.grid.get_model_count(use_limits=True)
            # Observational parameters grid
            count *= self.get_grid_param_count()
            return count
        elif self.sample_mode == 'random':
            return self.sample_count
        else:
            raise NotImplementedError()

    def get_wave_count(self):
        return self.pipeline.get_wave_count()

    def create_dataset(self, init_storage=True):
        return super(ModelGridDatasetBuilder, self).create_dataset(init_storage=init_storage)

    def process_item(self, i):
        super(ModelGridDatasetBuilder, self).process_item(i)

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

    def get_random_dist(self, dist):
        if dist is None:
            return None
        elif dist == 'normal':
            return self.random_state.normal
        elif dist == 'uniform':
            return self.random_state.uniform
        elif dist == 'lognormal':
            return self.random_state.lognormal
        elif dist == 'beta':
            return self.random_state.beta
        else:
            raise NotImplementedError()

    def draw_random_param(self, params, name, values, random_func):
        if values is not None and len(values) == 1:
            params[name] = values[0]
        elif values is not None and random_func is not None:
            params[name] = random_func(*values)

    def draw_random_params(self):
        # Always draw random parameters from self.random_state

        # Draw model physical parameters
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

        # Draw observational parameters
        self.draw_random_param(params, 'redshift', self.z, self.get_random_dist(self.z_dist))
        self.draw_random_param(params, 'mag', self.mag, self.get_random_dist(self.mag_dist))
        self.draw_random_param(params, 'extinction', self.ext, self.get_random_dist(self.ext_dist))

        # TODO: Do we want non-uniform here?
        self.draw_random_param(params, 'target_zenith_angle', self.target_zenith_angle, self.random_state.uniform)
        self.draw_random_param(params, 'target_field_angle', self.target_field_angle, self.random_state.uniform)
        self.draw_random_param(params, 'moon_zenith_angle', self.moon_zenith_angle, self.random_state.uniform)
        self.draw_random_param(params, 'moon_target_angle', self.moon_target_angle, self.random_state.uniform)
        self.draw_random_param(params, 'moon_phase', self.moon_phase, self.random_state.uniform)

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

        return self.dataset
