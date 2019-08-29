import numpy as np
import pandas as pd

from pfsspec.data.datasetbuilder import DatasetBuilder
from pfsspec.stellarmod.modelspectrum import ModelSpectrum

class ModelGridDatasetBuilder(DatasetBuilder):
    def __init__(self, orig=None):
        super(ModelGridDatasetBuilder, self).__init__(orig)
        if orig is None:
            self.grid = None
            self.interpolate = False
            self.spectrum_count = 0
        else:
            self.grid = orig.grid
            self.interpolate = orig.interpolate
            self.spectrum_count = orig.spectrum_count

    def add_args(self, parser):
        super(ModelGridDatasetBuilder, self).add_args(parser)

        parser.add_argument('--interp', type=int, default=None, help='Number of interpolations between models\n')

        for k in self.grid.params:
            parser.add_argument('--' + k, type=float, nargs=2, default=None, help='Limit ' + k)

    def init_from_args(self, args):
        super(ModelGridDatasetBuilder, self).init_from_args(args)

        if 'interp' in args and args['interp'] is not None:
            self.interpolate = True
            self.spectrum_count = args['interp']

            # Override grid range when interpolation is turned on and limits are set
            for k in self.grid.params:
                if args[k] is not None:
                    self.grid.params[k].min = args[k][0]
                    self.grid.params[k].max = args[k][1]

    def get_spectrum_count(self):
        if self.interpolate:
            return self.spectrum_count
        else:
            return self.index[0].shape[0]

    def get_wave_count(self):
        return self.pipeline.rebin.shape[0]

    def create_dataset(self):
        super(ModelGridDatasetBuilder, self).create_dataset()

    def process_item(self, i):
        if self.interpolate:
            return self.process_interpolating(i)
        else:
            return self.process_gridpoint(i)

    def process_interpolating(self, i):

        ## TODO: rewrite this

        spec = None
        while spec is None:
            # Generate random parameters
            Fe_H = np.random.uniform(self.grid.Fe_H_min, self.grid.Fe_H_max)
            T_eff = np.random.uniform(self.grid.T_eff_min, self.grid.T_eff_max)
            log_g = np.random.uniform(self.grid.log_g_min, self.grid.log_g_max)
            spec = self.grid.interpolate_model(Fe_H, T_eff, log_g)

        self.pipeline.run(spec)
        return spec

    # TODO: add non-interpolating random gridpoint sampling

    def process_gridpoint(self, i):
        fi = self.index[0][i]
        fj = self.index[1][i]
        fk = self.index[2][i]

        spec = self.grid.get_model(fi, fj, fk)
        self.pipeline.run(spec)
        return spec

    def build(self):
        # non-existing models have 0 flux
        self.nonempty = self.grid.flux_idx
        self.index = np.where(self.nonempty)

        super(ModelGridDatasetBuilder, self).build()

        self.dataset.wave[:] = self.pipeline.rebin
        return self.dataset