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
        spec = None
        while spec is None:
            # Generate random parameters
            M_H = np.random.uniform(self.grid.M_H_min, self.grid.M_H_max)
            T_eff = np.random.uniform(self.grid.T_eff_min, self.grid.T_eff_max)
            log_g = np.random.uniform(self.grid.log_g_min, self.grid.log_g_max)
            spec = self.grid.interpolate_model(M_H, T_eff, log_g)

        self.pipeline.run(spec)
        return spec

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