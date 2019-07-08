import numpy as np
import pandas as pd

from pfsspec.data.datasetbuilder import DatasetBuilder
from pfsspec.stellarmod.modelspectrum import ModelSpectrum

class ModelGridDatasetBuilder(DatasetBuilder):
    def __init__(self, orig=None):
        super(ModelGridDatasetBuilder, self).__init__(orig)
        if orig is not None:
            self.grid = orig.grid
        else:
            self.grid = None

    def get_spectrum_count(self):
        return self.index[0].shape[0]

    def get_wave_count(self):
        return self.pipeline.rebin.shape[0]

    def create_dataset(self):
        super(ModelGridDatasetBuilder, self).create_dataset()

    def process_item(self, i):
        fi = self.index[0][i]
        fj = self.index[1][i]
        fk = self.index[2][i]

        spec = ModelSpectrum()
        spec.redshift = 0
        spec.M_H = self.grid.M_H[fi]
        spec.T_eff = self.grid.T_eff[fj]
        spec.log_g = self.grid.log_g[fk]
        spec.alpha = None
        spec.N_He = None
        spec.v_turb = None
        spec.L_H = None

        spec.wave = self.grid.wave
        spec.flux = self.grid.flux[fi, fj, fk, :]

        self.pipeline.run(spec)
        return spec.flux

    def build(self):
        # non-existing models have 0 flux in bin 0
        self.nonempty = (self.grid.flux[:, :, :, 0] != 0)
        self.index = np.where(self.nonempty)

        super(ModelGridDatasetBuilder, self).build()

        self.dataset.wave[:] = self.pipeline.rebin
        self.dataset.params = pd.DataFrame(list(zip(
            self.grid.M_H[self.index[0]],
            self.grid.T_eff[self.index[1]],
            self.grid.log_g[self.index[2]])),
            columns=['fe_h', 't_eff', 'log_g']
        )

        return self.dataset