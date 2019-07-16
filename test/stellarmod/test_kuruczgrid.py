from test.test_base import TestBase
import os
import numpy as np

from pfsspec.stellarmod.kuruczspectrumreader import KuruczSpectrumReader
from pfsspec.stellarmod.kuruczgrid import KuruczGrid

class TestKuruczGrid(TestBase):
    def test_save(self):
        path = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/kurucz')
        file = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/compressed/test.npz')
        grid = KuruczSpectrumReader.read_grid(path, 'test')
        self.assertEqual((2, 61, 11, 1221), grid.flux.shape)
        grid.save(file)

    def test_load(self):
        file = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/compressed/kurucz.npz')
        grid = KuruczGrid()
        grid.load(file)
        self.assertEqual((18, 61, 11, 1221), grid.flux.shape)

    def test_get_nearest_model(self):
        file = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/compressed/kurucz.npz')
        grid = KuruczGrid()
        grid.load(file)

        spec = grid.get_nearest_model(0.11, 4900, 3.1)
        self.assertEqual((1221,), spec.wave.shape)
        self.assertEqual((1221,), spec.flux.shape)
        self.assertTrue(np.max(spec.flux) > 0)

        spec = grid.get_nearest_model(Fe_H=-0.1, T_eff=5200, log_g=4)
        self.assertEqual((1221,), spec.wave.shape)
        self.assertEqual((1221,), spec.flux.shape)
        self.assertTrue(np.max(spec.flux) > 0)

    def test_get_nearby_indexes(self):
        file = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/compressed/kurucz.npz')
        grid = KuruczGrid()
        grid.load(file)
        idx = grid.get_nearby_indexes(0.11, 4900, 3.1)
        self.assertEqual((13, 5, 6, 14, 6, 7), idx)

    def test_get_nearby_indexes_outside(self):
        file = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/compressed/kurucz.npz')
        grid = KuruczGrid()
        grid.load(file)
        # These are outside the grid completely
        idx = grid.get_nearby_indexes(-0.9, 14300, 5.2)
        self.assertIsNone(idx)
        # These are outside the parameter space but inside grid
        idx = grid.get_nearby_indexes(-0.9, 14300, 1.2)
        self.assertIsNone(idx)

    def test_interpolate_model(self):
        file = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/compressed/kurucz.npz')
        grid = KuruczGrid()
        grid.load(file)
        i1, j1, k1, i2, j2, k2 = grid.get_nearby_indexes(0.11, 4900, 3.1)
        a = grid.get_model(i1, j1, k1)
        b = grid.get_model(i2, j2, k2)
        a.plot()
        b.plot()
        spec = grid.interpolate_model(0.11, 4900, 3.1)
        self.assertEqual((1221,), spec.wave.shape)
        self.assertEqual((1221,), spec.flux.shape)
        spec.plot()
        self.save_fig()

    def test_interpolate_model_outside(self):
        file = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/compressed/kurucz.npz')
        grid = KuruczGrid()
        grid.load(file)
        # These are outside the parameter space but inside grid
        spec = grid.interpolate_model(-0.9, 14300, 1.2)
        self.assertIsNone(spec)