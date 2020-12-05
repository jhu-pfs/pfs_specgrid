from test.test_base import TestBase
import os
import numpy as np

from pfsspec.stellarmod.kuruczspectrumreader import KuruczSpectrumReader
from pfsspec.stellarmod.kuruczgridreader import KuruczGridReader
from pfsspec.stellarmod.kuruczgrid import KuruczGrid

class TestKuruczGrid(TestBase):
    def save_load_helper(self, format, ext):
        path = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/atlas9/')
        file = os.path.join(self.PFSSPEC_TEST_PATH, 'test_kurucz_grid_compressed' + ext)
        if os.path.exists(file):
            os.remove(file)

        grid = KuruczGrid('test')
        grid.preload_arrays = True
        reader = KuruczGridReader(grid, path)
        reader.read_grid()
        self.assertEqual((2, 61, 11, 1221), grid.values['flux'].shape)

        grid.save(file, format=format)

        grid = KuruczGrid(model='test')
        grid.preload_arrays = True
        grid.load(file, format=format)
        self.assertEqual((1221, ), grid.wave.shape)
        self.assertEqual((2, 61, 11, 1221), grid.values['flux'].shape)
        #self.assertIsNone(grid.cont)

    def test_save_numpy(self):
        self.save_load_helper('numpy', '.npy.gz')

    def test_save_pickle(self):
        self.save_load_helper('pickle', '.pickle.gz')

    def test_save_h5(self):
        self.save_load_helper('h5', '.h5')

    def import_kurucz_helper(self, file):
        path = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/atlas9/')
        if not os.path.exists(file):
            grid = KuruczGrid('kurucz')
            grid.preload_arrays = True
            reader = KuruczGridReader(grid, path)
            reader.read_grid()
            self.assertEqual((18, 61, 11, 1221), grid.values['flux'].shape)

            grid.save(file, 'h5')

    def load_kurucz_helper(self, preload_arrays=True):
        file = os.path.join(self.PFSSPEC_TEST_PATH, 'test_kurucz_grid_compressed.h5')
        self.import_kurucz_helper(file)
        grid = KuruczGrid(model='kurucz')
        grid.preload_arrays = preload_arrays
        grid.load(file, format='h5')
        return grid

    def test_get_nearest_model(self):
        grid = self.load_kurucz_helper()

        spec = grid.get_nearest_model(Fe_H=0.11, T_eff=4900, log_g=3.1)
        self.assertEqual((1221,), spec.wave.shape)
        self.assertEqual((1221,), spec.flux.shape)
        self.assertTrue(np.max(spec.flux) > 0)

        spec = grid.get_nearest_model(Fe_H=-0.1, T_eff=5200, log_g=4)
        self.assertEqual((1221,), spec.wave.shape)
        self.assertEqual((1221,), spec.flux.shape)
        self.assertTrue(np.max(spec.flux) > 0)

    def test_get_nearby_indexes(self):
        grid = self.load_kurucz_helper()

        idx = grid.get_nearby_indexes(Fe_H=0.11, T_eff=4900, log_g=3.1)
        self.assertEqual(((13, 5, 6), (14, 6, 7)), idx)

    def test_get_nearby_indexes_outside(self):
        grid = self.load_kurucz_helper()

        # These are outside the grid completely
        idx = grid.get_nearby_indexes(Fe_H=-0.9, T_eff=14300, log_g=5.2)
        self.assertIsNone(idx)

        # These are outside the parameter space but inside grid
        idx = grid.get_nearby_indexes(Fe_H=-0.9, T_eff=14300, log_g=1.2)
        self.assertIsNotNone(idx)

    def interpolate_model_linear_helper(self, preload_arrays, Fe_H=0.01, T_eff=4567, log_g=3.1):
        grid = self.load_kurucz_helper(preload_arrays=preload_arrays)

        idx1, idx2 = grid.get_nearby_indexes(Fe_H=Fe_H, T_eff=T_eff, log_g=log_g)
        a = grid.get_model(idx1)
        b = grid.get_model(idx2)
        a.plot()
        b.plot()
        spec = grid.interpolate_model_linear(Fe_H=Fe_H, T_eff=T_eff, log_g=log_g)
        self.assertEqual((1221,), spec.wave.shape)
        self.assertEqual((1221,), spec.flux.shape)
        spec.plot()
        self.save_fig()

    def test_interpolate_model_linear_preload(self):
        self.interpolate_model_linear_helper(preload_arrays=True)

    def test_interpolate_model_linear_lazy(self):
        self.interpolate_model_linear_helper(preload_arrays=False)

    def test_interpolate_model_linear_outside(self):
        self.interpolate_model_linear_helper(preload_arrays=False)

    def test_interpolate_model_linear_outside(self):
        grid = self.load_kurucz_helper(preload_arrays=True)

        # These are outside the parameter space but inside grid
        spec = grid.interpolate_model_linear(Fe_H=-0.9, T_eff=14300, log_g=1.2)
        self.assertIsNone(spec)

    def interpolate_model_spline_helper(self, preload_arrays, Fe_H=0.01, T_eff=4567, log_g=3.1):
        grid = self.load_kurucz_helper(preload_arrays=preload_arrays)

        idx1, idx2 = grid.get_nearby_indexes(Fe_H=Fe_H, T_eff=T_eff, log_g=log_g)
        a = grid.get_model(idx1)
        b = grid.get_model(idx2)
        a.plot()
        b.plot()

        spec = grid.interpolate_model_spline('T_eff', Fe_H=Fe_H, T_eff=T_eff, log_g=log_g)
        spec.plot()
        self.save_fig()

    def test_interpolate_model_spline_preload(self):
        self.interpolate_model_spline_helper(preload_arrays=True)

    def test_interpolate_model_spline_lazy(self):
        self.interpolate_model_spline_helper(preload_arrays=False)

    def test_interpolate_model_spline_outside(self):
        grid = self.load_kurucz_helper(preload_arrays=True)

        # These are outside the parameter space but inside grid
        spec = grid.interpolate_model_spline('T_eff', Fe_H=-0.9, T_eff=14300, log_g=1.2)
        self.assertIsNone(spec)