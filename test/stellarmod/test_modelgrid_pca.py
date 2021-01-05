import os
from test.test_base import TestBase

from pfsspec.data.arraygrid import ArrayGrid
from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.stellarmod.bosz import Bosz

class TestModelGrid_Pca(TestBase):
    def get_test_grid(self, args):
        file = os.path.join(self.PFSSPEC_DATA_PATH, '/scratch/ceph/dobos/temp/test071/spectra.h5')
        grid = ModelGrid(Bosz(pca=True), ArrayGrid)
        grid.load(file, format='h5')
        grid.init_from_args(args)

        return grid

    def test_init_from_args(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual(slice(None), grid.wave_slice)

    def test_get_axes(self):
        args = {}
        grid = self.get_test_grid(args)
        axes = grid.get_axes()
        self.assertEqual(5, len(axes))

    def test_get_shape(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual((14, 66, 11, 6, 4), grid.get_shape())

    def test_get_model_count(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual(116614, grid.get_model_count())

    def test_get_flux_shape(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual((14, 66, 11, 6, 4, 14272), grid.get_flux_shape())

        args = { 'T_eff': [3800, 5000] }
        grid = self.get_test_grid(args)
        self.assertEqual((14, 6, 11, 6, 4, 14272), grid.get_flux_shape())

        args = { 'T_eff': [5000] }
        grid = self.get_test_grid(args)
        self.assertEqual((14, 11, 6, 4, 14272), grid.get_flux_shape())

        args = { 'T_eff': [3800, 5000], 'lambda': [4800, 5600] }
        grid = self.get_test_grid(args)
        self.assertEqual((14, 6, 11, 6, 4, 1542), grid.get_flux_shape())

    def test_get_nearest_model(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(Fe_H=0., T_eff=4500, log_g=4, C_M=0, O_M=0)

    def test_interpolate_model_rbf(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.interpolate_model_rbf(Fe_H=-1.2, T_eff=4125, log_g=4.3)

        pass