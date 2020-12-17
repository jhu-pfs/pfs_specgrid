from test.test_base import TestBase
from pfsspec.stellarmod.boszmodelgrid import BoszModelGrid

class TestModelGrid(TestBase):
    def get_test_grid(self, args):
        grid = self.get_bosz_grid()
        grid.init_from_args(args)
        return grid

    def test_init_from_args(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual((slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, None)), grid.slice)

        args = { 'T_eff': [3800, 5000] }
        grid = self.get_test_grid(args)
        self.assertEqual(3800, grid.axes['T_eff'].min)
        self.assertEqual(5000, grid.axes['T_eff'].max)
        self.assertEqual(slice(1, 7, None), grid.slice[1])

        args = { 'T_eff': [3800, 4800] }
        grid = self.get_test_grid(args)
        self.assertEqual(slice(1, 6, None), grid.slice[1])

        args = { 'T_eff': [5000] }
        grid = self.get_test_grid(args)
        self.assertEqual(6, grid.slice[1])

    def test_get_axes(self):
        args = {}
        grid = self.get_test_grid(args)
        axes = grid.get_axes()
        self.assertEqual(5, len(axes))

        args = { 'T_eff': [3800, 4800] }
        grid = self.get_test_grid(args)
        axes = grid.get_axes()
        self.assertEqual(5, len(axes))

        args = { 'T_eff': [5000] }
        grid = self.get_test_grid(args)
        axes = grid.get_axes()
        self.assertEqual(4, len(axes))

    def test_get_shape(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual((14, 67, 11, 6, 4), grid.get_shape())

        args = { 'T_eff': [3800, 5000] }
        grid = self.get_test_grid(args)
        self.assertEqual((14, 6, 11, 6, 4), grid.get_shape())

        args = { 'T_eff': [5000] }
        grid = self.get_test_grid(args)
        self.assertEqual((14, 11, 6, 4), grid.get_shape())

    def test_get_model_count(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual(110455, grid.get_model_count())

        args = { 'T_eff': [3800, 5000] }
        grid = self.get_test_grid(args)
        self.assertEqual(15080, grid.get_model_count())

        args = { 'T_eff': [5000] }
        grid = self.get_test_grid(args)
        self.assertEqual(2520, grid.get_model_count())

    def test_get_flux_shape(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual((14, 67, 11, 6, 4, 16094), grid.get_flux_shape())

        args = { 'T_eff': [3800, 5000] }
        grid = self.get_test_grid(args)
        self.assertEqual((14, 6, 11, 6, 4, 16094), grid.get_flux_shape())

        args = { 'T_eff': [5000] }
        grid = self.get_test_grid(args)
        self.assertEqual((14, 11, 6, 4, 16094), grid.get_flux_shape())

