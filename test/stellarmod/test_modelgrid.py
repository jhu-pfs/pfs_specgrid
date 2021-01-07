from test.test_base import TestBase

class TestModelGrid(TestBase):
    def get_test_grid(self, args):
        grid = self.get_bosz_grid()
        grid.init_from_args(args)
        return grid

    def test_init_from_args(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual(slice(None), grid.wave_slice)

        args = { 'T_eff': [3800, 5000] }
        grid = self.get_test_grid(args)
        self.assertEqual(3800, grid.grid.axes['T_eff'].min)
        self.assertEqual(5000, grid.grid.axes['T_eff'].max)
        self.assertEqual(slice(1, 7, None), grid.grid.slice[1])

        args = { 'T_eff': [3800, 4800] }
        grid = self.get_test_grid(args)
        self.assertEqual(slice(1, 6, None), grid.grid.slice[1])

        args = { 'T_eff': [5000] }
        grid = self.get_test_grid(args)
        self.assertEqual(6, grid.grid.slice[1])

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
        self.assertEqual((14, 66, 11, 6, 4), grid.get_shape())

        args = { 'T_eff': [3800, 5000] }
        grid = self.get_test_grid(args)
        self.assertEqual((14, 6, 11, 6, 4), grid.get_shape())

        args = { 'T_eff': [5000] }
        grid = self.get_test_grid(args)
        self.assertEqual((14, 11, 6, 4), grid.get_shape())

    def test_get_model_count(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual(116614, grid.get_model_count())

        args = { 'T_eff': [3800, 5000] }
        grid = self.get_test_grid(args)
        self.assertEqual(18440, grid.get_model_count())

        args = { 'T_eff': [5000] }
        grid = self.get_test_grid(args)
        self.assertEqual(3080, grid.get_model_count())

    def test_get_flux_shape(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual((14, 66, 11, 6, 4, 21691), grid.get_flux_shape())

        args = { 'T_eff': [3800, 5000] }
        grid = self.get_test_grid(args)
        self.assertEqual((14, 6, 11, 6, 4, 21691), grid.get_flux_shape())

        args = { 'T_eff': [5000] }
        grid = self.get_test_grid(args)
        self.assertEqual((14, 11, 6, 4, 21691), grid.get_flux_shape())

        args = { 'T_eff': [3800, 5000], 'lambda': [4800, 5600] }
        grid = self.get_test_grid(args)
        self.assertEqual((14, 6, 11, 6, 4, 1542), grid.get_flux_shape())

    def test_get_nearest_model(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(Fe_H=0., T_eff=4500, log_g=4, C_M=0, O_M=0)
        self.assertIsNotNone(spec)

    def test_interpolate_model_linear(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.interpolate_model_linear(Fe_H=-1.2, T_eff=4125, log_g=4.3, C_M=0, O_M=0)
        self.assertIsNotNone(spec)