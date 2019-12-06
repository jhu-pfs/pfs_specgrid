from test.test_base import TestBase
import os
import numpy as np

from pfsspec.data.grid import Grid
from pfsspec.data.gridparam import GridParam

class TestGrid(TestBase):
    def create_empty_grid_inmem(self):
        grid = Grid()

        grid.preload_arrays = True
        grid.init_param('a', np.array([1., 2., 3., 4., 5.]))
        grid.init_param('b', np.array([10., 20., 30.]))
        grid.init_data_item('U', (10,))
        grid.init_data_item('V', (100,))

        grid.build_params_index()

        return grid

    def create_full_grid_inmem(self):
        grid = Grid()

        grid.preload_arrays = True
        grid.init_param('a', np.array([1., 2., 3., 4., 5.]))
        grid.init_param('b', np.array([10., 20., 30.]))
        grid.init_data_item('U', (10,))
        grid.init_data_item('V', (100,))

        grid.data['U'] = np.random.rand(*grid.data['U'].shape)
        grid.data['V'] = np.random.rand(*grid.data['V'].shape)

        grid.build_params_index()
        grid.build_data_index(rebuild=True)

        return grid

    def test_get_index(self):
        grid = self.create_full_grid_inmem()
        idx = grid.get_index(a=1, b=30)
        self.assertEqual((0, 2), idx)

    def test_get_nearest_index(self):
        grid = self.create_full_grid_inmem()

        idx = grid.get_nearest_index(a=1.7, b=14)
        self.assertEqual((1, 0), idx)

        idx = grid.get_nearest_index(a=5.6, b=14)
        self.assertEqual((4, 0), idx)

    def test_get_nearby_indexes(self):
        grid = self.create_full_grid_inmem()

        idx = grid.get_nearby_indexes(a=3.7, b=24)
        self.assertEqual(((2, 1), (3, 2)), idx)

        idx = grid.get_nearby_indexes(a=5.6, b=14)
        self.assertEqual(None, idx)

    def test_is_data_item_idx(self):
        grid = self.create_full_grid_inmem()

        self.assertTrue(grid.is_data_item_idx('U', (0, 0)))
        self.assertTrue(grid.is_data_item_idx('U', (0, 1)))

        grid.data['U'][0, 0, 7] = np.nan
        grid.build_data_item_index('U', rebuild=True)

        self.assertFalse(grid.is_data_item_idx('U', (0, 0)))
        self.assertTrue(grid.is_data_item_idx('U', (0, 1)))

    def test_set_data(self):
        grid = self.create_empty_grid_inmem()

        self.assertFalse(grid.is_data_item_idx('U', (1, 1)))

        grid.set_data({'U': np.random.rand(10)}, a=2, b=20)

        self.assertTrue(grid.is_data_item_idx('U', (1, 1)))

    def test_get_data(self):
        grid = self.create_full_grid_inmem()
        data = grid.get_data(a=1, b=20)
        self.assertEquals(2, len(data))
        self.assertEquals((10,), data['U'].shape)
        self.assertEquals((100,), data['V'].shape)

    def test_get_nearest_data(self):
        grid = self.create_full_grid_inmem()
        data = grid.get_nearest_data(a=1.1, b=27)
        self.assertEquals(2, len(data))
        self.assertEquals((10,), data['U'].shape)
        self.assertEquals((100,), data['V'].shape)

    def test_get_data_idx(self):
        grid = self.create_full_grid_inmem()
        data = grid.get_data_idx((1, 1))
        self.assertEquals(2, len(data))
        self.assertEquals((10,), data['U'].shape)
        self.assertEquals((100,), data['V'].shape)

    def test_get_data_item(self):
        grid = self.create_full_grid_inmem()
        data = grid.get_data_item('U', a=1, b=20)
        self.assertEquals((10,), data.shape)

    def test_get_nearest_data_item(self):
        grid = self.create_full_grid_inmem()
        data = grid.get_nearest_data_item('U', a=1.1, b=27)
        self.assertEquals((10,), data.shape)

    def test_get_data_item_idx(self):
        grid = self.create_full_grid_inmem()
        data = grid.get_data_item_idx('U', (1, 1))
        self.assertEquals((10,), data.shape)

    def test_interpolate_data_item_linear(self):
        grid = self.create_full_grid_inmem()
        data = grid.interpolate_data_item_linear('U', a=2.7, b=18)
        self.assertIsNotNone(data)

    def test_interpolate_data_item_spline(self):
        grid = self.create_full_grid_inmem()
        data, params = grid.interpolate_data_item_spline('U', 'a', a=2.7, b=18)
        self.assertIsNotNone(data)
        self.assertEquals({'a': 2.7, 'b': 20.0}, params)