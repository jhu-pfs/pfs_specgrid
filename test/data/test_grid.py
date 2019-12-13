from test.test_base import TestBase
import os
import numpy as np
import h5py

from pfsspec.data.grid import Grid
from pfsspec.data.gridparam import GridParam

class TestGrid(TestBase):
    def create_empty_grid(self, preload_arrays=True):
        grid = Grid()
        grid.preload_arrays = preload_arrays

        grid.init_param('a', np.array([1., 2., 3., 4., 5.]))
        grid.init_param('b', np.array([10., 20., 30.]))
        grid.init_data_item('U', (10,))
        grid.init_data_item('V', (100,))

        grid.build_params_index()

        return grid

    def create_full_grid(self, preload_arrays=True):
        grid = Grid()
        grid.preload_arrays = preload_arrays

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
        grid = self.create_full_grid()
        idx = grid.get_index(a=1, b=30)
        self.assertEqual((0, 2), idx)

    def test_get_nearest_index(self):
        grid = self.create_full_grid()

        idx = grid.get_nearest_index(a=1.7, b=14)
        self.assertEqual((1, 0), idx)

        idx = grid.get_nearest_index(a=5.6, b=14)
        self.assertEqual((4, 0), idx)

    def test_get_nearby_indexes(self):
        grid = self.create_full_grid()

        idx = grid.get_nearby_indexes(a=3.7, b=24)
        self.assertEqual(((2, 1), (3, 2)), idx)

        idx = grid.get_nearby_indexes(a=5.6, b=14)
        self.assertEqual(None, idx)

    def test_is_data_item_idx(self):
        grid = self.create_full_grid()

        self.assertTrue(grid.is_data_item_idx('U', (0, 0)))
        self.assertTrue(grid.is_data_item_idx('U', (0, 1)))

        grid.data['U'][0, 0, 7] = np.nan
        grid.build_data_item_index('U', rebuild=True)

        self.assertFalse(grid.is_data_item_idx('U', (0, 0)))
        self.assertTrue(grid.is_data_item_idx('U', (0, 1)))

    def test_set_data(self):
        grid = self.create_empty_grid()

        self.assertFalse(grid.is_data_item_idx('U', (1, 1)))

        grid.set_data({'U': np.random.rand(10)}, a=2, b=20)

        self.assertTrue(grid.is_data_item_idx('U', (1, 1)))

    def test_get_data(self):
        grid = self.create_full_grid()

        data = grid.get_data(a=1, b=20)
        self.assertEquals(2, len(data))
        self.assertEquals((10,), data['U'].shape)
        self.assertEquals((100,), data['V'].shape)

    def test_get_nearest_data(self):
        grid = self.create_full_grid()
        data = grid.get_nearest_data(a=1.1, b=27)
        self.assertEquals(2, len(data))
        self.assertEquals((10,), data['U'].shape)
        self.assertEquals((100,), data['V'].shape)

    def test_get_data_idx(self):
        grid = self.create_full_grid()
        data = grid.get_data_idx((1, 1))
        self.assertEquals(2, len(data))
        self.assertEquals((10,), data['U'].shape)
        self.assertEquals((100,), data['V'].shape)

    def test_get_data_item_whole(self):
        grid = self.create_full_grid()
        data = grid.get_data_item('U', a=1, b=20)
        self.assertEquals((10,), data.shape)

    def test_get_data_item_slice(self):
        grid = self.create_full_grid()
        data = grid.get_data_item('U', slice(2, 5), a=1, b=20)
        self.assertEquals((3,), data.shape)

    def test_get_nearest_data_item(self):
        grid = self.create_full_grid()

        data = grid.get_nearest_data_item('U', a=1.1, b=27)
        self.assertEquals((10,), data.shape)

        data = grid.get_nearest_data_item('U', slice(2, 5), a=1.1, b=27)
        self.assertEquals((3,), data.shape)

    def test_get_data_item_idx(self):
        grid = self.create_full_grid()

        data = grid.get_data_item_idx('U', (1, 1))
        self.assertEquals((10,), data.shape)

        data = grid.get_data_item_idx('U', ([1,1], [2,2]))
        self.assertEquals((2, 10,), data.shape)

    def test_interpolate_data_item_linear(self):
        grid = self.create_full_grid()
        data = grid.interpolate_data_item_linear('U', a=2.7, b=18)
        self.assertIsNotNone(data)

    def test_interpolate_data_item_spline(self):
        grid = self.create_full_grid()
        data, params = grid.interpolate_data_item_spline('U', 'a', a=2.7, b=18)
        self.assertIsNotNone(data)
        self.assertEquals({'a': 2.7, 'b': 20.0}, params)

    def test_save(self):
        grid = self.create_full_grid()
        grid.save(os.path.join(self.PFSSPEC_TEST_PATH, self.get_filename('.h5')), format='h5')

    def test_save_lazy_whole(self):
        grid = self.create_empty_grid(preload_arrays=False)
        grid.save(os.path.join(self.PFSSPEC_TEST_PATH, self.get_filename('.h5')), format='h5')
        grid.set_data_item('U', np.random.rand(*grid.data_shape['U']), a=3, b=20)
        grid.set_data_item('V', np.random.rand(*grid.data_shape['V']), a=3, b=20)

    def test_save_lazy_slice(self):
        grid = self.create_empty_grid(preload_arrays=False)
        grid.save(os.path.join(self.PFSSPEC_TEST_PATH, self.get_filename('.h5')), format='h5')
        grid.set_data_item('U', np.random.rand(*grid.data_shape['U']), s=slice(None), a=3, b=20)
        grid.set_data_item('V', np.random.rand(*grid.data_shape['V']), s=slice(None), a=3, b=20)
        grid.set_data_item('U', np.random.rand(3), s=slice(4, 7), a=3, b=20)
        grid.set_data_item('V', np.random.rand(2), s=slice(0, 2), a=3, b=20)

    def test_load(self):
        grid = self.create_full_grid()
        grid.save(os.path.join(self.PFSSPEC_TEST_PATH, self.get_filename('.h5')), format='h5')

        grid = self.create_empty_grid()
        grid.load(os.path.join(self.PFSSPEC_TEST_PATH, self.get_filename('.h5')), format='h5')

    def test_load_lazy_whole(self):
        grid = self.create_full_grid()
        grid.save(os.path.join(self.PFSSPEC_TEST_PATH, self.get_filename('.h5')), format='h5')

        grid = self.create_empty_grid(preload_arrays=False)
        grid.load(os.path.join(self.PFSSPEC_TEST_PATH, self.get_filename('.h5')), format='h5')

        self.assertIsNone(grid.data['U'])
        self.assertIsNone(grid.data['V'])

        data = grid.get_data_item('U', a=2, b=20)
        self.assertEquals((10,), data.shape)

        data = grid.get_data_item('U', slice(2, 5), a=2, b=20)
        self.assertEquals((3,), data.shape)

    def test_load_lazy_slice(self):
        grid = self.create_full_grid()
        grid.save(os.path.join(self.PFSSPEC_TEST_PATH, self.get_filename('.h5')), format='h5')

        grid = self.create_empty_grid(preload_arrays=False)
        grid.load(os.path.join(self.PFSSPEC_TEST_PATH, self.get_filename('.h5')), format='h5')

        self.assertIsNone(grid.data['U'])
        self.assertIsNone(grid.data['V'])

        data = grid.get_data_item('U', s=slice(None), a=2, b=20)
        self.assertEquals((10,), data.shape)
        data = grid.get_data_item('U', s=slice(2, 5), a=2, b=20)
        self.assertEquals((3,), data.shape)
