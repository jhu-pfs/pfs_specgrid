from test.test_base import TestBase
import os
import numpy as np
from numpy.testing import assert_array_equal
import h5py

from pfsspec.data.arraygrid import ArrayGrid
from pfsspec.data.gridaxis import GridAxis

class TestArrayGrid(TestBase):
    def create_new_grid(self, preload_arrays=True):
        grid = ArrayGrid()
        grid.preload_arrays = preload_arrays

        grid.init_axis('a', np.array([1., 2., 3., 4., 5.]))
        grid.init_axis('b', np.array([10., 20., 30.]))
        grid.build_axis_indexes()

        grid.init_value('U')
        grid.init_value('V')

        return grid

    def init_empty_grid(self, grid):
        grid.init_value('U', (10,))
        grid.init_value('V', (100,))

        return grid

    def init_full_grid(self, grid):
        grid.init_value('U', (10,))
        grid.init_value('V', (100,))

        grid.values['U'] = np.random.rand(*grid.values['U'].shape)
        grid.values['V'] = np.random.rand(*grid.values['V'].shape)
        
        grid.build_value_indexes(rebuild=True)

        return grid

    def init_jagged_grid(self, grid):
        self.init_full_grid(grid)

        grid.values['U'][2:,2:,:] = np.nan
        grid.values['V'][2:,2:,:] = np.nan

        grid.build_value_indexes(rebuild=True)

        return grid

    def create_new_grid_1D(self, preload_arrays=True):
        grid = ArrayGrid()
        grid.preload_arrays = preload_arrays

        grid.init_axis('a', np.array([1., 2., 3., 4., 5.]))
        grid.build_axis_indexes()

        grid.init_value('U')
        grid.init_value('V')

        return grid

    def init_full_grid_1D(self, grid):
        grid.init_value('U', (10,))
        grid.init_value('V', (100,))

        grid.values['U'] = np.random.rand(*grid.values['U'].shape)
        grid.values['V'] = np.random.rand(*grid.values['V'].shape)

        grid.build_value_indexes(rebuild=True)

        return grid

    def test_get_valid_value_count(self):
        #    1 2 3 4 5
        # 10 * * * * *
        # 20 * * * * *
        # 30 * * * * *
        grid = self.create_new_grid()
        self.init_full_grid(grid)

        grid.slice = None
        count = grid.get_valid_value_count('U')
        self.assertEqual(15, count)

        #    1 2 3 4 5
        # 10 o o o o o
        # 20 * * * * *
        # 30 * * * * *
        grid.axes['b'].min = 20
        grid.axes['b'].max = 30
        grid.slice = np.s_[:, 1:3]
        count = grid.get_valid_value_count('U')
        self.assertEqual(10, count)

        #    1 2 3 4 5
        # 10 * * * * *
        # 20 * * * * *
        # 30 * * o o o
        grid = self.create_new_grid()
        grid = self.init_jagged_grid(grid)
        grid.slice = None
        count = grid.get_valid_value_count('U')
        self.assertEqual(12, count)

        #    1 2 3 4 5
        # 10 o o o o o
        # 20 * * * * *
        # 30 * * o o o
        grid.axes['b'].min = 20
        grid.axes['b'].max = 30
        grid.slice = np.s_[:, 1:3]
        count = grid.get_valid_value_count('U')
        self.assertEqual(7, count)

    def test_get_index(self):
        grid = self.create_new_grid()
        self.init_full_grid(grid)

        idx = grid.get_index(a=1, b=30)
        self.assertEqual((0, 2), idx)

        idx = grid.get_index(a=1)
        self.assertEqual((0, slice(None)), idx)

        idx = grid.get_index(b=30)
        self.assertEqual((slice(None), 2), idx)

    def test_get_nearest_index(self):
        grid = self.create_new_grid()
        self.init_full_grid(grid)

        idx = grid.get_nearest_index(a=1.7, b=14)
        self.assertEqual((1, 0), idx)

        idx = grid.get_nearest_index(a=5.6, b=14)
        self.assertEqual((4, 0), idx)

        idx = grid.get_nearest_index(a=5.6)
        self.assertEqual((4, slice(None)), idx)

        idx = grid.get_nearest_index(b=14)
        self.assertEqual((slice(None), 0), idx)

    def test_get_nearby_indexes(self):
        grid = self.create_new_grid()
        self.init_full_grid(grid)

        idx = grid.get_nearby_indexes(a=3.7, b=24)
        self.assertEqual(((2, 1), (3, 2)), idx)

        idx = grid.get_nearby_indexes(a=5.6, b=14)
        self.assertEqual(None, idx)

    def test_has_value_at(self):
        grid = self.create_new_grid()
        self.init_full_grid(grid)

        self.assertTrue(grid.has_value_at('U', (0, 0)))
        self.assertTrue(grid.has_value_at('U', (0, 1)))

        grid.values['U'][0, 0, 7] = np.nan
        grid.build_value_index('U', rebuild=True)

        self.assertFalse(grid.has_value_at('U', (0, 0)))
        self.assertTrue(grid.has_value_at('U', (0, 1)))

    def test_set_values(self):
        grid = self.create_new_grid()
        self.init_empty_grid(grid)

        self.assertFalse(grid.has_value_at('U', (1, 1)))

        grid.set_values({'U': np.random.rand(10)}, a=2, b=20)

        self.assertTrue(grid.has_value_at('U', (1, 1)))

    def test_get_value(self):
        grid = self.create_new_grid()
        self.init_full_grid(grid)

        values = grid.get_values(a=1, b=20)
        self.assertEquals(2, len(values))
        self.assertEquals((10,), values['U'].shape)
        self.assertEquals((100,), values['V'].shape)

        values = grid.get_values(a=1)
        self.assertEquals(2, len(values))
        self.assertEquals((3, 10), values['U'].shape)
        self.assertEquals((3, 100), values['V'].shape)

        values = grid.get_values(b=20)
        self.assertEquals(2, len(values))
        self.assertEquals((5, 10), values['U'].shape)
        self.assertEquals((5, 100), values['V'].shape)

    def test_get_nearest_values(self):
        grid = self.create_new_grid()
        self.init_full_grid(grid)
        values = grid.get_nearest_values(a=1.1, b=27)
        self.assertEquals(2, len(values))
        self.assertEquals((10,), values['U'].shape)
        self.assertEquals((100,), values['V'].shape)

    def test_get_values_at(self):
        grid = self.create_new_grid()
        self.init_full_grid(grid)
        values = grid.get_values_at((1, 1))
        self.assertEquals(2, len(values))
        self.assertEquals((10,), values['U'].shape)
        self.assertEquals((100,), values['V'].shape)

    def test_get_value(self):
        grid = self.create_new_grid()
        self.init_full_grid(grid)
        
        value = grid.get_value('U')
        self.assertEquals((5, 3, 10,), value.shape)

        value = grid.get_value('U', a=1)
        self.assertEquals((3, 10,), value.shape)

        value = grid.get_value('U', slice(2, 5), b=20)
        self.assertEquals((5, 3,), value.shape)

        value = grid.get_value('U', slice(2, 5), a=1, b=20)
        self.assertEquals((3,), value.shape)

    def test_get_value_sliced(self):
        grid = self.create_new_grid()
        self.init_full_grid(grid)
        grid.slice = np.s_[0:4, 1:3]
        
        value = grid.get_value('U')
        self.assertEquals((4, 2, 10,), value.shape)

        value = grid.get_value('U', a=1)
        self.assertEquals((2, 10,), value.shape)

        value = grid.get_value('U', slice(2, 5), b=20)
        self.assertEquals((4, 3,), value.shape)

        value = grid.get_value('U', slice(2, 5), a=1, b=20)
        self.assertEquals((3,), value.shape)

    def test_get_nearest_value(self):
        grid = self.create_new_grid()
        self.init_full_grid(grid)

        value = grid.get_nearest_value('U', a=1.1, b=27)
        self.assertEquals((10,), value.shape)

        value = grid.get_nearest_value('U', slice(2, 5), a=1.1, b=27)
        self.assertEquals((3,), value.shape)

    def test_get_value_at(self):
        grid = self.create_new_grid()
        self.init_full_grid(grid)

        value = grid.get_value_at('U', (1, 1))
        self.assertEquals((10,), value.shape)

        value = grid.get_value_at('U', ([1,1], [2,2]))
        self.assertEquals((2, 10,), value.shape)

    def test_interpolate_value_linear1d(self):
        grid = self.create_new_grid_1D()
        self.init_full_grid_1D(grid)

        value = grid.interpolate_value_linear1d('U', a=2.7)
        self.assertIsNotNone(value)
        value = grid.interpolate_value_linear1d('U', a=2.0)
        self.assertIsNotNone(value)

    def test_interpolate_value_linearNd(self):
        grid = self.create_new_grid()
        self.init_full_grid(grid)
        value = grid.interpolate_value_linearNd('U', a=2.7, b=18)
        self.assertIsNotNone(value)

    def test_interpolate_value_spline(self):
        grid = self.create_new_grid()
        self.init_full_grid(grid)
        value, params = grid.interpolate_value_spline('U', 'a', a=2.7, b=18)
        self.assertIsNotNone(value)
        self.assertEquals({'a': 2.7, 'b': 20.0}, params)

    def test_save(self):
        grid = self.create_new_grid()
        self.init_full_grid(grid)
        grid.save(os.path.join(self.PFSSPEC_TEST_PATH, self.get_filename('.h5')), format='h5')

    def test_save_lazy_whole(self):
        grid = self.create_new_grid(preload_arrays=False)
        grid.save(os.path.join(self.PFSSPEC_TEST_PATH, self.get_filename('.h5')), format='h5')

        self.init_empty_grid(grid)

        grid.set_value('U', np.random.rand(*grid.value_shapes['U']), a=3, b=20)
        grid.set_value('V', np.random.rand(*grid.value_shapes['V']), a=3, b=20)

    def test_save_lazy_slice(self):
        grid = self.create_new_grid(preload_arrays=False)
        grid.save(os.path.join(self.PFSSPEC_TEST_PATH, self.get_filename('.h5')), format='h5')

        self.init_empty_grid(grid)

        grid.set_value('U', np.random.rand(*grid.value_shapes['U']), s=slice(None), a=3, b=20)
        grid.set_value('V', np.random.rand(*grid.value_shapes['V']), s=slice(None), a=3, b=20)
        grid.set_value('U', np.random.rand(3), s=slice(4, 7), a=3, b=20)
        grid.set_value('V', np.random.rand(2), s=slice(0, 2), a=3, b=20)

    def test_load(self):
        grid = self.create_new_grid()
        self.init_full_grid(grid)
        grid.save(os.path.join(self.PFSSPEC_TEST_PATH, self.get_filename('.h5')), format='h5')

        grid = self.create_new_grid()
        grid.load(os.path.join(self.PFSSPEC_TEST_PATH, self.get_filename('.h5')), format='h5')

    def test_load_lazy_whole(self):
        grid = self.create_new_grid()
        self.init_full_grid(grid)
        grid.save(os.path.join(self.PFSSPEC_TEST_PATH, self.get_filename('.h5')), format='h5')

        grid = self.create_new_grid(preload_arrays=False)

        self.assertIsNone(grid.values['U'])
        self.assertIsNone(grid.values['V'])

        grid.load(os.path.join(self.PFSSPEC_TEST_PATH, self.get_filename('.h5')), format='h5')

        value = grid.get_value('U', a=2, b=20)
        self.assertEquals((10,), value.shape)

        value = grid.get_value('U', slice(2, 5), a=2, b=20)
        self.assertEquals((3,), value.shape)

    def test_load_lazy_slice(self):
        grid = self.create_new_grid()
        self.init_full_grid(grid)
        grid.save(os.path.join(self.PFSSPEC_TEST_PATH, self.get_filename('.h5')), format='h5')

        grid = self.create_new_grid()

        self.assertIsNone(grid.values['U'])
        self.assertIsNone(grid.values['V'])

        grid.load(os.path.join(self.PFSSPEC_TEST_PATH, self.get_filename('.h5')), format='h5')

        value = grid.get_value('U', s=slice(None), a=2, b=20)
        self.assertEquals((10,), value.shape)
        value = grid.get_value('U', s=slice(2, 5), a=2, b=20)
        self.assertEquals((3,), value.shape)

    def test_get_value_padded(self):
        grid = self.create_new_grid()
        self.init_full_grid(grid)

        padded, paxes = grid.get_value_padded('U', interpolation='ijk')
        self.assertEqual((7, 5, 10), padded.shape)
        self.assertEqual(2, len(paxes))
        assert_array_equal(np.array([0., 1., 2., 3., 4., 5., 6.]), paxes['a'].values)
        assert_array_equal(np.array([0., 10., 20., 30., 40.]), paxes['b'].values)

        padded, paxes = grid.get_value_padded('U', interpolation='xyz')
        self.assertEqual((7, 5, 10), padded.shape)
        self.assertEqual(2, len(paxes))
        assert_array_equal(np.array([0., 1., 2., 3., 4., 5., 6.]), paxes['a'].values)
        assert_array_equal(np.array([ 0., 10., 20., 30., 40.]), paxes['b'].values)

    def test_fit_rbf(self):
        grid = self.create_new_grid()
        self.init_full_grid(grid)

        rbf, paxes = grid.fit_rbf('U')
        self.assertEqual((2, 35), rbf.xi.shape)
        self.assertEqual((35, 10), rbf.nodes.shape)

        rbf, paxes = grid.fit_rbf('U', padding_mode=None)
        self.assertEqual((2, 15), rbf.xi.shape)
        self.assertEqual((15, 10), rbf.nodes.shape)

        rbf, paxes = grid.fit_rbf('U', a=2)
        self.assertEqual((1, 5), rbf.xi.shape)
        self.assertEqual((5, 10), rbf.nodes.shape)

        rbf, paxes = grid.fit_rbf('U', s=np.s_[1:3], padding_mode='xyz', a=2)
        self.assertEqual((1, 5), rbf.xi.shape)
        self.assertEqual((5, 2), rbf.nodes.shape)

