from test.test_base import TestBase
import os
import numpy as np

from pfsspec.noisemod.sky import Sky


class TestSky(TestBase):
    def test_load(self):
        file = os.path.join(self.PFSSPEC_DATA_PATH, 'noise/sky/r/sky.h5')
        sky = Sky()
        sky.preload_arrays = True
        sky.load(file, format='h5')

        self.assertIsNotNone(sky.wave)
        self.assertIsNotNone(sky.data['counts'])
        self.assertIsNotNone(sky.data['conv'])

        sky = Sky()
        sky.preload_arrays = False
        sky.load(file, format='h5')

        self.assertIsNotNone(sky.wave)
        data = sky.get_nearest_data_item('conv', za=10, fa=0.5)
        self.assertIsNotNone(data)

    def test_interpolate(self):
        file = os.path.join(self.PFSSPEC_DATA_PATH, 'noise/sky/r/sky.h5')
        sky = Sky()
        sky.preload_arrays = True
        sky.load(file, format='h5')

        data, _ = sky.interpolate_data_item_linear('conv', za=10, fa=0.5)
        self.assertIsNotNone(data)