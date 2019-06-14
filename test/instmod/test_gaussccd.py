from test.testbase import TestBase
import numpy as np

from pfsspec.instmod.gaussccd import GaussCcd

class TestGaussCcd(TestBase):
    def test_create(self):
        ccd = GaussCcd()
        ccd.create(np.arange(6300, 9700, 2.7), 0.9, 7900, 1200)
        ccd.plot()

        self.assertEqual((1260,), ccd.wave.shape)
        self.assertEqual((1260,), ccd.qeff.shape)

        self.save_fig()