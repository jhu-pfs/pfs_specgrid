from test.testbase import TestBase
from unittest import TestCase
import numpy as np
import os

from pfsspec.filter import Filter

class TestFilter(TestBase):
    def test_read(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/hsc/hsc_r.dat')
        filter = Filter()
        filter.read(filename)
        filter.plot()

        self.assertEqual((251,), filter.wave.shape)
        self.assertEqual((251,), filter.thru.shape)

        self.save_fig()

class TestFilter(TestBase):
    def test_extend(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/hsc/hsc_r.dat')
        filter = Filter()
        filter.read(filename)
        filter.plot()

        self.assertEqual((251,), filter.wave.shape)
        self.assertEqual((251,), filter.thru.shape)

        filter.extend(3700, 12600, 2.7)
        filter.plot()

        self.assertEqual((2622,), filter.wave.shape)
        self.assertEqual((2622,), filter.thru.shape)

        self.save_fig()