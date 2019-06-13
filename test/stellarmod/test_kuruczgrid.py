from test.testbase import TestBase
from unittest import TestCase
import os

from pfsspec.stellarmod.io.kuruczspectrumreader import KuruczSpectrumReader
from pfsspec.stellarmod.kuruczgrid import KuruczGrid

class TestKuruczGrid(TestBase):

    def test_save(self):
        path = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/kurucz')
        file = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/compressed/test.npz')
        grid = KuruczSpectrumReader.read_grid(path, 'test')
        self.assertEqual((2, 61, 11, 1221), grid.flux.shape)
        grid.save(file)

    def test_load(self):
        file = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/compressed/kurucz.npz')
        grid = KuruczGrid()
        grid.load(file)
        self.assertEqual((18, 61, 11, 1221), grid.flux.shape)

    def test_get_nearest_model(self):
        file = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/compressed/kurucz.npz')
        grid = KuruczGrid()
        grid.load(file)
        spec = grid.get_nearest_model(0.11, 4900, 3.1)
        self.assertEqual((1221,), spec.wave.shape)
        self.assertEqual((1221,), spec.flux.shape)