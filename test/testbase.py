import os
from unittest import TestCase
import matplotlib.pyplot as plt

from pfsspec.stellarmod.kuruczgrid import KuruczGrid

class TestBase(TestCase):

    @classmethod
    def setUpClass(self):
        self.PFSSPEC_DATA_PATH = os.environ['PFSSPEC_DATA_PATH'].strip('"')
        self.PFSSPEC_TEST_PATH = os.environ['PFSSPEC_TEST_PATH'].strip('"')
        self.kurucz_grid = None

    def get_kurucz_grid(self):
        if self.kurucz_grid is None:
            file = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/compressed/kurucz.npz')
            self.kurucz_grid = KuruczGrid()
            self.kurucz_grid.load(file)

        return self.kurucz_grid

    def save_fig(self, f=None, filename=None):
        if f is None:
            f = plt
        if filename is None:
            filename = self._testMethodName[5:] + '.png'
        f.savefig(os.path.join(self.PFSSPEC_TEST_PATH, filename))