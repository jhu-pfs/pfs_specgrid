import os
from unittest import TestCase
import matplotlib.pyplot as plt

from pfsspec.data.dataset import Dataset
from pfsspec.stellarmod.kuruczgrid import KuruczGrid
from pfsspec.obsmod.filter import Filter

class TestBase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.PFSSPEC_DATA_PATH = os.environ['PFSSPEC_DATA'].strip('"') if 'PFSSPEC_DATA' in os.environ else None
        cls.PFSSPEC_TEST_PATH = os.environ['PFSSPEC_TEST'].strip('"') if 'PFSSPEC_TEST' in os.environ else None
        cls.PFSSPEC_SDSS_DATA_PATH = os.environ['PFSSPEC_SDSS_DATA'].strip('"') if 'PFSSPEC_SDSS_DATA' in os.environ else None
        cls.kurucz_grid = None
        cls.sdss_dataset = None
        cls.hsc_filters = None

    def setUp(self):
        plt.figure(figsize=(10, 6))

    def get_kurucz_grid(self):
        if self.kurucz_grid is None:
            file = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/compressed/kurucz.h5')
            self.kurucz_grid = KuruczGrid(model='test')
            self.kurucz_grid.load(file, s=None, format='h5')

        return self.kurucz_grid

    def get_sdss_dataset(self):
        if self.sdss_dataset is None:
            file = os.path.join(self.PFSSPEC_DATA_PATH, 'pfs_spec_test/sdss_test/dataset.dat')
            self.sdss_dataset = Dataset()
            self.sdss_dataset.load(file)

        return self.sdss_dataset

    def get_hsc_filter(self, band):
        if self.hsc_filters is None:
            ids = ['g', 'r', 'i', 'z', 'y']
            self.hsc_filters = dict()
            for i in ids:
                filter = Filter()
                filename = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/hsc', r'hsc_%s.dat' % (i))
                filter.read(filename)
                self.hsc_filters[i] = filter
        return self.hsc_filters[band]

    def get_filename(self, ext):
        filename = type(self).__name__[4:] + '_' + self._testMethodName[5:] + ext
        return filename

    def save_fig(self, f=None, filename=None):
        if f is None:
            f = plt
        if filename is None:
            filename = self.get_filename('.png')
        f.savefig(os.path.join(self.PFSSPEC_TEST_PATH, filename))