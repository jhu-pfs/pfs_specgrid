from test.test_base import TestBase
import os

from pfsspec.stellarmod.kuruczspectrumreader import KuruczSpectrumReader
from pfsspec.stellarmod.kuruczgridreader import KuruczGridReader
from pfsspec.stellarmod.kuruczgrid import KuruczGrid

class TestKuruczGridReader(TestBase):
    def test_read_grid_kurucz(self):
        path = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/atlas9/')
        grid = KuruczGrid('test')
        grid.preload_arrays = True
        reader = KuruczGridReader(grid, path)
        reader.read_grid()
        self.assertEqual((2, 61, 11, 1221), grid.data['flux'].shape)

    def test_get_filename(self):
        self.skipTest()