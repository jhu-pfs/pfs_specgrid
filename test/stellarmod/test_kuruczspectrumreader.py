from test.testbase import TestBase
import os

from pfsspec.stellarmod.kuruczspectrumreader import KuruczSpectrumReader

class TestKuruczSpectrumReader(TestBase):
    def test_read(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/kurucz/gridm05aodfnew/fm05ak2odfnew.pck')
        with open(filename) as f:
            r = KuruczSpectrumReader(f)
            spec = r.read()

    def test_read_all(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/kurucz/gridm05aodfnew/fm05ak2odfnew.pck')
        with open(filename) as f:
            r = KuruczSpectrumReader(f)
            specs = r.read_all()

    def test_read_grid_kurucz(self):
        path = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/kurucz')
        grid = KuruczSpectrumReader.read_grid(path, 'test')
        self.assertEqual((2, 61, 11, 1221), grid.flux.shape)

    def test_get_filename(self):
        self.fail()