from unittest import TestCase
import os

from pfsspec.stellarmod.io.kuruczspectrumreader import KuruczSpectrumReader

class TestKuruczSpectrumReader(TestCase):
    @classmethod
    def setUpClass(cls):
        global PFSSPEC_DATA_PATH
        PFSSPEC_DATA_PATH = os.environ['PFSSPEC_DATA_PATH']

    def test_read(self):
        filename = os.path.join(PFSSPEC_DATA_PATH, 'stellar/kurucz/gridm05aodfnew/fm05ak2odfnew.pck')
        with open(filename) as f:
            r = KuruczSpectrumReader(f)
            spec = r.read()

    def test_read_all(self):
        filename = os.path.join(PFSSPEC_DATA_PATH, 'stellar/kurucz/gridm05aodfnew/fm05ak2odfnew.pck')
        with open(filename) as f:
            r = KuruczSpectrumReader(f)
            specs = r.read_all()

    def test_read_grid_kurucz(self):
        path = os.path.join(PFSSPEC_DATA_PATH, 'stellar/kurucz')
        grid = KuruczSpectrumReader.read_grid(path, 'kurucz')