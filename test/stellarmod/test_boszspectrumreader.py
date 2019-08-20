from test.test_base import TestBase
import os

from pfsspec.stellarmod.boszspectrumreader import BoszSpectrumReader

class TestBoszSpectrumReader(TestBase):
    def test_read(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/bosz/amm03cm03om03t3500g25v20modrt0b5000rs.asc')
        with open(filename) as f:
            r = BoszSpectrumReader(f)
            spec = r.read()

    def test_read_bz2(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/bosz/amm03cm03om03t3500g25v20modrt0b5000rs.asc.bz2')
        r = BoszSpectrumReader(filename)
        spec = r.read()

    def test_read_grid(self):
        path = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/bosz')
        r = BoszSpectrumReader(wave_lim=(3600, 12560))
        grid = r.read_grid(path)
        self.assertEqual((14, 7, 6, 6, 4, 12496), grid.flux.shape)

    def test_get_filename(self):
        self.fail()