from test.test_base import TestBase
import os

from pfsspec.stellarmod.boszmodelgrid import BoszModelGrid
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
        grid = BoszModelGrid()
        r = BoszSpectrumReader(grid, wave_lim=(3600, 12560))
        r.read_grid(path)
        self.assertEqual((14, 7, 6, 6, 4, 12496), grid.flux.shape)

    def test_get_filename(self):
        self.fail()

    def test_parse_filename(self):
        fn = 'amm03cm03om03t3500g25v20modrt0b5000rs.asc.bz2'
        p = BoszSpectrumReader.parse_filename(fn)
        self.assertEqual(-0.25, p['Fe_H'])
        self.assertEqual(-0.25, p['C_M'])
        self.assertEqual(-0.25, p['O_M'])
        self.assertEqual(3500, p['T_eff'])
        self.assertEqual(2.5, p['log_g'])

    def test_enum_axes(self):
        grid = BoszModelGrid()
        r = BoszSpectrumReader(grid)

        g = BoszSpectrumReader.EnumAxesGenerator(grid)
        k = 0
        for i in g:
            k += 1

        s = 1
        for p in grid.axes:
            s *= grid.axes[p].values.shape[0]

        self.assertEqual(s, k)