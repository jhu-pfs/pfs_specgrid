from test.test_base import TestBase
import os

from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.stellarmod.phoenixspectrumreader import PhoenixSpectrumReader

class TestPhoenixSpectrumReader(TestBase):
    def test_read(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/phoenix/lte11600-2.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')

        r = PhoenixSpectrumReader(filename)
        spec = r.read()
        self.assertEqual(spec.wave.shape, spec.flux.shape)

        r = PhoenixSpectrumReader(filename, wave_lim=(4000, 6000))
        spec = r.read()
        self.assertEqual(spec.wave.shape, spec.flux.shape)

    def test_get_filename(self):
        fn = PhoenixSpectrumReader.get_filename(Fe_H=-2, T_eff=9600, log_g=4)
        self.assertEqual('lte09600-4.00-2.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', fn)

    def test_parse_filename(self):
        fn = 'lte09600-4.00-2.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        p = PhoenixSpectrumReader.parse_filename(fn)
        self.assertEqual(-2.0, p['Fe_H'])
        self.assertEqual(9600, p['T_eff'])
        self.assertEqual(4.0, p['log_g'])
