from test.testbase import TestBase
from unittest import TestCase
import numpy as np

from pfsspec.spectrum import Spectrum

class TestSpectrum(TestBase):
    def test_fnu_to_flam(self):
        self.fail()

    def test_flam_to_fnu(self):
        self.fail()

    def test_redshift(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(0.0, 7000, 1.45)
        spec.plot()

        spec.redshift(0.003)
        spec.plot()

        self.assertEqual(91.17269999999999, spec.wave[0])
        self.save_fig()

    def test_rebin(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(0.0, 7000, 1.45)
        spec.plot()

        nwave = np.arange(3800, 6500, 2.7) + 2.7 / 2
        nspec = spec.rebin(nwave)
        spec.plot()

        self.assertEqual((1000,), nspec.wave.shape)
        self.assertEqual((1000,), nspec.flux.shape)
        self.save_fig()

    def test_redden(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(0.0, 7000, 1.45)
        spec.plot()

        spec = spec.redden(1)
        spec.plot()

        self.save_fig()

    def test_synthflux(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(-0.5, 9600, 4.1)
        filter = self.get_hsc_filter('r')

        flux = spec.synthflux(filter)

        self.assertEqual(161746063.0325128, flux)

    def test_synthmag(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(-0.5, 9600, 4.1)
        filter = self.get_hsc_filter('r')

        flux = spec.synthmag(filter)

        self.assertEqual(-11.622084296395686, flux)