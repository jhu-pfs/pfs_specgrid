from test.test_base import TestBase
from unittest import TestCase
import numpy as np

from pfsspec.spectrum import Spectrum

class TestSpectrum(TestBase):
    def test_fnu_to_flam(self):
        self.fail()

    def test_flam_to_fnu(self):
        self.fail()

    def test_set_redshift(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(0.0, 7000, 1.45)
        spec.plot()

        spec.set_redshift(0.003)
        spec.plot()

        self.assertEqual(91.17269999999999, spec.wave[0])
        self.save_fig()

    def test_rebin(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(0.0, 7000, 1.45)
        spec.plot()

        nwave = np.arange(3800, 6500, 2.7) + 2.7 / 2
        spec.rebin(nwave)
        spec.plot()

        self.assertEqual((1000,), spec.wave.shape)
        self.assertEqual((1000,), spec.flux.shape)
        self.save_fig()

    def test_zero_mask(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(0.0, 7000, 1.45)
        spec.plot()

        spec.mask = np.zeros(spec.wave.shape)
        spec.mask[(spec.wave < 4500) | (spec.wave > 7500)] = 1
        spec.zero_mask()
        spec.plot()

        self.save_fig()

    def test_redden(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(0.0, 7000, 1.45)
        spec.plot()

        spec.redden(0.1)
        spec.plot()

        self.save_fig()

    def test_deredden(self):
        # This one doesn't make much sense with a model spectrum
        # pysynphot is tricked by using a negative extinction value
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(0.0, 7000, 1.45)
        spec.plot()

        spec.deredden(0.1)
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

    def test_running_mean(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(0.0, 7000, 1.45)
        spec.plot()

        rmean = Spectrum.running_filter(spec.wave, spec.flux, np.mean)
        spec.flux -= rmean
        spec.plot()
        spec.flux = rmean
        spec.plot(xlim=(2000, 12000))

        self.save_fig()

    def test_running_max(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(0.0, 7000, 1.45)
        spec.plot()

        rmean = Spectrum.running_filter(spec.wave, spec.flux, np.max)
        spec.flux -= rmean
        spec.plot()
        spec.flux = rmean
        spec.plot(xlim=(2000, 12000))

        self.save_fig()

    def test_running_median(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(0.0, 7000, 1.45)
        spec.plot()

        rmean = Spectrum.running_filter(spec.wave, spec.flux, np.median)
        spec.flux -= rmean
        spec.plot()
        spec.flux = rmean
        spec.plot(xlim=(2000, 12000))

        self.save_fig()

    def test_high_pass_filter(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(0.0, 7000, 1.45)
        spec.plot()

        spec.high_pass_filter()
        spec.plot(xlim=(2000, 12000))

        self.save_fig()