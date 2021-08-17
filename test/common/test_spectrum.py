from test.test_base import TestBase
import numpy as np

from pfsspec.common.spectrum import Spectrum

class TestSpectrum(TestBase):
    def test_set_redshift(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(Fe_H=0.0, T_eff=7000, log_g=1.45)
        spec.plot()

        spec.set_redshift(0.003)
        spec.plot()

        self.assertEqual(91.17269999999999, spec.wave[0])
        self.save_fig()

    def test_rebin(self):
        grid = self.get_bosz_grid()
        spec = grid.get_nearest_model(Fe_H=0.0, T_eff=7000, log_g=1.45, C_M=0, O_M=0)
        spec.plot()

        nwave = np.arange(3800, 6500, 2.7) + 2.7 / 2
        spec.rebin(nwave)
        spec.plot()

        self.assertEqual((1000,), spec.wave.shape)
        self.assertEqual((1000,), spec.flux.shape)
        self.save_fig()

    def test_rebin_with_mask(self):
        grid = self.get_bosz_grid()
        spec = grid.get_nearest_model(Fe_H=0.0, T_eff=7000, log_g=1.45, C_M=0, O_M=0)
        spec.mask = np.arange(spec.wave.shape[0], dtype=np.int64)       # fake mask
        spec.plot()

        nwave = np.arange(3800, 6500, 2.7) + 2.7 / 2
        spec.rebin(nwave)
        spec.plot()

        self.assertEqual((1000,), spec.wave.shape)
        self.assertEqual((1000,), spec.flux.shape)
        self.assertEqual((1000,), spec.mask.shape)
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

    def test_normalize_at(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(0.0, 7000, 1.45)
        spec.multiply(1e-7)
        spec.plot()

        spec.normalize_at(5513.7)
        spec.plot()

        self.save_fig()

    def test_normalize_in_single(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(0.0, 7000, 1.45)
        spec.multiply(1e-7)
        spec.plot()

        spec.normalize_in((5513.7, 6124.2), np.mean)
        spec.plot()

        self.save_fig()

    def test_normalize_in_multiple(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(0.0, 7000, 1.45)
        spec.multiply(1e-7)
        spec.plot()

        spec.normalize_in(((5513.7, 6124.2), (6513.7, 7124.2)), np.mean)
        spec.plot()

        self.save_fig()

    def test_normalize_by_T_eff(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(Fe_H=0.0, T_eff=5000, log_g=2.45)
        spec.normalize_by_T_eff(5000)
        spec.plot()

        self.save_fig()

    def test_convolve_gaussian(self):
        grid = self.get_kurucz_grid()

        spec = grid.get_nearest_model(Fe_H=0.0, T_eff=5000, log_g=2.45)
        spec.plot()

        spec.convolve_gaussian(dlambda=15, wlim=[3000, 9000])
        spec.plot()

        spec.convolve_gaussian(vdisp=1000, wlim=[4000, 8000])
        spec.plot()

        self.save_fig()

    def test_convolve_gaussian_log(self):
        grid = self.get_kurucz_grid()

        spec = grid.get_nearest_model(Fe_H=0.0, T_eff=5000, log_g=2.45)
        spec.plot()

        spec.convolve_gaussian_log(dlambda=15, wlim=[5000, 9000])
        spec.plot()

        spec.convolve_gaussian_log(vdisp=2000, wlim=[5500, 8000])
        spec.plot()

        self.save_fig()

    def test_redden(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(Fe_H=0.0, T_eff=7000, log_g=1.45)
        spec.plot()

        spec.redden(0.1)
        spec.plot()

        self.save_fig()

    def test_deredden(self):
        # This one doesn't make much sense with a model spectrum
        # pysynphot is tricked by using a negative extinction value
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(Fe_H=0.0, T_eff=7000, log_g=1.45)
        spec.plot()

        spec.deredden(0.1)
        spec.plot()

        self.save_fig()

    def test_synthflux(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(Fe_H=-0.5, T_eff=9600, log_g=4.1)
        filter = self.get_hsc_filter('r')

        flux = spec.synthflux(filter)

        #self.assertEqual(161746063.0325128, flux)
        self.assertEqual(5.0050577532662006e+19, flux)

    def test_synthmag(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(Fe_H=-0.5, T_eff=9600, log_g=4.1)
        filter = self.get_hsc_filter('r')

        flux = spec.synthmag(filter)

        #self.assertEqual(-11.622084296395686, flux)
        self.assertEqual(-40.34852273290004, flux)

    def test_running_mean(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(Fe_H=0.0, T_eff=7000, log_g=1.45)
        spec.plot()

        rmean = Spectrum.running_filter(spec.wave, spec.flux, np.mean)
        spec.flux -= rmean
        spec.plot()
        spec.flux = rmean
        spec.plot(xlim=(2000, 12000))

        self.save_fig()

    def test_running_max(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(Fe_H=0.0, T_eff=7000, log_g=1.45)
        spec.plot()

        rmean = Spectrum.running_filter(spec.wave, spec.flux, np.max)
        spec.flux -= rmean
        spec.plot()
        spec.flux = rmean
        spec.plot(xlim=(2000, 12000))

        self.save_fig()

    def test_running_median(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(Fe_H=0.0, T_eff=7000, log_g=1.45)
        spec.plot()

        rmean = Spectrum.running_filter(spec.wave, spec.flux, np.median)
        spec.flux -= rmean
        spec.plot()
        spec.flux = rmean
        spec.plot(xlim=(2000, 12000))

        self.save_fig()

    def test_high_pass_filter(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(Fe_H=0.0, T_eff=7000, log_g=1.45)
        spec.plot()

        spec.high_pass_filter()
        spec.plot(xlim=(2000, 12000))

        self.save_fig()

    def test_high_pass_filter_dlambda_1(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(Fe_H=0.0, T_eff=7000, log_g=1.45)
        spec.plot()

        spec.high_pass_filter(dlambda=200)
        spec.plot(xlim=(2000, 12000))

        self.save_fig()

    def test_high_pass_filter_dlambda_2(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(Fe_H=0.0, T_eff=7000, log_g=1.45)
        spec.plot()

        spec.high_pass_filter(dlambda=(200, 250))
        spec.plot(xlim=(2000, 12000))

        self.save_fig()

    def test_fit_envelope_chebyshev(self):
        grid = self.get_kurucz_grid()
        spec = grid.get_nearest_model(Fe_H=0.0, T_eff=7000, log_g=1.45)

        p, c = spec.fit_envelope_chebyshev(wlim=(6300, 9700), order=10)
        spec.flux /= c
        spec.plot(xlim=(2000, 12000), ylim=(0, 1.1))

        self.save_fig()

    def test_add_calib_bias(self):
        grid = self.get_bosz_grid()
        spec = grid.get_nearest_model(Fe_H=0.0, T_eff=7000, log_g=1.45, C_M=0, O_M=0)
        spec.rebin(np.linspace(6300, 9700, 1200))
        spec.plot(xlim=(5000, 10000))

        spec.add_calib_bias()
        spec.plot()

        self.save_fig()