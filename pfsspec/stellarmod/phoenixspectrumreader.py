import os
import numpy as np
from numpy.core.memmap import memmap
import pandas as pd
import re
from scipy.interpolate import interp1d
from astropy.io import fits

from pfsspec.data.spectrumreader import SpectrumReader
from pfsspec.stellarmod.phoenixspectrum import PhoenixSpectrum
from pfsspec.stellarmod.continuummodels.chebyshev import Chebyshev

#lte12000-6.00+1.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
class PhoenixSpectrumReader(SpectrumReader):
    INPUT_RESOLUTION = 500000
    WL_START_OPTICAL = 3000
    WL_END_OPTICAL = 25000

    def __init__(self, path=None, wave_lim=None, resolution=None):
        super(PhoenixSpectrumReader, self).__init__()

        self.path = path
        self.wave_lim = wave_lim
        self.resolution = resolution
        
        self.wave = None
        self.wave_mask = None
        self.wave_log = None
        self.wave_log_mask = None
        self.kernel = None
        

    def read(self, file=None):
        if file is None:
            file = self.path

        # Read flux
        with fits.open(file, memmap=False) as f:
            flux = f[0].data    

        # Wavelengths are read from a separate file which should be next to
        # the current one, in the same directory
        if self.wave is None:
            dir, _ = os.path.split(file)
            fn = os.path.join(dir, 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
            with fits.open(fn, memmap=False) as f:
                self.wave = f[0].data

        if self.wave_lim is None:
            self.wave_lim = (self.wave[0], self.wave[-1])

        # If target resolution is specified, spectrum needs to be convolved down
        # first. Phoenix uses a linear wavelength grid, so first resample it to
        # a log grid and then do the convolution.
        if self.resolution is not None:
            if self.wave_log is None:
                # TODO: move this to Spectrum class

                ratio = 1 + 1 / PhoenixSpectrumReader.INPUT_RESOLUTION

                # Allow and additional 10 angstroms at each end for convolution
                wlstart = max(self.wave[0], self.wave_lim[0] - 10)
                wlend = min(self.wave[-1], self.wave_lim[-1] + 10)

                ixwlstart = np.log10(wlstart) / np.log10(ratio)
                ixwlend = np.log10(wlend) / np.log10(ratio)
                length = int(ixwlend - ixwlstart + 1)

                self.wave_log = 10 ** np.linspace(np.log10(wlstart), np.log10(wlend), length)
                
                # Fix possible round-off errors at the ends
                self.wave_log[0] = max(self.wave_log[0], self.wave[0])
                self.wave_log[-1] = min(self.wave_log[-1], self.wave[-1])

                # Trim the logarithmic grid and reduce number of samples
                self.wave_log_mask = (self.wave_lim[0] <= self.wave_log) & (self.wave_log <= self.wave_lim[1])
                
                step = int(PhoenixSpectrumReader.INPUT_RESOLUTION / self.resolution / 2)
                m = np.full(self.wave_log_mask.shape, False)
                m[::step] = True
                self.wave_log_mask = (m & self.wave_log_mask)

            # Construct convolution kernel
            if self.kernel is None:
                wave_ref = 5000

                sigma_input = wave_ref / PhoenixSpectrumReader.INPUT_RESOLUTION
                sigma_output = wave_ref / self.resolution
                sigma_kernel = np.sqrt(sigma_output**2 - sigma_input**2)

                wave_ref_ix = np.digitize([wave_ref - 3.5 * sigma_kernel, wave_ref + 3.5 * sigma_kernel], self.wave_log)
                if (wave_ref_ix[1] - wave_ref_ix[0]) % 2 == 0:
                    wave_ref_ix[1] += 1
                wave_ref = self.wave_log[wave_ref_ix[0]:wave_ref_ix[1]]

                def gauss_kernel(dwave, sigma):
                    return np.exp(-dwave**2 / (2 * sigma**2))

                kernel = gauss_kernel(wave_ref - wave_ref[wave_ref.size // 2], sigma_kernel)
                kernel /= np.sum(kernel)

            # Interpolate flux to log wave grid then convolve with kernel
            flux = interp1d(self.wave, flux)(self.wave_log)
            flux = np.convolve(flux, kernel, mode='same')

            spec = PhoenixSpectrum()
            spec.wave = self.wave_log[self.wave_log_mask]
            spec.flux = flux[self.wave_log_mask]
        else:
            if self.wave_lim is not None:
                filt = (self.wave_lim[0] <= self.wave) & (self.wave <= self.wave_lim[1])
            else:
                filt = slice(None)

            spec = PhoenixSpectrum()
            spec.wave = self.wave[filt]
            spec.flux = flux[filt]

        # TODO: continuum?
    
        return spec

    @staticmethod
    def get_filename(**kwargs):
        # lte12000-6.00+1.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits

        Fe_H = kwargs.pop('Fe_H')
        T_eff = kwargs.pop('T_eff')
        log_g = kwargs.pop('log_g')

        fn = 'lte'
        fn += "{:05d}".format(int(T_eff))
        
        fn += '-'
        fn += '{:.02f}'.format(float(log_g))

        fn += '-' if Fe_H <= 0.0 else '+'
        fn += '{:.01f}'.format(np.abs(float(Fe_H)))

        fn += '.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'

        return fn

    @staticmethod
    def parse_filename(filename):

        #lte12000-6.00+1.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits

        p = r'lte(\d{5})([+-])(\d{1}).(\d{2})([+-])(\d{1}).(\d{1})'
        m = re.search(p, filename)

        return{
            'T_eff': float(m.group(1)),
            'log_g': float(m.group(3) + m.group(4)) / 100,
            'Fe_H': float(m.group(5) + m.group(6) + m.group(7)) / 10
        }

