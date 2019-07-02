import numpy as np
import matplotlib.pyplot as plt
import pysynphot
import pysynphot.binning
import pysynphot.spectrum
import pysynphot.reddening

from pfsspec.constants import Constants
from pfsspec.pfsobject import PfsObject

class Spectrum(PfsObject):
    def __init__(self, orig=None):
        super(Spectrum, self).__init__(orig=orig)
        if orig is None:
            self.redshift = 0.0
            self.redshift_err = 0.0
            self.snr = 1.0
            self.wave = None
            self.flux = None
            self.flux_err = None
            self.flux_sky = None
            self.mask = None
        else:
            self.redshift = orig.redshift
            self.redshift_err = orig.redshift_err
            self.snr = orig.snr
            self.wave = np.copy(orig.wave)
            self.flux = np.copy(orig.flux)
            self.flux_err = np.copy(orig.flux_err)
            self.flux_sky = np.copy(orig.flux_sky)
            self.mask = np.copy(orig.mask)

    def fnu_to_flam(self):
        # TODO: convert wave ?
        # ergs/cm**2/s/hz/ster to erg/s/cm^2/A surface flux
        self.flux /= 3.336e-19 * (self.wave) ** 2 / 4 / np.pi

    def flam_to_fnu(self):
        # TODO: convert wave ?
        self.flux *= 3.336e-19 * (self.wave) ** 2 / 4 / np.pi

    def set_redshift(self, z):
        self.redshift += z
        self.wave *= 1 + z

    def rebin(self, nwave):
        # TODO: how to rebin error and mask?

        filt = pysynphot.spectrum.ArraySpectralElement(self.wave, np.ones(len(self.wave)), waveunits='angstrom')

        spec = pysynphot.spectrum.ArraySourceSpectrum(wave=self.wave, flux=self.flux, keepneg=True)
        obs = pysynphot.observation.Observation(spec, filt, binset=nwave, force='taper')
        self.flux = obs.binflux

        if self.flux_sky is not None:
            spec = pysynphot.spectrum.ArraySourceSpectrum(wave=self.wave, flux=self.flux_sky, keepneg=True)
            obs = pysynphot.observation.Observation(spec, filt, binset=nwave, force='taper')
            self.flux_sky = obs.binflux

        self.wave = nwave
        self.flux_err = None
        self.mask = None

    def zero_mask(self):
        self.flux[self.mask != 0] = 0
        if self.flux_err is not None:
            self.flux_err[self.mask != 0] = 0
        if self.flux_sky is not None:
            self.flux_sky[self.mask != 0] = 0

    def redden(self, extval):
        spec = pysynphot.spectrum.ArraySourceSpectrum(wave=self.wave, flux=self.flux)
        obs = spec * pysynphot.reddening.Extinction(extval, 'mwavg')
        self.flux = obs.flux

    def deredden(self, extval):
        self.redden(-extval)

    def synthflux(self, filter):
        # Calculate error from error array?
        spec = pysynphot.spectrum.ArraySourceSpectrum(wave=self.wave, flux=self.flux)
        filt = pysynphot.spectrum.ArraySpectralElement(filter.wave, filter.thru, waveunits='angstrom')
        obs = pysynphot.observation.Observation(spec, filt)
        return obs.effstim('Jy')

    def synthmag(self, filter):
        flux = self.synthflux(filter)
        return -2.5 * np.log10(flux) + 8.90

    def running_filter(wave, data, func, vdisp=Constants.DEFAULT_FILTER_VDISP):
        # Don't care much about edges here, they'll be trimmed when rebinning
        z = vdisp / Constants.SPEED_OF_LIGHT
        ndata = np.empty(data.shape)
        for i in range(len(data)):
            mask = ((1 - z) * wave[i] < wave) & (wave < (1 + z) * wave[i])
            ndata[i] = func(data[mask])
        return ndata

    def high_pass_filter(self, func=np.median, vdisp=Constants.DEFAULT_FILTER_VDISP):
        self.flux -= Spectrum.running_filter(self.wave, self.flux, func, vdisp)
        if self.flux_sky is not None:
            self.flux_sky -= Spectrum.running_filter(self.wave, self.flux_sky, func, vdisp)

    def load(self, filename):
        raise NotImplementedError()

    def plot(self, ax=None, xlim=Constants.DEFAULT_PLOT_WAVE_RANGE, ylim=None, labels=True):
        ax = self.plot_getax(ax, xlim, ylim)
        ax.plot(self.wave, self.flux)
        if labels:
            ax.set_xlabel(r'$\lambda$ [A]')
            ax.set_ylabel(r'$F_\lambda$ [erg s-1 cm-2 A-1]')

        return ax