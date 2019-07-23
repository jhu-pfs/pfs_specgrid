import numpy as np
from scipy.interpolate import interp1d
import collections
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
        if isinstance(orig, Spectrum):
            self.redshift = orig.redshift
            self.redshift_err = orig.redshift_err
            self.snr = orig.snr
            self.mag = orig.mag
            self.wave = np.copy(orig.wave)
            self.flux = np.copy(orig.flux)
            self.flux_err = np.copy(orig.flux_err)
            self.flux_sky = np.copy(orig.flux_sky)
            self.mask = np.copy(orig.mask)
        else:
            self.redshift = 0.0
            self.redshift_err = 0.0
            self.snr = 0
            self.mag = 0
            self.wave = None
            self.flux = None
            self.flux_err = None
            self.flux_sky = None
            self.mask = None

    def copy(self, orig):
        return Spectrum(orig=orig)

    def get_param_names(self):
        return ['redshift',
                'redshift_err',
                'snr',
                'mag']

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
        # mask should be combined if flux comes from multiple bins

        filt = pysynphot.spectrum.ArraySpectralElement(self.wave, np.ones(len(self.wave)), waveunits='angstrom')

        if self.flux is not None:
            spec = pysynphot.spectrum.ArraySourceSpectrum(wave=self.wave, flux=self.flux, keepneg=True)
            obs = pysynphot.observation.Observation(spec, filt, binset=nwave, force='taper')
            self.flux = obs.binflux

        if self.flux_sky is not None:
            spec = pysynphot.spectrum.ArraySourceSpectrum(wave=self.wave, flux=self.flux_sky, keepneg=True)
            obs = pysynphot.observation.Observation(spec, filt, binset=nwave, force='taper')
            self.flux_sky = obs.binflux

        # For the error vector, use nearest-neighbor interpolations
        # later we can figure out how to do this correctly and add correlated noise, etc.
        if self.flux_err is not None:
            ip = interp1d(self.wave, self.flux_err, kind='nearest')
            self.flux_err = ip(nwave)

        self.wave = nwave
        self.mask = None

    def zero_mask(self):
        self.flux[self.mask != 0] = 0
        if self.flux_err is not None:
            self.flux_err[self.mask != 0] = 0
        if self.flux_sky is not None:
            self.flux_sky[self.mask != 0] = 0

    def multiply(self, a, silent=True):
        if np.isfinite(a):
            self.flux = self.flux * a
            if self.flux_err is not None:
                self.flux_err = self.flux_err * a
            if self.flux_sky is not None:
                self.flux_sky = self.flux_sky * a
        elif not silent:
            raise Exception('Cannot multiply by NaN of Inf')

    def normalize_at(self, lam, value=1.0):
        idx = np.digitize(lam, self.wave)
        flux = self.flux[idx]
        self.multiply(value / flux)

    def normalize_in(self, lam, func=np.median, value=1.0):
        idx = np.digitize(lam, self.wave)
        fl = self.flux[idx[0]:idx[1]]
        if fl.shape[0] < 2:
            raise Exception('Cannot get wavelength interval')
        fl = func(fl)
        self.multiply(value / fl)

    def normalize_to_mag(self, filt, mag):
        m = self.synthmag(filt)
        DM = mag - m
        D = 10 ** (DM / 5)

        if self.flux is not None:
            self.flux = self.flux / D**2
        if self.flux_err is not None:
            self.flux_err = self.flux_err / D**2

        self.mag = mag

    def add_noise(self, noise):
        # Assume noise is rebinned to the spectrum
        err = np.random.normal(size=self.flux.shape) * noise.noise
        self.snr = np.max(self.flux) / np.std(err)
        self.flux_err = noise.noise
        self.flux = self.flux + err

    def redden(self, extval):
        spec = pysynphot.spectrum.ArraySourceSpectrum(wave=self.wave, flux=self.flux)
        obs = spec * pysynphot.reddening.Extinction(extval, 'mwavg')
        self.flux = obs.flux

    def deredden(self, extval):
        self.redden(-extval)

    def synthflux(self, filter):
        # Calculate error from error array?
        spec = pysynphot.spectrum.ArraySourceSpectrum(wave=self.wave, flux=self.flux, fluxunits='flam')
        filt = pysynphot.spectrum.ArraySpectralElement(filter.wave, filter.thru, waveunits='angstrom')
        filt.binset = spec.wave     # supress warning from pysynphot
        obs = pysynphot.observation.Observation(spec, filt)
        return obs.effstim('Jy')

    def synthmag(self, filter):
        flux = self.synthflux(filter)
        return -2.5 * np.log10(flux) + 8.90

    def running_filter(wave, data, func, dlambda=None, vdisp=None):
        # Don't care much about edges here, they'll be trimmed when rebinning
        if dlambda is not None and vdisp is not None:
            raise Exception('Only one of dlambda and vdisp can be specified')
        elif dlambda is None and vdisp is None:
            vdisp = Constants.DEFAULT_FILTER_VDISP

        if vdisp is not None:
            z = vdisp / Constants.SPEED_OF_LIGHT

        ndata = np.empty(data.shape)
        for i in range(len(data)):
            if isinstance(dlambda, collections.Iterable):
                mask = (wave[i] - dlambda[0] <= wave) & (wave < wave[i] + dlambda[1])
            elif dlambda is not None:
                mask = (wave[i] - dlambda <= wave) & (wave < wave[i] + dlambda)
            else:
                mask = ((1 - z) * wave[i] < wave) & (wave < (1 + z) * wave[i])
            if mask.size < 2:
                ndata[i] = data[i]
            else:
                ndata[i] = func(data[mask])

        return ndata

    def high_pass_filter(self, func=np.median, dlambda=None, vdisp=None):
        # TODO: error array?
        self.flux -= Spectrum.running_filter(self.wave, self.flux, func, dlambda=dlambda, vdisp=vdisp)
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