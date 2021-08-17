import logging
import numpy as np
import numbers
from scipy.interpolate import interp1d
import collections
import matplotlib.pyplot as plt
import pysynphot
import pysynphot.binning
import pysynphot.spectrum
import pysynphot.reddening

from pfsspec.util.physics import Physics
from pfsspec.constants import Constants
from pfsspec.common.pfsobject import PfsObject
from pfsspec.obsmod.psf import Psf
from pfsspec.obsmod.pcapsf import PcaPsf
from pfsspec.obsmod.gausspsf import GaussPsf

class Spectrum(PfsObject):
    def __init__(self, orig=None):
        super(Spectrum, self).__init__(orig=orig)
        
        if isinstance(orig, Spectrum):
            self.index = orig.index
            self.id = orig.id
            self.redshift = orig.redshift
            self.redshift_err = orig.redshift_err
            self.exp_count = orig.exp_count
            self.exp_time = orig.exp_time
            self.extinction = orig.extinction
            self.target_zenith_angle = orig.target_zenith_angle
            self.target_field_angle = orig.target_field_angle
            self.moon_zenith_angle = orig.moon_zenith_angle
            self.moon_target_angle = orig.moon_target_angle
            self.moon_phase = orig.moon_phase
            self.snr = orig.snr
            self.mag = orig.mag
            self.wave = np.copy(orig.wave)
            self.flux = np.copy(orig.flux)
            self.flux_err = np.copy(orig.flux_err)
            self.flux_sky = np.copy(orig.flux_sky)
            self.mask = np.copy(orig.mask)
            self.cont = np.copy(orig.cont)
            self.cont_fit = orig.cont_fit
            self.random_seed = orig.random_seed
        else:
            self.index = None
            self.id = 0
            self.redshift = 0.0
            self.redshift_err = 0.0
            self.exp_count = 1
            self.exp_time = 450
            self.extinction = 0.0
            self.target_zenith_angle = None
            self.target_field_angle = None
            self.moon_zenith_angle = None
            self.moon_target_angle = None
            self.moon_phase = None
            self.snr = 0
            self.mag = 0
            self.wave = None
            self.flux = None
            self.flux_err = None
            self.flux_sky = None
            self.mask = None
            self.cont = None
            self.cont_fit = None
            self.random_seed = None

    def get_param_names(self):
        return ['id',
                'redshift',
                'redshift_err',
                'exp_count',
                'exp_time',
                'extinction',
                'target_zenith_angle',
                'target_field_angle',
                'moon_zenith_angle',
                'moon_target_angle',
                'moon_phase',
                'snr',
                'mag',
                'cont_fit',
                'random_seed']

    def get_params(self):
        params = {}
        for p in self.get_param_names():
            params[p] = getattr(self, p)
        return params

    def set_params(self, params):
        for p in self.get_param_names():
            if p in params:
                setattr(self, p, params[p])

    def get_params_as_datarow(self):
        row = {}
        for p in self.get_param_names():
            v = getattr(self, p)

            # If parameter is an array, assume it's equal length and copy to
            # pandas as a set of columns instead of a single column
            if v is None:
                row[p] = np.nan
            elif isinstance(v, np.ndarray):
                if len(v.shape) > 1:
                    raise Exception('Can only serialize one-dimensional arrays')
                for i in range(v.shape[0]):
                    row['{}_{}'.format(p, i)] = v[i]
            else:
                row[p] = v

        return row

    def set_redshift(self, z):
        # Assume zero redshift at start
        self.redshift = z
        self.wave *= 1 + z

    @staticmethod
    def rebin_vector(wave, nwave, data):
        if data is None:
            return None
        else:
            filt = pysynphot.spectrum.ArraySpectralElement(wave, np.ones(len(wave)), waveunits='angstrom')
            spec = pysynphot.spectrum.ArraySourceSpectrum(wave=wave, flux=data, keepneg=True)
            obs = pysynphot.observation.Observation(spec, filt, binset=nwave, force='taper')
            return obs.binflux

    def rebin(self, wave_centers, wave_edges=None):
        # TODO: how to rebin error and mask?
        #       mask should be combined if flux comes from multiple bins
        # TODO: does pysynphot support bin edges

        self.flux = Spectrum.rebin_vector(self.wave, wave_centers, self.flux)
        self.flux_sky = Spectrum.rebin_vector(self.wave, wave_centers, self.flux_err)

        # For the error vector, use nearest-neighbor interpolations
        # later we can figure out how to do this correctly and add correlated noise, etc.
        if self.flux_err is not None:
            ip = interp1d(self.wave, self.flux_err, kind='nearest')
            self.flux_err = ip(wave_centers)

        if self.cont is not None:
            self.cont = Spectrum.rebin_vector(self.wave, wave_centers, self.cont)

        # TODO: we only take the closest bin here which is incorrect
        #       mask values should combined with bitwise or across bins
        if self.mask is not None:
            wl_idx = np.digitize(wave_centers, self.wave)
            mask = self.mask[wl_idx]
        else:
            mask = None

        self.wave = wave_centers
        self.mask = mask

    def zero_mask(self):
        self.flux[self.mask != 0] = 0
        if self.flux_err is not None:
            self.flux_err[self.mask != 0] = 0
        if self.flux_sky is not None:
            self.flux_sky[self.mask != 0] = 0

    def multiply(self, a, silent=True):
        if np.all(np.isfinite(a)):
            self.flux = a * self.flux
            if self.flux_err is not None:
                self.flux_err = a * self.flux_err
            if self.flux_sky is not None:
                self.flux_sky = a * self.flux_sky
            if self.cont is not None:
                self.cont = a * self.cont
        elif not silent:
            raise Exception('Cannot multiply by NaN of Inf')

    def normalize_at(self, lam, value=1.0):
        idx = np.digitize(lam, self.wave)
        flux = self.flux[idx]
        self.multiply(value / flux)

    def normalize_in(self, lam, func=np.median, value=1.0):
        if type(lam[0]) is float:
            # single range
            idx = np.digitize(lam, self.wave)
            fl = self.flux[idx[0]:idx[1]]
        else:
            # list of ranges
            fl = []
            for rlam in lam:
                idx = np.digitize(rlam, self.wave)
                fl += list(self.flux[idx[0]:idx[1]])
            fl = np.array(fl)
        if fl.shape[0] < 2:
            raise Exception('Cannot get wavelength interval')
        fl = func(fl)
        self.multiply(value / fl)

    def normalize_to_mag(self, filt, mag):
        try:
            m = self.synthmag(filt)
        except Exception as ex:
            print('flux max', np.max(self.flux))
            print('mag', mag)
            raise ex
        DM = mag - m
        D = 10 ** (DM / 5)

        self.multiply(1 / D**2)
        self.mag = mag

    def normalize_by_continuum(self):
        self.multiply(1.0 / self.cont)

    @staticmethod
    def vdisp_to_z(vdisp):
        return vdisp * 1e3 / Physics.c

    @staticmethod
    def get_dispersion(dlambda=None, vdisp=None):
        if dlambda is not None and vdisp is not None:
            raise Exception('Only one of dlambda and vdisp can be specified')
        if dlambda is not None:
            if isinstance(dlambda, collections.Iterable):
                return dlambda, None
            else:
                return [dlambda, dlambda], None
        elif vdisp is not None:
            z = Spectrum.vdisp_to_z(vdisp)
            return None, z
        else:
            z = Spectrum.vdisp_to_z(Constants.DEFAULT_FILTER_VDISP)
            return None, z

    @staticmethod
    def convolve_vector(wave, data, kernel_func, dlambda=None, vdisp=None, wlim=None):
        """
        Convolve a data vector (flux) with a wavelength-dependent kernel provided as a callable.
        This is very generic, assumes no regular binning and constant-width kernel but bins should
        be similarly sized.
        :param data:
        :param kernel:
        :param dlam:
        :param wlim:
        :return:
        """

        # TODO: if bins are not similarly sized we need to compute bin widths and take that into
        # account

        # Start and end of convolution
        if wlim is not None:
            idx_lim = np.digitize(wlim, wave)
        else:
            idx_lim = [0, wave.shape[0] - 1]

        # Dispersion determines the width of the kernel to be taken into account
        dlambda, z = Spectrum.get_dispersion(dlambda, vdisp)

        # Construct results and fill-in parts that are not part of the original
        if not isinstance(data, tuple) and not isinstance(data, list):
            data = [data, ]
        res = [ np.zeros(d.shape) for d in data ]
        for d, r in zip(data, res):
            r[:idx_lim[0]] = d[:idx_lim[0]]
            r[idx_lim[1]:] = d[idx_lim[1]:]

        # Compute convolution
        for i in range(idx_lim[0], idx_lim[1]):
            if dlambda is not None:
                idx = np.digitize([wave[i] - dlambda[0], wave[i] + dlambda[0]], wave)
            else:
                idx = np.digitize([(1 - z) * wave[i], (1 + z) * wave[i]], wave)

            # Evaluate kernel
            k = kernel_func(wave[i], wave[idx[0]:idx[1]])

            # Sum up
            for d, r in zip(data, res):
                r[i] += np.sum(k * d[idx[0]:idx[1]])

        return res

    def convolve_gaussian(self, dlambda=None, vdisp=None, wlim=None):
        """
        Convolve with (a potentially wavelength dependent) Gaussian kernel)
        :param dlam: dispersion in units of Angstrom
        :param wave: wavelength in Ansgstroms
        :param wlim: limit convolution between wavelengths
        :return:
        """

        data = [self.flux, ]
        if self.cont is not None:
            data.append(self.cont)

        if dlambda is not None:
            def kernel_func(w, wave):
                # k = 1.0 / np.sqrt(2 * np.pi) / s2 ...
                k = 0.3989422804014327 / dlambda * np.exp(-(wave - w)**2 / (2 * dlambda**2))
                n = np.sum(k)
                return k / n
            res = Spectrum.convolve_vector(self.wave, data, kernel_func, dlambda=[4 * dlambda, 4 * dlambda], wlim=wlim)
        elif vdisp is not None:
            def kernel_func(w, wave):
                # k = 1.0 / np.sqrt(2 * np.pi) / s2 ...
                z = Spectrum.vdisp_to_z(vdisp)
                k = 0.3989422804014327 / (z * w) * np.exp(-(wave - w) ** 2 / (2 * (z * w)**2))
                n = np.sum(k)
                return k / n
            res = Spectrum.convolve_vector(self.wave, data, kernel_func, dlambda=[100, 100], wlim=wlim)
        else:
            raise Exception('dlambda or vdisp must be specified')

        self.flux = res[0]
        if self.cont is not None:
            self.cont = res[1]

    @staticmethod
    def convolve_vector_log(wave, data, kernel, wlim=None):
        # Start and end of convolution
        if wlim is not None:
            idx = np.digitize(wlim, wave)
            idx_lim = [max(idx[0] - kernel.shape[0] // 2, 0), min(idx[1] + kernel.shape[0] // 2, wave.shape[0] - 1)]
        else:
            idx_lim = [0, wave.shape[0] - 1]

        # Construct results
        if not isinstance(data, tuple) and not isinstance(data, list):
            data = [data, ]
        res = [np.empty(d.shape) for d in data]

        for d, r in zip(data, res):
            r[idx_lim[0]:idx_lim[1]] = np.convolve(d[idx_lim[0]:idx_lim[1]], kernel, mode='same')

        # Fill-in parts that are not convolved to save a bit of computation
        for d, r in zip(data, res):
            r[:idx[0]] = d[:idx[0]]
            r[idx[1]:] = d[idx[1]:]

        return res

    def convolve_gaussian_log(self, lambda_ref=5000,  dlambda=None, vdisp=None, wlim=None):
        """
        Convolve with a Gaussian filter, assume logarithmic binning in wavelength
        so the same kernel can be used across the whole spectrum
        :param dlambda:
        :param vdisp:
        :param wlim:
        :return:
        """

        data = [self.flux, ]
        if self.cont is not None:
            data.append(self.cont)

        if dlambda is not None:
            pass
        elif vdisp is not None:
            dlambda = lambda_ref * vdisp / Physics.c * 1000 # km/s
        else:
            raise Exception('dlambda or vdisp must be specified')

        # Make sure kernel size is always an odd number, starting from 3
        idx = np.digitize(lambda_ref, self.wave)
        i = 1
        while lambda_ref - 4 * dlambda < self.wave[idx - i] or \
              self.wave[idx + i] < lambda_ref + 4 * dlambda:
            i += 1
        idx = [idx - i, idx + i]

        wave = self.wave[idx[0]:idx[1]]
        k = 0.3989422804014327 / dlambda * np.exp(-(wave - lambda_ref) ** 2 / (2 * dlambda ** 2))
        k = k / np.sum(k)

        res = Spectrum.convolve_vector_log(self.wave, data, k, wlim=wlim)

        self.flux = res[0]
        if self.cont is not None:
            self.cont = res[1]

    @staticmethod
    def convolve_vector_varying_kernel(wave, data, kernel_func, size=None, wlim=None):
        """
        Convolve with a kernel that varies with wavelength. Kernel is computed
        by a function passed as a parameter.
        """

        # Get a test kernel from the middle of the wavelengt range to have its size
        kernel = kernel_func(wave[wave.shape[0] // 2], size=size)

        # Start and end of convolution
        # Outside this range, original values will be used
        if wlim is not None:
            idx = np.digitize(wlim, wave)
        else:
            idx = [0, wave.shape[0]]

        idx_lim = [
            max(idx[0], kernel.shape[0] // 2),
            min(idx[1], wave.shape[0] - kernel.shape[0] // 2)
        ]

        # Construct results
        if not isinstance(data, tuple) and not isinstance(data, list):
            data = [data, ]
        res = [np.zeros(d.shape) for d in data]

        # Do the convolution
        offset = 0 - kernel.shape[0] // 2
        for i in range(idx_lim[0], idx_lim[1]):
            kernel = kernel_func(wave[i], size=size)
            s = slice(i + offset, i + offset + kernel.shape[0])
            for d, r in zip(data, res):
                z = d[i] * kernel
                r[s] += z

        # Fill-in parts that are not convolved
        for d, r in zip(data, res):
            r[:idx_lim[0] - offset] = d[:idx_lim[0] - offset]
            r[idx_lim[1] + offset:] = d[idx_lim[1] + offset:]

        return res

    def convolve_varying_kernel(self, kernel_func, size=None, wlim=None):
        # NOTE: this function does not take model resolution into account!

        data = [self.flux, ]
        if self.cont is not None:
            data.append(self.cont)

        res = Spectrum.convolve_vector_varying_kernel(self.wave, data, kernel_func, size=size, wlim=wlim)

        self.flux = res[0]
        if self.cont is not None:
            self.cont = res[1]

    def convolve_psf(self, psf, model_res, wlim):
        # Note, that this is in addition to model spectrum resolution
        if isinstance(psf, Psf):
            self.convolve_varying_kernel(psf.get_kernel)
        elif isinstance(psf, numbers.Number) and model_res is not None:
            sigma = psf

            if model_res is not None:
                # Resolution at the middle of the detector
                fwhm = 0.5 * (wlim[0] + wlim[-1]) / model_res
                ss = fwhm / 2 / np.sqrt(2 * np.log(2))
                sigma = np.sqrt(sigma ** 2 - ss ** 2)
            self.convolve_gaussian(dlambda=sigma, wlim=wlim)
        elif isinstance(psf, numbers.Number):
            # This is a simplified version when the model resolution is not known
            # TODO: review this
            self.convolve_gaussian_log(5000, dlambda=psf)
        else:
            raise NotImplementedError()

    @staticmethod
    def generate_noise(flux, noise, error=None, random_seed=None):
        # Noise have to be reproducible when multiple datasets with different
        # post-processing are generated for autoencoders
        if random_seed is not None:
            np.random.seed(random_seed)

        if error is not None:
            # If error vector is present, use as sigma for additive noise
            err = noise * np.random.normal(size=flux.shape) * error
            return flux + err
        else:
            # Simple multiplicative noise, one random number per bin
            err = np.random.normal(1, noise, flux.shape)
            return flux * err

    def calculate_snr(self, weight=1.0):
        if self.flux_err is not None:
            # Make sure flux value is reasonable, otherwise exclude from averate SNR
            f = (self.flux_err <= 10 * self.flux) & (self.flux_err > 0)
            self.snr = np.mean(np.abs(self.flux[f] / self.flux_err[f])) / weight
        else:
            self.snr = 0
        
    def redden(self, extval=None):
        if extval is None:
            extval = self.extinction
        spec = pysynphot.spectrum.ArraySourceSpectrum(wave=self.wave, flux=self.flux, keepneg=True)
        # Cardelli, Clayton, & Mathis (1989, ApJ, 345, 245) R_V = 3.10.
        obs = spec * pysynphot.reddening.Extinction(extval, 'mwavg')
        self.flux = obs.flux

    def deredden(self, extval=None):
        extval = extval or self.extinction
        self.redden(-extval)

    def synthflux(self, filter):
        # TODO: Calculate error from error array?

        filt = pysynphot.spectrum.ArraySpectralElement(filter.wave, filter.thru, waveunits='angstrom')
        spec = pysynphot.spectrum.ArraySourceSpectrum(wave=self.wave, flux=self.flux, keepneg=True, fluxunits='flam')
        
        filt.binset = spec.wave     # supress warning from pysynphot
        obs = pysynphot.observation.Observation(spec, filt)
        return obs.effstim('Jy')
       
    def synthmag(self, filter, norm=1.0):
        flux = norm * self.synthflux(filter)
        return -2.5 * np.log10(flux) + 8.90

    @staticmethod
    def running_filter(wave, data, func, dlambda=None, vdisp=None):
        # TODO: use get_dispersion
        # Don't care much about edges here, they'll be trimmed when rebinning
        if dlambda is not None and vdisp is not None:
            raise Exception('Only one of dlambda and vdisp can be specified')
        elif dlambda is None and vdisp is None:
            vdisp = Constants.DEFAULT_FILTER_VDISP

        if vdisp is not None:
            z = vdisp * 1e3 / Physics.c

        ndata = np.empty(data.shape)
        for i in range(len(wave)):
            if isinstance(dlambda, collections.Iterable):
                mask = (wave[i] - dlambda[0] <= wave) & (wave < wave[i] + dlambda[1])
            elif dlambda is not None:
                mask = (wave[i] - dlambda <= wave) & (wave < wave[i] + dlambda)
            else:
                mask = ((1 - z) * wave[i] < wave) & (wave < (1 + z) * wave[i])

            if len(data.shape) == 1:
                if mask.size < 2:
                    ndata[i] = data[i]
                else:
                    ndata[i] = func(data[mask])
            elif len(data.shape) == 2:
                if mask.size < 2:
                    ndata[:, i] = data[:, i]
                else:
                    ndata[:, i] = func(data[:, mask], axis=1)
            else:
                raise NotImplementedError()

        return ndata

    def high_pass_filter(self, func=np.median, dlambda=None, vdisp=None):
        # TODO: error array?
        self.flux -= Spectrum.running_filter(self.wave, self.flux, func, dlambda=dlambda, vdisp=vdisp)
        if self.flux_sky is not None:
            self.flux_sky -= Spectrum.running_filter(self.wave, self.flux_sky, func, vdisp)

    def fit_envelope_chebyshev(self, wlim=None, iter=10, order=4, clip_top=3.0, clip_bottom=0.1):
        # TODO: use mask from spectrum
        wave = self.wave
        flux = np.copy(self.flux)
        if self.flux_err is not None:
            # TODO: maybe do some smoothing or filter out 0
            weight = 1 / self.flux_err
        else:
            weight = np.full(self.flux.shape, 1)

        mask = np.full(flux.shape, False)
        if wlim is not None and wlim[0] is not None and wlim[1] is not None:
            widx = np.digitize(wlim, wave)
            mask[:widx[0]] = True
            mask[widx[1]:] = True

        # Early stop if we have lost too many bins to do a meaningful fit
        while iter > 0 and np.sum(~mask) > 100:
            p = np.polynomial.chebyshev.chebfit(wave[~mask], flux[~mask], order, w=weight[~mask])
            c = np.polynomial.chebyshev.chebval(wave, p)
            s = np.sqrt(np.var(flux[~mask] - c[~mask]))
            mask |= ((flux > c + clip_top * s) | (flux < c - clip_bottom * s))

            iter -= 1

        return p, c

    @staticmethod
    def generate_calib_bias(wave, bandwidth=200, amplitude=0.05):
        """
        Simulate claibration error but multiplying with a slowly changing function

        :param bandwidth:
        :param amplitude:
        """

        def white_noise(N):
            s = 2 * np.exp(1j * np.random.rand(N // 2 + 1) * 2 * np.pi)
            noise = np.fft.irfft(s)
            return noise[:N]

        def wiener(N):
            step = white_noise(N + 1)
            walk = np.cumsum(step)[:N]
            return walk

        def damped_walk(N, alpha=1, D=1):
            step = white_noise(N)
            walk = np.zeros(N)
            for i in range(1, N):
                walk[i] = walk[i - 1] - alpha * walk[i - 1] + D * step[i]
            return walk

        def gauss(x, mu, sigma):
            return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

        def lowpass_filter(x, y, sigma):
            mid = np.mean(x)
            idx = np.digitize([mid - 3 * sigma, mid + 3 * sigma], wave)
            k = gauss(x[idx[0]:idx[1]], mid, sigma)
            k /= np.sum(k)
            return np.convolve(y, k, mode='same')

        bias = wiener(wave.shape[0])
        bias = lowpass_filter(wave, bias, 200)
        min = np.min(bias)
        max = np.max(bias)
        bias = 1.0 - amplitude * (bias - min) / (max - min)

        return bias

    def add_calib_bias(self, bandwidth=200, amplitude=0.05):
        bias = Spectrum.generate_calib_bias(self.wave, bandwidth=bandwidth, amplitude=amplitude)
        self.multiply(bias)

    def load(self, filename):
        raise NotImplementedError()

    def plot(self, ax=None, xlim=Constants.DEFAULT_PLOT_WAVE_RANGE, ylim=None, labels=True):
        ax = self.plot_getax(ax, xlim, ylim)
        ax.plot(self.wave, self.flux)
        if labels:
            ax.set_xlabel(r'$\lambda$ [A]')
            ax.set_ylabel(r'$F_\lambda$ [erg s-1 cm-2 A-1]')

        return ax

    def print_info(self):
        for p in self.get_param_names():
            print(p, getattr(self, p))