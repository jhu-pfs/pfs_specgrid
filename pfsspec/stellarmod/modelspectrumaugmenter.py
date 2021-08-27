import os
import logging
import numpy as np

import pysynphot, pysynphot.binning, pysynphot.spectrum, pysynphot.reddening

import pfsspec.util as util
from pfsspec.util.dist import *
from pfsspec.common.spectrum import Spectrum

class ModelSpectrumAugmenterMixin():
    def __init__(self, orig=None):

        if isinstance(orig, ModelSpectrumAugmenterMixin):            
            self.calib_bias = orig.calib_bias
            self.calib_bias_count = orig.calib_bias_count
            self.calib_bias_bandwidth = orig.calib_bias_bandwidth
            self.calib_bias_amplitude = orig.calib_bias_amplitude

            self.ext_A_lambda = orig.ext_A_lambda
            self.ext_dist = orig.ext_dist      
            self.ext = orig.ext
        else:          
            self.calib_bias = None
            self.calib_bias_count = None
            self.calib_bias_bandwidth = 200
            self.calib_bias_amplitude = 0.01

            self.ext_A_lambda = None
            self.ext_dist = None
            self.ext = None

    def add_args(self, parser):        
        parser.add_argument('--calib-bias', type=float, nargs=3, default=None, help='Add simulated calibration bias.')        

        parser.add_argument('--ext-dist', type=str, default=None, help='Extinction distribution.')
        parser.add_argument('--ext', type=float, nargs='*', default=None, help='Extinction distribution parameters.\n')

    def init_from_args(self, args):
        if self.is_arg('calib_bias', args):
            calib_bias = self.get_arg('calib_bias', self.calib_bias, args)
            self.calib_bias_count = int(calib_bias[0])
            self.calib_bias_bandwidth = calib_bias[1]
            self.calib_bias_amplitude = calib_bias[2]

        if self.is_arg('ext', args):
            self.ext_dist = self.get_arg('ext_dist', self.ext_dist, args)
            self.ext = self.get_arg('ext', self.ext, args)

    def on_epoch_end(self):
        self.calib_bias = None

    def apply_calib_bias(self, dataset, chunk_id, idx, flux):
        if self.calib_bias_count is not None:
            if self.calib_bias is None:
                if not dataset.constant_wave:
                    raise Exception('Random flux calibration bias can only be applied during data augmentation if a constant wave grid is used.')
                self.generate_calib_bias(dataset.wave)
        
            # Pick calibration bias curve
            i = np.random.randint(self.calib_bias.shape[0], size=(flux.shape[0],))
            flux *= self.calib_bias[i, :]
        
        return flux

    def generate_calib_bias(self, wave):
        self.logger.debug('Generating {} realization of calibration bias'.format(self.calib_bias_count))
        
        wave = wave if len(wave.shape) == 1 else wave[0]
        self.calib_bias = np.empty((self.calib_bias_count, wave.shape[0]))
        for i in range(self.calib_bias_count):
            self.calib_bias[i, :] = Spectrum.generate_calib_bias(wave,
                                                                 bandwidth=self.calib_bias_bandwidth,
                                                                 amplitude=self.calib_bias_amplitude)

        self.logger.debug('Generated {} realization of calibration bias'.format(self.calib_bias_count))

    def apply_ext(self, dataset, chunk_id, idx, flux):
        if self.ext is not None and self.ext_dist is not None:
            if self.ext_A_lambda is None:
                if not dataset.constant_wave:
                    raise Exception('Random extinction can only be applied during data augmentation if a constant wave grid is used.')
                self.generate_ext(dataset.wave)

            # Pick extinction value
            if len(self.ext) == 1:
                extval = self.ext[0]
            else:
                extval = get_random_dist(self.ext_dist, self.random_state)(*self.ext)

            flux *= 10 ** (self.ext_A_lambda * extval)

        return flux

    def generate_ext(self, wave):
        # This function generates a range of random extinction curves. Note that
        # it work only with fixed wavelength grids. We calculate and store
        # -0.4 * A_lambda in ext_A_lambda for E(B - V) = 1 so that the througput
        # at any E(B - V) can easily be computed as
        #     T = 10 ** (ext_A_lambda * ext)
        # where ext is a randomly drawn value of E(B - V)

        # TODO: this could be extended to use multiple extinction curves

        wave = wave if len(wave.shape) == 1 else wave[0]
        extval = 1
        spec = pysynphot.spectrum.ArraySourceSpectrum(wave=wave, flux=np.ones_like(wave), keepneg=True)
        spec = spec * pysynphot.reddening.Extinction(extval, 'mwavg')
        self.ext_A_lambda = np.log10(spec.flux)
