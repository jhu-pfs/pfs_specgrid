import os
import logging
import numpy as np

import pysynphot, pysynphot.binning, pysynphot.spectrum, pysynphot.reddening

import pfsspec.util as util
from pfsspec.common.spectrum import Spectrum

class KuruczAugmenter():
    def __init__(self, orig=None):
        if isinstance(orig, KuruczAugmenter):
            self.calib_bias = orig.calib_bias
            self.calib_bias_count = orig.calib_bias_count
            self.calib_bias_bandwidth = orig.calib_bias_bandwidth
            self.calib_bias_amplitude = orig.calib_bias_amplitude

            self.ext = orig.ext
            self.ext_count = orig.ext_count
            self.ext_dist = orig.ext_dist      
        else:          
            self.calib_bias = None
            self.calib_bias_count = None
            self.calib_bias_bandwidth = 200
            self.calib_bias_amplitude = 0.01

            self.ext = None
            self.ext_count = None
            self.ext_dist = None

    def add_args(self, parser):        
        parser.add_argument('--calib-bias', type=float, nargs=3, default=None, help='Add simulated calibration bias.')        
        parser.add_argument('--ext', type=float, nargs='*', default=None, help='Extinction or distribution parameters.\n')

    def init_from_args(self, args):
        if 'calib_bias' in args and args['calib_bias'] is not None:
            calib_bias = args['calib_bias']
            self.calib_bias_count = int(calib_bias[0])
            self.calib_bias_bandwidth = calib_bias[1]
            self.calib_bias_amplitude = calib_bias[2]

        if 'ext' in args and args['ext'] is not None:
            ext = args['ext']
            self.ext_count = int(ext[0])
            self.ext_dist = [ext[1], ext[2]]

    def on_epoch_end(self):
        self.calib_bias = None
        self.ext = None

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
        if self.ext_count is not None:
            if self.ext is None:
                if not dataset.constant_wave:
                    raise Exception('Random extinction can only be applied during data augmentation if a constant wave grid is used.')
                self.generate_ext(dataset.wave)

            # Pick extinction curve
            i = np.random.randint(self.ext.shape[0], size=(flux.shape[0],))
            flux *= self.ext[i, :]

        return flux

    def generate_ext(self, wave):
        # This function generates a range of random extinction curves. Note that
        # it work only with fixed wavelength grids

        self.logger.debug('Generating {} realization of extinction curve'.format(self.ext_count))

        wave = wave if len(wave.shape) == 1 else wave[0]
        self.ext = np.empty((self.ext_count, wave.shape[0]))
        for i in range(self.ext_count):
            extval = np.random.uniform(*self.ext_dist)
            spec = pysynphot.spectrum.ArraySourceSpectrum(wave=wave, flux=np.full(wave.shape, 1.0), keepneg=True)
            obs = spec * pysynphot.reddening.Extinction(extval, 'mwavg')
            self.ext[i, :] = obs.flux

        self.logger.debug('Generated {} realization of extinction curve'.format(self.ext_count))