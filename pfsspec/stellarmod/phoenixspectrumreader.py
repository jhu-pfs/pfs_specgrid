import os
import numpy as np
from numpy.core.memmap import memmap
import pandas as pd
import re
from astropy.io import fits

from pfsspec.data.spectrumreader import SpectrumReader
from pfsspec.stellarmod.phoenixspectrum import PhoenixSpectrum
from pfsspec.stellarmod.continuummodels.chebyshev import Chebyshev

#lte12000-6.00+1.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
class PhoenixSpectrumReader(SpectrumReader):
    '''MAP_FE_H = {
        'm03': -0.25,
        'm05': -0.5,
        'm08': -0.75,
        'm10': -1.0,
        'm13': -1.25,
        'm15': -1.5,
        'm18': -1.75,
        'm20': -2.0,
        'm23': -2.25,
        'm25': -2.5,
        'm28': -2.75,
        'm30': -3.0,
        'm35': -3.5,
        'm40': -4.0,
        'm45': -4.5,
        'm50': -5.0,
        'p00': 0.0,
        'p03': 0.25,
        'p05': 0.5,
        'p08': 0.75,
        'p10': 1.00,
        'p15': 1.50
    }'''

    def __init__(self, path=None, wave_lim=None, resolution=None):
        super(PhoenixSpectrumReader, self).__init__()

        self.path = path
        self.wave_lim = wave_lim
        self.resolution = resolution

    def read(self, file=None):
        if file is None:
            file = self.path

        # Read fluxe
        with fits.open(file, memmap=False) as f:
            flux = f[0].data    

        # Wavelengths are read from a separate file which should be next to
        # the current one, in the same directory
        dir, _ = os.path.split(file)
        fn = os.path.join(dir, 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
        with fits.open(fn, memmap=False) as f:
            wave = f[0].data    

        if self.wave_lim is not None:
            filt = (self.wave_lim[0] <= wave) & (wave <= self.wave_lim[1])
        else:
            filt = slice(None)

        spec = PhoenixSpectrum()
        spec.wave = wave[filt]
        spec.flux = flux[filt]

        # TODO: continuum?
        
        # estimate continuum using Alex method
        # trace = AlexContinuumModelTrace()
        # model = Alex(trace)
        # model = Chebyshev()
        # model.init_wave(spec.wave)
        # jk using chebyshev bc don't feed in continuum values (Which.... why would you build a model where you feed in
        # continuum values if you're trying to fit the continuum...?)
        # params = model.fit(spec)
        # wave, cont = model.eval(df['wave'][filt], params)
        # spec.cont = cont

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

