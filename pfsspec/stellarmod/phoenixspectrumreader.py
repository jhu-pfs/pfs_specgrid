import os
import logging
import math
import numpy as np
import pandas as pd
import re
import multiprocessing
import time
from astropy.io import fits
from pfsspec.data.spectrumreader import SpectrumReader
from pfsspec.stellarmod.kuruczspectrum import KuruczSpectrum
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



    def __init__(self, path=None, wave_lim=None):#, resolution=None):
        super(PhoenixSpectrumReader, self).__init__()
        self.path = path
        self.wave_lim = wave_lim
       # self.resolution = resolution

    def correct_wave_grid(self, wlim):
      #  RESOLU = self.resolution
        WLBEG = wlim[0]  # nm
        WLEND = wlim[1]  # nm
        RATIO = 1. + 1. / RESOLU
        RATIOLG = np.log10(RATIO)
        IXWLBEG = int(np.log10(WLBEG) / RATIOLG)
        WBEGIN = 10 ** (IXWLBEG * RATIOLG)

        if WBEGIN < WLBEG:
            IXWLBEG = IXWLBEG + 1
            WBEGIN = 10 ** (IXWLBEG * RATIOLG)
        IXWLEND = int(np.log10(WLEND) / RATIOLG)
        WLLAST = 10 ** (IXWLEND * RATIOLG)
        if WLLAST > WLEND:
            IXWLEND = IXWLEND - 1
            WLLAST = 10 ** (IXWLEND * RATIOLG)
        LENGTH = IXWLEND - IXWLBEG + 1
        DWLBEG = WBEGIN * RATIO - WBEGIN
        DWLLAST = WLLAST - WLLAST / RATIO

        a = np.linspace(np.log10(10 * WBEGIN), np.log10(10 * WLLAST), LENGTH)
        cwave = 10 ** a

        return cwave

    def read(self, file=None):
        compression = None
        if file is None:
            file = self.path
       # if type(file) is str:
        #    fn, ext = os.path.splitext(file)
         #   if ext == '.bz2':
          #      compression = 'bz2'

        # for some reason the C implementation of read_csv throws intermittent errors
        # when forking using multiprocessing
        # engine='python',
        #df = pd.read_csv(file, delimiter=r'\s+', header=None, compression=compression)
        d = fits.open(file)
        dat = d[0].data
        d.close()
        w = fits.open('/scratch/ceph/dobos/data/pfsspec/models/stellar/grid/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
        wav = w[0].data
        w.close()
        df = pd.DataFrame(np.array([wav,dat]).T,columns=['wave','flux'])
        spec = KuruczSpectrum()
        # cwave = self.correct_wave_grid((100, 32000))

        if self.wave_lim is not None:
            filt = (self.wave_lim[0] <= df['wave']) & (df['wave'] <= self.wave_lim[1])
        else:
            filt = slice(None)

        spec.wave = np.array(df['wave'][filt])
        spec.flux = np.array(df['flux'][filt])
        
        #estimate continuum using Alex method
        #trace = AlexContinuumModelTrace()
       # model = Alex(trace)
        model = Chebyshev()
        #model.init_wave(spec.wave)
        #jk using chebyshev bc don't feed in continuum values (Which.... why would you build a model where you feed in
        #continuum values if you're trying to fit the continuum...?)
        params = model.fit(spec)
        wave, cont = model.eval(df['wave'][filt], params)
        
        spec.cont = cont
        return spec

    @staticmethod
    def get_filename(Fe_H, T_eff, log_g):
        # amm03cm03om03t3500g25v20modrt0b5000rs.asc.bz2
        #lte12000-6.00+1.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits

        fn = 'lte'
        fn += '0' if T_eff < 10000 else ''
        fn += "{:.0f}".format(int(T_eff))
        
        fn += '-'
        fn += '{:.02f}'.format(float(log_g))

        fn += '{:+.01f}'.format(float(Fe_H)) if float(Fe_H) != 0 else '-0.0'
      #  fn += 'm' if O_M < 0 else 'p'
      #  fn += '%02d' % (int(abs(O_M) * 10 + 0.5))


        fn += '.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'

        return fn

    @staticmethod
    def parse_filename(filename):

        # amm03cm03om03t3500g25v20modrt0b5000rs.asc.bz2
        #lte12000-6.00+1.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
        p = r'lte(\d{5})([+-])(\d{1}).(\d{2})([+-])(\d{1}).(\d{1})'
        #r'am([pm]\d{2})c([pm]\d{2})o([pm]\d{2})t(\d{4,5})g(\d{2})v20'
        m = re.search(p, filename)

        return{
            'T_eff': float(m.group(1)),
            'log_g': float(m.group(3)+m.group(4))/100,
            'Fe_H': float(m.group(5)+m.group(6)+m.group(7))/10
        }

