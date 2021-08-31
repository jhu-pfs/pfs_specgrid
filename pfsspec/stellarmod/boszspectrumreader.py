import os
import logging
import math
import numpy as np
import pandas as pd
import re
import multiprocessing
import time

from pfsspec.data.spectrumreader import SpectrumReader
from pfsspec.stellarmod.boszspectrum import BoszSpectrum

class BoszSpectrumReader(SpectrumReader):
    MAP_FE_H = {
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
    }

    MAP_C_M = {
        'm03': -0.25,
        'm05': -0.5,
        'p00': 0.0,
        'p03': 0.25,
        'p05': 0.5,
    }

    MAP_O_M = {
        'm03': -0.25,
        'p00': 0.0,
        'p03': 0.25,
        'p05': 0.5,
    }

    def __init__(self, path=None, wave_lim=None, resolution=None):
        super(BoszSpectrumReader, self).__init__()

        self.path = path
        self.wave_lim = wave_lim
        self.resolution = resolution

    def correct_wave_grid(self, wlim):
        # BOSZ spectra are written to the disk with 3 decimals which aren't
        # enough to represent wavelength at high resolutions. This code is
        # from the original Kurucz SYNTHE to recalculate the wavelength grid.

        RESOLU = self.resolution
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
        if type(file) is str:
            fn, ext = os.path.splitext(file)
            if ext == '.bz2':
                compression = 'bz2'

        # for some reason the C implementation of read_csv throws intermittent errors
        # when forking using multiprocessing
        # engine='python',
        df = pd.read_csv(file, delimiter=r'\s+', header=None, compression=compression)
        df.columns = ['wave', 'flux', 'cont']

        # NOTE: wavelength values in the files have serious round-off errors
        # Correct wavelength grid here
        #spec.wave = np.array(df['wave'][filt])
        # cwave = self.correct_wave_grid((100, 32000))

        if self.wave_lim is not None:
            filt = (self.wave_lim[0] <= df['wave']) & (df['wave'] <= self.wave_lim[1])
        else:
            filt = slice(None)

        spec = BoszSpectrum()
        spec.wave = np.array(df['wave'][filt])
        spec.cont = np.array(df['cont'][filt])
        spec.flux = np.array(df['flux'][filt])

        return spec

    @staticmethod
    def get_filename(**kwargs):
        # amm03cm03om03t3500g25v20modrt0b5000rs.asc.bz2

        Fe_H = kwargs.pop('Fe_H')
        C_M = kwargs.pop('C_M')
        O_M = kwargs.pop('O_M')
        T_eff = kwargs.pop('T_eff')
        log_g = kwargs.pop('log_g')
        v_turb = kwargs.pop('v_turb', 0.2)
        v_rot = kwargs.pop('v_rop', 0)
        R = kwargs.pop('R', 5000)

        fn = 'a'

        fn += 'm'
        fn += 'm' if Fe_H < 0 else 'p'
        fn += '%02d' % (int(abs(Fe_H) * 10 + 0.5))

        fn += 'c'
        fn += 'm' if C_M < 0 else 'p'
        fn += '%02d' % (int(abs(C_M) * 10 + 0.5))

        fn += 'o'
        fn += 'm' if O_M < 0 else 'p'
        fn += '%02d' % (int(abs(O_M) * 10 + 0.5))

        fn += 't'
        fn += '%d' % (int(T_eff))

        fn += 'g'
        fn += '%02d' % (int(log_g * 10))

        fn += 'v'
        fn += '%02d' % (int(v_turb * 100))

        fn += 'mod'

        fn += 'rt'
        fn += '%d' % (int(v_rot))

        fn += 'b'
        fn += '%d' % (R)

        fn += 'rs.asc.bz2'

        return fn

    @staticmethod
    def parse_filename(filename):

        # amm03cm03om03t3500g25v20modrt0b5000rs.asc.bz2

        p = r'am([pm]\d{2})c([pm]\d{2})o([pm]\d{2})t(\d{4,5})g(\d{2})v20'
        m = re.search(p, filename)

        return{
            'Fe_H': BoszSpectrumReader.MAP_FE_H[m.group(1)],
            'C_M': BoszSpectrumReader.MAP_C_M[m.group(2)],
            'O_M': BoszSpectrumReader.MAP_O_M[m.group(3)],
            'T_eff': float(m.group(4)),
            'log_g': float(m.group(5)) / 10
        }
