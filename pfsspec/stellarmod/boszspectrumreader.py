import os
import logging
import math
import numpy as np
import pandas as pd
import re
import multiprocessing
import time

from pfsspec.stellarmod.modelgridspectrumreader import ModelGridSpectrumReader
from pfsspec.stellarmod.kuruczspectrum import KuruczSpectrum
from pfsspec.stellarmod.boszgrid import BoszGrid

class BoszSpectrumReader(ModelGridSpectrumReader):

    # TODO: Unify file open/close logig with other readers and
    # figure out how to use instance/class functions

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
        'p00': 0.0,
        'p03': 0.25,
        'p05': 0.5,
        'p08': 0.75
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

    def __init__(self, grid=None, path=None, wave_lim=None, max=None):
        super(BoszSpectrumReader, self).__init__(grid=grid)
        self.path = path
        self.wave_lim = wave_lim
        self.max = max

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

        if self.wave_lim is not None:
            filt = (self.wave_lim[0] <= df['wave']) & (df['wave'] <= self.wave_lim[1])
        else:
            filt = slice(None)

        spec = KuruczSpectrum()
        spec.wave = np.array(df['wave'][filt])
        spec.cont = np.array(df['cont'][filt])
        spec.flux = np.array(df['flux'][filt])

        return spec

    def process_item(self, i):
        logger = multiprocessing.get_logger()

        index, params = i
        fn = BoszSpectrumReader.get_filename(**params)
        fn = os.path.join(self.path, fn)

        if os.path.isfile(fn):
            tries = 3
            while True:
                try:
                    spec = self.read(fn)
                    return index, params, spec
                except Exception as e:
                    logger.error('Error parsing {}'.format(fn))
                    time.sleep(0.01)    # ugly hack
                    tries -= 1
                    if tries == 0:
                        raise e

        else:
            logger.debug('Cannot find file {}'.format(fn))
            return None

    def process_file(self, file):
        logger = multiprocessing.get_logger()

        params = BoszSpectrumReader.parse_filename(file)
        index = self.grid.get_index(**params)
        spec = self.read(file)

        return index, params, spec

    def get_filename(Fe_H, C_M, O_M, T_eff, log_g, v_turb=0.2, v_rot=0, R=5000):

        # amm03cm03om03t3500g25v20modrt0b5000rs.asc.bz2

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
        fn += '%d' % (int(log_g * 10))

        fn += 'v'
        fn += '%02d' % (int(v_turb * 100))

        fn += 'mod'

        fn += 'rt'
        fn += '%d' % (int(v_rot))

        fn += 'b'
        fn += '%d' % (R)

        fn += 'rs.asc.bz2'

        return fn

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
