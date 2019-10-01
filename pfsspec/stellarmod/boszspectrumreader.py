import os
import logging
import math
import numpy as np
import pandas as pd

from pfsspec.stellarmod.modelgridspectrumreader import ModelGridSpectrumReader
from pfsspec.stellarmod.kuruczspectrum import KuruczSpectrum
from pfsspec.stellarmod.boszgrid import BoszGrid

class BoszSpectrumReader(ModelGridSpectrumReader):

    # TODO: Unify file open/close logig with other readers and
    # figure out how to use instance/class functions

    def __init__(self, grid, path=None, wave_lim=None, max=None):
        super(BoszSpectrumReader, self).__init__(grid)
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
        index, params = i
        fn = BoszSpectrumReader.get_filename(**params)
        fn = os.path.join(self.path, fn)

        if os.path.isfile(fn):
            spec = self.read(fn)
            return index, params, spec
        else:
            logging.debug('Cannot find file {}'.format(fn))
            return None

    def get_filename(Fe_H, C_M, a_Fe, T_eff, log_g, v_turb=0.2, v_rot=0, R=5000):

        # amm03cm03om03t3500g25v20modrt0b5000rs.asc.bz2

        fn = 'a'

        fn += 'm'
        fn += 'm' if Fe_H < 0 else 'p'
        fn += '%02d' % (int(abs(Fe_H) * 10 + 0.5))

        fn += 'c'
        fn += 'm' if C_M < 0 else 'p'
        fn += '%02d' % (int(abs(C_M) * 10 + 0.5))

        fn += 'o'
        fn += 'm' if a_Fe < 0 else 'p'
        fn += '%02d' % (int(abs(a_Fe) * 10 + 0.5))

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