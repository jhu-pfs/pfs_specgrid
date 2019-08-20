import os
import logging
import math
import numpy as np
import pandas as pd

from pfsspec.data.spectrumreader import SpectrumReader
from pfsspec.stellarmod.kuruczspectrum import KuruczSpectrum
from pfsspec.stellarmod.boszgrid import BoszGrid

class BoszSpectrumReader(SpectrumReader):

    # TODO: Unify file open/close logig with other readers and
    # figure out how to use instance/class functions

    def __init__(self, file=None, wave_lim=None):
        super(BoszSpectrumReader, self).__init__()
        self.file = file
        self.wave_lim = wave_lim

    def read(self, file=None):
        compression = None
        if file is None:
            file = self.file
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
        spec.flux = np.array(df['flux'][filt])

        return spec

    def read_grid(self, path):
        grid = BoszGrid()
        grid.build_index()

        for fe_h in grid.params['Fe_H'].values:
            for t_eff in grid.params['T_eff'].values:
                for log_g in grid.params['log_g'].values:
                    for c_m in grid.params['C_M'].values:
                        for a_m in grid.params['alpha_M'].values:
                            fn = BoszSpectrumReader.get_filename(fe_h, c_m, a_m, t_eff, log_g)
                            fn = os.path.join(path, fn)
                            if os.path.isfile(fn):
                                spec = self.read(fn)
                                if grid.wave is None:
                                    grid.init_storage(spec.wave)
                                grid.set_flux(spec.flux, Fe_H=fe_h, T_eff=t_eff, log_g=log_g,
                                              C_M=c_m, alpha_M=a_m)

        logging.info("Grid loaded with flux grid shape {}".format(grid.flux.shape))

        return grid

    def get_filename(Fe_H, C_M, alpha_M, T_eff, log_g, v_turb=0.2, v_rot=0, R=5000):

        # amm03cm03om03t3500g25v20modrt0b5000rs.asc.bz2

        fn = 'a'

        fn += 'm'
        fn += 'm' if Fe_H < 0 else 'p'
        fn += '%02d' % (int(Fe_H + 0.05) * 10)

        fn += 'c'
        fn += 'm' if C_M < 0 else 'p'
        fn += '%02d' % (int(C_M + 0.05) * 10)

        fn += 'o'
        fn += 'm' if alpha_M < 0 else 'p'
        fn += '%02d' % (int(alpha_M + 0.05) * 10)

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