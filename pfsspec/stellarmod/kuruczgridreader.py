import os
import logging

from pfsspec.stellarmod.modelgridreader import ModelGridReader
from pfsspec.stellarmod.kuruczspectrumreader import KuruczSpectrumReader

class KuruczGridReader(ModelGridReader):
    # Implements a grid reader that reads Kurucz model spectra from files
    # where, for each metallicity, all spectra are stored in a single file

    def __init__(self, grid, parallel=True, threads=None, max=None):
        super(KuruczGridReader, self).__init__(grid, parallel=parallel, threads=threads, max=max)

    def read_grid(self, path):
        self.grid.build_axis_indexes()
        for m_h in self.grid.axes['Fe_H'].values:
            fn = KuruczGridReader.get_filename(m_h, 2.0, False, False, False)
            fn = os.path.join(path, fn)
            with open(fn) as f:
                r = KuruczSpectrumReader(f)
                specs = r.read_all()
                for spec in specs:
                    if self.grid.wave is None:
                        self.grid.wave = spec.wave
                        self.grid.init_values()
                        self.grid.allocate_values()
                    self.grid.set_flux(spec.flux, Fe_H=spec.Fe_H, T_eff=spec.T_eff, log_g=spec.log_g)
        
        self.grid.build_value_indexes(rebuild=True)

        self.logger.info("Grid loaded with flux shape {}".format(self.grid.get_value_shape('flux')))

    @staticmethod
    def get_filename(**kwargs):

        Fe_H = kwargs.pop('Fe_H')
        v_turb = kwargs.pop('v_turb')
        alpha = kwargs.pop('alpha',False)
        nover = kwargs.pop('nover', False)
        odfnew = kwargs.pop('odfnew', False)

        mh = "%02d" % (abs(Fe_H) * 10)

        dir = 'grid'
        dir += 'm' if Fe_H < 0 else 'p'
        dir += mh
        if alpha: dir += 'a'
        if nover: dir += 'nover'
        if odfnew: dir += 'odfnew'

        fn = 'f'
        fn += 'm' if Fe_H < 0 else 'p'
        fn += mh
        if alpha: dir += 'a'
        fn += 'k%01d' % (v_turb)
        if nover: fn += 'nover'
        if odfnew: fn += 'odfnew'
        fn += '.pck'

        return os.path.join(dir, fn)