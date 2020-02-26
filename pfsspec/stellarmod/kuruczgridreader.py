import os
import logging

from pfsspec.stellarmod.modelgridreader import ModelGridReader
from pfsspec.stellarmod.kuruczspectrumreader import KuruczSpectrumReader

class KuruczGridReader(ModelGridReader):
    def __init__(self, grid, path):
        super(KuruczGridReader, self).__init__(grid)
        self.path = path

    def read_grid(self):
        self.grid.build_params_index()
        for m_h in self.grid.params['Fe_H'].values:
            fn = KuruczGridReader.get_filename(m_h, 2.0, False, False, False)
            fn = os.path.join(self.path, fn)
            with open(fn) as f:
                r = KuruczSpectrumReader(f)
                specs = r.read_all()
                for spec in specs:
                    if self.grid.wave is None:
                        self.grid.wave = spec.wave
                        self.grid.init_data()
                        self.grid.allocate_data()
                    self.grid.set_flux(spec.flux, Fe_H=spec.Fe_H, T_eff=spec.T_eff, log_g=spec.log_g)
        
        self.grid.build_data_index(rebuild=True)

        logging.info("Grid loaded with flux shape {}".format(self.grid.get_data_item_shape('flux')))

    @staticmethod
    def get_filename(Fe_H, v_turb, alpha=False, nover=False, odfnew=False):
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