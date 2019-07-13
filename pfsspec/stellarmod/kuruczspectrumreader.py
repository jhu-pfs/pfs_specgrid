import os
import logging
import math
import numpy as np

from pfsspec.data.spectrumreader import SpectrumReader
from pfsspec.stellarmod.kuruczspectrum import KuruczSpectrum
from pfsspec.stellarmod.kuruczgrid import KuruczGrid

class KuruczSpectrumReader(SpectrumReader):
    HEADER_LINES = 22
    WAVELENGTHS = 1221
    FLUX_COLUMNS = 10

    def __init__(self, file):
        super(KuruczSpectrumReader, self).__init__()
        self.file = file
        self.state = 'start'
        self.wave = None

    def read(self):
        if self.state == 'start':
            self.skip_file_header()
            self.state = 'wave'
            self.wave = self.read_wavelengths(KuruczSpectrumReader.WAVELENGTHS)
            self.state = 'spec'

        spec = KuruczSpectrum()
        spec.wave = self.wave
        # read stellar parameters
        if not self.read_spec_header(spec):
            return None
        # read absorbtion spectrum, ergs/cm**2/s/hz/ster
        spec.flux = self.read_fluxes(KuruczSpectrumReader.WAVELENGTHS)
        # skip continuum
        self.read_fluxes(KuruczSpectrumReader.WAVELENGTHS)
        spec.fnu_to_flam()

        return spec

    def read_all(self):
        wave = None
        specs = []
        i = 0
        while True:
            spec = self.read()
            if spec is None:
                break
            specs.append(spec)
        return specs

    def skip_file_header(self):
        for i in range(KuruczSpectrumReader.HEADER_LINES):
            self.file.readline()

    def read_spec_header(self, spec):
        line = self.file.readline()
        parts = line.split()

        try:
            if len(parts) == 0:
                return False
            elif len(parts) == 12:
                # oldest format with L/H specified
                spec.T_eff = float(parts[1])
                spec.log_g = float(parts[3])
                spec.M_H = float(parts[6].strip('[]aA'))
                spec.alpha = (('a' in parts[6]) or ('A' in parts[6]))
                spec.N_He = -1
                spec.v_turb = float(parts[8])
                spec.L_H = float(parts[11])
            elif len(parts) == 10:
                # oldest format without L/H specified
                spec.T_eff = float(parts[1])
                spec.log_g = float(parts[3])
                spec.M_H = float(parts[6].strip('[]aA'))
                spec.alpha = (('a' in parts[6]) or ('A' in parts[6]))
                spec.N_He = -1
                spec.v_turb = float(parts[8])
                spec.L_H = -1
            else:
                spec.T_eff = float(parts[1])
                spec.log_g = float(parts[3])
                spec.M_H = float(parts[4].strip('[]aA'))
                spec.alpha = (('a' in parts[4]) or ('A' in parts[4]))
                spec.N_He = float(parts[5].split('=')[1])
                spec.v_turb = float(parts[6].split('=')[1])
                spec.L_H = float(parts[7].split('=')[1])
        except Exception:
            print(line)
            raise Exception()

        return True

    def read_wavelengths(self, n):
        wave = np.empty(n)
        i = 0
        while i < n:
            line = self.file.readline()
            parts = line.split()
            for p in parts:
                wave[i] = 10 * float(p)  # models use nm
                i += 1
        return wave

    def read_fluxes(self, n):
        flux = np.empty(n)
        i = 0
        while i < n:
            line = self.file.readline()
            parts = [line[i:i + KuruczSpectrumReader.FLUX_COLUMNS] for i in range(0, len(line), KuruczSpectrumReader.FLUX_COLUMNS)]
            for p in parts:
                if len(p) == KuruczSpectrumReader.FLUX_COLUMNS:
                    flux[i] = float(p)
                    i += 1
        return flux

    def read_grid(path, model):
        grid = KuruczGrid(model)
        grid.build_index()

        for m_h in grid.M_H:
            fn = KuruczSpectrumReader.get_filename(m_h, 2.0, False, False, False)
            fn = os.path.join(path, fn)
            with open(fn) as f:
                r = KuruczSpectrumReader(f)
                specs = r.read_all()
                for spec in specs:
                    if grid.wave is None:
                        grid.init_storage(spec.wave)
                    grid.set_flux(spec.M_H, spec.T_eff, spec.log_g, spec.flux)

        logging.info("Grid loaded with flux grid shape {}".format(grid.flux.shape))

        return grid

    def get_filename(M_H, v_turb, alpha=False, nover=False, odfnew=False):
        mh = "%02d" % (abs(M_H) * 10)

        dir = 'grid'
        dir += 'm' if M_H < 0 else 'p'
        dir += mh
        if alpha: dir += 'a'
        if nover: dir += 'nover'
        if odfnew: dir += 'odfnew'

        fn = 'f'
        fn += 'm' if M_H < 0 else 'p'
        fn += mh
        if alpha: dir += 'a'
        fn += 'k%01d' % (v_turb)
        if nover: fn += 'nover'
        if odfnew: fn += 'odfnew'
        fn += '.pck'

        return os.path.join(dir, fn)