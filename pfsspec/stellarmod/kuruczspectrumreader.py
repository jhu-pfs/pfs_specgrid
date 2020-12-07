import os
import logging
import math
import numpy as np

from pfsspec.data.spectrumreader import SpectrumReader
from pfsspec.physics import Physics
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
        # skip continuum, it's all zero
        self.read_fluxes(KuruczSpectrumReader.WAVELENGTHS)
        # convert from model surface intensity in erg/s/cm^2/sterad to erg/s/cm^2/A
        spec.flux = Physics.fnu_to_flam(spec.wave, spec.flux) * 4 * np.pi

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
                spec.Fe_H = float(parts[6].strip('[]aA'))
                spec.a_Fe = (('a' in parts[6]) or ('A' in parts[6]))
                spec.N_He = -1
                spec.v_turb = float(parts[8])
                spec.L_H = float(parts[11])
            elif len(parts) == 10:
                # oldest format without L/H specified
                spec.T_eff = float(parts[1])
                spec.log_g = float(parts[3])
                spec.Fe_H = float(parts[6].strip('[]aA'))
                spec.a_Fe = (('a' in parts[6]) or ('A' in parts[6]))
                spec.N_He = -1
                spec.v_turb = float(parts[8])
                spec.L_H = -1
            else:
                spec.T_eff = float(parts[1])
                spec.log_g = float(parts[3])
                spec.Fe_H = float(parts[4].strip('[]aA'))
                spec.a_Fe = (('a' in parts[4]) or ('A' in parts[4]))
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
