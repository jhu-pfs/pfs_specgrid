import os
import logging
import numpy as numpy

from pfsspec.data.atmreader import AtmReader
from pfsspec.stellarmod.kuruczatm import KuruczAtm

class KuruczAtmReader(AtmReader):
    def __init__(self):
        super(KuruczAtmReader, self).__init__()
        self.atm = None
        self.layer = 0

    def read(self, file):
        self.atm = KuruczAtm()
        
        state = 'start'
        while True:
            line = file.readline()
            if line == '':
                break

            if state == 'start':
                if line.startswith('TEFF'):
                    self.read_params(line)
                elif line.startswith('TITLE'):
                    self.read_title(line)
                elif line.startswith(' OPACITY'):
                    pass
                elif line.startswith(' CONVECTION'):
                    pass
                elif line.startswith('ABUNDANCE'):
                    self.read_abundance_first(line)
                elif line.startswith(' ABUNDANCE'):
                    self.read_abundance(line)
                elif line.startswith('READ'):
                    state = 'atm'
                    self.layer = 0
                else:
                    raise Exception('Invalid line: {}'.format(line))
            elif state == 'atm':
                if line.startswith(' '):
                    self.read_atm(line)
                else:
                    # Don't read beyond the layers
                    break

        return self.atm

    def read_params(self, line):
        self.atm.T_eff = float(line[4:14])
        self.atm.log_g = float(line[21:30])

    def read_title(self, line):
        self.atm.title = line[7:]

    def read_abundance_first(self, line):
        # This is H and He which are saved with higher accurancy
        self.atm.ABUNDANCE[0] = float(line[45:52])
        self.atm.ABUNDANCE[1] = float(line[55:62])

    def read_abundance(self, line):
        for i in range(6):
            if len(line) > 19 + 10 * i and line[19 + 10 * i] != ' ':
                self.read_abundance_item(line[18 + 10 * i:])

    def read_abundance_item(self, line):
        idx = int(line[0:2])
        val = float(line[2:9])
        self.atm.ABUNDANCE[idx - 1] = val

    def read_atm(self, line):
        parts = list(filter(None, line.split()))
        self.atm.RHOX[self.layer] = float(parts[0])
        self.atm.T[self.layer] = float(parts[1])
        self.atm.P[self.layer] = float(parts[2])
        self.atm.XNE[self.layer] = float(parts[3])
        self.atm.ABROSS[self.layer] = float(parts[4])
        self.atm.ACCRAD[self.layer] = float(parts[5])
        self.atm.VTURB[self.layer] = float(parts[6])
        self.atm.FLXCNV[self.layer] = float(parts[7])
        self.atm.VCONV[self.layer] = float(parts[8])
        self.atm.VELSND[self.layer] = float(parts[9])
        self.layer += 1
