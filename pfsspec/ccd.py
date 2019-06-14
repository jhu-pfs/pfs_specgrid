import numpy as np
import matplotlib.pyplot as plt

from pfsspec.constants import Constants
from pfsspec.pfsobject import PfsObject

class Ccd(PfsObject):
    def __init__(self, name='(unnamed)'):
        super(Ccd, self).__init__()
        self.name = name
        self.wave = None
        self.qeff = None

    def read(self, file):
        [self.wave, self.qeff] = np.loadtxt(file).transpose()

    def plot(self, ax=None, xlim=Constants.DEFAULT_PLOT_WAVE_RANGE, ylim=None, labels=True):
        ax = self.plot_getax(ax, xlim, ylim)
        ax.plot(self.wave, self.qeff, label=self.name)
        if labels:
            ax.set_xlabel(r'$\lambda$ [A]')
            ax.set_ylabel(r'$Q_\lambda$')

        return ax