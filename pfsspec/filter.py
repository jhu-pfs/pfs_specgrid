import numpy as np
import matplotlib.pyplot as plt

from pfsspec.constants import Constants
from pfsspec.pfsobject import PfsObject

class Filter(PfsObject):
    def __init__(self, name='(unnamed)'):
        super(Filter, self).__init__()
        self.name = name
        self.wave = None
        self.thru = None

    def read(self, file):
        [self.wave, self.thru] = np.loadtxt(file).transpose()

    def extend(self, min, max, res):
        wavemin = self.wave.min()
        wavemax = self.wave.max()
        if (min < wavemin):
            w = np.arange(min, wavemin, res)
            self.wave = np.hstack((w, self.wave))
            self.thru = np.hstack((np.zeros(len(w)), self.thru))

        if (max > wavemax):
            w = np.arange(wavemax, max, res) + res
            self.wave = np.hstack((self.wave, w))
            self.thru = np.hstack((self.thru, np.zeros(len(w))))


    def plot(self, ax=None, xlim=Constants.DEFAULT_PLOT_WAVE_RANGE, ylim=None, labels=True):
        ax = self.plot_getax(ax, xlim, ylim)
        ax.plot(self.wave, self.thru, label=self.name)
        if labels:
            ax.set_xlabel(r'$\lambda$ [A]')
            ax.set_ylabel(r'$R_\lambda$')

        return ax