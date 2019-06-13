import numpy as np

class Spectrum():
    def __init__(self):
        self.wave = None
        self.flux = None

    def fnu_to_flam(self):
        # ergs/cm**2/s/hz/ster to erg/s/cm^2/A surface flux
        self.flux /= 3.336e-19 * (self.wave) ** 2 / 4 / np.pi

    def flam_to_fnu(self):
        self.flux *= 3.336e-19 * (self.wave) ** 2 / 4 / np.pi

    def plot(self, ax, xlim=(3500, 9000), labels=True):
        ax.plot(self.wave, self.flux)
        if xlim:
            ax.set_xlim(xlim)
        if labels:
            ax.set_xlabel(r'$\lambda$ [A]')
            ax.set_ylabel(r'$F_\lambda$ [erg s-1 cm-2 A-1]')