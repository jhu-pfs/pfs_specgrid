import numpy as np

from pfsspec.physics import Physics
from pfsspec.pfsobject import PfsObject

class Noise(PfsObject):
    def __init__(self, orig=None):
        super(Noise, self).__init__(orig=orig)
        self.sky = None
        self.moon = None
        self.detector = None

    def get_stray_light(self, counts):
        return np.zeros(counts.shape)

    def get_sky_subtraction_error(self, sky):
        return np.zeros(sky.shape)

    def get_flux(self, counts, exp_count, exp_time, za, fa):
        conv = self.sky.get_conversion(exp_count, exp_time, za=za, fa=fa)
        flux = counts / conv
        return flux

    def get_counts(self, wave, flux, exp_count, exp_time, za, fa):
        # ETC uses 1e-17 erg/cm2/s as the unit of flux
        # Spectra are in erg/cm2/s/A
        # TODO: take dlambda from data files
        # dlambda = 0.659
        fnu = Physics.flam_to_fnu(wave, flux)
        conv = self.sky.get_conversion(exp_count, exp_time, za=za, fa=fa)
        counts = fnu * conv
        return counts

    def get_skycounts(self, exp_count, exp_time, za, fa, lunar_ta, lunar_za, lunar_phase):
        # Here we sum up sky and moon contribution only, because these are defined
        # in units of e/pix, whereas detector noise is e^2/pix
        counts = self.sky.get_counts(exp_count, exp_time, za, fa)
        counts += self.moon.get_counts(exp_count, exp_time, za, fa, lunar_ta, lunar_za, lunar_phase)
        return counts

    def get_noise(self, exp_count, exp_time, za, fa, lunar_ta, lunar_za, lunar_phase):
        counts = self.get_counts()
        counts = np.sqrt(counts)
        sigma = self.get_flux(counts, za, fa)
        return sigma

        counts += exp_time / self.ref_exp_time * exp_count / self.ref_exp_count * self.detector.dark
        counts += exp_count / self.ref_exp_count * self.readout
        counts *= self.sample_factor

        # TODO: add stray lights and sky subtraction error