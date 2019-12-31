import numpy as np

from pfsspec.pfsobject import PfsObject

class Noise(PfsObject):
    def __init__(self, orig=None):
        super(Noise, self).__init__(orig=orig)
        self.ref_exp_count = 1
        self.ref_exp_time = 450

    def get_stray_light(self, counts):
        return np.zeros(counts.shape)

    def get_sky_subtraction_error(self, sky):
        return np.zeros(sky.shape)

    def get_counts(self, exp_count, exp_time, za, fa, lunar_ta, lunar_za, lunar_phase):
        counts = self.sky.get_counts(exp_count, exp_time, za, fa)
        counts += self.moon.get_counts(exp_count, exp_time, za, fa, lunar_ta, lunar_za, lunar_phase)
        counts += exp_time / self.ref_exp_time * exp_count / self.ref_exp_count * self.dark
        counts += exp_count / self.ref_exp_count * self.readout
        counts *= self.sample_factor

        return counts

    def get_flux(self, counts, za, fa):
        conv = self.sky.interpolate_data_item_linear('conv', za=za, fa=fa)
        flux = counts / conv
        return flux

    def get_noise(self, exp_count, exp_time, za, fa, lunar_ta, lunar_za, lunar_phase):
        counts = self.get_counts()
        counts = np.sqrt(counts)
        sigma = self.get_flux(counts, za, fa)
        return sigma

        # TODO: add stray lights and sky subtraction error