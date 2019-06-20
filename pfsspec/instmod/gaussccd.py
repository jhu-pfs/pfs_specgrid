import scipy.stats

from pfsspec.obsmod.ccd import Ccd

class GaussCcd(Ccd):
    def create(self, wave, A=1, mu=6500, sigma=3000):
        self.wave = wave
        norm = scipy.stats.norm(mu, sigma)
        self.qeff = A * norm.pdf(wave) / norm.pdf(mu)