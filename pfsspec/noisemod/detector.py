import numpy as np
from scipy.interpolate import interp1d

from pfsspec.pfsobject import PfsObject

class Detector(PfsObject):
    def __init__(self):
        super(Detector, self).__init__()
        self.wave = None
        self.dark = None
        self.readout = None
        self.sample_factor = None       # multiply noise by this (used with subtraction error and stray light only)

    def save_items(self):
        self.save_item('wave', self.wave)
        self.save_item('dark', self.dark)
        self.save_item('readout', self.readout)
        self.save_item('sample_factor', self.sample_factor)

    def load_items(self, s=None):
        self.wave = self.load_item('wave', np.ndarray)
        self.dark = self.load_item('dark', np.ndarray)
        self.readout = self.load_item('readout', np.ndarray)
        self.sample_factor = self.load_item('sample_factor', np.ndarray)

    def get_noise(self, exp_count, exp_time):
        # These data vectors already contain the sampling factor
        # Units of counts at this point is e^2 / pix
        noise = self.dark
        noise += self.readout
        return noise