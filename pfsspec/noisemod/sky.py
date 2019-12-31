import numpy as np
from scipy.interpolate import interp1d

from pfsspec.constants import Constants
from pfsspec.data.grid import Grid

class Sky(Grid):
    def __init__(self):
        super(Sky, self).__init__()
        self.ref_exp_count = 1
        self.ref_exp_time = 450

        self.wave = None

    def init_params(self):
        self.init_param('fa')
        self.init_param('za')

    def init_data(self):
        self.init_data_item('counts')        # Photon count originating from the sky
        self.init_data_item('conv')          # Conversion function including atmosphere

    def allocate_data(self):
        self.allocate_data_item('counts', self.wave.shape)  # Photon count originating from the sky
        self.allocate_data_item('conv', self.wave.shape)  # Conversion function including atmosphere

    def save_items(self):
        self.save_item('wave', self.wave)
        super(Sky, self).save_items()

    def load_items(self, s=None):
        self.wave = self.load_item('wave', np.ndarray)
        self.init_data()
        super(Sky, self).load_items(s=s)

    def get_counts(self, exp_count, exp_time, za, fa):
        counts, _ = self.interpolate_data_item_linear('counts', fa=fa, za=za)
        counts = counts * exp_count / self.ref_exp_count
        counts = counts * exp_time / self.ref_exp_time
        return counts
