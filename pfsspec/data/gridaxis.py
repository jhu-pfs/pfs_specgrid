import numpy as np
from scipy.interpolate import interp1d

class GridAxis():
    def __init__(self, name, values=None, orig=None):

        if isinstance(orig, GridAxis):
            self.name = orig.name
            self.values = orig.values if values is None else values
            self.index = orig.index
            self.min = orig.min
            self.max = orig.max
            self.ip_to_index = orig.ip_to_index
            self.ip_to_value = orig.ip_to_value
        else:
            self.name = name
            self.values = values
            self.index = None
            self.min = None
            self.max = None
            self.ip_to_index = None
            self.ip_to_value = None

    def build_index(self):
        # NOTE: assume one dimension here
        self.index = {v: i[0] for i, v in np.ndenumerate(self.values)}
        self.min = np.min(self.values)
        self.max = np.max(self.values)
        if self.values.shape[0] > 1:
            self.ip_to_index = interp1d(self.values, np.arange(self.values.shape[0]), fill_value='extrapolate')
            self.ip_to_value = interp1d(np.arange(self.values.shape[0]), self.values, fill_value='extrapolate')
        else:
            self.ip_to_index = None
            self.ip_to_value = None

    def get_index(self, value):
        return self.index[value]

    def get_nearest_index(self, value):
        return np.abs(self.values - value).argmin()

    def get_nearby_indexes(self, value):
        i1 = self.get_nearest_index(value)
        if value < self.values[i1]:
            i1, i2 = i1 - 1, i1
        else:
            i1, i2 = i1, i1 + 1

        # Verify if indexes inside bounds
        if i1 < 0 or i2 >= self.values.shape[0]:
            return None

