import numpy as np
import pandas as pd
import pickle

class DataSet():
    def __init__(self):
        self.params = None
        self.spectra = None

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.params, f)
            pickle.dump(self.spectra, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.params = pickle.load(f)
            self.spectra = pickle.load(f)