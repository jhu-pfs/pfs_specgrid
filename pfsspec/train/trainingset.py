import numpy as np
import pandas as pd
import pickle

class TrainingSet():
    def __init__(self):
        self.params = None
        self.wave = None
        self.flux = None

    def init_storage(self, wcount, scount):
        self.wave = np.empty(wcount)
        self.flux = np.empty((scount, wcount))

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.params, f)
            pickle.dump(self.wave)
            pickle.dump(self.flux)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.params = pickle.load(f)
            self.wave = pickle.load(f)
            self.flux = pickle.load(f)
