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
            np.savez(f, wave=self.wave, flux=self.flux)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.params = pickle.load(f)
            data = np.load(f, allow_pickle=True)
            self.wave = data['wave']
            self.flux = data['flux']
