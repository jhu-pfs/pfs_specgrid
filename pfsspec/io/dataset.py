import logging
import numpy as np
import pandas as pd
import pickle

class Dataset():
    def __init__(self):
        self.params = None
        self.wave = None
        self.flux = None

    def init_storage(self, wcount, scount):
        self.wave = np.empty(wcount)
        self.flux = np.empty((scount, wcount))

    def save(self, filename):
        logging.info("Saving dataset to file {}...".format(filename))

        with open(filename, 'wb') as f:
            pickle.dump(self.params, f)
            np.savez(f, wave=self.wave, flux=self.flux)

        logging.info("Saved dataset.")

    def load(self, filename):
        logging.info("Loading dataset from file {}...".format(filename))

        with open(filename, 'rb') as f:
            self.params = pickle.load(f)
            data = np.load(f)
            self.wave = data['wave']
            self.flux = data['flux']

        logging.info("Loaded dataset with shapes:")
        logging.info("  params:  {}".format(self.params.shape))
        logging.info("  wave:    {}".format(self.wave.shape))
        logging.info("  flux:    {}".format(self.flux.shape))
        logging.info("  columns: {}".format(self.params.columns))

    def get_split_index(self, split_value):
        split_index = int((1 - split_value) *  self.flux.shape[0])
        return split_index

    def split(self, split_value):
        a = Dataset()
        b = Dataset()

        split_index = self.get_split_index(split_value)
        a_range = [i for i in range(0, split_index)]
        b_range = [i for i in range(split_index, self.flux.shape[0])]

        a.params = self.params.iloc[a_range]
        a.wave = self.wave
        a.flux = self.flux[a_range]

        b.params = self.params.iloc[b_range]
        b.wave = self.wave
        b.flux = self.flux[b_range]

        return split_index, a, b

    def filter(self, f):
        a = Dataset()
        b = Dataset()

        a.params = self.params.ix[f]
        a.wave = self.wave
        a.flux = self.flux[f]

        b.params = self.params.ix[~f]
        b.wave = self.wave
        b.flux = self.flux[~f]

        return a, b

    def merge(self, b):
        a = Dataset()

        a.params = pd.concat([self.params, b.params], axis=0)
        a.wave = self.wave
        a.flux = np.concatenate([self.flux, b.flux], axis=0)

        return a
