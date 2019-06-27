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
        logging.info("  params: ", self.params.shape)
        logging.info("  wave:   ", self.wave.shape)
        logging.info("  flux:   ", self.flux.shape)
        logging.info("  columns:", self.params.columns)

    def get_split_index(self, split_value):
        split_index = int(self.flux.shape[0] * split_value)
        return split_index

    def create_split(self):
        return Dataset(), Dataset()

    def split(self, split_value):
        split_index = self.get_split_index(split_value)
        a_range = [i for i in range(0, split_index)]
        b_range = [i for i in range(split_index, self.flux.shape[0])]
        a, b = self.create_split()

        a.params = self.params.iloc[a_range]
        a.wave = self.wave
        a.flux = self.flux[a_range]

        b.params = self.params.iloc[b_range]
        b.wave = self.wave
        b.flux = self.flux[b_range]

        return split_index, a, b
