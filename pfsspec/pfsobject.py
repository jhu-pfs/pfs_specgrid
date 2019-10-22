import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gzip, pickle
import h5py

from pfsspec.constants import Constants

class PfsObject():
    def __init__(self, orig=None):
        self.file = None
        self.filename = None
        self.fileformat = None
        self.filedata = None

    def get_format(self, filename):
        fn, ext = os.path.splitext(filename)
        if ext == '.h5':
            return 'h5'
        elif ext == '.npz':
            return 'npz'
        elif ext == '.gz':
            fn, ext = os.path.splitext(fn)
            if ext == '.npy':
                return 'numpy'
            elif ext == '.dat':
                return 'pickle'
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def save(self, filename, format='pickle'):
        logging.info("Saving {} to file {}...".format(type(self).__name__, filename))

        self.filename = filename
        self.fileformat = format

        if self.fileformat in ['numpy', 'pickle']:
            with gzip.open(self.filename, 'wb') as f:
                self.file = f
                self.save_items()
                self.file = None
        elif self.fileformat == 'npz':
            self.filedata = {}
            self.save_items()
            np.savez(filename, **self.filedata)
            self.filedata = None
        elif self.fileformat == 'h5':
            self.save_items()
        else:
            raise NotImplementedError()

        logging.info("Saved {} to file {}.".format(type(self).__name__, filename))

    def save_items(self):
        raise NotImplementedError()

    def save_item(self, name, item):
        logging.debug('Saving item {} with type {}'.format(name, type(item).__name__))

        if self.fileformat == 'numpy':
            np.save(self.file, item, allow_pickle=True)
        elif self.fileformat == 'pickle':
            pickle.dump(item, self.file, protocol=4)
        elif self.fileformat == 'npz':
            self.filedata[name] = item
        elif self.fileformat == 'h5':
            if item is None:
                # Do not save if value is None
                pass
            elif isinstance(item, pd.DataFrame):
                item.to_hdf(self.filename, name, mode='a')
            elif isinstance(item, np.ndarray):
                with h5py.File(self.filename, 'a') as f:
                    if name in f:
                        del f[name]
                    f.create_dataset(name, data=item)
            else:
                raise NotImplementedError('Unsupported type: {}'.format(type(item).__name__))
        else:
            raise NotImplementedError()

    def load(self, filename, format=None):
        logging.info("Loading {} from file {}...".format(type(self).__name__, filename))

        if format is None:
            format = self.get_format(filename)

        self.filename = filename
        self.fileformat = format

        if self.fileformat in ['numpy', 'pickle']:
            with gzip.open(self.filename, 'rb') as f:
                self.file = f
                self.load_items()
                self.file = None
        if self.fileformat == 'npz':
            self.filedata = np.load(self.filename, allow_pickle=True)
            logging.debug('Found items: {}'.format([k for k in self.filedata]))
            self.load_items()
            self.filedata = None
        elif self.fileformat == 'h5':
            self.load_items()
        else:
            raise NotImplementedError()

        logging.info("Loaded {} from file {}.".format(type(self).__name__, filename))

    def load_items(self):
        raise NotImplementedError()

    def load_item(self, name, type):
        logging.debug('Loading item {} with type {}'.format(name, type.__name__))

        if self.fileformat == 'numpy':
            data = np.load(self.file, allow_pickle=True)
            return self.load_none_array(data)
        elif self.fileformat == 'pickle':
            return pickle.load(self.file)
        elif self.fileformat == 'npz':
            if name in self.filedata:
                data = self.filedata[name]
                return self.load_none_array(data)
            else:
                return None
        elif self.fileformat == 'h5':
            if type == pd.DataFrame:
                return pd.read_hdf(self.filename, name)
            elif type == np.ndarray:
                with h5py.File(self.filename, 'r') as f:
                    if name in f.keys():
                        return f[name][:]
                    else:
                        return None
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def load_none_array(self, data):
        if isinstance(data, np.ndarray) and data.shape == ():
            return None
        else:
            return data

    def plot_getax(self, ax=None, xlim=Constants.DEFAULT_PLOT_WAVE_RANGE, ylim=None):
        if ax is None:
            ax = plt.gca()
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        return ax

    def plot(self, ax=None, xlim=Constants.DEFAULT_PLOT_WAVE_RANGE, labels=True):
        raise NotImplementedError()