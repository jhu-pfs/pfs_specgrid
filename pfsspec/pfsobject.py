import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gzip, pickle
import h5py

from pfsspec.constants import Constants
import pfsspec.util as util

class PfsObject():
    def __init__(self, orig=None):
        self.file = None
        self.filename = None
        self.fileformat = None
        self.filedata = None

    def get_arg(self, name, old_value, args=None):
        args = args or self.args
        return util.get_arg(name, old_value, args)

    def is_arg(self, name, args=None):
        args = args or self.args
        return util.is_arg(name, args)

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

    def save(self, filename, format='pickle', save_items_func=None):
        logging.info("Saving {} to file {}...".format(type(self).__name__, filename))

        save_items_func = save_items_func or self.save_items

        self.filename = filename
        self.fileformat = format

        if self.fileformat in ['numpy', 'pickle']:
            with gzip.open(self.filename, 'wb') as f:
                self.file = f
                save_items_func()
                self.file = None
        elif self.fileformat == 'npz':
            self.filedata = {}
            save_items_func()
            np.savez(filename, **self.filedata)
            self.filedata = None
        elif self.fileformat == 'h5':
            save_items_func()
        else:
            raise NotImplementedError()

        logging.info("Saved {} to file {}.".format(type(self).__name__, filename))

    def save_items(self):
        raise NotImplementedError()

    def allocate_item(self, name, shape, dtype=np.float):
        if self.fileformat != 'h5':
            raise NotImplementedError()

        if self.fileformat == 'h5':
            with h5py.File(self.filename, 'a') as f:
                if name not in f.keys():
                    chunks = self.get_chunks(shape)
                    f.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks)

    def save_item(self, name, item, slice=None):
        logging.debug('Saving item {} with type {}'.format(name, type(item).__name__))

        if self.fileformat != 'h5' and slice is not None:
            raise NotImplementedError()

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
                    if slice is not None:
                        # in-place update
                        f[name][slice] = item
                    else:
                        if name in f.keys():
                            del f[name]
                        chunks = self.get_chunks(item.shape)
                        if chunks is not None:
                            f.create_dataset(name, data=item, chunks=chunks)
                            logging.debug('Saving item {} with chunks {}'.format(name, chunks))
                        else:
                            f.create_dataset(name, data=item)
            else:
                raise NotImplementedError('Unsupported type: {}'.format(type(item).__name__))
        else:
            raise NotImplementedError()

    def get_chunks(self, shape):
        size = 1
        for s in shape:
            size *= s

        if size > 0x100000:
            chunks = list(shape)
            for i in range(len(chunks)):
                if chunks[i] <= 0x80:  # 128
                    chunks[i] = 1
                elif chunks[i] > 0x400:  # 1k
                    pass
                else:
                    chunks[i] = 64
            return tuple(chunks)
        else:
            return None

    def load(self, filename, slice=None, format=None, load_items_func=None):
        logging.info("Loading {} from file {} with slices {}...".format(type(self).__name__, filename, slice))

        load_items_func = load_items_func or self.load_items
        format = format or self.get_format(filename)

        self.filename = filename
        self.fileformat = format

        if self.fileformat in ['numpy', 'pickle']:
            with gzip.open(self.filename, 'rb') as f:
                self.file = f
                self.load_items(slice=slice)
                self.file = None
        elif self.fileformat == 'npz':
            self.filedata = np.load(self.filename, allow_pickle=True)
            logging.debug('Found items: {}'.format([k for k in self.filedata]))
            self.load_items(slice=slice)
            self.filedata = None
        elif self.fileformat == 'h5':
            self.load_items(slice=slice)
        else:
            raise NotImplementedError()

        logging.info("Loaded {} from file {}.".format(type(self).__name__, filename))

    def load_items(self, slice=None):
        raise NotImplementedError()

    def load_item(self, name, type, slice=None):
        logging.debug('Loading item {} with type {} and slices {}'.format(name, type.__name__, slice))

        if self.fileformat == 'numpy':
            data = np.load(self.file, allow_pickle=True)
            data = self.load_none_array(data)
            if data is not None and slice is not None:
                return data[slice]
            else:
                return data
        elif self.fileformat == 'pickle':
            data = pickle.load(self.file)
            if data is not None and slice is not None:
                return data[slice]
            else:
                return data
        elif self.fileformat == 'npz':
            if name in self.filedata:
                data = self.filedata[name]
                data = self.load_none_array(data)
                if data is not None and slice is not None:
                    return data[slice]
                else:
                    return data
            else:
                return None
        elif self.fileformat == 'h5':
            if type == pd.DataFrame:
                if slice is not None:
                    return pd.read_hdf(self.filename, name, start=slice.start, stop=slice.stop)
                else:
                    return pd.read_hdf(self.filename, name)
            elif type == np.ndarray:
                with h5py.File(self.filename, 'r') as f:
                    if name in f.keys():
                        #a = np.empty(f[name].shape, dtype=f[name].dtype)
                        #if slice is not None:
                        #    f[name].read_direct(a, source_sel=slice, dest_sel=slice)
                        #else:
                        #    f[name].read_direct(a)
                        #return a
                        if slice is not None:
                            return f[name][slice]
                        else:
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