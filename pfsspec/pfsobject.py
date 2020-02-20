import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gzip, pickle
import h5py
import json
import numbers
from collections import Iterable

from pfsspec.constants import Constants
import pfsspec.util as util

class PfsObject():
    def __init__(self, orig=None):
        self.jsonomit = set([
            'jsonomit',
            'file',
            'filename',
            'fileformat',
            'filedata'
        ])

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

    def save_json(self, filename):
        d = self.__dict__.copy()
        for k in self.jsonomit:
            if k in d:
                del d[k]
        with open(filename, 'w') as f:
            f.write(json.dumps(d, default=self.save_json_default, indent=4))

    def save_json_default(self, o):
        return None

    def load_json(self, filename):
        with open(filename, 'r') as f:
            d = json.loads(f.read())
            for k in d:
                if k not in self.jsonomit and k in self.__dict__:
                    self.__dict__[k] = d[k]

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

    def save_item(self, name, item, s=None):
        logging.debug('Saving item {} with type {}'.format(name, type(item).__name__))

        if self.fileformat != 'h5' and s is not None:
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
                    if s is not None:
                        # in-place update
                        f[name][s] = item
                    else:
                        if name in f.keys():
                            del f[name]
                        chunks = self.get_chunks(item.shape)
                        if chunks is not None:
                            f.create_dataset(name, data=item, chunks=chunks)
                            logging.debug('Saving item {} with chunks {}'.format(name, chunks))
                        else:
                            f.create_dataset(name, data=item)
            elif isinstance(item, numbers.Number):
                with h5py.File(self.filename, 'a') as f:
                    if name in f.keys():
                        del f[name]
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

    def load(self, filename, s=None, format=None, load_items_func=None):
        logging.info("Loading {} from file {} with slices {}...".format(type(self).__name__, filename, slice))

        load_items_func = load_items_func or self.load_items
        format = format or self.get_format(filename)

        self.filename = filename
        self.fileformat = format

        if self.fileformat in ['numpy', 'pickle']:
            with gzip.open(self.filename, 'rb') as f:
                self.file = f
                self.load_items(s=s)
                self.file = None
        elif self.fileformat == 'npz':
            self.filedata = np.load(self.filename, allow_pickle=True)
            logging.debug('Found items: {}'.format([k for k in self.filedata]))
            self.load_items(s=s)
            self.filedata = None
        elif self.fileformat == 'h5':
            self.load_items(s=s)
        else:
            raise NotImplementedError()

        logging.info("Loaded {} from file {}.".format(type(self).__name__, filename))

    def load_items(self, s=None):
        raise NotImplementedError()

    def load_item(self, name, type, s=None):
        logging.debug('Loading item {} with type {} and slices {}'.format(name, type.__name__, s))

        if self.fileformat == 'numpy':
            data = np.load(self.file, allow_pickle=True)
            data = self.load_none_array(data)
            if data is not None and s is not None:
                return data[s]
            else:
                return data
        elif self.fileformat == 'pickle':
            data = pickle.load(self.file)
            if data is not None and s is not None:
                return data[s]
            else:
                return data
        elif self.fileformat == 'npz':
            if name in self.filedata:
                data = self.filedata[name]
                data = self.load_none_array(data)
                if data is not None and s is not None:
                    return data[s]
                else:
                    return data
            else:
                return None
        elif self.fileformat == 'h5':
            if type == pd.DataFrame:
                if s is not None:
                    return pd.read_hdf(self.filename, name, start=s.start, stop=s.stop)
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

                        # Do some smart indexing magic here because index arrays are not supported by h5py
                        # This is not full fancy indexing!
                        shape = None
                        idxshape = None
                        if isinstance(s, Iterable):
                            for i in range(len(s)):
                                if isinstance(s[i], (np.int32, np.int64)):
                                    if shape is None:
                                        shape = (1,)
                                elif isinstance(s[i], np.ndarray):
                                    if shape is None or shape == (1,):
                                        shape = s[i].shape
                                    if idxshape is not None and idxshape != s[i].shape:
                                        raise Exception('Incompatible shapes')
                                    idxshape = s[i].shape
                                elif isinstance(s[i], slice):
                                    k = len(range(*s[i].indices(f[name].shape[i])))
                                    if shape is None:
                                        shape = (k, )
                                    else:
                                        shape = shape + (k, )

                            if shape is None:
                                shape = f[name].shape
                            else:
                                shape = shape + f[name].shape[len(s):]

                            if idxshape is None:
                                data = f[name][s]
                            else:
                                data = np.empty(shape)
                                for idx in np.ndindex(idxshape):
                                    ii = []
                                    for i in range(len(s)):
                                        if isinstance(s[i], (np.int32, np.int64)):
                                            ii.append(s[i])
                                        elif isinstance(s[i], np.ndarray):
                                            ii.append(s[i][idx])
                                        elif isinstance(s[i], slice):
                                            ii.append(s[i])
                                    data[idx] = f[name][tuple(ii)]
                            return data
                        elif s is not None:
                            return f[name][s]
                        else:
                            return f[name][:]
                    else:
                        return None
            elif type == np.float or type == np.int:
                with h5py.File(self.filename, 'r') as f:
                    if name in f.keys():
                        data = f[name][()]
                        return data
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

    def get_item_shape(self, name):
        if self.fileformat == 'h5':
            with h5py.File(self.filename, 'r') as f:
                if name in f.keys():
                    return f[name].shape
                else:
                    return None
        else:
            raise NotImplementedError()

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