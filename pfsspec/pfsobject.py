import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gzip, pickle
import h5py
import json
import multiprocessing
import numbers
from collections import Iterable

from pfsspec.constants import Constants
import pfsspec.util as util

class PfsObject():
    def __setstate__(self, state):
        self.__dict__ = state

        # When a child process is starting use multiprocessing logger
        # if multiprocessing.current_process()._inheriting:
        #    self.logger = multiprocessing.get_logger()

    def __init__(self, orig=None):
        self.jsonomit = set([
            'jsonomit',
            'file',
            'filename',
            'fileformat',
            'filedata,'
            'hdf5file'
        ])

        self.logger = logging.getLogger()

        if isinstance(orig, PfsObject):
            self.file = None

            self.filename = orig.filename
            self.fileformat = orig.fileformat
            self.filedata = orig.filedata
        else:
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
        self.logger.info("Saving {} to file {}...".format(type(self).__name__, filename))

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

        self.logger.info("Saved {} to file {}.".format(type(self).__name__, filename))

    def save_items(self):
        raise NotImplementedError()

    def allocate_item(self, name, shape, dtype=np.float):
        if self.fileformat != 'h5':
            raise NotImplementedError()

        if self.fileformat == 'h5':
            with h5py.File(self.filename, 'a') as f:
                if name not in f.keys():
                    chunks = self.get_chunks(name, shape, None)
                    return f.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks)

    def save_item(self, name, item, s=None, min_string_length=None):
        self.logger.debug('Saving item {} with type {}'.format(name, type(item).__name__))

        if self.fileformat != 'h5' and s is not None:
            raise NotImplementedError()

        if self.fileformat == 'numpy':
            np.save(self.file, item, allow_pickle=True)
        elif self.fileformat == 'pickle':
            pickle.dump(item, self.file, protocol=4)
        elif self.fileformat == 'npz':
            self.filedata[name] = item
        elif self.fileformat == 'h5':
            self.save_item_hdf5(name, item, s=s, min_string_length=min_string_length)
        else:
            raise NotImplementedError()

    def save_item_hdf5(self, name, item, s=None, min_string_length=None):
        def open_hdf5():
            if self.file is None:
                return h5py.File(self.filename, 'a')
            else:
                return self.file

        def close_hdf5(f):
            if self.file is None:
                f.close()
            else:
                pass

        f = open_hdf5()
        g, name = self.get_hdf5_group(f, name, create=True)

        if item is None:
            # Do not save if value is None
            pass
        elif isinstance(item, pd.DataFrame):
            # This is a little bit different for DataFrames when s is defined. When slicing,
            # arrays are update in place, an operation not supported by HDFStore
            # TODO: this issue could be solved by first removing the matching rows, then
            #       appending them.

            if s is None:
                item.to_hdf(self.filename, name, mode='a', min_itemsize=min_string_length)
            else:
                item.to_hdf(self.filename, name, mode='a', format='table', append=True, min_itemsize=min_string_length)
        elif isinstance(item, np.ndarray):
            if s is not None:
                # in-place update
                g[name][s] = item
            else:
                if name in g.keys():
                    del g[name]
                chunks = self.get_chunks(name, item.shape, s=s)
                if chunks is not None:
                    g.create_dataset(name, data=item, chunks=chunks)
                    self.logger.debug('Saving item {} with chunks {}'.format(name, chunks))
                else:
                    g.create_dataset(name, data=item)
        elif isinstance(item, numbers.Number):
            # TODO: now storing a single number in a separate dataset. Change this
            #       to store as an attribute. Keeping it now for compatibility.
            if name in g.keys():
                del g[name]
            g.create_dataset(name, data=item)
        elif isinstance(item, str):
            g.attrs[name] = item
        else:
            raise NotImplementedError('Unsupported type: {}'.format(type(item).__name__))

        close_hdf5(f)

    def get_chunks(self, name, shape, s=None):
        needchunk = False
        size = 1
        for s in shape:
            needchunk |= (s > 0x80)
            size *= s

        # TODO: Review this logic here. For some reason, tiny dimensions are chunked
        #       While longer arrays aren't. This function is definitely called during
        #       model grid import when chunking is important.

        if needchunk and size > 0x100000:
            chunks = list(shape)
            for i in range(len(chunks)):
                if chunks[i] <= 0x80:  # 128
                    chunks[i] = 1
                elif chunks[i] > 0x10000:  # 64k
                    chunks[i] = 0x1000     # 4k 
                elif chunks[i] > 0x400:  # 1k
                    pass
                else:
                    chunks[i] = 64
            return tuple(chunks)
        else:
            return None

    def load(self, filename, s=None, format=None, load_items_func=None):
        self.logger.info("Loading {} from file {} with slices {}...".format(type(self).__name__, filename, slice))

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
            self.logger.debug('Found items: {}'.format([k for k in self.filedata]))
            self.load_items(s=s)
            self.filedata = None
        elif self.fileformat == 'h5':
            self.load_items(s=s)
        else:
            raise NotImplementedError()

        self.logger.info("Loaded {} from file {}.".format(type(self).__name__, filename))

    def load_items(self, s=None):
        raise NotImplementedError()

    def load_item(self, name, type, s=None):
        # self.logger.debug('Loading item {} with type {} and slices {}'.format(name, type.__name__, s))

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
            return self.load_item_hdf5(name, type, s=s)
        else:
            raise NotImplementedError()

    def get_hdf5_group(self, f, name, create=False):
        # If name contains /, split and dig down in the hierarchy
        parts = name.split('/')
        g = f
        for part in parts[:-1]:
            if create and part not in g:
                g = g.create_group(part)
            elif part not in g:
                return None, None
            else:
                g = g[part]
        return g, parts[-1]

    def load_item_hdf5(self, name, type, s=None):
        if type == pd.DataFrame:
            if s is not None:
                return pd.read_hdf(self.filename, name, start=s.start, stop=s.stop)
            else:
                return pd.read_hdf(self.filename, name)
        elif type == np.ndarray:
            with h5py.File(self.filename, 'r') as f:
                g, name = self.get_hdf5_group(f, name, create=False)
                if g is not None and name in g:
                    #a = np.empty(g[name].shape, dtype=g[name].dtype)
                    #if slice is not None:
                    #    g[name].read_direct(a, source_sel=slice, dest_sel=slice)
                    #else:
                    #    g[name].read_direct(a)
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
                                k = len(range(*s[i].indices(g[name].shape[i])))
                                if shape is None:
                                    shape = (k, )
                                else:
                                    shape = shape + (k, )

                        if shape is None:
                            shape = g[name].shape
                        else:
                            shape = shape + g[name].shape[len(s):]

                        if idxshape is None:
                            data = g[name][s]
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
                                data[idx] = g[name][tuple(ii)]
                    elif s is not None:
                        data = g[name][s]
                    else:
                        data = g[name][:]
                    if not isinstance(data, np.ndarray):
                        data = np.ndarray(data)
                    return data
                else:
                    return None
        elif type == np.float or type == np.int:
            # TODO: rewrite this to use attributes
            with h5py.File(self.filename, 'r') as f:
                g, name = self.get_hdf5_group(f, name, create=False)
                if g is not None and name in g:
                    data = g[name][()]
                    return data
                else:
                    return None
        elif type == str:
            with h5py.File(self.filename, 'r') as f:
                g, name = self.get_hdf5_group(f, name, create=False)
                if g is not None and name in g.attrs:
                    return g.attrs[name]
                else:
                    return None
        else:
            raise NotImplementedError()

    def load_none_array(self, data):
        if isinstance(data, np.ndarray) and data.shape == ():
            return None
        else:
            return data

    def has_item(self, name):
        if self.fileformat == 'h5':
            with h5py.File(self.filename, 'r') as f:
                g, name = self.get_hdf5_group(f, name, create=False)
                return (g is not None) and (name in g)
        else:
            raise NotImplementedError()

    def get_item_shape(self, name):
        if self.fileformat == 'h5':
            with h5py.File(self.filename, 'r') as f:
                g, name = self.get_hdf5_group(f, name, create=False)
                if (g is None) or (name not in g):
                    return None
                else:
                    return g[name].shape
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