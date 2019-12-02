import logging

from pfsspec.parallel import SmartParallel
from pfsspec.data.spectrumreader import SpectrumReader

class ModelGridSpectrumReader(SpectrumReader):
    class EnumParamsGenerator():
        def __init__(self, grid=None, max=None):
            self.grid = grid
            self.limits = [grid.params[p].values.shape[0] for p in grid.params]
            self.i = 0
            self.max = max
            self.current = [0 for p in grid.params]
            self.stop = False

        def __iter__(self):
            return self

        def __len__(self):
            if self.max is not None:
                return self.max
            else:
                s = 1
                for p in self.grid.params:
                    s *= self.grid.params[p].values.shape[0]
                return s

        def __next__(self):
            if self.stop:
                raise StopIteration()
            else:
                ci = self.current.copy()
                cr = {p: self.grid.params[p].values[self.current[i]] for i, p in enumerate(self.grid.params)}

                k = len(self.limits) - 1
                while k >= 0:
                    self.current[k] += 1
                    if self.current[k] == self.limits[k]:
                        self.current[k] = 0
                        k -= 1
                        continue
                    else:
                        break

                if k == -1 or self.i == self.max:
                    self.stop = True

                self.i += 1
                return ci, cr

    def __init__(self, grid, parallel=True, max=None):
        super(ModelGridSpectrumReader, self).__init__()
        self.grid = grid
        self.parallel = parallel
        self.max = max

    def read_grid(self, stop=None):
        shape = self.grid.get_flux_shape()
        logging.info("Loading grid with flux grid shape {}".format(shape))
        if self.max is not None:
            logging.info("Loading will stop after {} spectra".format(self.max))

        g = ModelGridSpectrumReader.EnumParamsGenerator(self.grid, max=self.max)
        with SmartParallel(verbose=True, parallel=self.parallel) as p:
            for r in p.map(self.process_item, g):
                if r is not None:
                    index, params, spec = r
                    self.grid.set_flux_idx(index, spec.flux, spec.cont)

        logging.info("Grid loaded with flux shape {}".format(shape))

    def process_item(self, i):
        raise NotImplementedError()

    def read_files(self, files, stop=None):
        shape = self.grid.get_flux_shape()
        logging.info("Loading grid with flux grid shape {}".format(shape))
        if self.max is not None:
            logging.info("Loading will stop after {} spectra".format(self.max))

        k = 0
        with SmartParallel(verbose=True, parallel=self.parallel) as p:
            for r in p.map(self.process_file, files):
                if r is not None:
                    index, params, spec = r
                    self.grid.set_flux_idx(index, spec.flux, spec.cont)
                k += 1

        logging.info('{} files loaded.'.format(k))

    def process_file(self, file):
        raise NotImplementedError()