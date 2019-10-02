import logging

from pfsspec.parallel import srl_map, prll_map
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
        self.grid.build_index()

        logging.info("Loading grid with flux grid shape {}".format(self.grid.flux.shape))
        if self.max is not None:
            logging.info("Loading will stop after {} spectra".format(self.max))

        g = ModelGridSpectrumReader.EnumParamsGenerator(self.grid, max=self.max)
        if self.parallel:
            res = prll_map(self.process_item, g, verbose=True)
        else:
            res = srl_map(self.process_item, g, verbose=True)

        for r in res:
            if r is not None:
                index, params, spec = r
                self.grid.set_flux_idx(index, spec.flux, spec.cont)

        logging.info("Grid loaded with flux grid shape {}".format(self.grid.flux.shape))

    def process_item(self, i):
        raise NotImplementedError()

    def read_files(self, files, stop=None):
        self.grid.build_index()

        logging.info("Loading grid with flux grid shape {}".format(self.grid.flux.shape))
        if self.max is not None:
            logging.info("Loading will stop after {} spectra".format(self.max))

        if self.parallel:
            res = prll_map(self.process_file, files, verbose=True)
        else:
            res = srl_map(self.process_file, files, verbose=True)

        k = 0
        for r in res:
            if r is not None:
                k += 1
                index, params, spec = r
                self.grid.set_flux_idx(index, spec.flux, spec.cont)

        logging.info('{} files loaded.'.format(k))

    def process_file(self, file):
        raise NotImplementedError()