import logging
import numpy as np

from pfsspec.parallel import SmartParallel
from pfsspec.pfsobject import PfsObject

class GridReader(PfsObject):
    class EnumAxesGenerator():
        def __init__(self, grid=None, max=None, resume=False):
            self.grid = grid
            self.limits = [grid.axes[p].values.shape[0] for p in grid.axes]
            self.i = 0
            self.max = max
            self.current = [0 for p in grid.axes]
            self.stop = False
            self.resume = resume

        def __iter__(self):
            return self

        def __len__(self):
            if self.max is not None:
                return self.max
            else:
                s = 1
                for p in self.grid.axes:
                    s *= self.grid.axes[p].values.shape[0]
                return s

        def __next__(self):
            if self.stop:
                raise StopIteration()
            else:
                # If in continue mode, we need to skip item that are already in the grid
                while True:
                    ci = self.current.copy()
                    cr = {p: self.grid.axes[p].values[self.current[i]] for i, p in enumerate(self.grid.axes)}

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

                    if self.resume:
                        # Test if item is already in the grid
                        mask = None
                        for name in self.grid.value_indexes:
                            m = self.grid.has_value_at(name, ci)
                            if mask is None:
                                mask = m
                            else:
                                mask |= m

                        if not mask:
                            # Item does not exist
                            break
                    else:
                        break

                return ci, cr

    def __init__(self, grid, orig=None, parallel=True, threads=None, max=None, resume=False):
        super(GridReader, self).__init__(orig=orig)

        self.grid = grid
        self.parallel = parallel
        self.threads = threads
        self.max = max
        self.resume = resume

    def read_grid(self, resume=False):
        if not resume:
            self.grid.init_values()
            self.grid.allocate_values()

        # Iterate over the grid points and call a function for each
        self.logger.info("Reading grid {}.".format(type(self.grid).__name__))
        if self.max is not None:
            self.logger.info("Reading grid will stop after {} items.".format(self.max))

        g = GridReader.EnumAxesGenerator(self.grid, max=self.max, resume=resume)
        with SmartParallel(verbose=True, parallel=self.parallel, threads=self.threads) as p:
            for res in p.map(self.process_item, g):
                self.store_item(res)

        self.logger.info("Grid loaded.")

    def read_files(self, files, resume=False):
        if resume:
            # TODO: get ids from existing params data frame and remove the ones
            #       from the list that have already been imported
            raise NotImplementedError()

        # Iterate over a list of files and call a function for each
        self.logger.info("Loading {}".format(type(self.grid).__name__))
        if self.max is not None:
            self.logger.info("Loading will stop after {} spectra".format(self.max))
            files = files[:min(self.max, len(files))]

        k = 0
        with SmartParallel(verbose=True, parallel=self.parallel, threads=self.threads) as p:
            for res in p.map(self.process_file, files):
                self.store_item(res)
                k += 1

        self.logger.info('{} files loaded.'.format(k))

    def process_item(self, i):
        raise NotImplementedError()

    def process_file(self, file):
        raise NotImplementedError()

    def store_item(self, res):
        raise NotImplementedError()