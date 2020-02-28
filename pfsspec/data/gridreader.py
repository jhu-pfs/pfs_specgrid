import logging

from pfsspec.parallel import SmartParallel

class GridReader():
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
        self.grid = grid
        self.parallel = parallel
        self.max = max

    def read_grid(self):
        self.grid.init_data()
        self.grid.allocate_data()

        # Iterate over the grid points and call a function for each
        logging.info("Loading {}.".format(type(self.grid).__name__))
        if self.max is not None:
            logging.info("Loading will stop after {} items.".format(self.max))

        g = GridReader.EnumParamsGenerator(self.grid, max=self.max)
        with SmartParallel(verbose=True, parallel=self.parallel) as p:
            for res in p.map(self.process_item, g):
                self.store_item(res)

        logging.info("Grid loaded.")

    def read_files(self, files):
        # Iterate over a list of files and call a function for each
        logging.info("Loading {}".format(type(self.grid).__name__))
        if self.max is not None:
            logging.info("Loading will stop after {} spectra".format(self.max))
            files = files[:min(self.max, len(files))]

        k = 0
        with SmartParallel(verbose=True, parallel=self.parallel) as p:
            for res in p.map(self.process_file, files):
                self.store_item(res)
                k += 1

        logging.info('{} files loaded.'.format(k))

    def process_item(self, i):
        raise NotImplementedError()

    def process_file(self, file):
        raise NotImplementedError()

    def store_item(self, res):
        raise NotImplementedError()