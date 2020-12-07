import logging

from pfsspec.data.gridreader import GridReader

class ModelGridReader(GridReader):
    def __init__(self, grid, reader=None, parallel=True, threads=None, max=None):
        super(ModelGridReader, self).__init__(grid, parallel=parallel, threads=threads, max=max)

        self.reader = reader