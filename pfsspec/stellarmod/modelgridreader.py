import logging

from pfsspec.data.gridreader import GridReader

class ModelGridReader(GridReader):
    def store_item(self, res):
        if res is not None:
            # This is read specific!
            index, params, spec = res
            self.grid.set_flux_idx(index, spec.flux, spec.cont)