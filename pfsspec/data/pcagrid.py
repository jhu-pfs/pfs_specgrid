import logging
import numpy as np

from pfsspec.data.gridaxis import GridAxis

# TODO: this should be independent of the Grid class, not using anything from there
class PCAGrid(PfsObject):
    # Implements a grid that supports interpolation based on continuum fits
    # and eigenvalues

    def __init__(self, orig=None):
        super(PCAGrid, self).__init__(orig=orig)

        if isinstance(orig, PCAGrid):
            self.truncate = orig.truncate

            self.eigs = orig.eigs
            self.eigv = orig.eigv
        else:
            self.truncate = None

            self.eigs = None
            self.eigv = None

    def init_values(self):
        super(PCAGrid, self).init_values()

        self.init_value('coeffs')       # Principal components

    def allocate_values(self):
        super(PCAGrid, self).allocate_values()

        self.allocate_value('coeffs')

    def save_items(self):
        self.save_item('eigs', self.eigs)
        self.save_item('eigv', self.eigv)

        super(PCAGrid, self).save_items()

    def load_items(self, s=None):
        self.eigs = self.load_item('eigs', np.ndarray)
        self.eigv = self.load_item('eigv', np.ndarray)
        self.init_values()

        super(PCAGrid, self).load_items(s=s)