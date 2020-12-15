import os
import numpy as np

from test.test_base import TestBase
from pfsspec.stellarmod.boszmodelgrid import BoszModelGrid

class TestBoszModelGrid(TestBase):
    def get_grid(self):
        fn = os.path.join(self.PFSSPEC_DATA_PATH, 'import/stellar/grid/bosz_50000/spectra.h5')

        grid = BoszModelGrid()
        grid.preload_arrays = False
        grid.load(fn, format='h5')

        return grid

    def test_get_slice_rbf(self):
        grid = self.get_grid()

        wl_idx = np.digitize([6565], grid.wave)
        flux, cont, axes = grid.get_slice_rbf(s=wl_idx, O_M=0, C_M=0)
        self.assertEqual((3, 8752), flux.xi.shape)
        self.assertEqual((8752,), flux.nodes.shape)

        wl_idx = np.digitize([6565, 6575], grid.wave)
        flux, cont, axes = grid.get_slice_rbf(s=slice(wl_idx[0], wl_idx[1]), O_M=0, C_M=0)
        self.assertEqual((3, 8752), flux.xi.shape)
        self.assertEqual((8752, 152), flux.nodes.shape)

        pass

    def test_get_slice_rbn_nopadding(self):
        grid = self.get_grid()

        wl_idx = np.digitize([6565], grid.wave)
        flux, cont, axes = grid.get_slice_rbf(padding=False, s=wl_idx, O_M=0, C_M=0)
        self.assertEqual((3, 6336), flux.xi.shape)
        self.assertEqual((6336,), flux.nodes.shape)