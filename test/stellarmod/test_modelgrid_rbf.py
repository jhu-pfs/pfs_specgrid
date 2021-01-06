import os
from test.test_base import TestBase

from pfsspec.data.rbfgrid import RbfGrid
from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.stellarmod.bosz import Bosz

class TestModelGrid_Rbf(TestBase):
    def get_test_grid(self, args):
        #file = os.path.join(self.PFSSPEC_DATA_PATH, '/scratch/ceph/dobos/temp/test072/spectra.h5')
        file = '/scratch/ceph/dobos/data/pfsspec/import/stellar/grid/bosz_5000_nb515_rbf/spectra.h5'
        grid = ModelGrid(Bosz(), RbfGrid)
        grid.load(file, format='h5')
        grid.init_from_args(args)

        return grid

    def test_init_from_args(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual(slice(None), grid.wave_slice)

    def test_get_axes(self):
        args = {}
        grid = self.get_test_grid(args)
        axes = grid.get_axes()
        self.assertEqual(3, len(axes))

    def test_get_nearest_model(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(Fe_H=0., T_eff=4500, log_g=4, C_M=0, O_M=0)

    def test_interpolate_model_rbf(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.interpolate_model_rbf(Fe_H=-1.2, T_eff=4125, log_g=4.3)