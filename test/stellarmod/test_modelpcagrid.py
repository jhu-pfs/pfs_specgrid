from test.test_base import TestBase
from pfsspec.stellarmod.boszpcagrid import BoszPCAGrid

class TestModelPCAGrid(TestBase):
    def get_test_grid(self, args):
        grid = BoszPCAGrid()
        grid.load('/scratch/ceph/dobos/temp/test041/spectra.h5')
        grid.init_from_args(args)
        return grid

    def test_interpolate_model_rbf(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.interpolate_model_rbf(Fe_H=-1.2, T_eff=4125, log_g=4.3)

        pass