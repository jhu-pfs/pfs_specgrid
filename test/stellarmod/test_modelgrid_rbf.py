import os
import logging
from tqdm import tqdm
import numpy as np
from test.test_base import TestBase

from pfsspec.data.rbfgrid import RbfGrid
from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.stellarmod.bosz import Bosz

class TestModelGrid_Rbf(TestBase):
    def get_test_grid(self, args):
        #file = os.path.join(self.PFSSPEC_DATA_PATH, '/scratch/ceph/dobos/temp/test072/spectra.h5')
        file = '/scratch/ceph/dobos/data/pfsspec/import/stellar/rbf/bosz_5000_full/fitrbf_3/spectra.h5'
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
        self.assertIsNotNone(spec)
        
    def test_interpolate_model_rbf(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.interpolate_model_rbf(Fe_H=-1.2, T_eff=4125, log_g=4.3, O_M=0.1, C_M=-0.1)
        self.assertIsNotNone(spec)

    def test_get_continuum_parameters(self):
        args = {}
        grid = self.get_test_grid(args)
        params = grid.get_continuum_parameters(Fe_H=-1.2, T_eff=4125, log_g=4.3, O_M=0.1, C_M=-0.1)

    def test_get_continuum_parameters_at(self):
        args = {}
        grid = self.get_test_grid(args)
        idx = grid.get_index(Fe_H=-1.2, T_eff=4125, log_g=4.3, O_M=0.1, C_M=-0.1)
        params = grid.get_continuum_parameters_at(idx)

    def test_interpolate_model_rbf_performance(self):
        # logging.basicConfig(level=logging.DEBUG)

        args = {}
        grid = self.get_test_grid(args)
        rbf_raw = grid.grid

        for i in tqdm(range(1000000)):
            # Draw a value for each parameters
            # Note that these are indices and not physical values!
            params = {}
            for k in rbf_raw.axes:
                v = rbf_raw.axes['Fe_H'].values
                params[k] = np.random.uniform(v[0], v[-1])
            
            rbf_raw.get_values(names=['blended_0', 'blended_1', 'blended_2'], **params)