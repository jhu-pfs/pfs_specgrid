import numpy as np
from test.test_base import TestBase
from pfsspec.stellarmod.alexcontinuummodel import AlexContinuumModel, AlexContinuumModelTrace

class TestAlexContinuumModel(TestBase):
    def get_test_grid(self, args):
        grid = self.get_bosz_grid()
        grid.init_from_args(args)
        return grid

    def test_get_nearest_model(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(Fe_H=0., T_eff=4500, log_g=4, C_M=0, O_M=0)
        self.assertIsNotNone(spec)

    def test_find_limits(self):
        args = {}
        grid = self.get_test_grid(args)
        trace = AlexContinuumModelTrace()
        model = AlexContinuumModel(trace)
        model.find_limits(grid.wave)
        self.assertEqual(3, len(model.blended_fit_masks))

    def test_fit(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(Fe_H=0., T_eff=3800, log_g=1, C_M=0, O_M=0)
        trace = AlexContinuumModelTrace()
        model = AlexContinuumModel(trace)
        model.init_wave(spec.wave)
        params = model.fit(spec)
        self.assertEqual((36,), params.shape)

    def test_eval(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(Fe_H=0., T_eff=4500, log_g=4, C_M=0, O_M=0)
        trace = AlexContinuumModelTrace()
        model = AlexContinuumModel(trace)
        model.init_wave(spec.wave)
        params = model.fit(spec)
        cont = model.eval(params)

    def test_normalize(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(Fe_H=0., T_eff=4500, log_g=4, C_M=0, O_M=0)
        trace = AlexContinuumModelTrace()
        model = AlexContinuumModel(trace)
        model.init_wave(spec.wave)
        params = model.fit(spec)
        model.normalize(spec, params)

    def test_denormalize(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(Fe_H=0., T_eff=4500, log_g=4, C_M=0, O_M=0)
        trace = AlexContinuumModelTrace()
        model = AlexContinuumModel(trace)
        model.init_wave(spec.wave)
        params = model.fit(spec)
        model.normalize(spec, params)
        model.denormalize(spec, params)
