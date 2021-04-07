from test.test_base import TestBase
from pfsspec.stellarmod.alexcontinuummodel import AlexContinuumModel

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

    def test_fit_legendre(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(Fe_H=2, T_eff=6000, log_g=1, C_M=0, O_M=0)
        alex = AlexContinuumModel()
        log_flux, log_cont = alex.prepare(spec)
        fits = alex.fit_legendre(log_cont)
        self.assertIsNotNone(fits)
        model_cont = alex.eval_legendre(fits)
        self.assertIsNotNone(model_cont)

    def test_get_norm_flux(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(Fe_H=2, T_eff=6000, log_g=1, C_M=0, O_M=0)
        alex = AlexContinuumModel()
        norm_flux = alex.get_norm_flux(spec)
        self.assertIsNotNone(norm_flux)
    
    def test_check_gap_6(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(Fe_H=2, T_eff=6000, log_g=1, C_M=0, O_M=0)
        alex = AlexContinuumModel()
        norm_flux = alex.get_norm_flux(spec)
        params_gap_6_left, params_gap_6_right, mask_left_6, mask_right_6 = alex.fit_gap_6(norm_flux)
        norm_cont= alex.eval_gap_6(params_gap_6_left, params_gap_6_right, mask_left_6, mask_right_6)
        self.assertIsNotNone(params_gap_6_left)
        self.assertIsNotNone(params_gap_6_right)
        self.assertIsNotNone(mask_left_6)
        self.assertIsNotNone(mask_right_6)
        self.assertIsNotNone(norm_flux)

    def test_fit_gap_2(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(Fe_H=2, T_eff=6000, log_g=1, C_M=0, O_M=0)
        alex = AlexContinuumModel()
        norm_flux = alex.get_norm_flux(spec)
        gap_2_control_pts, mask, sigfun, params_2 = alex.fit_gap_2(norm_flux, isEval=True)
        norm_cont = alex.eval_gap_2(gap_2_control_pts, mask, sigfun, params_2)
        self.assertIsNotNone(gap_2_control_pts)
        self.assertIsNotNone(mask)
        self.assertIsNotNone(params_2)
        self.assertIsNotNone(norm_cont)

    def test_fit_gaps(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(Fe_H=2, T_eff=6000, log_g=1, C_M=0, O_M=0)
        alex = AlexContinuumModel()
        norm_flux = alex.get_norm_flux(spec)
        # params_gap_0, params_gap_2, params_gap_4, params_gap_6 = self.fit_gaps(norm_flux)
        params_gap_2, params_gap_4 = alex.fit_gaps(norm_flux)

        self.assertIsNotNone(params_gap_2)
        self.assertIsNotNone(params_gap_4)
        # self.assertIsNotNone(params_gap_6)
        # self.assertIsNotNone(params_gap_0)

    def test_fit_gap_sigmoid(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(Fe_H=2, T_eff=6000, log_g=1, C_M=0, O_M=0)
        alex = AlexContinuumModel()
        norm_flux = self.get_norm_flux(spec)
        params_gap_0, params_gap_2, params_gap_4, params_gap_6 = self.fit_gaps(norm_flux)
        self.assertIsNotNone(params_gap_0)
        self.assertIsNotNone(params_gap_2)
        self.assertIsNotNone(params_gap_4)
        self.assertIsNotNone(params_gap_6)






    