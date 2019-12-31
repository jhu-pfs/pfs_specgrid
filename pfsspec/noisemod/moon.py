import numpy as np
from scipy.interpolate import interp1d

from pfsspec.constants import Constants
from pfsspec.physics import Physics
from pfsspec.data.grid import Grid

class Moon(Grid):
    def __init__(self):
        super(Moon, self).__init__()
        self.ref_exp_count = 1
        self.ref_exp_time = 450
        self.ref_target_za = 0
        self.ref_target_angle = 60              # Reference values, distance from the moon
        self.ref_lunar_za = 60                  # Reference value at which models are precomputed
        self.ref_lunar_phase = 1.0              # Reference value at which models are precomputed

        self.wave = None

    def init_params(self):
        self.init_param('fa')

    def init_data(self):
        self.init_data_item('counts')  # Photon count originating from the Moon

    def allocate_data(self):
        self.allocate_data_item('counts', self.wave.shape)  # Photon count originating from the Moon

    def save_items(self):
        self.save_item('wave', self.wave)
        super(Moon, self).save_items()

    def load_items(self, s=None):
        self.wave = self.load_item('wave', np.ndarray)
        self.init_data()
        super(Moon, self).load_items(s=s)

    def get_lunar_zenith_angle_coeff(self, lunar_za):
        # kV=0.12-0.01*(lambda-550.)/ 50.
        # * pow(10., -0.4*kV/sqrt(1.-0.96*sin(obs->lunarZA*DEGREE)*sin(obs->lunarZA*DEGREE)))

        # continuum opacity is calculated at lambda = 550
        l = 550.
        kV = 0.12 - 0.01 * (l - 550.) / 50.
        x = 10**(-0.4 * kV / np.sqrt(1.0 - 0.96 * np.sin(self.ref_lunar_za * Constants.DEGREE)**2))  # old
        y = 10**(-0.4 * kV / np.sqrt(1.0 - 0.96 * np.sin(lunar_za * Constants.DEGREE)**2))  # new
        return y / x

    def get_target_zenith_angle_coeff(self, target_za):
        # kV=0.12-0.01*(lambda-550.)/ 50.
        # * (1. - pow(10., -0.4 * kV / sqrt(1. - 0.96 * sin(obs->zenithangle * DEGREE) * sin(obs->zenithangle * DEGREE))))

        # continuum opacity is calculated at lambda = 550
        l = 550.
        kV = 0.12 - 0.01 * (l - 550.) / 50.
        x = 1 - 10**(-0.4 * kV / np.sqrt(1.0 - 0.96 * np.sin(self.ref_target_za * Constants.DEGREE)**2))  # old
        y = 1 - 10**(-0.4 * kV / np.sqrt(1.0 - 0.96 * np.sin(target_za * Constants.DEGREE)**2))  # new
        return y / x

    def get_target_angle_f1_f2(self, target_angle):
        f1 = 2.29e5 * (1.06 + np.cos(target_angle * Constants.DEGREE) * np.cos(target_angle * Constants.DEGREE))
        f2 = np.power(10., 6.15 - target_angle / 40.)
        return f1, f2

    def get_target_angle_coeff(self, target_angle):
        scale_RS = (np.exp(2480. / 550.) - 1.) / (np.exp(2480. / self.wave) - 1.) * np.power(self.wave / 550., -7.0)
        scale_MS = (np.exp(2480. / 550.) - 1.) / (np.exp(2480. / self.wave) - 1.) * np.power(self.wave / 550., -4.3)

        f1, f2 = self.get_target_angle_f1_f2(self.ref_target_angle)
        x = f1 * scale_RS + f2 * scale_MS

        f1, f2 = self.get_target_angle_f1_f2(target_angle)
        y = f1 * scale_RS + f2 * scale_MS

        return y / x

    def get_lunar_Istar(self, lunar_phase):
        alpha = 360 * np.abs(lunar_phase - 0.5);
        Istar = pow(10., -0.4 * (3.84 + 0.026 * alpha + 4e-9 * alpha * alpha * alpha * alpha));
        if (alpha < 7):
            Istar *= 1.35-0.05 * alpha;
        return Istar

    def get_lunar_phase_coeff(self, lunar_phase):
        x = self.get_lunar_Istar(self.ref_lunar_phase)
        y = self.get_lunar_Istar(lunar_phase)
        return y / x

    def get_counts(self, exp_count, exp_time, target_za, fa, target_angle, lunar_za, lunar_phase):
        # Effect of lunar zenith angle is computed here but target zenith angle and field angle affect the
        # overall transmission function, hence it is computed by the ETC and pretabulated
        # It is not checked if target_za, lunar_za and ta are geometrically meaningful although TA is measured
        # as a great circle distance
        counts, _ = self.interpolate_data_item_linear('counts', fa=fa)
        counts = counts * self.get_target_angle_coeff(target_angle)
        counts = counts * self.get_target_zenith_angle_coeff(target_za)
        counts = counts * self.get_lunar_zenith_angle_coeff(lunar_za)
        counts = counts * self.get_lunar_phase_coeff(lunar_phase)
        counts = counts * exp_count / self.ref_exp_count
        counts = counts * exp_time / self.ref_exp_time
        return counts