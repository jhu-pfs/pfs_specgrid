import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy import stats

class AlexSigmoid():
    def __init__(self, domain=None, bounds=None):
        self.domain = domain
        
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = (-np.inf, np.inf)

    def get_param_count(self):
        return 5

    def fit(self, x, y, w=None, p0=None, jac=None):
        # TODO: use the Jacobian

        # Make sure p0 is within the bounds
        pp0 = np.where(self.bounds[0] > p0, self.bounds[0], p0)
        pp0 = np.where(self.bounds[1] < pp0, self.bounds[1], pp0)

        sigma = 1 / w if w is not None else None

        try:
            #pp, _ = curve_fit(AlexSigmoid.f, x, y, pp0, sigma=sigma, jac=AlexSigmoid.jac, bounds=self.bounds)
            pp, _ = curve_fit(AlexSigmoid.f, x, y, pp0, sigma=sigma, bounds=self.bounds)
        except RuntimeError as ex:
            # No convergence
            raise ex
        except Exception as ex:
            raise ex

        return pp

    def eval(self, x, params):
        return AlexSigmoid.f(x, *params)

    def find_p0(self, x, y, w=None, method='interp1d'):
        # Find the initial guess for the parameters from the control points

        # Amplitude and midpoints
        y_min, y_max = y.min(), y.max()
        a = y_max - y_min
        y_mid = 0.5 * (y_min + y_max)

        # Midpoint x and slope
        if method == "siegel":
            res = stats.siegelslopes(y, x)
            f_inverse = lambda z: (z - res[1]) / res[0] 
            slope_mid = res[0]
        elif method == "interp1d":
            f_inverse = interp1d(y, x, bounds_error=False)
            delta = abs(y_mid / 4)
            slope_mid = 2* delta / (f_inverse(y_mid + delta) - f_inverse(y_mid - delta))
        # elif method == 'dd':
        #     idx = np.abs(y - y[0] / 2).argmin()
        #     slope_mid = np.mean(dd[(idx-2): (idx+2)])
        else:
            raise NotImplementedError()

        x_mid = f_inverse(y_mid)

        # Inflection points
        s0 = 0.5
        s1 = 0.5

        # If amplitude is smaller than a limit
        if a < self.bounds[0][0]:
            return False, np.array([0, np.nan, np.nan, np.nan, np.nan])

        return True, np.array([a, slope_mid / a, x_mid, s0, s1])

    @staticmethod
    def f(x, a, b, c, r0, r1):
        #---------------------------------------------------------
        # splice a sigmoid-like curve from three pieces:
        #  - a linear segment in the middle, with a slope of b
        #     and a value of 1/2 at c
        #  - two exponential pieces, both left and right
        #    which are tangential to the line in their
        #    respective quadrants
        # The three curves are merged seamlessly, with a
        # conntinous function and derivative
        # 2021-02-14   Alex Szalay
        #---------------------------------------------------------

        # If the amplitude is zero, return an all zero vector, even if other
        # parameters are nan. This is necessary to handle the case of very
        # small amplitude blended regions.
        if a == 0.0 or b <= 0 or r0 <= 0 or r1 <= 0:
            return np.zeros(x.shape)
        else:
            x0 = c - 1 / (2 * b)
            x1 = c + 1 / (2 * b)

            beta0  = 2 * b / r0
            alpha0 = r0 / (2 * np.e)   
            beta1  = 2 * b / r1
            alpha1 = r1 / (2 * np.e)
            
            t0 = x0 + 1 / beta0
            t1 = x1 - 1 / beta1
            i0 = (x <= t0)
            i1 = (x >= t1)
            im = (x > t0) & (x < t1)
            
            y = np.zeros(x.shape)
            arg0 = beta0 * (x[i0] - x0)
            arg1 = -beta1 * (x[i1] - x1)

            y[i0] = alpha0 * np.exp(arg0)
            y[i1] = 1 - alpha1 * np.exp(arg1)
            y[im] = b * (x[im] - c) + 0.5

            return a * (y - 1)

    @staticmethod
    def jac(x, a, b, c, r0, r1):
        # Find the merge point t0, t1
        t0 = c - (1 - r0) / (2 * b)
        t1 = c + (1 - r1) / (2 * b)        

        # Mask for different region
        mask_left = x <= t0
        mask_mid = (x > t0) & (x < t1)
        mask_right = x >= t1

        # Initialize Jacobian with zero matrix
        jac = np.zeros((len(x), 5))
        
        xx = x[mask_left] - c
        EE = np.exp(2 * b / r0 * (xx + 1 / (2 * b)))
        jac[mask_left, 0] = r0 / (2 * np.e) * EE - 1
        jac[mask_left, 1] = a * xx / np.e * EE
        jac[mask_left, 2] = -a * b / np.e * EE
        jac[mask_left, 3] = a / (2 * np.e) * (1 - (2 * b * xx + 1) / r0) * EE
        jac[mask_left, 4] = 0

        xx = x[mask_right] - c
        EE = np.exp(-2 * b / r1 * (xx - 1 / (2 * b)))
        jac[mask_right, 0] = -r1 / (2 * np.e) * EE
        jac[mask_right, 1] = a * xx / np.e * EE
        jac[mask_right, 2] = a * b / np.e * EE
        jac[mask_right, 3] = 0
        jac[mask_right, 4] = -a / (2 * np.e) * (1 + (2 * b * xx - 1) / r1) * EE

        xx = x[mask_mid] - c
        jac[mask_mid, 0] = b * xx - 0.5
        jac[mask_mid, 1] = a * xx
        jac[mask_mid, 2] = -a - b
        jac[mask_mid, 3] = 0
        jac[mask_mid, 4] = 0

        return jac
