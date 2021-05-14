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
            pp, _ = curve_fit(AlexSigmoid.f, x, y, pp0, sigma=sigma, jac=AlexSigmoid.jac, bounds=self.bounds)
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

        return np.array([a, slope_mid / a, x_mid, s0, s1])

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

        # Initialize Jacobian with zero matrix
        jac = np.zeros((len(x), 5))
        
        # creating mask for different region
        mask_left = x <= t0
        mask_mid = (x > t0) & (x < t1)
        mask_right = x >= t1
        
        # calculating jac for different region
        jac[mask_left,:] = AlexSigmoid.jac_left(x[mask_left],a,b,c,r0,r1)
        jac[mask_mid,:] = AlexSigmoid.jac_mid(x[mask_mid],a,b,c,r0,r1)    
        jac[mask_right,:] = AlexSigmoid.jac_right(x[mask_right],a,b,c,r0,r1)        
        
        return jac

    @staticmethod
    def jac_left(x,a,b,c,r0,r1):
        # Find the jacobian for x <= t0 region
        c0 = (1 - 2 * b * (c - x)) / r0 - 1
        exp = np.exp(c0)

        # Calculate derivative for 5 variables
        da = r0 * exp / 2.
        db = a * (x - c) * exp
        dc = -a * b * exp
        dr0 = -a * exp * c0 /2.
        dr1 = np.zeros_like(x)

        return np.vstack((da,db,dc,dr0,dr1)).T

    @staticmethod
    def jac_mid(x,a,b,c,r0,r1):
        # Find the jacobian for t0 < x < t1 region

        # Calculate derivative for a,b,c,r0,r1
        da = 0.5 + b * (x - c)
        db = a * (x - c)
        dc = -a * b * (np.zeros_like(x) + 1)
        dr0 = np.zeros_like(x)
        dr1 = np.zeros_like(x)

        return np.vstack((da,db,dc,dr0,dr1)).T    
    
    @staticmethod
    def jac_right(x,a,b,c,r0,r1):
        # Find the jacobian for x > t1 region

        c0 = (1 + 2 * b * (c - x)) / r1 - 1
        exp = np.exp(c0)
        # Calculate derivative for a,b,c,r0,r1
        da = 1 - r1 * exp / 2.
        db = a * (x - c) * exp
        dc = - a * b * exp 
        dr0 = np.zeros_like(x)
        dr1 = a * exp * c0 / 2.

        return np.vstack((da,db,dc,dr0,dr1)).T   