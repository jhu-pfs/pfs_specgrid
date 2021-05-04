import copy
import numpy as np
import pandas as pd
import scipy as sp
import logging
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from pfsspec.physics import Physics
from pfsspec.stellarmod.continuummodel import ContinuumModel
from pfsspec.util.array_filters import *

class AlexContinuumModelTrace():
    def __init__(self, orig=None):
        if isinstance(orig, AlexContinuumModelTrace):
            self.limit_fit = orig.limit_fit
            self.norm_flux = orig.norm_flux
            self.params = orig.params
        else:
            self.limit_fit = None
            self.norm_flux = None
            self.params = None

class AlexContinuumModel(ContinuumModel):
    # Fit the upper envelope of a stellar spectrum model. The theoretical continuum
    # is first fitted with Lengendre polinomials between the Hzdrogen photoionization
    # limits, then the model is normalized and the remaining blended line regions
    # are fitted with a modified sigmoid function to remove all non-linearities
    # from the continuum.

    def __init__(self, orig=None, trace=None):
        super(AlexContinuumModel, self).__init__(orig=orig)

        # Trace certain variables for debugging purposes
        self.trace = trace

        if isinstance(orig, AlexContinuumModel):
            pass
        else:
            # Global wavelength limits that we can fit
            self.wave_min = 3000
            self.wave_max = 14000

            # The wave vector is assumed to be constant for all spectra and cached
            self.wave_mask = None
            self.wave = None
            self.log_wave = None

            # Masks of continuum intervals
            self.cont_fit_masks = None      # Continuum intervals for fitting
            self.cont_eval_masks = None     # Intervals to evaluate continuum on

            # Parameters of continuum Legendre fits
            self.legendre_rate = None       # How many points to skip when fitting Legendre to continuum
            self.legendre_rank = 6
            self.legendre_deg = 7

            # Bounds and masks of blended regions near photoionization limits

            self.limit_wave = None                      # Limit wavelengths, including model lower and upper bounds
            self.limit_map = None                       # Map to continuum intervals the blended regions are associated with
            
            # Blended region upper limits
            self.blended_bounds = [3200.0, 4200, 12000]
            
            # Optional blue-side offset (in term of spectral elements) of fitting
            # of blended regions. By default, blue end comes from the physical value
            self.blended_offset = [0, 0, 0]
            
            self.blended_fit_masks = None                 # Masks where limits are fitted
            self.blended_eval_masks = None                # Masks where limits are evaluated
            
            self.blended_dx = None            # TODO: what is this for?

            # Parameters of blended region upper envelope fits
                                    
            # self.sigmoid_fn = self.sigmoid
            # self.init_s0s1 = [[0.15, 1], [None], [0.15, 0.1], [None], [2., 3.1], [None], [None]] 
            
            self.blended_model_fn = self.sigmoid2
            self.blended_param_count = 5
            self.blended_default_params = np.full((self.blended_param_count,), np.nan)       # Default return value in case of bad fit

            # TODO: rename these
            self.init_s0s1 = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
            self.slope_cutoff = 25
            self.x1_y_ub = 0.001            # TODO: used by get_upper_points_for_gap

            self.smoothing_iter = 5
            self.smoothing_option = 1
            self.smoothing_kappa = 50
            self.smoothing_gamma = 0.1

    def add_args(self, parser):
        super(AlexContinuumModel, self).add_args(parser)

        parser.add_argument('--smoothing-iter', type=int, help='Smoothing iterations.\n')
        parser.add_argument('--smoothing-option', type=int, help='Smoothing kernel function.\n')
        parser.add_argument('--smoothing-kappa', type=float, help='Smoothing kappa.\n')
        parser.add_argument('--smoothing-gamma', type=float, help='Smoothing gamma.\n')

    def init_from_args(self, args):
        super(AlexContinuumModel, self).init_from_args(args)

        if 'smoothing_iter' in args and args['smoothing_iter'] is not None:
            self.smoothing_iter = args['smoothing_iter']
        if 'smoothing_option' in args and args['smoothing_option'] is not None:
            self.smoothing_option = args['smoothing_option']
        if 'smoothing_kappa' in args and args['smoothing_kappa'] is not None:
            self.smoothing_kappa = args['smoothing_kappa']
        if 'smoothing_gamma' in args and args['smoothing_gamma'] is not None:
            self.smoothing_gamma = args['smoothing_gamma']

    def get_constants(self):
        return np.array([])

    def set_constants(self, constants):
        pass

    def init_wave(self, wave):
        self.find_limits(wave)

#region Utility functions

    def safe_log(self, x):
        return np.log(np.where(x <= 1, 1, x))

#endregion
#region Main entrypoints: fit, eval and normalize

    def fit(self, spec):
        # Fit the spectrum and return the parameters
        log_flux = self.safe_log(spec.flux[self.wave_mask])
        log_cont = self.safe_log(spec.cont[self.wave_mask])
        
        # Fit continuum and normalize spectrum to fit blended lines as a next step
        try:
            fits, cont_params = self.fit_continuum(log_cont)
            model_cont = self.eval_continuum(fits=fits)
        except Exception as e:
            raise e
        norm_flux = log_flux - model_cont

        # Fit blended lines of the photoionization limits
        try:
            limit_params = self.fit_blended_all(norm_flux)
        except Exception as e:
            raise e

        if self.trace is not None:
            self.trace.norm_flux = norm_flux

        params = np.concatenate((cont_params, limit_params))

        return params

    def eval(self, params):
        # Evaluate the continuum model over the wave grid

        model_cont = self.eval_continuum(params=params)
        model_cont += self.eval_blended_all(params)

        return self.wave, model_cont
    
    def normalize(self, spec, params):
        # Normalize the spectrum given the fit params and constants
        # Returns normalized log flux

        # Continuum
        model_cont = self.eval_continuum(params=params)
        if spec.cont is not None:
            norm_cont = self.safe_log(spec.cont[self.wave_mask]) - model_cont
        else:
            norm_cont = None

        # Continuum and blended regions
        model_cont += self.eval_blended_all(params)
        norm_flux = self.safe_log(spec.flux[self.wave_mask]) - model_cont

        if self.trace is not None:
            self.trace.norm_flux = norm_flux
            self.trace.norm_cont = norm_cont
            
        spec.wave = self.wave
        spec.cont = norm_cont
        spec.flux = norm_flux

    def denormalize(self, spec, params):
        # Denormalize the spectrum given the fit params
        # Expects normalized log flux

        model_cont = self.eval_continuum(params=params)
        if spec.cont is not None:
            cont = np.exp(spec.cont + model_cont)
        else:
            cont = None

        model_cont += self.eval_blended_all(params)
        flux = np.exp(spec.flux + model_cont)
        
        spec.flux = flux
        spec.cont = cont

    def smooth_params(self, params):
        # Smooth the parameter grid
        # Apply only to parameters of the blended region fits, not the
        # Legendre coefficients

        k = self.legendre_deg * len(self.cont_fit_masks)
        l = k + len(self.limit_map) * self.blended_param_count

        smooth_params = np.full(params.shape, np.nan)
        smooth_params[..., :k] = params[..., :k]

        for i in range(k, l):
            # Fill in holes of the grid
            fp = fill_holes_filter(params[..., i], fill_filter=np.nanmean, value_filter=np.nanmin)

            # Smooth the parameters.
            shape = fp.shape
            fp = fp.squeeze()
            sp = anisotropic_diffusion(fp, 
                                        niter=self.smoothing_iter,
                                        kappa=self.smoothing_kappa,
                                        gamma=self.smoothing_gamma)
            smooth_params[..., i] = sp.reshape(shape)

        return smooth_params

#endregion            
#region Limits and mask

    def get_limit_wavelengths(self):
        # Returns the wavelength associated with the Hydrogen ionization limits
        # and the global limits of what we can fit
        limits = np.array(Physics.HYDROGEN_LIMITS)
        limits = limits[(self.wave_min < limits) & (limits < self.wave_max)]
        limits = np.array([self.wave_min, *Physics.air_to_vac(limits), self.wave_max])
        return limits

    def find_limits(self, wave):
        # Exact wavelengths of the limits and the lower and upper bounds of what we can fit
        if self.limit_wave is None:
            self.limit_wave = self.get_limit_wavelengths()

        # Mask that defines the entire range we can fit. Precalculate masked wavelength
        # grid and log lambda grid for convenience.
        if self.wave_mask is None:
            [self.wave_mask], _ = self.find_cont_masks(wave, [self.limit_wave[0], self.limit_wave[-1]], dlambda=0)
            self.wave = wave[self.wave_mask]
            self.log_wave = np.log(self.wave)

        # Every mask below will refer to the grid defined by wave_mask

        # Masks that define the regions where we fit the continuum
        if self.cont_fit_masks is None:
            self.cont_fit_masks, _ = self.find_cont_masks(self.wave, self.limit_wave, dlambda=0.5)

        # Disjoint masks that define where we evaluate the continuum, no gaps here
        if self.cont_eval_masks is None:
            self.cont_eval_masks, _ = self.find_cont_masks(self.wave, self.limit_wave, dlambda=0.0)

        # Masks where we will fit the blended lines' upper envelope. These are a
        # little bit redward from the photoionization limit.
        if self.blended_fit_masks is None:
            self.blended_fit_masks, self.limit_map = self.find_blended_masks(self.wave, self.cont_fit_masks)

        # Masks where we should evaluate the blended lines's upper envelope.
        if self.blended_eval_masks is None:
            self.blended_eval_masks, _ = self.find_blended_masks(self.wave, self.cont_eval_masks)

        # TODO: what is this exactly?
        if self.blended_dx is None:
            mask = (self.wave > 3000) & (self.wave < 3006) 
            dx = int(len(self.wave[mask]))
            # [200, 0, 400, 0, 200, 0, 200]
            self.blended_dx = [dx, 2 * dx, dx]
        
        # Downsampling of the wavelength grid for fitting the continuum
        if self.legendre_rate is None:
            dx = int(self.blended_dx[0] / 2)
            self.legendre_rate = [dx, 3 * dx, 2 * dx, 25]

    def find_cont_masks(self, wave, limits, dlambda):
        # Find intervals between the limits
        masks = []
        bounds = []
        for i in range(len(limits) - 1):
            mask = (wave >= limits[i] + dlambda) & (wave < limits[i + 1] - dlambda)
            masks.append(mask)
            wm = wave[mask]
            if wm.size > 0:
                bounds.append([wm[0], wm[-1]])
            else:
                bounds.append([np.nan, np.nan])

        return masks, bounds

    def find_blended_masks(self, wave, cont_masks):
        blended_masks = []
        limit_map = []
        for i in range(len(self.blended_bounds)):
            # Find the continuum region the limit is associated with
            for j in range(len(self.limit_wave)):
                if j == len(self.limit_wave) - 1:
                    limit_map.append(None)
                    break
                if self.limit_wave[j] < self.blended_bounds[i] and self.blended_bounds[i] < self.limit_wave[j + 1]:
                    limit_map.append(j)
                    break

            if limit_map[i] is not None:
                mask = cont_masks[limit_map[i]]
                blended_masks.append(mask & (wave < self.blended_bounds[i]))
            else:
                blended_masks.append(None)

        return blended_masks, limit_map

#endregion
#region Blended region fitting

    def fit_blended_all(self, norm_flux):
        if self.trace is not None:
            self.trace.x1 = {}
            self.trace.hb = {}
            self.trace.hull = {}
            self.trace.params = {}
            self.trace.params_est = {}

        params = []
        for i in range(len(self.limit_map)):
            pp = self.fit_blended(norm_flux, gap_id=i)
            params.append(pp)
            if self.trace is not None:
                self.trace.params[i] = pp

        params = np.concatenate(params, axis = 0)
        return params

    def get_blended_params(self, params, i):
        k = len(self.cont_fit_masks) * self.legendre_deg
        p = params[k + i * self.blended_param_count:k + (i + 1) * self.blended_param_count]
        return p
        
    def eval_blended_all(self, params):
        # Evaluate model around the limits
        model = np.zeros_like(self.wave)
        for i in range(len(self.limit_map)):
            p = self.get_blended_params(params, i)
            flux, mask = self.eval_blended(p, i)
            model[mask] += flux
        return model

    def eval_blended(self, params, i):
        mask = self.blended_eval_masks[i]
        if np.any(np.isnan(params) | np.isinf(params)) or abs(params).sum() == 0:
            return np.zeros_like(self.log_wave[mask]), mask
        else:
            model = self.blended_model_fn(self.log_wave[mask], *params)
            return model, mask

    def check_fit(self, y):
        # Make sure number of fitted parameters is correct and in the right range.
        return (len(y) > 3) and (y[0] < -0.001)

    def fit_blended(self, norm_flux, gap_id=None):
        gap_control_pts = self.get_upper_points_for_gap(norm_flux, gap_id=gap_id)
        # if self.limit_fit[gap_id] is False: 
        # x1 = np.max(gap_control_pts[:, 0])
        if not self.check_fit(gap_control_pts[:, 1]):
            if self.trace is not None:
                self.trace.limit_fit[gap_id] = False
            return self.blended_default_params 

        gap_hull_x, gap_hull_y, dd = self.get_slope_filtered_robust(gap_control_pts[:, 0],\
                                            gap_control_pts[:, 1])
        # self.check_fit_gap(gap_hull_y, gap_id, message = 'no hull left')
        # if self.limit_fit[gap_id] is False:
        if not self.check_fit(gap_hull_y):
            if self.trace is not None:
                self.trace.limit_fit[gap_id] = False
            return self.blended_default_params

        y0, slope_mid, x_mid = self.get_init_sigmoid_estimation(gap_hull_x, gap_hull_y, method = "interp1d")
        pmt = np.append([y0, slope_mid, x_mid], self.init_s0s1[gap_id])

        if self.trace is not None:
            self.trace.hb[gap_id] = gap_control_pts
            self.trace.hull[gap_id] = np.column_stack((gap_hull_x, gap_hull_y))
            self.trace.params_est[gap_id] = pmt

        try:
            # bnds = ([0, 0, 0, 0, 0], \
            #         [3., np.inf, np.log(self.blended_bounds[gap_id]), 20., 20.])
            bnds = ([0, 0, 0, 0, 0], \
                    [10., 1000, np.log(self.blended_bounds[gap_id]), 1., 1.])
            # bnds = (0, np.inf)
            pmt, _ = curve_fit(self.blended_model_fn, gap_hull_x, gap_hull_y, pmt, bounds=bnds) 
            # pmt, _ = curve_fit(self.blended_model_fn, gap_hull_x, gap_hull_y, pmt) 
        except:
            if self.trace is not None:
                self.trace.limit_fit[gap_id] = False
            # logging.warning('curve')
            return self.blended_default_params
        return pmt

    def fit_sigmoid_iterate(self, pts, w0, sigfun, pmt, thr = None):
        if (pts.shape[0]<7): return pmt
        for i in range(len(thr)):
            pts = self.clip_fitted_points(pts, sigfun, pmt, thr[i])
            pmt = self.fit_curve_with_sigmoid(pts[:, 0], pts[:, 1], w0, sigfun, pmt, (), type = 2)
        return pmt

    def get_min_max_norm(self, x):
        xmin, xmax = np.min(x), np.max(x)
        return (x - xmin)/(xmax - xmin)

    def get_slope_filtered(self, x, y, cutoff = 0):
        xx = self.get_min_max_norm(x)
        yy = self.get_min_max_norm(y)

        dd = np.diff(yy) / np.diff(xx)
        dd = np.abs(np.append(dd, dd[-1]))
        dd_median, dd_std = np.median(dd), dd.std()
        dd_high = dd_median + dd_std * 3.0
        # print(dd_high, cutoff)
        slope_cut = np.min([dd_high, cutoff])
        mask = (dd < slope_cut)
        return x[mask], y[mask], dd[mask]
    
    def get_slope_filtered_robust(self, x, y):
        x, y, dd = self.get_slope_filtered(x, y, self.slope_cutoff)
        # self.check_fit_gap(y, 3, 'slope')
        # idx = np.abs(y - y[0] / 2).argmin()
        # slope_mid = np.mean(dd[(idx - 2) : (idx + 2)])
        # x_mid = x[idx]
        # mask = dd < (abs(slope_mid) * scale)
        # x1, y1, dd1 = x[mask], y[mask], dd[mask]
        return x, y, dd
        # mask_dy = self.mask_outlier(y)
        # return x[mask_dy], y[mask_dy], dd[mask_dy]

    def get_init_sigmoid_estimation(self, x, y, dd = None, method = None):
        y_mid = y[0] / 2.
        # fit_mask = (y > (y_mid / 2 + y_mid)) & (y < (y_mid - y_mid / 2))
        # fit_mask = 
        # y_fit, x_fit = y[fit_mask], x[fit_mask]
        # if dd is None:
        #     dd = 
        y_fit, x_fit = y, x
        if method == "siegel":
            res = stats.siegelslopes(y_fit, x_fit)
            f_inverse = lambda y: (y - res[1]) / res[0] 
            slope_mid = res[0]
        elif method == "interp1d":
            f_inverse = interp1d(y_fit, x_fit, bounds_error=0)
            delta = abs(y_mid / 4)
            slope_mid = (delta*2) / (f_inverse(y_mid + delta) - f_inverse(y_mid - delta))
        elif method == 'dd':
            idx = np.abs(y - y[0] / 2).argmin()
            slope_mid = np.mean(dd[(idx-2): (idx+2)])
        else:
            raise "choose method for slope"
        x_mid = f_inverse(y_mid)
        # print(slope_mid, res[0])
        # a = 2 * abs(half_y)
        a = -y[0]
        # print(a, slope_mid)
        return a, slope_mid / a, x_mid
        # return a, slope_mid_dd, x_mid

    def get_upper_points_for_gap(self, norm_flux, gap_id=None):
        mask = self.blended_fit_masks[gap_id]
        offset = self.blended_offset[gap_id]
        
        x = self.log_wave[mask][offset:]
        y = norm_flux[mask][offset:]
        dx = self.blended_dx[gap_id]
        x1 = x[np.abs(y + self.x1_y_ub).argmin()]
        x_mask = (x < x1)

        if self.trace is not None:
            self.trace.x1[gap_id] = x1
        
        if len(x[x_mask]) > 5:
            internal_maxs = self.get_interval_max(x[x_mask], y[x_mask], dx)
            hb = self.get_accumulate_max(internal_maxs[:, 0], internal_maxs[:, 1])
        else:
            hb = np.array([[0, 0]])
        # self.check_fit_gap(hb[:, 1], gap_id, message = 'no hb left')
        return hb
        # max_hull_cut = self.max_hull_cut[gap_id]
        # if max_hull_cut is not None:
        #     mask = (internal_maxs[:,0] < max_hull_cut)
        #     cc1 = internal_maxs[mask]
        #     cc1_hull = self.get_accumulate_max(cc1[:, 0], cc1[:, 1])
        #     cc2 = internal_maxs[~mask]
        #     hb = np.concatenate((cc1_hull, cc2), axis = 0)
        # else:

    def get_accumulate_max(self, x, y):
        y_accumulated = np.maximum.accumulate(y)
        mask = (y >= y_accumulated)
        hb = np.column_stack((x[mask], y[mask])) 
        return hb

    def get_interval_max(self, x, y, dx=500):
        # Get the maximum in every interval of dx

        N = x.shape[0]
        pad_row = np.int(np.floor(N / dx)) + 1 
        pad_num = pad_row*dx - N
        pad_val = np.min(y)-1

        x_reshaped = np.pad(x, (0, pad_num), constant_values=pad_val).reshape(pad_row,dx)
        y_reshaped = np.pad(y, (0, pad_num), constant_values=pad_val).reshape(pad_row,dx)

        max_idx = np.argmax(y_reshaped, axis = 1)
        h = np.column_stack((np.take_along_axis(x_reshaped, max_idx[:,None], 1), \
                            np.take_along_axis(y_reshaped, max_idx[:,None], 1)))
        return h

    def mask_outlier(self, x):
        x = np.append(x, x[-1])
        dx = abs(np.diff(x))
        dx_mean, dx_std = np.median(dx), np.std(dx)
        mask = dx < dx_mean + 1.5 * dx_std 
        return mask

#endregion
#region Modified sigmoid fitting to blended regions

    def fit_curve_with_sigmoid(self, wx, wy, w0, sigfun, pmt, bnds, type='nbnd'):
        #---------------------------------------------------------
        # fit a sigmoid with 2 saturation parameters
        # input is log(wave), log(flux) points
        # parameters:
        #   a: height of the jump in the log(flux), positive
        #   b: slope of the exponential
        #   c: is the wavelength where the function is halfway
        #      between the two asymptotics
        #   d: potential small offset from 0 at the right end
        #---------------------------------------------------------
        # use linear wavelength
        # wx = h[:,0]
        # wy = h[:,1]
        if type == 'nbnd':
            pmt, pcov = curve_fit(sigfun, wx, wy, pmt) 
            return pmt

        if (len(pmt)==0):
            pmt = [-wy[0],0.01,w0,0.25,0.25]     
        if (len(bnds)==0):
            if type == 2:
                bnds=np.array([[0, 0, 0, 0, 0],[8.0,0.1,w0,20,20]])
            elif type == 4:
                bnds=np.array([[0, 0, 0, 0, 0],[0.5,3.0,0.1,w0,20,20]])

        pmt, pcov = curve_fit(sigfun, wx, wy, pmt, bounds=bnds) 
        return pmt

    def sigmoid2(self, x, a, b, c, r0, r1):
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
        # a1 = a / (y[t1] - y[]) 
        return a * (y - 1)

#endregion
#region Continuum fitting with Legendre polynomials

    def fit_continuum(self, log_cont):
        fits = []
        cc = len(self.cont_fit_masks)
        params = np.zeros((cc, self.legendre_deg), dtype=log_cont.dtype)
        for i in range(cc):
            ff = self.fit_legendre(i, log_cont)
            params[i] = ff.coef
            fits.append(ff)
        return fits, params.flatten()
        
    def eval_continuum(self, params=None, fits=None):
        # Evaluate the fitted continuum model (Legendre polynomial) over the
        # wavelength grid. Either params (the coefficients) or fits (a list of already
        # initialized Legendre objects) must be specified.

        model_cont = np.zeros_like(self.log_wave)

        if params is not None:
            for n in range(len(self.cont_eval_masks)):
                pp = params[n * self.legendre_deg:(n + 1) * self.legendre_deg]
                # TODO: this is a repetition whats in fit_legendre_by_n,
                #       store the xx grid and the domains somewhere
                rate = self.legendre_rate[n]
                xx = self.log_wave[self.cont_fit_masks[n]][::rate]
                ff = np.polynomial.legendre.Legendre(pp, domain=(xx[0], xx[-1]))
                model_cont[self.cont_eval_masks[n]] = ff(self.log_wave[self.cont_eval_masks[n]])
        elif fits is not None:
            for n in range(len(fits)):
                ff = fits[n]
                model_cont[self.cont_eval_masks[n]] = ff(self.log_wave[self.cont_eval_masks[n]])
        else:
            raise Exception('Either params or fits must be specified.')

        return model_cont

    def fit_legendre(self, n, log_cont):
        # Fit the nth continuum segment with a Legendre polynomial
        # TODO: we could do some caching of xx here
        rate = self.legendre_rate[n]
        xx = self.log_wave[self.cont_fit_masks[n]][::rate]
        yy = log_cont[self.cont_fit_masks[n]][::rate]

        ff, res = np.polynomial.legendre.Legendre.fit(xx, yy, self.legendre_rank,
                                                      domain=(xx[0], xx[-1]), full=True)
        assert np.sqrt(res[0] / xx.shape[0]).round(1) == 0
        return ff

#endregion