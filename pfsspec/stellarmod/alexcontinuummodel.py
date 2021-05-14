import copy
import numpy as np
import pandas as pd
import scipy as sp
import logging

from pfsspec.physics import Physics
from pfsspec.stellarmod.continuummodel import ContinuumModel
from pfsspec.util.array_filters import *

from pfsspec.fit.legendre import Legendre
from pfsspec.fit.alexsigmoid import AlexSigmoid

class AlexContinuumModelTrace():
    def __init__(self):
        self.model_cont = None
        self.model_blended = None
        self.norm_flux = None
        self.norm_cont = None

        self.blended_control_points = {}
        self.blended_p0 = {}
        self.blended_params = {}
        self.blended_fit = {}
        self.x1 = {}

        self.fill_holes = False

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
            self.cont_fit_rate_multiplier = np.array([1, 3, 2])
            self.cont_fit_rate = None       # How many points to skip when fitting Legendre to continuum            
            self.cont_fit_masks = None      # Continuum intervals for fitting
            self.cont_eval_masks = None     # Intervals to evaluate continuum on
            self.cont_models = None

            # Parameters of continuum Legendre fits
            self.legendre_deg = 6

            # Bounds and masks of blended regions near photoionization limits

            self.limit_wave = None                      # Limit wavelengths, including model lower and upper bounds
            self.limit_map = None                       # Map to continuum intervals the blended regions are associated with
            
            # Blended region upper limits
            # self.blended_bounds = np.array([3200.0, 5000, 12000])
            self.blended_bounds = np.array([3400.0, 8000, 13000])
            self.blended_count = self.blended_bounds.size
            
            self.blended_fit_masks = None                 # Masks where limits are fitted
            self.blended_eval_masks = None                # Masks where limits are evaluated
            self.blended_models = None

            self.blended_dx_multiplier = np.array([1, 2, 1])
            self.blended_dx = None

            self.blended_slope_cutoff = 25  # Cutoff to filter out very steep intial part of blended regions

            # Parameters of blended region upper envelope fits
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
            cont_params = self.fit_continuum_all(log_flux, log_cont)
            model_cont = self.eval_continuum_all(cont_params)
        except Exception as e:
            raise e
        norm_flux = log_flux - model_cont

        if self.trace is not None:
            self.trace.norm_flux = norm_flux
            
        # Fit blended lines of the photoionization limits
        try:
            limit_params = self.fit_blended_all(norm_flux)
        except Exception as e:
            raise e

        params = np.concatenate((cont_params, limit_params))
        return params

    def eval(self, params):
        # Evaluate the continuum model over the wave grid

        model_cont = self.eval_continuum_all(params)
        model_cont += self.eval_blended_all(params)

        return self.wave, model_cont
    
    def normalize(self, spec, params):
        # Normalize the spectrum given the fit params and constants
        # Returns normalized log flux

        # Continuum
        model_cont = self.eval_continuum_all(params=params)
        if spec.cont is not None:
            norm_cont = self.safe_log(spec.cont[self.wave_mask]) - model_cont
        else:
            norm_cont = None

        # Continuum and blended regions
        model_blended = self.eval_blended_all(params)
        cont_norm_flux = self.safe_log(spec.flux[self.wave_mask]) - model_cont
        norm_flux = cont_norm_flux - model_blended

        if self.trace is not None:
            self.trace.cont_norm_flux = cont_norm_flux
            self.trace.model_cont = model_cont
            self.trace.model_blended = model_blended
            self.trace.norm_flux = norm_flux
            self.trace.norm_cont = norm_cont
            
        spec.wave = self.wave
        spec.cont = norm_cont
        spec.flux = norm_flux

    def denormalize(self, spec, params):
        # Denormalize the spectrum given the fit params
        # Expects normalized log flux

        model_cont = self.eval_continuum_all(params=params)
        if spec.cont is not None:
            cont = np.exp(spec.cont + model_cont)
        else:
            cont = None

        model_blended = self.eval_blended_all(params)
        flux = np.exp(spec.flux + model_cont + model_blended)

        if self.trace is not None:
            self.trace.model_cont = model_cont
            self.trace.model_blended = model_blended
            self.trace.norm_flux = spec.flux
            self.trace.norm_cont = spec.cont
        
        spec.flux = flux
        spec.cont = cont

    def smooth_params(self, params):
        # Smooth the parameter grid
        # Apply only to parameters of the blended region fits, not the
        # Legendre coefficients

        k = 0
        for m in self.cont_models:
            k += m.get_param_count()
        l = k
        for m in self.blended_models:
            l += m.get_param_count()

        # Create the output params grid with all nans
        smooth_params = np.full(params.shape, np.nan)

        # Copy Legendre polynomial coefficients as they are
        smooth_params[..., :k] = params[..., :k]

        # Smooth all blended region fit parameters one by one
        for i in range(k, l):
            # Fill in holes of the grid
            if self.fill_holes:
                fp = fill_holes_filter(params[..., i], fill_filter=np.nanmean, value_filter=np.nanmin)
            else:
                fp = params[..., i]

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
        self.limit_wave = self.get_limit_wavelengths()

        # Mask that defines the entire range we can fit. Precalculate masked wavelength
        # grid and log lambda grid for convenience.
        [self.wave_mask], _ = self.find_cont_masks(wave, [self.limit_wave[0], self.limit_wave[-1]], dlambda=0)
        self.wave = wave[self.wave_mask]
        self.log_wave = np.log(self.wave)

        # Every mask below will refer to the grid defined by wave_mask

        # Masks that define the regions where we fit the continuum
        self.cont_fit_masks, _ = self.find_cont_masks(self.wave, self.limit_wave, dlambda=0.5)
        # Disjoint masks that define where we evaluate the continuum, no gaps here
        self.cont_eval_masks, _ = self.find_cont_masks(self.wave, self.limit_wave, dlambda=0.0)
        # Continuum models
        self.cont_models = []
        for i in range(len(self.cont_fit_masks)):
            w0 = np.log(self.limit_wave[i])
            w1 = np.log(self.limit_wave[i + 1])
            m = Legendre(self.legendre_deg, domain=[w0, w1])
            self.cont_models.append(m)

        # Masks where we will fit the blended lines' upper envelope. These are a
        # little bit redward from the photoionization limit.
        self.blended_fit_masks, self.limit_map = self.find_blended_masks(self.wave, self.cont_fit_masks)
        # Masks where we should evaluate the blended lines's upper envelope.
        self.blended_eval_masks, _ = self.find_blended_masks(self.wave, self.cont_eval_masks)
        # Blended region models
        self.blended_models = []
        for i in range(len(self.blended_fit_masks)):
            # amplitude, slope, midpoint, inflexion points s0, s1
            bounds = ([.0, 0., np.log(self.limit_wave[self.limit_map[i]]), 0., 0.], \
                      [10., 1000, np.log(self.blended_bounds[i]), 1., 1.])
            m = AlexSigmoid(bounds=bounds)
            self.blended_models.append(m)

        mask = (self.wave > 3000) & (self.wave < 3006) 
        dx = int(len(self.wave[mask]))

        # TODO: what is this exactly?
        self.blended_dx = self.blended_dx_multiplier * dx
        
        # Downsampling of the wavelength grid for fitting the continuum
        self.cont_fit_rate = self.cont_fit_rate_multiplier * dx

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

    def get_blended_params(self, params, i):
        # Get blended parameters from the params array

        # Sum up all continuum and preceeding blended region pieces
        k = 0
        for j in range(len(self.cont_models)):
            k += self.cont_models[j].get_param_count()
        for j in range(i):
            k += self.blended_models[j].get_param_count()

        # Current blended region model
        kk = self.blended_models[i].get_param_count()
        p = params[k:k + kk]
        return p

    def fit_blended_all(self, norm_flux):
        params = []
        for i in range(len(self.limit_map)):
            pp = self.fit_blended(norm_flux, i)
            params.append(pp)
        params = np.concatenate(params, axis = 0)
        return params
        
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
        model = self.blended_models[i]
        if np.any(np.isnan(params) | np.isinf(params)) or abs(params).sum() == 0:
            return np.zeros_like(self.log_wave[mask]), mask
        else:
            flux = model.eval(self.log_wave[mask], params)
            return flux, mask

    def fit_blended(self, norm_flux, i):
        # Fit a blended line region

        model = self.blended_models[i]

        # Try to fit and handle gracefully if fails
        try:
            # Get control points using the maximum hull method
            x, y = self.get_blended_control_points(norm_flux, i)

            # Check if control points are good enough for a fit
            if x is None or y is None:
                raise Exception('No valid control points.')

            if self.trace is not None:
                self.trace.blended_control_points[i] = (x, y)

            # Estimate the initial value of the parameters
            p0 = model.find_p0(x, y)
            if self.trace is not None:
                self.trace.blended_p0[i] = p0
        
            pp = model.fit(x, y, w=None, p0=p0)

            if self.trace is not None:
                self.trace.blended_fit[i] = True
                self.trace.blended_params[i] = pp

            return pp
        except Exception as ex:
            # logging.warning(ex)
            if self.trace is not None:
                self.trace.blended_fit[i] = False
            return np.array(model.get_param_count() * [np.nan])

    def get_blended_control_points(self, norm_flux, i):
        # Find control points for fitting a modified sigmoid function
        # to a blended line region redward of the photoionization limits.

        # Make sure number of fitted parameters is correct and in the right range.
        def validate_control_points(y):
            return len(y) > 3 and y[0] < -0.001

        mask = self.blended_fit_masks[i]
        dx = self.blended_dx[i]

        x, y = self.log_wave[mask], norm_flux[mask]

        # Find the maximum in intervals of dx
        x, y = self.get_max_interval(x, y, dx=dx)

        # Determine the maximum hull
        x, y = self.get_max_hull(x, y)
        if not validate_control_points(y):
            return None, None

        # Calculate the differential and drop the very steep part at the
        # beginning of the interval, as it may be a narrow line instead of a
        # blended region
        x, y = self.get_slope_filtered(x, y, cutoff=self.blended_slope_cutoff)
        if not validate_control_points(y):
            return None, None

        # TODO: itt volt még a 0.001-es vágás

        return x, y

    def get_slope_filtered(self, x, y, cutoff=0):
        def get_min_max_norm(x):
            xmin, xmax = np.min(x), np.max(x)
            return (x - xmin) / (xmax - xmin)

        xx = get_min_max_norm(x)
        yy = get_min_max_norm(y)

        dd = np.diff(yy) / np.diff(xx)
        dd = np.abs(np.append(dd, dd[-1]))
        dd_median, dd_std = np.median(dd), dd.std()
        dd_high = dd_median + dd_std * 3.0
        slope_cut = np.min([dd_high, cutoff])
        mask = (dd < slope_cut)
        return x[mask], y[mask]
    

    def get_max_hull(self, x, y):
        # Get the maximum hull
        y_accumulated = np.maximum.accumulate(y)
        mask = (y >= y_accumulated)
        return x[mask], y[mask]

    def get_max_interval(self, x, y, dx=500):
        # Get the maximum in every interval of dx

        N = x.shape[0]
        pad_row = np.int(np.floor(N / dx)) + 1 
        pad_num = pad_row * dx - N
        pad_val = np.min(y) - 1

        x_reshaped = np.pad(x, (0, pad_num), constant_values=pad_val).reshape(pad_row, dx)
        y_reshaped = np.pad(y, (0, pad_num), constant_values=pad_val).reshape(pad_row, dx)

        max_idx = np.argmax(y_reshaped, axis = 1)
        max_x = np.take_along_axis(x_reshaped, max_idx[..., np.newaxis], axis=1)[:, 0]
        max_y = np.take_along_axis(y_reshaped, max_idx[..., np.newaxis], axis=1)[:, 0]

        return max_x, max_y

#endregion
#region Modified sigmoid fitting to blended regions


#endregion
#region Continuum fitting with Legendre polynomials

    def get_cont_params(self, params, i):
        # Count the number of parameters before i
        l, u = 0, 0
        for k in range(i + 1):
            c = self.cont_models[i].get_param_count()
            if k < i:
                l += c
            u += c
        p = params[l:u]
        return p

    def fit_continuum_all(self, log_flux, log_cont):
        params = []
        for i in range(len(self.cont_models)):
            p = self.fit_continuum(log_flux, log_cont, i)
            params.append(p)
        return np.concatenate(params)

    def fit_continuum(self, log_flux, log_cont, i):
        mask = self.cont_fit_masks[i]
        x = self.log_wave[mask]
        y = log_cont[mask]
        model = self.cont_models[i]

        params = self.fit_model_simple(model, x, y)
        
        # Find the minimum difference between the model fitted to the continuum
        # and the actual flux and shift the model to avoid big jumps.
        v = model.eval(x, params)
        if np.any(v > log_flux[mask]):
            offset = np.min((v - log_flux[mask])[v > log_flux[mask]])
            if offset > 1e-2:
                model.shift(-offset, params)

        return params
        
    def eval_continuum_all(self, params):
        # Evaluate the fitted continuum model (Legendre polynomials) over the
        # wavelength grid.
        model_cont = np.zeros_like(self.log_wave)
        for i in range(len(self.cont_models)):
            cont, mask = self.eval_continuum(params, i)
            model_cont[mask] = cont
        return model_cont

    def eval_continuum(self, params, i):
        pp = self.get_cont_params(params, i)
        mask = self.cont_eval_masks[i]
        wave = self.log_wave[mask]
        cont = self.cont_models[i].eval(wave, pp)
        return cont, mask

#endregion