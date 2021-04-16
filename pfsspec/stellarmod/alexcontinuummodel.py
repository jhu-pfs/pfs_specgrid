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


class AlexContinuumModel(ContinuumModel):
    def __init__(self, orig=None, debug = False, test = None):
        if isinstance(orig, AlexContinuumModel):
            pass
        else:
            self.photo_limits = self.get_photo_limits(d = 0.5)
            self.legendre_rate = None
            self.legendre_rank = 6
            self.legendre_deg = 7
            self.dx = None
            self.gap_ids = [0, 2, 4]
            self.offset = [0, None, 0, None, 0, None, None]

            self.slope_cutoff = 8
            self.init_s0s1 = [[0.15, 1], [None], [0.15, 0.1], [None],
                                [2., 3.1], [None], [None]] 

            self.fit_gap = {0: True, 2: True, 4: True, 6: False}

            self.wave = None
            self.log_wave = None
            self.wave_mask = None
            self.masks = None
            self.limits = None
            self.eval_masks = None
            self.gap_masks = None
            self.debug = debug
            # self.method = method
            self.test = test


    def add_args(self, parser):
        super(AlexContinuumModel, self).add_args(parser)

    def init_from_args(self, args):
        super(AlexContinuumModel, self).init_from_args(args)

    def prepare(self, spec):
        self.find_limits(spec.wave)
        # norm = 4 * np.log(spec.T_eff)
        norm = 0
        log_cont = self.get_log(spec.cont[self.wave_mask]) - norm
        log_flux = self.get_log(spec.flux[self.wave_mask]) - norm
        return log_flux, log_cont

    def get_norm_flux_n_params(self, spec):
        log_flux, log_cont = self.prepare(spec)
        try:
            fits, params = self.fit_legendre(log_cont)
            model_cont = self.eval_legendre(fits)
        except Exception as e:
            raise e
            
        norm_flux = log_flux - model_cont
        return norm_flux, params
    
    def normalize(self, spec):
        norm_flux, norm_params = self.get_norm_flux_n_params(spec)
        norm_cont = np.zeros_like(self.wave)
        # gap_params = np.array([])

        norm_params = np.array([])
        gap_params = self.fit_gaps(norm_flux)
        # print(gap_params)
        norm_cont = self.eval_gaps(norm_cont, gap_params)
        if self.debug:
            self.norm_flux = norm_flux
            self.norm_cont = norm_cont

        spec.wave = self.wave
        spec.cont = norm_cont
        spec.flux = norm_flux - norm_cont
        
        # params = np.concatenate((params_gap_2, params_gap_4), axis = 0)\
        params = np.concatenate((norm_params, gap_params))
        if not self.check(params): params[0] = np.inf 
        return params

    def check(self, value):
        return np.logical_not(np.any(np.isnan(value), axis=-1)) & ((value.max(axis=-1) != 0) | (value.min(axis=-1) != 0))
    
    # def fit_gaps(self, norm_flux):
    #     params_gap_0, params_gap_2, params_gap_4, params_gap_6 = self.fit_gaps(norm_flux)
    #     params = np.concatenate((params_gap_0, params_gap_2, params_gap_4, params_gap_6), axis = 0)
    #     assert (len(params) == 16)
    #     return params
    
    # def eval(self, norm_cont, params):
    #     # if params[0] == np.inf: 
    #     #     params[0] = 0
    #     # params_legendre, params_gap = params[:self.legendre_deg], params[28:]
    #     params_gap = params
    #     params_gap_0 = params_gap[0:5]
    #     params_gap_2 = params_gap[5:10]
    #     params_gap_4 = params_gap[10:15]
    #     params_gap_6 = params_gap[-1]
    #     assert (len(params_gap_0) == 5) & (len(params_gap_2) == 5) & (len(params_gap_4) == 5)
    #     norm_cont = self.eval_gaps(norm_cont, params_gap_0, params_gap_2, params_gap_4, params_gap_6)
    #     return norm_cont

    def fit_gaps(self, norm_flux):
        if self.debug:
            self.x1 = {}
            self.hb = {}
            self.hull = {}
            self.params = {}
            self.params_est = {}

        if self.test is not None:
            if self.test == 0:
                params_gap_0 = self.fit_gap_sigmoid(norm_flux, gap_id = 0)
                params_gap_2, params_gap_4 = np.zeros(5), np.zeros(5)
            elif self.test == 2:
                params_gap_2 = self.fit_gap_sigmoid(norm_flux, gap_id = 2)
                params_gap_0, params_gap_4 = np.zeros(5), np.zeros(5)
            elif self.test == 4:
                params_gap_4 = self.fit_gap_sigmoid(norm_flux, gap_id = 4)
                params_gap_0, params_gap_2 = np.zeros(5), np.zeros(5)
        else:
            ################### GAP 0 #####################
            params_gap_0 = self.fit_gap_sigmoid(norm_flux, gap_id = 0)
            ################### GAP 2 #####################
            params_gap_2 = self.fit_gap_sigmoid(norm_flux, gap_id = 2)
            ################### GAP 4 #####################
            params_gap_4 = self.fit_gap_sigmoid(norm_flux, gap_id = 4)
            ################### GAP 6 #####################
            # params_gap_6 = self.fit_gap_6(norm_flux, isEval=False)
        params_gap_6 = np.array([0])

        if self.debug:
            self.params[0] = params_gap_0
            self.params[2] = params_gap_2
            self.params[4] = params_gap_4
            self.params[6] = params_gap_6

        params = np.concatenate((params_gap_0, params_gap_2, params_gap_4, params_gap_6), axis = 0)
        return params
        
    def eval_gaps(self, norm_cont, gap_params):
        self.find_limits(self.wave)
        gap_params = np.where(gap_params == np.inf, 0, gap_params)
        params_gap_0 = gap_params[0:5]
        params_gap_2 = gap_params[5:10]
        params_gap_4 = gap_params[10:15]
        ################### GAP 0 #####################
        norm_cont = self.eval_gap_sigmoid(norm_cont, params_gap_0, gap_id = 0)
        ################### GAP 2 #####################
        norm_cont = self.eval_gap_sigmoid(norm_cont, params_gap_2, gap_id = 2)
        ################### GAP 4 #####################
        norm_cont = self.eval_gap_sigmoid(norm_cont, params_gap_4, gap_id = 4)
        ################### GAP 6 #####################
        # params_gap_6 = gap_params[-1]

        return norm_cont

    # def check_fit_gap(self, y, gap_id, message = ''):
    #     fit_gap = (len(y) > 3) and (y[0] < -0.005)
    #     if not fit_gap:
    #         # print(message)
    #         self.fit_gap[gap_id] = False
    #         return None
    #     else:
    #         self.fit_gap[gap_id] = True
    
    def check_fit(self, y):
        return (len(y) > 3) and (y[0] < -0.001)

    def fit_gap_sigmoid(self, norm_flux, gap_id = None):
        gap_control_pts = self.get_upper_points_for_gap(norm_flux, gap_id = gap_id)
        # if self.fit_gap[gap_id] is False: 
        if not self.check_fit(gap_control_pts[:, 1]):
            if self.debug: self.fit_gap[gap_id] = False
            return np.zeros(5)

        gap_hull_x, gap_hull_y, dd = self.get_slope_filtered_robust(gap_control_pts[:, 0],\
                                            gap_control_pts[:, 1])
        # self.check_fit_gap(gap_hull_y, gap_id, message = 'no hull left')
        # if self.fit_gap[gap_id] is False:
        if not self.check_fit(gap_hull_y):
            if self.debug: self.fit_gap[gap_id] = False
            return np.zeros(5)

        y0, slope_mid, x_mid = self.get_init_sigmoid_estimation(gap_hull_x, gap_hull_y, method = "interp1d")
        pmt = np.append([y0, slope_mid, x_mid], self.init_s0s1[gap_id])

        if self.debug:
            self.hb[gap_id] = gap_control_pts
            self.hull[gap_id] = np.column_stack((gap_hull_x, gap_hull_y))
            self.params_est[gap_id] = pmt

        try:
            # bnds = (0, np.inf)
            # bnds = (0, [np.inf, np.inf, np.inf, , 20])
            bnds = (0, np.inf)

            pmt, _ = curve_fit(self.sigmoid, gap_hull_x, gap_hull_y, pmt, bounds=bnds) 
            # pmt, _ = curve_fit(self.sigmoid, gap_hull_x, gap_hull_y, pmt) 
        except:
            self.fit_gap[gap_id] = False
            return np.zeros(5)
        return pmt

    def eval_gap_sigmoid(self, norm_cont, params, gap_id = None):
        if abs(params).sum() == 0:
            return norm_cont
        # if norm_cont is None: norm_cont = np.zeros_like(self.wave)
        mask = self.gap_masks[gap_id]
        model = self.sigmoid(self.wave[mask], *params)        
        norm_cont[mask] = model
        return norm_cont

    def fit_sigmoid_iterate(self, pts, w0, sigfun, pmt, thr = None):
        if (pts.shape[0]<7): return pmt
        for i in range(len(thr)):
            pts = self.clip_fitted_points(pts, sigfun, pmt, thr[i])
            pmt = self.fit_curve_with_sigmoid(pts[:, 0], pts[:, 1], w0, sigfun, pmt, (), type = 2)
        return pmt

    # def get_
    def get_min_max_norm(self, x):
        xmin, xmax = np.min(x), np.max(x)
        return (x - xmin)/(xmax - xmin)

    def get_slope_filtered(self, x, y, cutoff = 0):
        xx = self.get_min_max_norm(x)
        yy = self.get_min_max_norm(y)

        dd = np.diff(yy) / np.diff(xx)
        dd = np.abs(np.append(dd, dd[-1]))
        dd_median, dd_std = np.median(dd), dd.std()
        dd_high = dd_median + dd_std * 2
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

    def get_upper_points_for_gap(self, norm_flux, gap_id = None):
        mask = self.gap_masks[gap_id]
        offset = self.offset[gap_id]
        
        x = self.log_wave[mask][offset:]
        y = norm_flux[mask][offset:]
        dx = self.dx[gap_id]
        x1 = x[np.abs(y + 0.005).argmin()]
        if self.debug:
            self.x1[gap_id] = x1
        x_mask = (x < x1)

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

    def get_interval_max(self, x, y, dx = 500):
        N = x.shape[0]
        pad_row = np.int(np.floor(N/dx))+1 
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
############################################### Fit Functions ###############################

    def fit_linear(self, x, a, b):
        return  x*a + b

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
        #print('[fitS]',pmt)
        # pmt, pcov = curve_fit(sigfun, wx, wy, pmt)

        pmt, pcov = curve_fit(sigfun, wx, wy, pmt, bounds=bnds) 
        #print('[fitSigmoid2]','pmt:',pmt)
        return pmt

    def sigmoid2(self, x, a, b, c, t0, t1):
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
        s0 = s0+1
        s1 = s1+1
        x0 = c-1/(2*b)
        x1 = c+1/(2*b)

        beta0  = 2*b*s0
        alpha0 = 1/(2*s0*np.e)   
        beta1  = 2*b*s1
        alpha1 = 1/(2*s1*np.e)
        
        x0 = t0 - 1 / beta0
        x1 = t1 + 1 / beta1

        # t0 = x0+1/beta0
        # t1 = x1-1/beta1
        i0 = (x<=t0)
        i1 = (x>=t1)
        im = (x>t0) & (x<t1)
        
        y = np.zeros(x.shape)
        y[i0] = alpha0*np.exp(beta0*(x[i0]-x0))
        y[i1] = 1-alpha1*np.exp(-beta1*(x[i1]-x1))
        y[im] = b*(x[im]-c)+1/2
        
        return a*(y-1)
    
    def sigmoid_jac(self, x, a, b, c, s0, s1):
        s0 = s0 + 1
        s1 = s1 + 1
        x0 = c - 1 / (2 * b)
        x1 = c + 1 / (2 * b)

        beta0  = 2 * b * s0
        alpha0 = 1 / (2 * s0 * np.e)   
        beta1  = 2 * b * s1
        alpha1 = 1 / (2 * s1 * np.e)
        
        t0 = x0 + 1 / beta0
        t1 = x1 - 1 / beta1
        i0 = (x <= t0)
        i1 = (x >= t1)
        im = (x > t0) & (x < t1)
        
        y = np.zeros(x.shape)
        y[i0] = alpha0 * np.exp(beta0 * (x[i0] - x0))
        arg = - beta1 * (x[i1] - x1)
        # exp_arg = 0 if arg.all() < -1e4 else np.exp(arg)
        y[i1] = 1 - alpha1 * np.exp(arg)
        y[im] = b * (x[im] - c) + 0.5
        ans = a * (y - 1)

        # return np.derivative(ans, a), d ans/d b, d ans/ dc,  d ans/ d s0 

    def sigmoid(self, x, a, b, c, s0, s1):
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
        s0 = s0 + 1
        s1 = s1 + 1
        x0 = c - 1 / (2 * b)
        x1 = c + 1 / (2 * b)

        beta0  = 2 * b * s0
        alpha0 = 1 / (2 * s0 * np.e)   
        beta1  = 2 * b * s1
        alpha1 = 1 / (2 * s1 * np.e)
        
        t0 = x0 + 1 / beta0
        t1 = x1 - 1 / beta1
        i0 = (x <= t0)
        i1 = (x >= t1)
        im = (x > t0) & (x < t1)
        
        y = np.zeros(x.shape)
        y[i0] = alpha0 * np.exp(beta0 * (x[i0] - x0))
        arg = - beta1 * (x[i1] - x1)
        # exp_arg = 0 if arg.all() < -1e4 else np.exp(arg)
        y[i1] = 1 - alpha1 * np.exp(arg)
        y[im] = b * (x[im] - c) + 0.5

        return a * (y - 1)

    def sigmoid_fixing_right(self,x,rLevel,a,b,c,s0,s1):
        # q is the negative of the DC level at the right end
        return self.sigmoid(x,a-rLevel,b,c,s0,s1)+rLevel

    def fit_poly(self, hp,rank=4,plot=0):
        #-----------------------------------------------------
        # fits an orthogonal polynomial to a spectrum segment.
        # the fit is done in linear wavelength space
        # input/output is
        #    x: wavelength
        #    y: log(flux)
        #-----------------------------------------------------
        x, y = hp[:,0],  hp[:,1] 
        # set domain
        dom = (min(x),max(x))
        # check the number of points
        nn = x.shape[0]
        if (nn>4):
            fit, resid = np.polynomial.legendre.Legendre.fit(x, y, rank, domain=dom, full=True)
        elif (nn>=2) :
            fit, resid = np.polynomial.legendre.Legendre.fit(x, y, 1, domain=dom, full=True)
        else:
            logging.error('not enough points', nn)
            return None
        return fit, resid

    def fit_poly_clipped(self, pts, pmt = [], thr = None):
        fit, resid = self.fit_poly(pts,1)
        pts2       = self.clip_fitted_points(pts,fit,pmt,thr)
        fit, resid = self.fit_poly(pts2,1)
        return fit, resid

########################### FIT LEGENDRE #############################

    def fit_legendre(self, log_cont):
        fits = []
        params = np.zeros((len(self.gap_ids), self.legendre_deg))
        for i, n in enumerate(self.gap_ids):
            ff = self.fit_legendre_by_n(n, log_cont)
            params[i] = ff.coef
            fits.append(ff)
        return fits, params.flatten()
        
    def eval_legendre(self, fits):
        model_cont = np.zeros_like(self.log_wave)
        for n in range(len(fits)):
            ff = fits[n]
            model_cont[self.eval_masks[n]] = ff(self.log_wave[self.eval_masks[n]])
        return model_cont

    def fit_legendre_by_n(self, n, log_cont):
        #--------------------------------------------------------------
        # fit the nth segment n=(0,2,4,6) with a Legendre polynomial
        #--------------------------------------------------------------      
        rate = self.legendre_rate[n]
        xx = self.log_wave[self.masks[n]][::rate]
        yy = log_cont[self.masks[n]][::rate]

        ff, res = np.polynomial.legendre.Legendre.fit(xx, yy, self.legendre_rank,\
                domain=(xx[0],xx[-1]), full=True)
        assert np.sqrt(res[0]/xx.shape[0]).round(1) == 0
        return ff

    # def set_constants(self, wave, constants= [6,1]):
    #     self.chebyshev_degrees = int(constants[0])
    #     self.limits_dlambda = constants[1]
    #     self.find_limits(wave, self.limits_dlambda)

    #     for i in range(len(self.fit_limits)):
    #         self.fit_limits[i][0] = constants[2 + 2 * i + 0]
    #         self.fit_limits[i][1] = constants[2 + 2 * i + 1]

###########################DONE#############################
    def get_log(self, x):
        return np.log(np.where(x <= 1, 1, x))

    def get_photo_limits(self, d = 0.5):
        limits = np.array(Physics.HYDROGEN_LIMITS)
        cuts = np.sort((np.append(limits - d, limits + d)))
        return np.array([3000, *Physics.air_to_vac(cuts), 15000 + 1])

    def find_masks_between_limits(self, wave, dlambda):
        masks = []
        limits = []
        mm = np.zeros_like(wave)
        for i in range(len(self.photo_limits) - 1):
            mask = (wave >= self.photo_limits[i] + dlambda) & (wave < self.photo_limits[i + 1] - dlambda)

            masks.append(mask)
            wm = wave[mask]
            
            if wm.size > 0:
                limits.append([wave[mask].min(), wave[mask].max()])
            else:
                limits.append([np.nan, np.nan])
            mm[mask] = i
        self.mm = mm
        return masks, limits

    def find_limits(self, wave):
        if self.wave_mask is None:
            self.wave_mask = (wave > 3000.) & (wave < 14000.)

        if self.wave is None:
            self.wave = wave[self.wave_mask]
            self.log_wave = np.log(self.wave)

        if self.masks is None:
            self.masks, self.limits = self.find_masks_between_limits(self.wave, dlambda=0)

        if self.eval_masks is None:
            self.eval_masks = [self.masks[0], self.masks[1] | self.masks[2],
                    self.masks[3] | self.masks[4], self.masks[5] | self.masks[6]]

        if self.gap_masks is None:
            self.gap_masks = self.get_gap_masks()

        if self.dx is None:
            mask = (self.wave > 3000) & (self.wave < 3006) 
            dx = int(len(self.wave[mask]))
            # [200, 0, 400, 0, 200, 0, 200]
            self.dx = [dx, 0, 2 * dx, 0, dx, 0, dx]
        
        if self.legendre_rate is None:
            dx = int(self.dx[0] / 2)
            self.legendre_rate = [dx, 1, 3 * dx, 1, 2 * dx, 1, 25]
            # print(legendre_rate)
    
    def get_gap_masks(self):
        hi = [3600, None, 5000, None, 10000, None, 15000]
        gap_masks = {}
        for i in self.gap_ids:
            mask = self.masks[i] 
            gap_masks[i] = mask & (self.wave < hi[i])
        return gap_masks
            
    # def fit_between_limits_cheby(self, wave, flux):
    #     self.find_limits(wave, self.limits_dlambda)

    #     pp = []
    #     for i in range(len(self.fit_masks)):
    #         mask = self.fit_masks[i]
    #         wave_min, wave_max = self.fit_limits[i]
            
    #         p = np.polynomial.chebyshev.chebfit(
    #             (wave[mask] - wave_min) / (wave_max - wave_min), 
    #             flux[mask], 
    #             deg=self.chebyshev_degrees)
    #         pp.append(p)
        
    #     return np.concatenate(pp)

    def eval_between_limits_cheby(self, wave, pp):
        self.find_limits()

        flux = np.full(wave.shape, np.nan)

        for i in range(len(self.eval_masks)):
            mask = self.eval_masks[i]
            wave_min, wave_max = self.eval_limits[i]

            if wave_min is not None and wave_max is not None:
                flux[mask] = np.polynomial.chebyshev.chebval(
                    (wave[mask] - wave_min) / (wave_max - wave_min), 
                    pp[i * (self.chebyshev_degrees + 1): (i + 1) * (self.chebyshev_degrees + 1)])

        return flux



    def denormalize(self, spec, params):
        wave = spec.wave
        norm = 4 * np.log10(spec.T_eff)
        model = self.eval(wave, params)
        
        spec.flux = 10**(spec.flux + norm + model)
        if spec.cont is not None:
            spec.cont = 10**(spec.cont + norm)

    
    def fit_gap_6(self, norm_flux, isEval = True):
        na, nb, lo, hi, dx, thr = self.gap_params[3]       
        wa, qa, mask_left =  self.get_margin(na, norm_flux, lo = lo)
        ha = self.get_control_points(wa, qa, dx)
        ca = ha[-10:,:]

        wb, qb, mask_right =  self.get_margin(nb, norm_flux, hi = hi)
        hb = self.get_control_points(wb, qb, dx)
        cb = hb[:10,:]
        if isEval: 
            return ca, cb, mask_left, mask_right
        return np.concatenate((ca, cb))
        # return ca, wa,qa,wb,qb,lo,hi,fita, fitb

    def eval_gap_6(self, norm_cont, ca, cb, mask_left, mask_right, pmt = []):
        thr = self.sigma_clip_threshold[3][3]
        eval_gap_6_left, ra = self.fit_poly_clipped(ca, thr = thr)
        eval_gap_6_right, rb = self.fit_poly_clipped(cb, thr = thr)
        left_index = np.where(mask_left)[0][-800]
        mask_left[:left_index] = False
        model_left = eval_gap_6_left(self.wave[mask_left], *pmt)
        model_right = eval_gap_6_right(self.wave[mask_right], *pmt)
        norm_cont[mask_left] = model_left
        norm_cont[mask_right] = model_right

        # mask = mask_left | mask_right
        # model = np.concatenate((model_left, model_right))
        return norm_cont


    def get_constants(self, wave):
        return np.array([])

####################################################################################################################
    # def fit_gap_4(self, norm_flux):
    #     sigfun = self.sigmoid
    #     gap_4_control_pts = self.get_control_points_gap(norm_flux, gap_id = 4)
    #     cb2 = copy.deepcopy(gap_4_control_pts)

    #     w0 = 8500      
    #     pmt = [-cb2[0,1], 0.005, w0, 0.25, 0.1]
    #     pmt0 = self.fit_curve_with_sigmoid(cb2[:, 0], cb2[:, 1], w0, sigfun, pmt, (), type = 2)
    #     pmt = self.fit_sigmoid_iterate(cb2, w0, sigfun, pmt0, thr = [3, 2])
    #     if isEval: 
    #         return gap_4_control_pts, sigfun, pmt
    #     return pmt

    # def fit_gap_2(self, norm_flux, isEval = True):
    #     sigfun = self.sigmoid
    #     gap_2_control_pts = self.get_control_points_gap(norm_flux, gap_id=2)
    #     cb2 = copy.deepcopy(gap_2_control_pts)

    #     w0 = 3750      
    #     pmt = [-cb2[0,1], 0.008, w0, 0.15, 0.1]
    #     pmt0 = self.fit_curve_with_sigmoid(cb2[:, 0], cb2[:, 1], w0, sigfun, pmt, (), type = 2)
    #     pmt = self.fit_sigmoid_iterate(cb2, w0, sigfun, pmt0, thr = [2, 0.5])
    #     if isEval: 
    #         return gap_2_control_pts, sigfun, pmt
    #     return pmt

    # def fit_gap_4(self, norm_flux, isEval = True):
    #     gap_4_control_pts, mask, sigfun, pmt = self.fit_gap_sigmoid(norm_flux)
    #     if isEval: return gap_4_control_pts, mask, sigfun, pmt
    #     return pmt

    # def fit_gap_0(self, norm_flux):
    #     gap_id = 0
    #     sigfun = self.sigfun[gap_id]
    #     gap_control_pts = self.get_control_points_for_gap(norm_flux, gap_id = gap_id)
    #     if self.debug:
    #         self.cc[gap_id] = gap_control_pts
    #     gap_0_control_pts, mask = self.get_control_points_for_gap(norm_flux, gap_id=0)
    #     cc = [0,0,0,0,0,0]
    #     return cc

        # def get_control_points_for_gap(self, norm_flux, ):
    #     hb = self.get_control_points_by_segment(gap_id, norm_flux)
        # if self.debug:
        #     self.hb[gap_id] = hb
        # hb = hb[5:, :]
        # cb = hb[hb[:,0]<9200,:]
        # cc = self.fit_gap_robust(cb, gap_id = gap_id,  isEval = False)
        # return cc
        # if self.method == 'acc':
        #     return hb[10:,:]
        # elif self.method == 'dx':
        #     cb = hb[hb[:,0]<9200,:]
        #     cc = self.fit_gap_robust(cb, gap_id = gap_id,  isEval = False)
        #     return cc
        # else:
        #     return hb

    # def get_control_points_for_gap_accumulate(self, norm_flux, gap_id = None):

    #     hb = self.get_control_points_by_segment(gap_id, norm_flux)
    #     cb = hb[hb[:,0]<9200,:]
    #     cc = self.fit_gap_robust(cb, gap_id = gap_id,  isEval = False)
    #     return cc