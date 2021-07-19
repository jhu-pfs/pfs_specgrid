import numpy as np

from pfsspec.physics import Physics
from pfsspec.stellarmod.continuummodels.continuummodel import ContinuumModel

class Chebyshev(ContinuumModel):
    def __init__(self, orig=None):
        if isinstance(orig, Chebyshev):
            self.photo_limits = orig.photo_limits

            self.chebyshev_degrees = orig.chebyshev_degrees
            self.limits_dlambda = orig.limits_dlambda

            self.fit_masks = orig.fit_masks
            self.fit_limits = orig.fit_limits
            self.eval_masks = orig.eval_masks
            self.eval_limits = orig.eval_limits
        else:
            limits = [2530,] + Physics.HYDROGEN_LIMITS + [17500,]
            self.photo_limits = Physics.air_to_vac(np.array(limits))

            self.chebyshev_degrees = 6
            self.limits_dlambda = 1

            self.fit_masks = None
            self.fit_limits = None
            self.eval_masks = None
            self.eval_limits = None

    @property
    def name(self):
        return "chebyshev"

    def add_args(self, parser):
        super(Chebyshev, self).add_args(parser)

    def init_from_args(self, args):
        super(Chebyshev, self).init_from_args(args)

    def get_constants(self, wave):
        self.find_limits(wave, self.limits_dlambda)

        constants = []
        constants.append(self.chebyshev_degrees)
        constants.append(self.limits_dlambda)
        for i in range(len(self.fit_limits)):
            constants.append(self.fit_limits[i][0])
            constants.append(self.fit_limits[i][1])

        return np.array(constants)

    def set_constants(self, wave, constants):
        self.chebyshev_degrees = int(constants[0])
        self.limits_dlambda = constants[1]
        self.find_limits(wave, self.limits_dlambda)

        for i in range(len(self.fit_limits)):
            self.fit_limits[i][0] = constants[2 + 2 * i + 0]
            self.fit_limits[i][1] = constants[2 + 2 * i + 1]

    def find_masks_between_limits(self, wave, dlambda):
        masks = []
        limits = []

        for i in range(len(self.photo_limits) - 1):
            mask = (wave >= self.photo_limits[i] + dlambda) & (wave < self.photo_limits[i + 1] - dlambda)

            masks.append(mask)
            wm = wave[mask]
            
            if wm.size > 0:
                limits.append([wave[mask].min(), wave[mask].max()])
            else:
                limits.append([np.nan, np.nan])

        return masks, limits

    def find_limits(self, wave, dlambda):
        if self.fit_masks is None:
            self.fit_masks, self.fit_limits = self.find_masks_between_limits(wave, dlambda=dlambda)
        
        if self.eval_masks is None:
            self.eval_masks, self.eval_limits = self.find_masks_between_limits(wave, dlambda=0)
            
            # Extrapolate continuum to the edges
            # Equality must be allowed here because eval_limits are calculated by taking
            # wave[mask].min/max which are the actual wavelength grid values
            self.eval_masks[0] = (wave <= self.eval_limits[0][1])
            self.eval_masks[-1] = (wave >= self.eval_limits[-1][0])

    def fit_between_limits(self, wave, flux):
        self.find_limits(wave, self.limits_dlambda)

        pp = []
        for i in range(len(self.fit_masks)):
            mask = self.fit_masks[i]
            wave_min, wave_max = self.fit_limits[i]
            
            p = np.polynomial.chebyshev.chebfit(
                (wave[mask] - wave_min) / (wave_max - wave_min), 
                flux[mask], 
                deg=self.chebyshev_degrees)
            pp.append(p)
        
        return np.concatenate(pp)

    def eval_between_limits(self, wave, pp):
        self.find_limits(wave, self.limits_dlambda)

        flux = np.full(wave.shape, np.nan)

        for i in range(len(self.eval_masks)):
            mask = self.eval_masks[i]
            wave_min, wave_max = self.eval_limits[i]

            if wave_min is not None and wave_max is not None:
                flux[mask] = np.polynomial.chebyshev.chebval(
                    (wave[mask] - wave_min) / (wave_max - wave_min), 
                    pp[i * (self.chebyshev_degrees + 1): (i + 1) * (self.chebyshev_degrees + 1)])

        return flux

    def fit(self, spec):
        params = self.fit_between_limits(spec.wave, spec.flux)
        return params

    def eval(self, wave, params):
        flux = self.eval_between_limits(wave, params)
        return wave, flux

    def normalize(self, spec, params):
        wave = spec.wave
        norm = 4 * np.log10(spec.T_eff)
        cont = np.log10(spec.cont) - norm

        params = self.fit(wave, cont)       ## TODO
        model = self.eval(wave, params)

        spec.cont = cont
        spec.flux = np.log10(spec.flux) - norm - model
        
        return params

    def denormalize(self, spec, params):
        wave = spec.wave
        norm = 4 * np.log10(spec.T_eff)
        model = self.eval(wave, params)
        
        spec.flux = 10**(spec.flux + norm + model)
        if spec.cont is not None:
            spec.cont = 10**(spec.cont + norm)
        else:
            spec.cont = 10**norm