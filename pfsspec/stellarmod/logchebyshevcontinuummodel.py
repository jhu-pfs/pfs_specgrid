import numpy as np

from pfsspec.physics import Physics
from pfsspec.stellarmod.continuummodel import ContinuumModel

class LogChebyshevContinuumModel(ContinuumModel):
    def __init__(self, orig=None):
        if isinstance(orig, LogChebyshevContinuumModel):
            self.photo_limits = orig.photo_limits

            self.fit_masks = orig.fit_masks
            self.fit_limits = orig.fit_limits
            self.cont_masks = orig.cont_masks
            self.cont_limits = orig.cont_limits
        else:
            self.photo_limits = Physics.air_to_vac(Physics.HYDROGEN_LIMITS)

            self.fit_masks = None
            self.fit_limits = None
            self.cont_masks = None
            self.cont_limits = None

    def add_args(self, parser):
        super(LogChebyshevContinuumModel, self).add_args(parser)

    def parse_args(self, args):
        super(LogChebyshevContinuumModel, self).parse_args(args)

    def find_masks_between_limits(self, wave, dlambda):
        wave_min = wave.min()
        wave_max = wave.max()

        masks = []
        limits = []

        for i in range(len(self.photo_limits) + 1):
            if i == 0:
                mask = wave < self.photo_limits[i] - dlambda
            elif i == len(self.photo_limits):
                mask = wave >= self.photo_limits[-1] + dlambda
            else:
                mask = (wave >= self.photo_limits[i - 1] + dlambda) & (wave < self.photo_limits[i] - dlambda)

            masks.append(mask)
            wm = wave[mask]
            if wm.size > 0:
                limits.append((wave[mask].min(), wave[mask].max()))
            else:
                limits.append((None, None))

        return masks, limits

    def fit_between_limits(self, wave, flux):
        # TODO: These should go into parameters
        chebdeg = 6
        dlambda = 1

        if self.fit_masks is None:
            self.fit_masks, self.fit_limits = self.find_masks_between_limits(wave, dlambda=dlambda)

        pp = []
        for i in range(len(self.fit_masks)):
            mask = self.fit_masks[i]
            wave_min, wave_max = self.fit_limits[i]
            
            p = np.polynomial.chebyshev.chebfit((wave[mask] - wave_min) / (wave_max - wave_min), flux[mask], deg=chebdeg)
            pp.append(p)
        
        return np.concatenate(pp)

    def eval_between_limits(self, wave, pp):
        # TODO: These should go into parameters
        chebdeg = 6

        if self.cont_masks is None:
            self.cont_masks, self.cont_limits = self.find_masks_between_limits(wave, dlambda=0)

        flux = np.full(wave.shape, np.nan)

        for i in range(len(self.cont_masks)):
            mask = self.cont_masks[i]
            wave_min, wave_max = self.cont_limits[i]

            if wave_min is not None and wave_max is not None:
                flux[mask] = np.polynomial.chebyshev.chebval((wave[mask] - wave_min) / (wave_max - wave_min), pp[i * (chebdeg + 1): (i + 1) * (chebdeg + 1)])

        return flux

    def fit(self, wave, flux):
        params = self.fit_between_limits(wave, flux)
        return params

    def eval(self, wave, params):
        flux = self.eval_between_limits(wave, params)
        return flux

    def normalize(self, spec):
        wave = spec.wave
        norm = 4 * np.log10(spec.T_eff)
        cont = np.log10(spec.cont) - norm

        params = self.fit(wave, cont)
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