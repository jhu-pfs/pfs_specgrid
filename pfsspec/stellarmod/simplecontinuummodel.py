import numpy as np

from pfsspec.physics import Physics
from pfsspec.stellarmod.continuummodel import ContinuumModel

class SimpleContinuumModel(ContinuumModel):
    def __init__(self, orig=None):
        if isinstance(orig, SimpleContinuumModel):
            self.photo_limits = orig.photo_limits

            self.fit_masks = orig.fit_masks
            self.fit_limits = orig.fit_limits
            self.cont_masks = orig.cont_masks
        else:
            self.photo_limits = Physics.air_to_vac(Physics.HYDROGEN_LIMITS)

            self.fit_masks = None
            self.fit_limits = None
            self.cont_masks = None

    def add_args(self, parser):
        super(SimpleContinuumModel, self).add_args(parser)

    def parse_args(self, args):
        super(SimpleContinuumModel, self).parse_args(args)

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
            limits.append((wave[mask].min(), wave[mask].max()))

        return masks, limits

    def fit_between_limits(self, spec):
        # TODO: These should go into parameters
        chebdeg = 6
        dlambda = 1

        if self.fit_masks is None:
            self.fit_masks, self.fit_limits = self.find_masks_between_limits(spec.wave, dlambda=dlambda)

        wave = spec.wave
        norm = 4 * np.log10(spec.T_eff)
        cont = np.log10(spec.cont) - norm

        pp = []
        pp.append(np.array([norm]))
        for i in range(len(self.fit_masks)):
            mask = self.fit_masks[i]
            wave_min, wave_max = self.fit_limits[i]
            
            p = np.polynomial.chebyshev.chebfit((wave[mask] - wave_min) / (wave_max - wave_min), cont[mask], deg=chebdeg)
            pp.append(p)
        
        return np.concatenate(pp)

    def eval_between_limits(self, spec, pp):
        # TODO: These should go into parameters
        chebdeg = 6

        if self.cont_masks is None:
            self.cont_masks, _ = self.find_masks_between_limits(spec.wave, dlambda=0)

        wave = spec.wave
        cont = np.full(spec.wave.shape, np.nan)

        for i in range(len(self.cont_masks)):
            mask = self.cont_masks[i]
            wave_min, wave_max = self.fit_limits[i]

            cont[mask] = pp[0] + np.polynomial.chebyshev.chebval((wave[mask] - wave_min) / (wave_max - wave_min), pp[1 + i * (chebdeg + 1):1 + (i + 1) * (chebdeg + 1)])

        return cont

    def fit(self, spec):
        return self.fit_between_limits(spec)

    def eval(self, spec, params):
        spec.cont = self.eval_between_limits(spec, params)
        
        mask = spec.flux == 0
        spec.flux = np.log10(spec.flux)
        spec.flux[mask] = -3