import numpy as np

from pfsspec.pfsobject import PfsObject

class ContinuumModel(PfsObject):
    def __init__(self, orig=None):
        super(ContinuumModel, self).__init__()

    def add_args(self, parser):
        pass

    def init_from_args(self, parser):
        pass

    def get_constants(self):
        raise NotImplementedError()

    def set_constants(self, constants):
        raise NotImplementedError()

    def init_wave(self, wave):
        raise NotImplementedError()

    def fit(self, spec):
        raise NotImplementedError()

    def eval(self, params):
        raise NotImplementedError()

    def normalize(self, spec, params):
        raise NotImplementedError()

    def denormalize(self, spec, params):
        raise NotImplementedError()

    def smooth_params(self, params):
        raise NotImplementedError()

    def fit_model_simple(self, model, x, y, w=None, p0=None):
        # Simple chi2 fit to x and y with optional weights
        params = model.fit(x, y, w=w, p0=p0)
        return params

    def fit_model_sigmaclip(self, model, x, y, w=None, p0=None, sigma_low=1, sigma_high=1):
        w = w if w is not None else np.full(x.shape, 1.0)
        m = np.full(x.shape, True)
        p = p0
        for i in range(5):
            p = model.fit(x[m], y[m], w=w[m], p0=p)
            f = model.eval(x, p)
            
            # Sigma clipping
            s = np.std(f - y)
            m = (y < f + sigma_high * s) & (f - sigma_low * s < y)
        return p

    def eval_model(self, model, x, params):
        return model.eval(x, params)