from pfsspec.pfsobject import PfsObject

class ContinuumModel(PfsObject):
    def __init__(self, orig=None):
        super(ContinuumModel, self).__init__()

    def init_from_args(self, parser):
        pass

    def parse_args(self, args):
        pass

    def fit(self, wave, flux):
        raise NotImplementedError()

    def eval(self, wave, params):
        raise NotImplementedError()

    def normalize(self, spec, params):
        raise NotImplementedError()

    def denormalize(self, spec, params):
        raise NotImplementedError()