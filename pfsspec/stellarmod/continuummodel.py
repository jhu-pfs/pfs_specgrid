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