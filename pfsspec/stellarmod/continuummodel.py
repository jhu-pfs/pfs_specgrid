from pfsspec.pfsobject import PfsObject

class ContinuumModel(PfsObject):
    def __init__(self, orig=None):
        super(ContinuumModel, self).__init__()

    def add_args(self, parser):
        pass

    def parse_args(self, args):
        pass

    def fit(self, spec):
        raise NotImplementedError()

    def eval(self, spec, params):
        raise NotImplementedError()