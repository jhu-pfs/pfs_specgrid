from pfsspec.common.pfsobject import PfsObject

class Importer(PfsObject):
    def __init__(self, orig=None):
        super(Importer, self).__init__()

    def add_args(self, parser):
        pass

    def init_from_args(self, args):
        pass

    def execute_notebooks(self, script):
        pass