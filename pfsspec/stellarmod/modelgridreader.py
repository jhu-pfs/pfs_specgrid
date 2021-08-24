import logging

from pfsspec.data.gridreader import GridReader

class ModelGridReader(GridReader):
    def __init__(self, grid=None, orig=None):
        super(ModelGridReader, self).__init__(grid=grid, orig=orig)

        if isinstance(orig, ModelGridReader):
            self.wave = orig.wave
            self.resolution = orig.resolution

            self.reader = orig.reader
        else:
            self.wave = None
            self.resolution = 5000

            self.reader = None

    def add_args(self, parser):
        super(ModelGridReader, self).add_args(parser)
        parser.add_argument("--lambda", type=float, nargs=2, default=None, help="Wavelength limits.\n")
        parser.add_argument("--resolution", type=int, default=None, help="Resolution.\n")

    def init_from_args(self, args):
        super(ModelGridReader, self).init_from_args(args)

        self.wave = self.get_arg('lambda', self.wave, args)
        self.resolution = self.get_arg('resolution', self.resolution, args)
