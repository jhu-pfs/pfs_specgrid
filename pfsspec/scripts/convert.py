import os
import logging
import numpy as np

from pfsspec.data.dataset import Dataset
from pfsspec.scripts.script import Script
from pfsspec.obsmod.filter import Filter

class Convert(Script):
    def __init__(self):
        super(Convert, self).__init__()
        self.outdir = None

    def add_args(self):
        super(Convert, self).add_args()
        self.parser.add_argument('--in', type=str, help="Data set directory\n")
        self.parser.add_argument('--out', type=str, help='Training set output directory\n')
        self.parser.add_argument('--norm', action='store_true', help='Normalize\n')
        self.parser.add_argument('--normfilt', type=str, default=None, help='Normalize in filter\n')
        self.parser.add_argument('--normmag', type=float, default=None, help='Normalize to mag in filter\n')
        self.parser.add_argument('--hipass', type=float, help='High-pass filter v_disp\n')
        self.parser.add_argument('--wave', type=float, nargs=2, default=(3400, 8800), help='Wavelength range\n')
        self.parser.add_argument('--wavebins', type=int, default=2000, help='Number of wavelength bins\n')
        self.parser.add_argument('--wavelog', action='store_true', help='Logarithmic wavelength binning\n')

    def init_pipeline(self, pipeline):
        pipeline.normalize = self.args['norm']

        if self.args['normfilt'] is not None:
            pipeline.normalize = True
            pipeline.normalize_filter = Filter()
            pipeline.normalize_filter.read(self.args['normfilt'])
            pipeline.normalize_mag = self.args['normmag']

        if 'hipass' in self.args and self.args['hipass'] is not None:
            pipeline.high_pass_vdisp = self.args['hipass']

        if self.args['wavelog']:
            pipeline.rebin = np.logspace(np.log10(self.args['wave'][0]), np.log10(self.args['wave'][1]), self.args['wavebins'])
        else:
            pipeline.rebin = np.linspace(self.args['wave'][0], self.args['wave'][1], self.args['wavebins'])

    def prepare(self):
        super(Convert, self).prepare()
        self.outdir = self.args['out']
        self.create_output_dir(self.outdir)

    def run(self):
        super(Convert, self).run()
