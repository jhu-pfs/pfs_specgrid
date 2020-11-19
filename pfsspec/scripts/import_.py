import os
import logging
import numpy as np

from pfsspec.data.dataset import Dataset
from pfsspec.scripts.script import Script

class Import(Script):
    def __init__(self):
        super(Import, self).__init__()
        self.path = None
        self.outdir = None
        self.resume = False

    def add_args(self, parser):
        super(Import, self).add_args(parser)
        parser.add_argument("--path", type=str, required=True, help="Model/data directory base path\n")
        parser.add_argument("--out", type=str, required=True, help="Output file, must be .h5 or .npz\n")
        parser.add_argument('--resume', action='store_true', help='Resume existing but aborted import.\n')

    def parse_args(self):
        super(Import, self).parse_args()

        self.path = self.get_arg('path', self.path)
        self.outdir = self.get_arg('out', self.outdir)
        self.resume = self.get_arg('resume', self.resume)

    def prepare(self):
        super(Import, self).prepare()
        
        self.create_output_dir(self.args['out'], resume=self.resume)
        self.save_command_line(os.path.join(self.outdir, 'command.sh'))

    def run(self):
        self.init_logging(self.outdir)