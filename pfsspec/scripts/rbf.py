import os
import logging
import numpy as np

from pfsspec.scripts.configurations import RBF_CONFIGURATIONS
from pfsspec.scripts.script import Script

class Rbf(Script):
    def __init__(self):
        super(Rbf, self).__init__()

        self.rbf = None

    def add_subparsers(self, parser):
        sps = parser.add_subparsers(dest='source')
        for src in RBF_CONFIGURATIONS:
            ps = sps.add_parser(src)
            spp = ps.add_subparsers(dest='rbf')
            for rbf in RBF_CONFIGURATIONS[src]:
                pp = spp.add_parser(rbf)
                self.add_args(pp)
                c = RBF_CONFIGURATIONS[src][rbf]['config']
                p = RBF_CONFIGURATIONS[src][rbf]['class'](c)
                p.add_args(pp)

    def add_args(self, parser):
        super(Rbf, self).add_args(parser)

        parser.add_argument('--in', type=str, help="Input data path.\n")
        parser.add_argument('--out', type=str, help='Output data path.\n')

    def create_rbf(self):
        config = RBF_CONFIGURATIONS[self.args['source']][self.args['rbf']]['config']
        self.rbf = RBF_CONFIGURATIONS[self.args['source']][self.args['rbf']]['class'](config)
        self.rbf.args = self.args
        self.rbf.parse_args() 

    def open_data(self):
        self.rbf.open_data(self.args['in'], self.outdir)

    def save_data(self):
        self.rbf.save_data(self.outdir)

    def prepare(self):
        super(Rbf, self).prepare()

        self.outdir = self.args['out']
        self.create_output_dir(self.outdir, resume=False)
        self.save_command_line(os.path.join(self.outdir, 'command.sh'))

        self.create_rbf()
        self.open_data()

    def run(self):
        self.rbf.run()
        self.save_data()

def main():
    script = Rbf()
    script.execute()

if __name__ == "__main__":
    main()