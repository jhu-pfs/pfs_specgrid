import os
import logging
import numpy as np

from pfsspec.scripts.configurations import PCA_CONFIGURATIONS
from pfsspec.scripts.script import Script

class Pca(Script):
    def __init__(self):
        super(Pca, self).__init__()

        self.pca = None

    def add_subparsers(self, parser):
        sps = parser.add_subparsers(dest='type')
        for src in PCA_CONFIGURATIONS:
            ps = sps.add_parser(src)
            spp = ps.add_subparsers(dest='source')
            for pca in PCA_CONFIGURATIONS[src]:
                pp = spp.add_parser(pca)
                self.add_args(pp)
                c = PCA_CONFIGURATIONS[src][pca]['config']
                p = PCA_CONFIGURATIONS[src][pca]['class'](c)
                p.add_args(pp)

    def add_args(self, parser):
        super(Pca, self).add_args(parser)

        parser.add_argument('--in', type=str, help="Input data path.\n")
        parser.add_argument('--out', type=str, help='Output data path.\n')
        parser.add_argument('--params', type=str, help='Parameters grid, if different from input.\n')

    def create_pca(self):
        config = PCA_CONFIGURATIONS[self.args['type']][self.args['source']]['config']
        self.pca = PCA_CONFIGURATIONS[self.args['type']][self.args['source']]['class'](config)
        self.pca.args = self.args
        self.pca.parse_args() 

    def open_data(self):
        self.pca.open_data(self.args['in'], self.outdir, self.args['params'])

    def save_data(self):
        self.pca.save_data(self.outdir)

    def prepare(self):
        super(Pca, self).prepare()

        self.outdir = self.args['out']
        self.create_output_dir(self.outdir, resume=False)
        self.save_command_line(os.path.join(self.outdir, 'command.sh'))

        self.create_pca()
        self.open_data()

    def run(self):
        self.pca.run()
        self.save_data()

def main():
    script = Pca()
    script.execute()

if __name__ == "__main__":
    main()