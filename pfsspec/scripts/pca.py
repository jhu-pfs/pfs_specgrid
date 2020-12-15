import os
import logging
import numpy as np

from pfsspec.scripts.configurations import PCA_CONFIGURATIONS
from pfsspec.scripts.script import Script

class PCA(Script):
    def __init__(self):
        super(PCA, self).__init__()

        self.pca = None

    def add_subparsers(self, parser):
        sps = parser.add_subparsers(dest='source')
        for src in PCA_CONFIGURATIONS:
            ps = sps.add_parser(src)
            spp = ps.add_subparsers(dest='pca')
            for pca in PCA_CONFIGURATIONS[src]:
                pp = spp.add_parser(pca)
                
                self.add_args(pp)
                p = PCA_CONFIGURATIONS[src][pca]()
                p.add_args(pp)

    def add_args(self, parser):
        super(PCA, self).add_args(parser)

        parser.add_argument('--in', type=str, help="Input data path.\n")
        parser.add_argument('--out', type=str, help='Output data path.\n')

    def create_pca(self):
        self.pca = PCA_CONFIGURATIONS[self.args['source']][self.args['pca']]()
        self.pca.parse_args(self.args) 

    def open_data(self):
        self.pca.open_data(self.args['in'], self.outdir)

    def save_data(self):
        self.pca.save_data(self.outdir)

    def prepare(self):
        super(PCA, self).prepare()

        self.outdir = self.args['out']
        self.create_output_dir(self.outdir, resume=False)
        self.save_command_line(os.path.join(self.outdir, 'command.sh'))

        self.create_pca()
        self.open_data()

    def run(self):
        self.pca.run()

        self.save_data()

def main():
    script = PCA()
    script.execute()

if __name__ == "__main__":
    main()