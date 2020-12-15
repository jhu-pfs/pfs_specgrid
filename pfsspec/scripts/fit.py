import os
import logging
import numpy as np

from pfsspec.scripts.configurations import FIT_CONFIGURATIONS
from pfsspec.scripts.script import Script

class Fit(Script):
    def __init__(self):
        super(Fit, self).__init__()

        self.pca = None

    def add_subparsers(self, parser):
        tps = parser.add_subparsers(dest='type')
        for t in FIT_CONFIGURATIONS:
            tp = tps.add_parser(t)
            sps = tp.add_subparsers(dest='source')
            for s in FIT_CONFIGURATIONS[t]:
                sp = sps.add_parser(s)
                mps = sp.add_subparsers(dest='model')
                for m in FIT_CONFIGURATIONS[t][s]['models']:
                    mp = mps.add_parser(m)
                    self.add_args(mp)

                    gg = FIT_CONFIGURATIONS[t][s]['grid']()
                    gg.add_args(mp)

                    mm = FIT_CONFIGURATIONS[t][s]['models'][m]()
                    mm.add_args(mp)

    def add_args(self, parser):
        super(Fit, self).add_args(parser)

        parser.add_argument('--in', type=str, help="Input data path.\n")
        parser.add_argument('--out', type=str, help='Output data path.\n')

    def create_fit(self):
        self.fit = FIT_CONFIGURATIONS[self.args['type']][self.args['source']]['grid']()
        self.fit.parallel = self.threads != 1
        self.fit.threads = self.threads
        self.fit.parse_args(self.args)
        self.fit.model = self.model

    def create_model(self):
        self.model = FIT_CONFIGURATIONS[self.args['type']][self.args['source']]['models'][self.args['model']]()
        self.model.parse_args(self.args)

    def open_data(self):
        self.fit.open_data(self.args['in'], self.outdir)

    def prepare(self):
        super(Fit, self).prepare()

        self.outdir = self.args['out']
        self.create_output_dir(self.outdir, resume=False)
        self.save_command_line(os.path.join(self.outdir, 'command.sh'))

        self.create_model()
        self.create_fit()
        self.open_data()

    def run(self):
        self.fit.run()
        self.fit.save_data()

def main():
    script = Fit()
    script.execute()

if __name__ == "__main__":
    main()