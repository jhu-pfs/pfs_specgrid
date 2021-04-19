import os
import logging

from pfsspec.scripts.configurations import FIT_CONFIGURATIONS
from pfsspec.scripts.script import Script

class Fit(Script):
    def __init__(self):
        super(Fit, self).__init__()

        self.fit = None

    def add_subparsers(self, parser):
        tps = parser.add_subparsers(dest='type')
        for t in FIT_CONFIGURATIONS:
            tp = tps.add_parser(t)
            sps = tp.add_subparsers(dest='source')
            for s in FIT_CONFIGURATIONS[t]:
                sp = sps.add_parser(s)
                self.add_args(sp)
                cc = FIT_CONFIGURATIONS[t][s]['config']
                gg = FIT_CONFIGURATIONS[t][s]['class'](cc)
                gg.add_args(sp)

    def add_args(self, parser):
        super(Fit, self).add_args(parser)

        parser.add_argument('--in', type=str, help="Input data path.\n")
        parser.add_argument('--out', type=str, help='Output data path.\n')

    def create_fit(self):
        config = FIT_CONFIGURATIONS[self.args['type']][self.args['source']]['config']
        self.fit = FIT_CONFIGURATIONS[self.args['type']][self.args['source']]['class'](config)
        self.fit.parallel = self.threads != 1
        self.fit.threads = self.threads
        self.fit.args = self.args
        self.fit.parse_args()

    def open_data(self):
        self.fit.open_data(self.args['in'], self.outdir)

    def prepare(self):
        super(Fit, self).prepare()

        self.outdir = self.args['out']
        self.create_output_dir(self.outdir, resume=False)
        self.save_command_line(os.path.join(self.outdir, 'command.sh'))

        self.create_fit()
        self.open_data()

    def run(self):
        self.init_logging(self.outdir)
        self.fit.run()
        self.fit.save_data(self.outdir)

def main():
    script = Fit()
    script.execute()

if __name__ == "__main__":
    main()