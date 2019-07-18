#!python

import os

from pfsspec.scripts.import_ import Import
from pfsspec.stellarmod.kuruczspectrumreader import KuruczSpectrumReader

class ImportKurucz(Import):
    def __init__(self):
        super(ImportKurucz, self).__init__()

    def add_args(self):
        super(ImportKurucz, self).add_args()
        self.parser.add_argument("--grid", type=str,
                                 choices=['kurucz', 'nover', 'anover', 'odfnew', 'aodfnew'],
                                 default='kurucz', help="Model subtype\n")

    def run(self):
        super(ImportKurucz, self).run()

        grid = KuruczSpectrumReader.read_grid(self.args['path'], self.args['grid'])
        grid.save(os.path.join(self.args['out'], 'spectra.npz'))

def main():
    script = ImportKurucz()
    script.execute()

if __name__ == "__main__":
    main()