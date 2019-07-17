#!python

import os
import logging

from pfsspec.scripts.convert import Convert
from pfsspec.stellarmod.kuruczgrid import KuruczGrid
from pfsspec.stellarmod.modelgriddatasetbuilder import ModelGridDatasetBuilder
from pfsspec.pipelines.kuruczbasicpipeline import KuruczBasicPipeline

class ConvertKurucz(Convert):
    def __init__(self):
        super(ConvertKurucz, self).__init__()

    def add_args(self):
        super(ConvertKurucz, self).add_args()
        self.parser.add_argument('--interp', type=int, default=None, help='Number of interpolations between models\n')

    def init_pipeline(self, pipeline):
        super(ConvertKurucz, self).init_pipeline(pipeline)

    def run(self):
        super(ConvertKurucz, self).run()

        grid = KuruczGrid()
        grid.load(os.path.join(self.args.__dict__['in'], 'spectra.npz'))

        pipeline = KuruczBasicPipeline()
        self.init_pipeline(pipeline)
        self.dump_json(pipeline, os.path.join(self.args.out, 'pipeline.json'))

        tsbuilder = ModelGridDatasetBuilder()
        tsbuilder.parallel = not self.args.debug
        tsbuilder.grid = grid
        tsbuilder.pipeline = pipeline
        if self.args.interp is not None:
            tsbuilder.interpolate = True
            tsbuilder.spectrum_count = self.args.interp
        tsbuilder.build()

        logging.info(tsbuilder.dataset.params.head())

        tsbuilder.dataset.save(os.path.join(self.args.out, 'dataset.dat'))

        logging.info('Done.')

def main():
    script = ConvertKurucz()
    script.execute()

if __name__ == "__main__":
    main()