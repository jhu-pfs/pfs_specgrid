import os
import logging
import numpy as np

from pfsspec.data.dataset import Dataset
from pfsspec.scripts.script import Script

class Convert(Script):
    def __init__(self):
        super(Convert, self).__init__()
        self.outdir = None
        self.pipeline = None
        self.tsbuilder = None

    def add_subparsers(self, parser):
        subparsers = self.parser.add_subparsers(dest='pipeline')
        for k in self.PIPELINE_TYPES:
            p = subparsers.add_parser(k)
            self.add_args(p)
            self.PIPELINE_TYPES[k]().add_args(p)

    def add_args(self, parser):
        super(Convert, self).add_args(parser)
        parser.add_argument('--in', type=str, help="Data set directory\n")
        parser.add_argument('--out', type=str, help='Training set output directory\n')

    def create_pipeline(self):
        raise NotImplementedError()

    def init_pipeline(self, pipeline):
        pipeline.init_from_args(self.args)

    def create_tsbuilder(self):
        raise NotImplementedError()

    def init_tsbuilder(self, tsbuilder):
        tsbuilder.init_from_args(self.args)

    def prepare(self):
        super(Convert, self).prepare()
        self.outdir = self.args['out']
        self.create_output_dir(self.outdir)

        self.pipeline = self.create_pipeline()
        self.init_pipeline(self.pipeline)
        self.dump_json(self.pipeline, os.path.join(self.args['out'], 'pipeline.json'))

        self.tsbuilder = self.create_tsbuilder()
        self.tsbuilder.pipeline = self.pipeline

    def run(self):
        super(Convert, self).run()

        self.tsbuilder.build()
        self.tsbuilder.dataset.save(os.path.join(self.args['out'], 'dataset.h5'), 'h5')

        logging.info(self.tsbuilder.dataset.params.head())
        logging.info('Results are written to {}'.format(self.args['out']))

    def execute_notebooks(self):
        super(Convert, self).execute_notebooks()

        self.execute_notebook('eval_dataset', parameters={
                                  'DATASET_PATH': self.args['out']
                              })
