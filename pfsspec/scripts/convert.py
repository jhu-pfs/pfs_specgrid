import os
import logging
import numpy as np

from pfsspec.data.dataset import Dataset
from pfsspec.scripts.script import Script
from pfsspec.surveys.sdssdatasetbuilder import SdssDatasetBuilder
from pfsspec.pipelines.sdssbasicpipeline import SdssBasicPipeline
from pfsspec.stellarmod.modelgriddatasetbuilder import ModelGridDatasetBuilder
from pfsspec.pipelines.kuruczbasicpipeline import KuruczBasicPipeline

class Convert(Script):


    def __init__(self):
        super(Convert, self).__init__()

        self.CONVERTER_TYPES = {
            'sdss': SdssDatasetBuilder,
            'kurucz': ModelGridDatasetBuilder
        }

        self.PIPELINE_TYPES = {
            'sdss': { 'basic': SdssBasicPipeline },
            'kurucz': { 'basic': KuruczBasicPipeline }
        }

        self.outdir = None
        self.pipeline = None
        self.tsbuilder = None

    def add_subparsers(self, parser):
        spc = self.parser.add_subparsers(dest='dataset')
        for kc in self.CONVERTER_TYPES:
            pc = spc.add_parser(kc)
            spp = pc.add_subparsers(dest='pipeline')
            for kp in self.PIPELINE_TYPES[kc]:
                pp = spp.add_parser(kp)
                self.add_args(pp)
                self.CONVERTER_TYPES[kc]().add_args(pp)
                self.PIPELINE_TYPES[kc][kp]().add_args(pp)

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
        tsbuilder.pipeline = self.pipeline
        tsbuilder.init_from_args(self.args)

    def prepare(self):
        super(Convert, self).prepare()
        self.outdir = self.args['out']
        self.create_output_dir(self.outdir)

        self.pipeline = self.create_pipeline()
        self.init_pipeline(self.pipeline)
        self.dump_json(self.pipeline, os.path.join(self.args['out'], 'pipeline.json'))

        self.tsbuilder = self.create_tsbuilder()
        self.init_tsbuilder(self.tsbuilder)

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
