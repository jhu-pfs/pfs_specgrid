#!python

import os
import logging

from pfsspec.scripts.convert import Convert
from pfsspec.surveys.survey import Survey
from pfsspec.surveys.sdssdatasetbuilder import SdssDatasetBuilder
from pfsspec.pipelines.sdssbasicpipeline import SdssBasicPipeline

class ConvertSdss(Convert):
    def __init__(self):
        super(ConvertSdss, self).__init__()

    def add_args(self):
        super(ConvertSdss, self).add_args()
        self.parser.add_argument('--rest', action='store_true', help='Convert to rest-frame\n')

    def init_pipeline(self, pipeline):
        super(ConvertSdss, self).init_pipeline(pipeline)
        pipeline.restframe = self.args.rest

    def run(self):
        super(ConvertSdss, self).run()

        survey = Survey()
        survey.load(os.path.join(self.args.__dict__['in'], 'spectra.dat'))

        pipeline = SdssBasicPipeline()
        self.init_pipeline(pipeline)
        self.dump_json(pipeline, os.path.join(self.args.out, 'pipeline.json'))

        tsbuilder = SdssDatasetBuilder()
        tsbuilder.parallel = not self.args.debug
        tsbuilder.survey = survey
        tsbuilder.params = survey.params
        tsbuilder.pipeline = pipeline
        tsbuilder.build()
        tsbuilder.dataset.save(os.path.join(self.args.out, 'dataset.dat'))

        logging.info('Done.')

def main():
    script = ConvertSdss()
    script.execute()

if __name__ == "__main__":
    main()