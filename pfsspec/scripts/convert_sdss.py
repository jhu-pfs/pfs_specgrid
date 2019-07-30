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

    def create_pipeline(self):
        return SdssBasicPipeline()

    def create_tsbuilder(self):
        tsbuilder = SdssDatasetBuilder()
        tsbuilder.survey = Survey()
        tsbuilder.survey.load(os.path.join(self.args['in'], 'spectra.dat'))
        tsbuilder.params = tsbuilder.survey.params
        return tsbuilder

def main():
    script = ConvertSdss()
    script.execute()

if __name__ == "__main__":
    main()