#!python

import argparse
import numpy as np

from pfsspec.surveys.survey import Survey
from pfsspec.surveys.sdssdatasetbuilder import SdssDatasetBuilder
from pfsspec.pipelines.sdssbasicpipeline import SdssBasicPipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", type=str, help="Data set file\n")
    parser.add_argument('--out', type=str, help='Training set file\n')
    return parser.parse_args()

def process_spectra(args):
    survey = Survey()
    survey.load(args.__dict__['in'])
    print(survey.params.head(10))

    pipeline = SdssBasicPipeline()
    pipeline.rebin = np.arange(3500, 8800, 2.7)
    pipeline.normalize = True

    tsbuilder = SdssDatasetBuilder()
    tsbuilder.dataset = survey
    tsbuilder.params = survey.params
    tsbuilder.pipeline = pipeline
    ts = tsbuilder.build()
    print(ts.wave.shape, ts.flux.shape)
    ts.save(args.out)

def __main__(args):
    process_spectra(args)

__main__(parse_args())