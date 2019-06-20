#!python

import os
import argparse
import numpy as np

from pfsspec.survey.dataset import DataSet
from pfsspec.train.sdsstrainingsetbuilder import SdssTrainingSetBuilder
from pfsspec.pipelines.sdssbasicpipeline import SdssBasicPipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", type=str, help="Data set file\n")
    parser.add_argument('--out', type=str, help='Training set file\n')
    return parser.parse_args()

def process_spectra(args):
    dataset = DataSet()
    dataset.load(args.__dict__['in'])
    print(dataset.params.head(10))

    pipeline = SdssBasicPipeline()
    pipeline.rebin = np.arange(6300, 8500, 2.7)
    pipeline.normalize = True

    tsbuilder = SdssTrainingSetBuilder()
    tsbuilder.dataset = dataset
    tsbuilder.params = dataset.params
    tsbuilder.pipeline = pipeline
    ts = tsbuilder.build()
    print(ts.wave.shape, ts.flux.shape)
    ts.save(args.out)

def __main__(args):
    process_spectra(args)

__main__(parse_args())