#!python

import os
import logging
import argparse
import numpy as np

from pfsspec.util import *
from pfsspec.surveys.survey import Survey
from pfsspec.surveys.sdssdatasetbuilder import SdssDatasetBuilder
from pfsspec.pipelines.sdssbasicpipeline import SdssBasicPipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", type=str, help="Data set file\n")
    parser.add_argument('--out', type=str, help='Training set output directory\n')
    parser.add_argument('--wave', type=float, nargs=2, default=(3400, 8800), help='Wavelength range\n')
    parser.add_argument('--wavebins', type=int, default=2000, help='Number of wavelength bins\n')
    parser.add_argument('--wavelog', type=bool, default=False, help='Logarithmic wavelength binning\n')
    parser.add_argument('--wavelog', type=bool, default=False, help='Logarithmic wavelength binning\n')
    return parser.parse_args()

def process_spectra(args):
    create_output_dir(args.out)

    dump_json(args, os.path.join(args.out, 'args.json'))

    survey = Survey()
    survey.load(args.__dict__['in'])

    pipeline = SdssBasicPipeline()
    if args.wavelog:
        pipeline.rebin = np.logspace(np.log10(args.wave[0]), np.log10(args.wave[1]), args.wavebins)
    else:
        pipeline.rebin = np.linspace(args.wave[0], args.wave[1], args.wavebins)
    pipeline.normalize = True
    dump_json(pipeline, os.path.join(args.out, 'pipeline.json'))

    tsbuilder = SdssDatasetBuilder()
    tsbuilder.survey = survey
    tsbuilder.params = survey.params
    tsbuilder.pipeline = pipeline
    ts = tsbuilder.build()
    ts.save(os.path.join(args.out, 'dataset.dat'))

    logging.info('Done.')

def __main__(args):
    process_spectra(args)

__main__(parse_args())