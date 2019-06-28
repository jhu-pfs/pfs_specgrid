#!/usr/bin/env python

import os
import argparse
import logging
import numpy as np
from tensorflow.python.client import device_lib

from pfsspec.util import *
from pfsspec.io.dataset import Dataset
from pfsspec.ml.dnn.keras.densepyramid import DensePyramid
from pfsspec.ml.dnn.keras.cnnpyramid import CnnPyramid
from pfsspec.surveys.sdssdatasetaugmenter import SdssDatasetAugmenter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", type=str, help="Training set file\n")
    parser.add_argument("--out", type=str, help="Output directory\n")
    parser.add_argument('--param', type=str, nargs='+', help='Parameter to ml\n')
    parser.add_argument('--gpus', type=str, help='GPUs to use\n')
    parser.add_argument('--type', type=str, help='Type of network\n')
    parser.add_argument('--levels', type=int, help='Number of levels\n')
    parser.add_argument('--units', type=int, help='Number of units\n')
    parser.add_argument('--split', type=float, default=0.5, help='Training/validation split\n')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs\n')
    parser.add_argument('--batch', type=int, default=None, help='Batch size\n')
    parser.add_argument('--patience', type=int, default=None, help='Number of epochs to wait before early stop.\n')
    parser.add_argument('--aug', action='store_true', help='Augment data.\n')
    return parser.parse_args()

def train_dnn(args):
    if args.type == 'dense':
        model = DensePyramid()
    elif args.type == 'cnn':
        model = CnnPyramid()
    else:
        raise NotImplementedError()

    if args.levels is not None:
        model.levels = args.levels
    if args.units is not None:
        model.units = args.units

    model.gpus = args.gpus
    model.validation_split = args.split
    model.patience = args.patience
    model.epochs = args.epochs
    model.batch_size = args.batch
    model.generate_name()

    outdir = model.name + '_' + '_'.join(args.param)
    outdir = os.path.join(args.out, outdir)
    create_output_dir(outdir)

    setup_logging(os.path.join(outdir, 'training.log'))
    model.checkpoint_path = os.path.join(outdir, 'best_model_weights.dat')

    dataset = Dataset()
    dataset.load(args.__dict__['in'])

    if args.aug:
        _, ts, vs = dataset.split(args.split)

        training_generator = SdssDatasetAugmenter(ts, args.param, batch_size=args.batch)
        training_generator.multiplicative_bias = True
        training_generator.additive_bias = True

        validation_generator = SdssDatasetAugmenter(vs, args.param, batch_size=args.batch)

        logging.info("Data input and labels shape: {}, {}"
                     .format(training_generator.input_shape, training_generator.labels_shape))
        logging.info("Validation input and labels shape: {}, {}"
                     .format(validation_generator.input_shape, validation_generator.labels_shape))

        model.ensure_model(training_generator.input_shape, training_generator.labels_shape)
        model.print()
        model.train_with_generator(training_generator, validation_generator)
    else:
        input = dataset.flux
        labels = np.array(dataset.params[args.param])

        logging.info("Input and labels shape: {}, {}"
                     .format(input.shape, labels.shape))

        model.ensure_model(input.shape, labels.shape)
        model.print()
        model.train(input, labels)

    if args.out is not None:
        model.save(os.path.join(outdir, 'model.json'))
        model.save_history(os.path.join(outdir, 'history.csv'))

    output = model.predict(dataset.flux)    # TODO: could use the data generators here
    np.savez(os.path.join(outdir, 'prediction.npz'), output)

    logging.info('Results are written to {}'.format(outdir))

def __main__(args):
    setup_logging()
    train_dnn(args)

__main__(parse_args())