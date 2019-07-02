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
    parser.add_argument("--in", type=str, help="Training set or data file\n")
    parser.add_argument("--out", type=str, help="Model directory\n")
    parser.add_argument("--name", type=str, help="Model name prefix\n")
    parser.add_argument('--labels', type=str, nargs='+', help='Labels to train for\n')
    parser.add_argument('--wave', action='store_true', help='Include wavelength vector in training.\n')
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

    model.include_wave = args.wave
    model.gpus = args.gpus
    model.validation_split = args.split
    model.patience = args.patience
    model.epochs = args.epochs
    model.batch_size = args.batch
    model.generate_name()

    labels, coeffs = parse_labels_coeffs(args)

    outdir = model.name + '_' + '_'.join(labels)
    if args.name is not None:
        outdir = args.name + '_' + outdir
    outdir = os.path.join(args.out, outdir)
    create_output_dir(outdir)
    setup_logging(os.path.join(outdir, 'training.log'))
    dump_json(args, os.path.join(outdir, 'args.json'))
    model.checkpoint_path = os.path.join(outdir, 'best_model_weights.dat')

    dataset = Dataset()
    dataset.load(os.path.join(args.__dict__['in'], 'dataset.dat'))

    _, ts, vs = dataset.split(args.split)

    training_generator = SdssDatasetAugmenter(ts, labels, coeffs, batch_size=args.batch)
    training_generator.include_wave = args.wave
    training_generator.multiplicative_bias = True
    training_generator.additive_bias = True

    validation_generator = SdssDatasetAugmenter(vs, labels, coeffs, batch_size=args.batch)
    validation_generator.include_wave = args.wave

    logging.info("Data input and labels shape: {}, {}"
                 .format(training_generator.input_shape, training_generator.labels_shape))
    logging.info("Validation input and labels shape: {}, {}"
                 .format(validation_generator.input_shape, validation_generator.labels_shape))

    model.ensure_model(training_generator.input_shape, training_generator.labels_shape)
    model.print()

    model.train_with_generator(training_generator, validation_generator)
    model.save(os.path.join(outdir, 'model.json'))
    model.save_history(os.path.join(outdir, 'history.csv'))

    predict_generator = SdssDatasetAugmenter(dataset, labels, coeffs, batch_size=dataset.flux.shape[0], shuffle=False)
    predict_generator.shuffle = False
    predict_generator.include_wave = args.wave
    flux, _ = predict_generator.next_batch(0)
    output = model.predict(flux) * coeffs
    np.savez(os.path.join(outdir, 'prediction.npz'), output)

    logging.info('Results are written to {}'.format(outdir))

def __main__(args):
    setup_logging()
    train_dnn(args)

__main__(parse_args())