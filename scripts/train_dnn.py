#!python

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
    parser.add_argument('--param', type=str, help='Parameter to ml\n')
    parser.add_argument('--gpus', type=str, help='GPUs to use\n')
    parser.add_argument('--type', type=str, help='Type of network\n')
    parser.add_argument('--split', type=float, default=0.5, help='Training/validation split\n')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs\n')
    parser.add_argument('--batch', type=int, default=None, help='Batch size\n')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait before early stop.\n')
    parser.add_argument('--aug', action='store_true', help='Augment data.\n')
    return parser.parse_args()

def train_dnn(args):
    create_output_dir(args.out)

    if args.type == 'dense':
        model = DensePyramid()
    elif args.type == 'cnn':
        model = CnnPyramid()
    else:
        raise NotImplementedError()

    model.gpus = args.gpus
    model.validation_split = args.split
    model.patience = args.patience
    model.epochs = args.epochs
    model.batch_size = args.batch

    dataset = Dataset()
    dataset.load(args.__dict__['in'])

    if args.aug:
        _, ds, vs = dataset.split(args.split)

        data_generator = SdssDatasetAugmenter(ds, [args.param,], batch_size=args.batch)
        validation_generator = SdssDatasetAugmenter(vs, [args.param,], batch_size=args.batch)

        logging.info("Data input and labels shape:")
        logging.info(data_generator.input_shape, data_generator.labels_shape)
        logging.info("Validation input and labels shape:")
        logging.info(validation_generator.input_shape, validation_generator.labels_shape)

        model.ensure_model(data_generator.input_shape, data_generator.labels_shape)
        model.print()
        model.train_with_generator(data_generator, validation_generator)
    else:
        input = dataset.flux
        labels = np.array(dataset.params[args.param])

        logging.info("Input and labels shape:")
        logging.info(input.shape, labels.shape)

        model.ensure_model(input.shape, labels.shape)
        model.print()
        model.train(input, labels)

    if args.out is not None:
        model.save(os.path.join(args.out, 'model.json'))
        model.save_history(os.path.join(args.out, 'history.csv'))

    output = model.predict(dataset.flux)    # TODO: could use the data generators here
    np.savez(os.path.join(args.out, 'prediction.npz'), output)

def __main__(args):
    print(device_lib.list_local_devices())
    train_dnn(args)

__main__(parse_args())