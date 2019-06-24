#!python

import argparse
import numpy as np

from pfsspec.io.dataset import Dataset
from pfsspec.ml.dnn.keras.densepyramid import DensePyramid
from pfsspec.ml.dnn.keras.cnnpyramid import CnnPyramid

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", type=str, help="Training set file\n")
    parser.add_argument("--out", type=str, help="Output directory\n")
    parser.add_argument('--param', type=str, help='Parameter to ml\n')
    parser.add_argument('--gpus', type=str, help='GPUs to use\n')
    parser.add_argument('--type', type=str, help='Type of network\n')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs\n')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait before early stop.\n')
    return parser.parse_args()

def train_dnn(args):
    ts = Dataset()
    ts.load(args.__dict__['in'])
    print(ts.params.shape, ts.wave.shape, ts.flux.shape)
    print(ts.params.columns)

    #model = pfsspec.ml.dnn.keras.models.create_dense_pyramid(ts.flux.shape[1], 1)
    #pfsspec.ml.dnn.keras.ml.train_dnn(ts.flux, ts.params[args.param], model, epochs=args.epochs)

    input = ts.flux
    labels = np.array(ts.params[args.param])

    if args.type == 'dense':
        model = DensePyramid()
    elif args.type == 'cnn':
        model = CnnPyramid()
    else:
        raise NotImplementedError()

    model.gpus = args.gpus
    model.patience = args.patience
    model.epochs = args.epochs

    model.train(input, labels)

    # TODO: add various types of outputs

def __main__(args):
    train_dnn(args)

__main__(parse_args())