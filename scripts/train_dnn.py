#!python

import os
import argparse
import numpy as np
from tensorflow.python.client import device_lib

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
    parser.add_argument('--split', type=float, default=0.5, help='Training/validation split\n')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs\n')
    parser.add_argument('--batch', type=int, default=None, help='Batch size\n')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait before early stop.\n')
    return parser.parse_args()

def train_dnn(args):
    print('Output directory is {}'.format(args.out))
    if not os.path.exists(args.out):
        print('Creating output directory {}'.format(args.out))
        os.makedirs(args.out)

    ts = Dataset()
    ts.load(args.__dict__['in'])

    print("Dataset shapes:")
    print(ts.params.shape, ts.wave.shape, ts.flux.shape)
    print(ts.params.columns)

    input = ts.flux
    labels = np.array(ts.params[args.param])

    print("Input and labels shape:")
    print(input.shape, labels.shape)

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

    if args.out is not None:
        model.checkpoint_path = os.path.join(args.out, 'best_weigths.dat')

    model.set_model_shapes(input, labels)
    model.create_model()
    model.compile_model()
    model.print()

    model.train(input, labels)

    if args.out is not None:
        model.save(os.path.join(args.out, 'model.json'))
        model.save_history(os.path.join(args.out, 'history.csv'))

    output = model.predict(input)
    np.savez(os.path.join(args.out, 'prediction.npz'), output)

def __main__(args):
    print(device_lib.list_local_devices())
    train_dnn(args)

__main__(parse_args())