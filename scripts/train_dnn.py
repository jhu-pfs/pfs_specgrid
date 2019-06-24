#!python

import argparse

import tensorflow as tf
from pfsspec.io.dataset import Dataset
import pfsspec.ml.dnn.keras.models
import pfsspec.ml.dnn.keras.train

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", type=str, help="Training set file\n")
    parser.add_argument('--param', type=str, help='Parameter to ml\n')
    parser.add_argument('--gpu', type=str, help='GPUs to use\n')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs\n')
    return parser.parse_args()

def init_tensorflow():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

def train_dnn(args):
    ts = Dataset()
    ts.load(args.__dict__['in'])
    print(ts.params.shape, ts.wave.shape, ts.flux.shape)
    print(ts.params.columns)

    #model = pfsspec.ml.dnn.keras.models.create_dense_pyramid(ts.flux.shape[1], 1)
    #pfsspec.ml.dnn.keras.ml.train_dnn(ts.flux, ts.params[args.param], model, epochs=args.epochs)

    nn_input = ts.flux.reshape((ts.flux.shape[0], ts.flux.shape[1], 1))
    nn_labels = ts.params[args.param]
    model = pfsspec.ml.dnn.keras.models.create_cnn_pyramid(nn_input.shape[1:], 1, levels=4)
    pfsspec.ml.dnn.keras.train.train_dnn(nn_input, nn_labels, model, epochs=args.epochs)


def __main__(args):
    init_tensorflow()
    train_dnn(args)

__main__(parse_args())