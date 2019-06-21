#!python

import os
import argparse
import numpy as np

import tensorflow as tf
from pfsspec.train.trainingset import TrainingSet
import pfsspec.train.dnn.keras.models
import pfsspec.train.dnn.keras.train

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", type=str, help="Training set file\n")
    parser.add_argument('--param', type=str, help='Parameter to train\n')
    parser.add_argument('--gpu', type=str, help='GPUs to use\n')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs\n')
    return parser.parse_args()

def init_tensorflow():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

def train_dnn(args):
    ts = TrainingSet()
    ts.load(args.__dict__['in'])
    print(ts.params.shape, ts.wave.shape, ts.flux.shape)
    print(ts.params.columns)

    #model = pfsspec.train.dnn.keras.models.create_dense_pyramid(ts.flux.shape[1], 1)
    model = pfsspec.train.dnn.keras.models.create_cnn_pyramid(ts.flux.shape[1], 1)
    pfsspec.train.dnn.keras.train.train_dnn(ts.flux, ts.params['t_eff'], model, epochs=args.epochs)


def __main__(args):
    init_tensorflow()
    train_dnn(args)

__main__(parse_args())