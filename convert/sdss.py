#!python

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import getpass
import SciServer.Authentication as Authentication

from pfsspec.survey.io.sdssspectrumreader import SdssSpectrumReader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('verb', nargs=1, default=os.getcwd())
    parser.add_argument('--user', type=str, help='SciServer username\n')
    parser.add_argument('--token', type=str, help='SciServer auth token\n')
    parser.add_argument("--path", type=str, help="Spectrum data directory base path\n")
    parser.add_argument('--out', type=str, help='Output file, must be .npz\n')
    parser.add_argument('--top', type=int, default=None, help='Limit number of results')
    parser.add_argument('--plate', type=int, default=None, help='Limit to a single plate')

    return parser.parse_args()

def get_auth_token(args):
    if args.token is not None:
        token = args.token
    else:
        if args.user is None:
            user = input('SciServer username: ')
        else:
            user = args.user
        password = getpass.getpass()
        token = Authentication.login(user, password)
    print('SciServer token:', token)
    return token

def export_params(args):
    reader = SdssSpectrumReader()
    reader.sciserver_token = get_auth_token(args)
    params = reader.find_stars(top=args.top, plate=args.plate)
    print(params.head(10))
    params.to_csv(args.out)

def export_spectra(args):
    reader = SdssSpectrumReader()
    reader.sciserver_token = get_auth_token(args)
    params = reader.find_stars(top=args.top, plate=args.plate)
    dataset = reader.load_dataset(args.path, params)
    print(dataset.params.head(10))
    dataset.save(args.out)

def __main__(args):
    verb = args.verb[0]
    if verb == 'params':
        export_params(args)
    elif verb == 'spectra':
        export_spectra(args)
    else:
        raise NotImplementedError()

__main__(parse_args())