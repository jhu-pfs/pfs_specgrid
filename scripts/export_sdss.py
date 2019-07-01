#!python

import os
import argparse
import getpass
import asyncio
import SciServer.Authentication as Authentication

from pfsspec.surveys.sdssspectrumreader import SdssSpectrumReader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('verb', nargs=1)
    parser.add_argument('--user', type=str, help='SciServer username\n')
    parser.add_argument('--token', type=str, help='SciServer auth token\n')
    parser.add_argument("--path", type=str, help="Spectrum data directory base path\n")
    parser.add_argument('--out', type=str, help='Output file, must be .npz\n')
    parser.add_argument('--top', type=int, default=None, help='Limit number of results')
    parser.add_argument('--plate', type=int, default=None, help='Limit to a single plate')
    parser.add_argument('--feh', type=float, nargs=2, default=None, help='Limit [Fe/H]')
    parser.add_argument('--teff', type=float, nargs=2, default=None, help='Limit T_eff')
    parser.add_argument('--logg', type=float, nargs=2, default=None, help='Limit log_g')
    parser.add_argument('--afe', type=float, nargs=2, default=None, help='Limit [a/Fe]')

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

def get_reader(args):
    reader = SdssSpectrumReader()
    reader.path = args.path
    reader.sciserver_token = get_auth_token(args)
    return reader

def find_stars(reader, args):
    return reader.find_stars(top=args.top, plate=args.plate, Fe_H=args.feh, T_eff=args.teff,
                             log_g=args.logg, a_fe=args.afe)

def export_params(args):
    reader = get_reader(args)
    params = find_stars(reader, args)
    print(params.head(10))
    params.to_csv(args.out)

def export_spectra(args):
    reader = get_reader(args)
    params = find_stars(reader, args)
    survey = reader.load_survey(params)
    print(survey.params.head(10))
    survey.save(args.out)
    print('Saved %d spectra.' % len(survey.spectra))

def __main__(args):
    verb = args.verb[0]
    if verb == 'params':
        export_params(args)
    elif verb == 'spectra':
        export_spectra(args)
    else:
        raise NotImplementedError()

__main__(parse_args())

#futures = [__main__(parse_args())]
#loop = asyncio.get_event_loop()
#loop.run_until_complete(asyncio.wait(futures))
#loop.close()