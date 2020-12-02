#!/usr/bin/env python

import os
import logging
import getpass
import SciServer.Authentication as Authentication

from pfsspec.scripts.import_ import Import
from pfsspec.surveys.sdssspectrumreader import SdssSpectrumReader

class ImportSdss(Import):
    def __init__(self):
        super(ImportSdss, self).__init__()
        self.user = None
        self.token = None
        self.reader = None

    def add_args(self, parser):
        super(ImportSdss, self).add_args(parser)
        parser.add_argument('--user', type=str, help='SciServer username\n')
        parser.add_argument('--token', type=str, help='SciServer auth token\n')
        parser.add_argument('--top', type=int, default=None, help='Limit number of results')
        parser.add_argument('--plate', type=int, default=None, help='Limit to a single plate')
        parser.add_argument('--feh', type=float, nargs=2, default=None, help='Limit [Fe/H]')
        parser.add_argument('--teff', type=float, nargs=2, default=None, help='Limit T_eff')
        parser.add_argument('--logg', type=float, nargs=2, default=None, help='Limit log_g')
        parser.add_argument('--afe', type=float, nargs=2, default=None, help='Limit [a/Fe]')

    def create_auth_token(self):
        if 'token' in self.args and self.args['token'] is not None:
            self.token = self.args['token']
        else:
            if 'user' in self.args and self.args['user'] is None:
                self.user = input('SciServer username: ')
            else:
                self.user = self.args['user']
            password = getpass.getpass()
            self.token = Authentication.login(self.user, password)
        self.logger.info('SciServer token: {}'.format(self.token))

    def create_reader(self):
        self.reader = SdssSpectrumReader(verbose=True, parallel=not self.debug, threads=self.threads)
        self.reader.path = self.args['path']
        self.reader.sciserver_token = self.token

    def find_stars(self):
        return self.reader.find_stars(top=self.args['top'], plate=self.args['plate'], Fe_H=self.args['feh'],
                                      T_eff=self.args['teff'], log_g=self.args['logg'], a_Fe=self.args['afe'])

    def prepare(self):
        super(ImportSdss, self).prepare()
        self.create_auth_token()
        self.create_reader()

    def run(self):
        super(ImportSdss, self).run()
        params = self.find_stars()
        self.logger.info(params.head(10))
        survey = self.reader.load_survey(params)
        survey.save(os.path.join(self.args['out'], 'spectra.dat'))
        self.logger.info('Saved %d spectra.' % len(survey.spectra))

    def execute_notebooks(self):
        super(ImportSdss, self).execute_notebooks()
        self.execute_notebook('eval_import_sdss', parameters={
                                'PFSSPEC_ROOT': os.environ['PFSSPEC_ROOT'],
                                'PFSSPEC_DATA': os.environ['PFSSPEC_DATA'],
                                'DATASET_PATH': self.outdir,
                            },
                            output_notebook_name='eval_import_sdss',
                            outdir=self.outdir)


def main():
    script = ImportSdss()
    script.execute()

if __name__ == "__main__":
    main()