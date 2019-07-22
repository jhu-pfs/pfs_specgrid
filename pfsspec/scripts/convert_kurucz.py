#!python

import os
import logging
import numpy as np

from pfsspec.scripts.convert import Convert
from pfsspec.stellarmod.kuruczgrid import KuruczGrid
from pfsspec.stellarmod.modelgriddatasetbuilder import ModelGridDatasetBuilder
from pfsspec.pipelines.kuruczbasicpipeline import KuruczBasicPipeline
from pfsspec.obsmod.noise import Noise

class ConvertKurucz(Convert):
    def __init__(self):
        super(ConvertKurucz, self).__init__()

    def add_args(self):
        super(ConvertKurucz, self).add_args()
        self.parser.add_argument('--interp', type=int, default=None, help='Number of interpolations between models\n')
        self.parser.add_argument('--feh', type=float, nargs=2, default=None, help='Limit [Fe/H]')
        self.parser.add_argument('--teff', type=float, nargs=2, default=None, help='Limit T_eff')
        self.parser.add_argument('--logg', type=float, nargs=2, default=None, help='Limit log_g')
        self.parser.add_argument('--afe', type=float, nargs=2, default=None, help='Limit [a/Fe]')

        self.parser.add_argument('--rndz', type=float, default=None, help='Radial velocity dispersion')
        self.parser.add_argument('--rndm', type=float, nargs=2, default=None, help='Apparent magnitude mean and sigma')
        self.parser.add_argument('--noiz', type=str, default=None, help='Noise model')

    def init_pipeline(self, pipeline):
        super(ConvertKurucz, self).init_pipeline(pipeline)

        if self.args['rndz'] is not None:
             pipeline.random_redshift = lambda: np.random.normal(0, self.args['rndz'])

        if self.args['rndm'] is not None:
            pipeline.normalize = True
            pipeline.normalize_mag = None
            pipeline.random_mag = lambda: np.random.normal(self.args['rndm'][0], self.args['rndm'][1])

        if self.args['noiz'] is not None:
            pipeline.noise = Noise()
            pipeline.noise.read(self.args['noiz'])
            pipeline.noise.resample(pipeline.rebin)

    def run(self):
        super(ConvertKurucz, self).run()

        grid = KuruczGrid()
        grid.load(os.path.join(self.args['in'], 'spectra.npz'))

        pipeline = KuruczBasicPipeline()
        self.init_pipeline(pipeline)
        self.dump_json(pipeline, os.path.join(self.args['out'], 'pipeline.json'))

        tsbuilder = ModelGridDatasetBuilder()
        tsbuilder.parallel = not self.args['debug']
        tsbuilder.grid = grid
        tsbuilder.pipeline = pipeline
        if 'interp' in self.args and self.args['interp'] is not None:
            tsbuilder.interpolate = True
            tsbuilder.spectrum_count = self.args['interp']

            # Override grid range when interpolation is turned on and limits are set
            if self.args['feh'] is not None:
                tsbuilder.grid.Fe_H_min = self.args['feh'][0]
                tsbuilder.grid.Fe_H_max = self.args['feh'][1]
            if self.args['teff'] is not None:
                tsbuilder.grid.T_eff_min = self.args['teff'][0]
                tsbuilder.grid.T_eff_max = self.args['teff'][1]
            if self.args['logg'] is not None:
                tsbuilder.grid.log_g_min = self.args['logg'][0]
                tsbuilder.grid.log_g_max = self.args['logg'][1]
            if self.args['afe'] is not None:
                tsbuilder.grid.a_Fe_min = self.args['afe'][0]
                tsbuilder.grid.a_Fe_max = self.args['afe'][1]

        tsbuilder.build()

        logging.info(tsbuilder.dataset.params.head())

        tsbuilder.dataset.save(os.path.join(self.args['out'], 'dataset.dat.gz'))

        self.execute_notebook(r'nb/eval_dataset.ipynb',
                              'eval_dataset.ipynb',
                              'eval_dataset.html',
                              {
                                  'DATASET_PATH': self.args['out']
                              })

        logging.info('Done.')

def main():
    script = ConvertKurucz()
    script.execute()

if __name__ == "__main__":
    main()