import os
import logging
import numpy as np

from pfsspec.scripts.script import Script
from pfsspec.surveys.survey import Survey
from pfsspec.surveys.sdssdatasetbuilder import SdssDatasetBuilder
from pfsspec.obsmod.sdssbasicpipeline import SdssBasicPipeline
from pfsspec.stellarmod.kuruczgrid import KuruczGrid
from pfsspec.stellarmod.boszgrid import BoszGrid
from pfsspec.stellarmod.modelgriddatasetbuilder import ModelGridDatasetBuilder
from pfsspec.obsmod.kuruczbasicpipeline import KuruczBasicPipeline
from pfsspec.obsmod.boszbasicpipeline import BoszBasicPipeline

class Convert(Script):
    def __init__(self):
        super(Convert, self).__init__()

        self.DATASET_TYPES = {
            'sdss':
                {
                    'builder': SdssDatasetBuilder,
                    'survey': Survey,
                    'pipelines': {
                        'basic': SdssBasicPipeline
                    }
                },
            'kurucz':
                {
                    'builder': ModelGridDatasetBuilder,
                    'grid': KuruczGrid,
                    'pipelines': {
                        'basic': KuruczBasicPipeline
                    }
                },
            'bosz':
                {
                    'builder': ModelGridDatasetBuilder,
                    'grid': BoszGrid,
                    'pipelines': {
                        'basic': BoszBasicPipeline
                    }
                }
        }

        self.outdir = None
        self.pipeline = None
        self.dsbuilder = None
        self.params = None

    def add_subparsers(self, parser):
        spc = parser.add_subparsers(dest='dataset')
        for ds in self.DATASET_TYPES:
            pc = spc.add_parser(ds)
            spp = pc.add_subparsers(dest='pipeline')
            for pl in self.DATASET_TYPES[ds]['pipelines']:
                pp = spp.add_parser(pl)
                self.add_args(pp)
                self.create_dsbuilder(ds).add_args(pp)
                self.create_pipeline(ds, pl).add_args(pp)

    def add_args(self, parser):
        super(Convert, self).add_args(parser)
        parser.add_argument('--in', type=str, help="Data set directory\n")
        parser.add_argument('--out', type=str, help='Training set output directory\n')
        parser.add_argument('--params', type=str, help="Take spectrum params from dataset.\n")

    def parse_args(self):
        super(Convert, self).parse_args()
        if 'params' in self.args and self.args['params'] is not None:
            self.params = self.args['params']

    def create_pipeline(self, dataset, pipeline):
        return self.DATASET_TYPES[dataset]['pipelines'][pipeline]()

    def init_pipeline(self, pipeline):
        pipeline.init_from_args(self.args)

    def create_dsbuilder(self, dataset):
        ds = self.DATASET_TYPES[dataset]
        dsbuilder = ds['builder'](random_seed=self.random_seed)
        if 'survey' in ds:
            dsbuilder.survey = ds['survey']()
        elif 'grid' in ds:
            dsbuilder.grid = ds['grid']()
        else:
            raise NotImplementedError()
        return dsbuilder

    def load_data(self, dsbuilder):
        ds = self.DATASET_TYPES[self.args['dataset']]
        if 'survey' in ds:
            dsbuilder.survey.load(os.path.join(self.args['in'], 'spectra.dat'))
            dsbuilder.params = dsbuilder.survey.params
        elif 'grid' in ds:
            dsbuilder.grid.use_cont = dsbuilder.use_cont
            # If a limit is specified on any of the parameters, try to slice the
            # grid while loading from HDF5
            s = []
            for k in dsbuilder.grid.params:
                if self.args[k] is not None:
                    idx = np.digitize([self.args[k][0], self.args[k][1]], dsbuilder.grid.params[k].values)
                    s.append(slice(idx[0], idx[1] + 1, None))
                else:
                    s.append(slice(None))
            s.append(slice(None))  # wave axis
            s = tuple(s)

            fn = os.path.join(self.args['in'], 'spectra')
            if os.path.isfile(fn + '.h5'):
                dsbuilder.grid.load(fn + '.h5', slice=s, format='h5')
            elif os.path.isfile(fn + '.npz'):
                dsbuilder.grid.load(fn + '.npz', slice=s, format='npz')

            # Load source parameters, if necessary
            if self.params is not None:
                fn = os.path.join(self.params, 'dataset.h5')
                logging.info('Taking parameters from existing dataset: {}'.format(fn))
                ds = dsbuilder.create_dataset(init_storage=False)
                ds.load(fn, format='h5')
                dsbuilder.params = ds.params
        else:
            raise NotImplementedError()

    def init_dsbuilder(self, dsbuilder):
        dsbuilder.pipeline = self.pipeline
        dsbuilder.init_from_args(self.args)

    def prepare(self):
        super(Convert, self).prepare()
        self.outdir = self.args['out']
        self.create_output_dir(self.outdir)
        self.save_command_line(os.path.join(self.outdir, 'command.sh'))

        self.pipeline = self.create_pipeline(self.args['dataset'], self.args['pipeline'])
        self.init_pipeline(self.pipeline)
        self.dump_json(self.pipeline, os.path.join(self.args['out'], 'pipeline.json'))

        self.dsbuilder = self.create_dsbuilder(self.args['dataset'])
        self.init_dsbuilder(self.dsbuilder)
        self.load_data(self.dsbuilder)
        # Do this again because grid sampling overrides might have changed when
        # loading the grid
        self.dsbuilder.init_from_args(self.args)

    def run(self):
        self.init_logging(self.outdir)
        self.dsbuilder.build()
        self.dsbuilder.dataset.save(os.path.join(self.args['out'], 'dataset.h5'), 'h5')

        logging.info(self.dsbuilder.dataset.params.head())
        logging.info('Results are written to {}'.format(self.args['out']))

    def execute_notebooks(self):
        super(Convert, self).execute_notebooks()

        self.execute_notebook('eval_dataset', parameters={
                                  'DATASET_PATH': self.args['out']
                              })

def main():
    script = Convert()
    script.execute()

if __name__ == "__main__":
    main()