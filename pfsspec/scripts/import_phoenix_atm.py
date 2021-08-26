#!python

import os
import glob
import logging

from pfsspec.scripts.import_ import Import
from pfsspec.stellarmod.phoenixspectrumreader import PhoenixSpectrumReader
from pfsspec.stellarmod.phoenixatmreader import PhoenixAtmReader
from pfsspec.stellarmod.phoenixatmgridreader import PhoenixAtmGridReader
from pfsspec.stellarmod.phoenixatmgrid import PhoenixAtmGrid
from pfsspec.stellarmod.modelgridconfig import ModelGridConfig



class ImportPhoenixAtm(Import):
    def __init__(self):
        super(ImportPhoenixAtm, self).__init__()

    def add_args(self, parser):
        super(ImportPhoenixAtm, self).add_args(parser)
        parser.add_argument("--max", type=int, default=None, help="Stop after this many items.\n")
        parser.add_argument('--preload-arrays', action='store_true', help='Do not preload flux arrays to save memory\n')
        print('attempting to add continuum model argument in import_phoenix_atm')
        choices = [k for k in ModelGridConfig.CONTINUUM_MODEL_TYPES.keys()] ######
        parser.add_argument('--continuum_model', type=str, choices=choices, help='Continuum model.\n') ####


    

    def run(self):
        super(ImportPhoenixAtm, self).run()
        print('args in importphoenixatm', self.args)
        grid = PhoenixAtmGrid()
        r = PhoenixAtmReader(self.args['path'])
        gr = PhoenixAtmGridReader(grid, r, self.args['max']) 
        gr.parallel = self.threads != 1

        if 'preload_arrays' in self.args and self.args['preload_arrays'] is not None:
            grid.preload_arrays = self.args['preload_arrays']

        if os.path.isdir(self.args['path']):
            self.logger.info('Running in grid mode')
            raise NotImplementedError()
        else:
            self.logger.info('Running in file list mode')
            files = glob.glob(os.path.expandvars(self.args['path']))
            self.logger.info('Found {} files.'.format(len(files)))

        grid.init_values()
        grid.build_axis_indexes()
        grid.save(os.path.join(self.args['out'], 'atm.h5'), 'h5')

        if os.path.isdir(self.args['path']):
            raise NotImplementedError()
            #gr.read_grid()
        else:
            gr.read_files(files)

        #r.grid.build_flux_index(rebuild=True)
        grid.save(os.path.join(self.args['out'], 'atm.h5'), 'h5')

def main():
    script = ImportPhoenixAtm()
    script.execute()

if __name__ == "__main__":
    main()