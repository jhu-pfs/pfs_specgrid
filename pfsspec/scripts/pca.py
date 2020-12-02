import os
import logging
import numpy as np

from pfsspec.scripts.convert import Convert

class Pca(Convert):
    def __init__(self):
        super(Pca, self).__init__()

        self.truncate = None

    def add_args(self, parser):
        super(Pca, self).add_args(parser)
        parser.add_argument('--truncate', type=int, default=None, help="Data set directory\n")

    def run(self):
        super(Convert, self).run()

        self.dsbuilder.build()
        self.dsbuilder.dataset.run_pca(self.args['truncate'])
        self.dsbuilder.dataset.save(os.path.join(self.args['out'], 'dataset.h5'), 'h5')
        self.dsbuilder.dataset.save_pca(os.path.join(self.args['out'], 'pca.h5'), 'h5')

        self.logger.info(self.dsbuilder.dataset.params.head())
        self.logger.info('Results are written to {}'.format(self.args['out']))

    def execute_notebooks(self):
        super(Convert, self).execute_notebooks()

        self.execute_notebook('eval_pca', parameters={
                                  'DATASET_PATH': self.args['out']
                              })

def main():
    script = Pca()
    script.execute()

if __name__ == "__main__":
    main()