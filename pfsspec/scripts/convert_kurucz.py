#!python

import os
import logging
import numpy as np

from pfsspec.scripts.convert import Convert
from pfsspec.stellarmod.kuruczgrid import KuruczGrid
from pfsspec.stellarmod.modelgriddatasetbuilder import ModelGridDatasetBuilder
from pfsspec.pipelines.kuruczbasicpipeline import KuruczBasicPipeline

class ConvertKurucz(Convert):
    def __init__(self):
        super(ConvertKurucz, self).__init__()
        self.PIPELINE_TYPES = {'basic': KuruczBasicPipeline}

    def create_pipeline(self):
        return KuruczBasicPipeline()

    def create_tsbuilder(self):
        tsbuilder = ModelGridDatasetBuilder()
        tsbuilder.grid = KuruczGrid()
        tsbuilder.grid.load(os.path.join(self.args['in'], 'spectra.npz'))
        return tsbuilder

def main():
    script = ConvertKurucz()
    script.execute()

if __name__ == "__main__":
    main()