import os
import numpy as np

from test.test_base import TestBase
from pfsspec.stellarmod.boszpcagrid import BoszPCAGrid

class TestBoszPCAGrid(TestBase):
    def test_load(self):
        fn = '/scratch/ceph/dobos/temp/test024/spectra.h5'
        pca = BoszPCAGrid()
        pca.preload_arrays = False
        pca.load(fn, format='h5')