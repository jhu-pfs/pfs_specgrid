import os
import numpy as np

from test.test_base import TestBase
from pfsspec.data.pcagrid import PcaGrid

class TestBoszPcaGrid(TestBase):
    def test_load(self):
        fn = '/scratch/ceph/dobos/temp/test024/spectra.h5'
        pca = BoszPcaGrid()
        pca.preload_arrays = False
        pca.load(fn, format='h5')