from test.test_base import TestBase
import os

from pfsspec.io.dataset import Dataset

class TestDataset(TestBase):
    def test_split(self):
        ds = self.get_sdss_dataset()
        split_index, a, b = ds.split(0.2)

    def test_filter(self):
        ds = self.get_sdss_dataset()
        filter = (ds.params['snr'] > 60)
        a, b = ds.filter(filter)

        self.assertEqual(ds.flux.shape[0], a.flux.shape[0] + b.flux.shape[0])

    def test_merge(self):
        ds = self.get_sdss_dataset()
        filter = (ds.params['snr'] > 60)
        a, b = ds.filter(filter)

        a = a.merge(b)

        self.assertEqual(ds.flux.shape[0], a.flux.shape[0])