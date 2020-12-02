from test.test_base import TestBase
import h5py as h5
import os
import numpy as np
import logging, multiprocessing

from pfsspec.ml.dnn.keras.kerasdatagenerator import KerasDataGenerator
from pfsspec.data.regressionaldatasetaugmenter import RegressionalDatasetAugmenter
from pfsspec.data.dataset import Dataset

class TestDataset(TestBase):
    def get_test_dataset(self):
        filename = '/scratch/ceph/dobos/data/pfsspec/train/sdss_stellar_model/dataset/bosz/nowave/test_T_eff_100k/dataset.h5'
        ds = Dataset(preload_arrays=False)
        ds.load(filename)
        return ds

    def test_shuffle_get_item(self):
        multiprocessing.log_to_stderr(logging.DEBUG)

        def helper(chunk_size=None, batch_size=16, thread_count=4, top=None):
            labels = 'T_eff'
            ds = self.get_test_dataset()
            da = RegressionalDatasetAugmenter.from_dataset(ds, labels, [1.0], partitions=4, batch_size=batch_size, chunk_size=chunk_size)
            if top is not None:
                da.filter = np.concatenate([np.full((top,), True), np.full((ds.shape[0] - top,), False)])
            da.reshuffle()
            data_count = top if top is not None else ds.shape[0]
            self.assertEqual(da.data_index.shape[0], data_count)
            self.assertEqual(len(da.batch_index), thread_count)
            # TODO: verify partial chunk and partial batch here
            # self.assertEqual(da.batch_index[0].shape[0], np.int32(np.ceil(np.ceil(data_count / batch_size) / thread_count)))
            
            dg = KerasDataGenerator(augmenter=da, threads=4)
            batch_count = da.get_batch_count()
            last = False
            for batch_id in range(batch_count):
                input, output, weight = dg.__getitem__(batch_id)
                if input.shape[0] != da.batch_size:
                    self.assertEqual(input.shape[0], data_count % da.batch_size)
                    self.assertFalse(last)
                    last = True
                else:
                    self.assertEqual(input.shape[0], da.batch_size)

        # helper(top=11500)
        helper(chunk_size=1600)
