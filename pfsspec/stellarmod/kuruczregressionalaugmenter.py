import os
import numpy as np

from pfsspec.util import *
from pfsspec.data.regressionaldatasetaugmenter import RegressionalDatasetAugmenter
from pfsspec.stellarmod.kuruczaugmenter import KuruczAugmenter

class KuruczRegressionalAugmenter(RegressionalDatasetAugmenter, KuruczAugmenter):
    def __init__(self, orig=None):
        RegressionalDatasetAugmenter.__init__(self, orig=orig)
        KuruczAugmenter.__init__(self, orig=orig)

    @classmethod
    def from_dataset(cls, dataset, labels, coeffs, weight=None, partitions=None, batch_size=None, shuffle=None, chunk_size=None, seed=None):
        d = super(KuruczRegressionalAugmenter, cls).from_dataset(dataset, labels, coeffs, weight,
                                  partitions=partitions, batch_size=batch_size, shuffle=shuffle, chunk_size=chunk_size, seed=seed)
        return d

    def add_args(self, parser):
        RegressionalDatasetAugmenter.add_args(self, parser)
        KuruczAugmenter.add_args(self, parser)

    def init_from_args(self, args):
        RegressionalDatasetAugmenter.init_from_args(self, args)
        KuruczAugmenter.init_from_args(self, args)

    def on_epoch_end(self):
        super(KuruczRegressionalAugmenter, self).on_epoch_end()
        KuruczAugmenter.on_epoch_end(self)

    def augment_batch(self, chunk_id, idx):
        flux, labels, weight = RegressionalDatasetAugmenter.augment_batch(self, chunk_id, idx)

        mask = np.full(flux.shape, False)
        mask = self.get_data_mask(chunk_id, idx, flux, mask)
        mask = self.generate_random_mask(chunk_id, idx, flux, mask)
        
        flux = self.apply_ext(self.dataset, chunk_id, idx, flux)
        flux = self.apply_calib_bias(self.dataset, chunk_id, idx, flux)
        flux, error = self.generate_noise(chunk_id, idx, flux)
        flux = self.augment_flux(chunk_id, idx, flux)

        flux = self.cut_lowsnr(flux, error)
        flux = self.cut_extreme(flux, error)
        flux = self.apply_mask(flux, error, mask)

        return flux, labels, weight