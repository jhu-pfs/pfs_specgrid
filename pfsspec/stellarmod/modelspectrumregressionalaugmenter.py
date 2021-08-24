import os
import numpy as np

from pfsspec.util import *
from pfsspec.ml.dnn.regressionaldatasetaugmenter import RegressionalDatasetAugmenter
from pfsspec.stellarmod.modelspectrumaugmenter import ModelSpectrumAugmenter

class ModelSpectrumRegressionalAugmenter(RegressionalDatasetAugmenter, ModelSpectrumAugmenter):
    def __init__(self, orig=None):
        RegressionalDatasetAugmenter.__init__(self, orig=orig)
        ModelSpectrumAugmenter.__init__(self, orig=orig)

    @classmethod
    def from_dataset(cls, dataset, labels, coeffs, weight=None, partitions=None, batch_size=None, shuffle=None, chunk_size=None, seed=None):
        d = super(ModelSpectrumRegressionalAugmenter, cls).from_dataset(dataset, labels, coeffs, weight,
                                  partitions=partitions, batch_size=batch_size, shuffle=shuffle, chunk_size=chunk_size, seed=seed)
        return d

    def add_args(self, parser):
        RegressionalDatasetAugmenter.add_args(self, parser)
        ModelSpectrumAugmenter.add_args(self, parser)

    def init_from_args(self, args):
        RegressionalDatasetAugmenter.init_from_args(self, args)
        ModelSpectrumAugmenter.init_from_args(self, args)

    def on_epoch_end(self):
        super(ModelSpectrumRegressionalAugmenter, self).on_epoch_end()
        ModelSpectrumAugmenter.on_epoch_end(self)

    def augment_batch(self, chunk_id, idx):
        flux, labels, weight = RegressionalDatasetAugmenter.augment_batch(self, chunk_id, idx)

        mask = np.full(flux.shape, False)
        mask = self.get_data_mask(chunk_id, idx, flux, mask)
        mask = self.generate_random_mask(chunk_id, idx, flux, mask)
        
        flux = self.apply_ext(self.dataset, chunk_id, idx, flux)
        flux = self.apply_calib_bias(self.dataset, chunk_id, idx, flux)
        flux, error = self.generate_noise(chunk_id, idx, flux)
        flux = self.augment_flux(chunk_id, idx, flux)

        flux = self.substitute_lowsnr(flux, error)
        flux = self.substitute_outlier(flux, error)
        flux = self.substitute_mask(flux, error, mask)

        return flux, labels, weight