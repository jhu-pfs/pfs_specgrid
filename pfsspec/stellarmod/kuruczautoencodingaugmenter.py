import numpy as np

from pfsspec.data.autoencodingdatasetaugmenter import AutoencodingDatasetAugmenter
from pfsspec.stellarmod.kuruczaugmenter import KuruczAugmenter

class KuruczAutoencodingAugmenter(AutoencodingDatasetAugmenter, KuruczAugmenter):
    def __init__(self):
        super(KuruczAutoencodingAugmenter, self).__init__()
        KuruczAugmenter.__init__(self)

    @classmethod
    def from_datasets(cls, input_dataset, output_dataset, weight=None, batch_size=1, shuffle=True, seed=None):
        d = super(KuruczAutoencodingAugmenter, cls).from_datasets(input_dataset, output_dataset, weight,
                                                               batch_size=batch_size,
                                                               shuffle=shuffle,
                                                               seed=seed)
        return d

    def copy(self):
        new = KuruczAutoencodingAugmenter(self.dataset, self.labels, self.coeffs,
                                        self.batch_size, self.shuffle, self.seed)
        return new

    def add_args(self, parser):
        AutoencodingDatasetAugmenter.add_args(self, parser)
        KuruczAugmenter.add_args(self, parser)

    def init_from_args(self, args):
        AutoencodingDatasetAugmenter.init_from_args(self, args)
        KuruczAugmenter.init_from_args(self, args)

    def on_epoch_end(self):
        super(KuruczAutoencodingAugmenter, self).on_epoch_end()
        KuruczAugmenter.on_epoch_end(self)

    def augment_batch(self, idx):
        input, output, weight = AutoencodingDatasetAugmenter.augment_batch(self, idx)
        input, _, weight = KuruczAugmenter.augment_batch(self, self.input_dataset, idx, input, None, weight)
        return input, output, weight