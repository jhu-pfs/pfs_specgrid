import numpy as np

from pfsspec.ml.dnn.autoencodingdatasetaugmenter import AutoencodingDatasetAugmenter
from pfsspec.stellarmod.modelspectrumaugmenter import ModelSpectrumAugmenterMixin

class ModelSpectrumAutoencodingAugmenter(AutoencodingDatasetAugmenter, ModelSpectrumAugmenterMixin):
    def __init__(self, orig=None):
        AutoencodingDatasetAugmenter.__init__(self, orig=orig)
        ModelSpectrumAugmenterMixin.__init__(self, orig=orig)

    @classmethod
    def from_datasets(cls, input_dataset, output_dataset, weight=None, batch_size=1, shuffle=True, random_seed=None):
        d = super(ModelSpectrumAutoencodingAugmenter, cls).from_datasets(input_dataset, output_dataset, weight,
                                                               batch_size=batch_size,
                                                               shuffle=shuffle,
                                                               random_seed=random_seed)
        return d

    def add_args(self, parser):
        AutoencodingDatasetAugmenter.add_args(self, parser)
        ModelSpectrumAugmenterMixin.add_args(self, parser)

    def init_from_args(self, args):
        AutoencodingDatasetAugmenter.init_from_args(self, args)
        ModelSpectrumAugmenterMixin.init_from_args(self, args)

    def on_epoch_end(self):
        super(ModelSpectrumAutoencodingAugmenter, self).on_epoch_end()
        ModelSpectrumAugmenterMixin.on_epoch_end(self)

    def augment_batch(self, chunk_id, idx):
        # TODO: extend this to read chunks and slice into those with idx from HDF5
        raise NotImplementedError()

        input, output, weight = AutoencodingDatasetAugmenter.augment_batch(self, chunk_id, idx)
        input, _, weight = ModelSpectrumAugmenterMixin.augment_batch(self, self.input_dataset, chunk_id, idx, input, None, weight)
        return input, output, weight