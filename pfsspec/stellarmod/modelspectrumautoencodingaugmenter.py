import numpy as np

from pfsspec.ml.dnn.autoencodingdatasetaugmenter import AutoencodingDatasetAugmenter
from pfsspec.stellarmod.modelspectrumaugmenter import ModelSpectrumAugmenter

class ModelSpectrumAutoencodingAugmenter(AutoencodingDatasetAugmenter, ModelSpectrumAugmenter):
    def __init__(self, orig=None):
        AutoencodingDatasetAugmenter.__init__(self, orig=orig)
        ModelSpectrumAugmenter.__init__(self, orig=orig)

    @classmethod
    def from_datasets(cls, input_dataset, output_dataset, weight=None, batch_size=1, shuffle=True, seed=None):
        d = super(ModelSpectrumAutoencodingAugmenter, cls).from_datasets(input_dataset, output_dataset, weight,
                                                               batch_size=batch_size,
                                                               shuffle=shuffle,
                                                               seed=seed)
        return d

    def add_args(self, parser):
        AutoencodingDatasetAugmenter.add_args(self, parser)
        ModelSpectrumAugmenter.add_args(self, parser)

    def init_from_args(self, args):
        AutoencodingDatasetAugmenter.init_from_args(self, args)
        ModelSpectrumAugmenter.init_from_args(self, args)

    def on_epoch_end(self):
        super(ModelSpectrumAutoencodingAugmenter, self).on_epoch_end()
        ModelSpectrumAugmenter.on_epoch_end(self)

    def augment_batch(self, chunk_id, idx):
        # TODO: extend this to read chunks and slice into those with idx from HDF5
        raise NotImplementedError()

        input, output, weight = AutoencodingDatasetAugmenter.augment_batch(self, chunk_id, idx)
        input, _, weight = ModelSpectrumAugmenter.augment_batch(self, self.input_dataset, chunk_id, idx, input, None, weight)
        return input, output, weight