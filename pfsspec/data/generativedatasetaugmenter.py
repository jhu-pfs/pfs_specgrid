import numpy as np

from pfsspec.data.datasetaugmenter import DatasetAugmenter

class GenerativeDatasetAugmenter(DatasetAugmenter):
    def __init__(self, orig=None):
        super(GenerativeDatasetAugmenter, self).__init__(orig=orig)

    @classmethod
    def from_dataset(cls, dataset, labels, coeffs, weight=None, batch_size=1, shuffle=True, chunk_size=None, seed=None):
        input_shape = (dataset.shape[0], len(labels),)
        output_shape = dataset.shape[1:]
        d = super(GenerativeDatasetAugmenter, cls).from_dataset(input_shape, output_shape,
                                                                  dataset, labels, coeffs,
                                                                  weight=weight,
                                                                  batch_size=batch_size,
                                                                  shuffle=shuffle,
                                                                  chunk_size=chunk_size,
                                                                  seed=seed)
        return d

    def scale_input(self, input):
        values = input / self.coeffs
        return super(GenerativeDatasetAugmenter, self).scale_input(values)

    def rescale_input(self, input):
        values = input * self.coeff
        return super(GenerativeDatasetAugmenter, self).rescale_input(values)

    def augment_batch(self, chunk_id, idx):
        input, output, weight = super(GenerativeDatasetAugmenter, self).augment_batch(chunk_id, idx)

        # TODO: extend this to read chunks and slice into those with idx from HDF5
        raise NotImplementedError()

        input = np.array(self.dataset.params[self.labels].iloc[idx], copy=True, dtype=np.float)
        output = np.array(self.dataset.flux[idx], copy=True, dtype=np.float)

        return input, output, weight

    def get_output_mean(self):
        return np.mean(self.dataset.flux, axis=0)
