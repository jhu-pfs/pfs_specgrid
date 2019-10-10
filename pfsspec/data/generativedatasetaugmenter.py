import numpy as np

from pfsspec.data.datasetaugmenter import DatasetAugmenter

class GenerativeDatasetAugmenter(DatasetAugmenter):
    def __init__(self):
        super(GenerativeDatasetAugmenter, self).__init__()

    @classmethod
    def from_dataset(cls, dataset, labels, coeffs, weight=None, batch_size=1, shuffle=True, seed=None):
        input_shape = (dataset.flux.shape[0], len(labels),)
        output_shape = dataset.flux.shape[1:]
        d = super(GenerativeDatasetAugmenter, cls).from_dataset(input_shape, output_shape,
                                                                  dataset, labels, coeffs,
                                                                  weight=weight,
                                                                  batch_size=batch_size,
                                                                  shuffle=shuffle,
                                                                  seed=seed)
        return d

    def scale_input(self, input):
        values = input / self.coeffs
        return super(GenerativeDatasetAugmenter, self).scale_input(values)

    def rescale_input(self, input):
        values = input * self.coeff
        return super(GenerativeDatasetAugmenter, self).rescale_input(values)

    def augment_batch(self, batch_index):
        input, output, weight = super(GenerativeDatasetAugmenter, self).augment_batch(batch_index)

        input = np.array(self.dataset.params[self.labels].iloc[batch_index], copy=True, dtype=np.float)
        output = np.array(self.dataset.flux[batch_index], copy=True, dtype=np.float)

        return input, output, weight

    def get_average(self):
        return np.mean(self.dataset.flux, axis=0)
