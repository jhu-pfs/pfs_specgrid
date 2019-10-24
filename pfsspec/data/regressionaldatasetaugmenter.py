import numpy as np

from pfsspec.data.datasetaugmenter import DatasetAugmenter

class RegressionalDatasetAugmenter(DatasetAugmenter):
    def __init__(self):
        super(RegressionalDatasetAugmenter, self).__init__()

    @classmethod
    def from_dataset(cls, dataset, labels, coeffs, weight=None, batch_size=1, shuffle=True, seed=None):
        input_shape = dataset.flux.shape
        output_shape = (len(labels),)
        d = super(RegressionalDatasetAugmenter, cls).from_dataset(input_shape, output_shape,
                                                                  dataset, labels, coeffs,
                                                                  weight=weight,
                                                                  batch_size=batch_size,
                                                                  shuffle=shuffle,
                                                                  seed=seed)
        return d

    def scale_output(self, output):
        values = output / self.coeffs
        return super(RegressionalDatasetAugmenter, self).scale_output(values)

    def rescale_output(self, output):
        values = output * self.coeffs
        return super(RegressionalDatasetAugmenter, self).rescale_output(values)

    def augment_batch(self, idx):
        input, output, weight = super(RegressionalDatasetAugmenter, self).augment_batch(idx)

        input = np.array(self.dataset.flux[idx], copy=True, dtype=np.float)
        output = np.array(self.dataset.params[self.labels].iloc[idx], copy=True, dtype=np.float)

        return input, output, weight

    def get_output_mean(self):
        return np.mean(np.array(self.dataset.params[self.labels]), axis=0) / self.coeffs