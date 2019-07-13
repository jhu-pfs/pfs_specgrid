import numpy as np

from pfsspec.data.datasetaugmenter import DatasetAugmenter

class KuruczGenerativeAugmenter(DatasetAugmenter):
    def __init__(self, dataset, labels, coeffs, batch_size=1, shuffle=True, seed=None):
        input_shape = (dataset.flux.shape[0], len(labels),)
        output_shape = dataset.flux.shape[1:]
        super(KuruczGenerativeAugmenter, self).__init__(dataset, labels, coeffs,
                                                        input_shape, output_shape,
                                                        batch_size=batch_size, shuffle=shuffle, seed=seed)

    def copy(self):
        new = KuruczGenerativeAugmenter(self.dataset, self.labels, self.coeffs,
                                        self.batch_size, self.shuffle, self.seed)
        return new

    def scale_input(self, input):
        return input / self.coeffs

    def rescale_input(self, input):
        return input * self.coeff

    def augment_batch(self, batch_index):
        input = np.array(self.dataset.params[self.labels].iloc[batch_index], copy=True, dtype=np.float)
        output = np.array(self.dataset.flux[batch_index], copy=True, dtype=np.float)

        # Add minimal Gaussian noise on output
        # output *= np.random.normal(1, 0.01, output.shape)

        return input, output