import numpy as np

from pfsspec.data.generativedatasetaugmenter import GenerativeDatasetAugmenter

class KuruczGenerativeAugmenter(GenerativeDatasetAugmenter):
    def __init__(self):
        super(KuruczGenerativeAugmenter, self).__init__()

    @classmethod
    def from_dataset(cls, dataset, labels, coeffs, weight=None, batch_size=1, shuffle=True, seed=None):
        d = super(KuruczGenerativeAugmenter, cls).from_dataset(dataset, labels, coeffs, weight,
                                                               batch_size=batch_size,
                                                               shuffle=shuffle,
                                                               seed=seed)
        return d

    def copy(self):
        new = KuruczGenerativeAugmenter(self.dataset, self.labels, self.coeffs,
                                        self.batch_size, self.shuffle, self.seed)
        return new

    def augment_batch(self, batch_index):
        labels, flux, weight = super(KuruczGenerativeAugmenter, self).augment_batch(batch_index)

        # Add minimal Gaussian noise on output
        # output *= np.random.normal(1, 0.01, output.shape)

        # TODO: what type of augmentation can we do here?
        # Cubic spline interpolation along grid lines?

        return labels, flux, weight