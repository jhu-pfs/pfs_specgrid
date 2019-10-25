import numpy as np

from pfsspec.data.autoencodingdatasetaugmenter import AutoencodingDatasetAugmenter

class KuruczAutoencodingAugmenter(AutoencodingDatasetAugmenter):
    def __init__(self):
        super(KuruczAutoencodingAugmenter, self).__init__()

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

    def augment_batch(self, batch_id):
        labels, flux, weight = super(KuruczAutoencodingAugmenter, self).augment_batch(batch_id)

        # Add minimal Gaussian noise on output
        # output *= np.random.normal(1, 0.01, output.shape)

        # TODO: what type of augmentation can we do here?
        # Cubic spline interpolation along grid lines?

        return labels, flux, weight