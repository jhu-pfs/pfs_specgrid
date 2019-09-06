import numpy as np

from pfsspec.ml.dnn.keras.kerasdatagenerator import KerasDataGenerator

class DatasetAugmenter(KerasDataGenerator):
    def __init__(self):
        super(DatasetAugmenter, self).__init__()
        self.dataset = None
        self.labels = None
        self.coeffs = None
        self.weight = None

        self.multiplicative_bias = False
        self.additive_bias = False
        self.include_wave = False

    @classmethod
    def from_dataset(cls, dataset, labels, coeffs, weight=None, batch_size=1, shuffle=True, seed=None):
        input_shape = dataset.flux.shape
        output_shape = (len(labels),)
        d = super(DatasetAugmenter, cls).from_shapes(input_shape, output_shape, batch_size=batch_size, shuffle=shuffle, seed=seed)
        d.dataset = dataset
        d.labels = labels
        d.coeffs = coeffs
        d.weight = weight

        return d

    def add_args(self, parser):
        parser.add_argument('--aug', type=str, default=None, help='Augment data.\n')

    def init_from_args(self, args, mode):
        # TODO: extend with more options
        if 'aug' in args and args['aug'] == 'no':
            self.multiplicative_bias = False
            self.additive_bias = False
        else:
            self.multiplicative_bias = True
            self.additive_bias = True

        # Copy weight column to new column called weight and normalize
        # We use the median since SNR is exponentially distributed
        # the clip normalized weights at 1
        if self.weight is not None and 'weight' not in self.dataset.params.columns:
            m = self.dataset.params[self.weight].median()
            self.dataset.params['weight'] = 0.5 * self.dataset.params[self.weight] / m
            self.dataset.params['weight'][self.dataset.params['weight'] > 1] = 1

    def augment_batch(self, batch_index):
        flux = np.array(self.dataset.flux[batch_index], copy=True, dtype=np.float)
        labels = np.array(self.dataset.params[self.labels].iloc[batch_index], copy=True, dtype=np.float)

        if self.weight is not None:
            weight = np.array(self.dataset.params['weight'].iloc[batch_index], copy=True, dtype=np.float)
        else:
            weight = None

        return flux, labels, weight

    def get_average(self):
        return np.mean(np.array(self.dataset.params[self.labels]), axis=0) / self.coeffs