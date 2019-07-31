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
        parser.add_argument('--aug', action='store_true', help='Augment data.\n')

    def init_from_args(self, args, mode):
        self.multiplicative_bias = args['aug']
        self.additive_bias = args['aug']

    def augment_batch(self, batch_index):
        flux = np.array(self.dataset.flux[batch_index], copy=True, dtype=np.float)
        labels = np.array(self.dataset.params[self.labels].iloc[batch_index], copy=True, dtype=np.float)

        if self.weight is not None:
            weight = np.array(self.dataset.params[self.weight].iloc[batch_index], copy=True, dtype=np.float)
        else:
            weight = None

        return flux, labels, weight