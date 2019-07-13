import numpy as np

from pfsspec.ml.dnn.keras.kerasdatagenerator import KerasDataGenerator

class DatasetAugmenter(KerasDataGenerator):
    def __init__(self, dataset, labels, coeffs, input_shape, output_shape, batch_size=1, shuffle=True, seed=None):
        self.dataset = dataset
        self.labels = labels
        self.coeffs = coeffs
        super(DatasetAugmenter, self).__init__(input_shape, output_shape,
                                                 batch_size=batch_size, shuffle=shuffle, seed=seed)
