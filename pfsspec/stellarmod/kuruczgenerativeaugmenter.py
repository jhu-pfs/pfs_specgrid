import numpy as np

from pfsspec.ml.dnn.keras.kerasdatagenerator import KerasDataGenerator

class KuruczGenerativeAugmenter(KerasDataGenerator):
    def __init__(self, dataset, labels, coeffs, batch_size=1, shuffle=True, seed=0):
        self.dataset = dataset
        self.labels = labels
        self.coeffs = coeffs

        input_shape = (self.dataset.flux.shape[0], len(self.labels), )
        output_shape = self.dataset.flux.shape[1:]
        super(KuruczGenerativeAugmenter, self).__init__(input_shape, output_shape,
                                                   batch_size=batch_size, shuffle=shuffle, seed=seed)

    def next_batch(self, batch_index):
        bs = self.next_batch_size(batch_index)

        input = np.array(self.dataset.params[self.labels].iloc[self.index[batch_index * self.batch_size:batch_index * self.batch_size + bs]], copy=True, dtype=np.float)
        output = np.array(self.dataset.flux[self.index[batch_index * self.batch_size:batch_index * self.batch_size + bs]], copy=True, dtype=np.float)

        input, output = self.augment_batch(self.dataset.wave, input, output)
        input /= self.coeffs

        return input, output

    def scale_input(self, input):
        return input / self.coeffs

    def rescale_input(self, input):
        return input * self.coeff

    def augment_batch(self, wave, input, output):
        return input, output