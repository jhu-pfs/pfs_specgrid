import numpy as np

from pfsspec.ml.dnn.keras.kerasdatagenerator import KerasDataGenerator

class DatasetAugmenter(KerasDataGenerator):
    def __init__(self, orig=None):
        super(DatasetAugmenter, self).__init__(orig=orig)

        if isinstance(orig, DatasetAugmenter):
            self.dataset = orig.dataset
            self.labels = orig.labels
            self.coeffs = orig.coeffs
            self.weight = orig.weight
        else:
            self.dataset = None
            self.labels = None
            self.coeffs = None
            self.weight = None

    @classmethod
    def from_dataset(cls, input_shape, output_shape, dataset, labels, coeffs, weight=None, batch_size=1, shuffle=True, chunk_size=None, seed=None):
        d = super(DatasetAugmenter, cls).from_shapes(input_shape, output_shape, batch_size=batch_size, shuffle=shuffle, chunk_size=chunk_size, seed=seed)

        d.dataset = dataset
        d.labels = labels
        d.coeffs = coeffs
        d.weight = weight
               
        return d

    def add_args(self, parser):
        super(DatasetAugmenter, self).add_args(parser)

    def init_from_args(self, args):
        super(DatasetAugmenter, self).init_from_args(args)

    def augment_batch(self, chunk_id, idx):
        input = None
        output = None

        if self.weight is not None and 'weight' in self.dataset.params.columns:
            weight = np.array(self.dataset.params['weight'].iloc[idx], copy=True, dtype=np.float)
        else:
            weight = None

        return input, output, weight

    def get_output_mean(self):
        raise NotImplementedError()

    def get_output_labels(self, model):
        # Override this to return list of labels and postfixes for prediction
        pass

    def init_output_labels(self, labels, postfixes):
        # Override this if new labels need to be created for prediction
        pass
