import numpy as np

from pfsspec.ml.dnn.keras.kerasdatagenerator import KerasDataGenerator

class AutoencodingDatasetAugmenter(KerasDataGenerator):
    def __init__(self):
        super(AutoencodingDatasetAugmenter, self).__init__()
        self.input_dataset = None
        self.output_dataset = None
        self.weight = None

    @classmethod
    def from_datasets(cls, input_dataset, output_dataset, weight=None, batch_size=1, shuffle=True,
                     seed=None):
        input_shape = input_dataset.flux.shape
        output_shape = (output_dataset.flux.shape[1], )
        d = super(AutoencodingDatasetAugmenter, cls).from_shapes(input_shape, output_shape, batch_size=batch_size, shuffle=shuffle,
                                                     seed=seed)
        d.input_dataset = input_dataset
        d.output_dataset = output_dataset
        d.weight = weight

        return d

    def add_args(self, parser):
        parser.add_argument('--aug', type=str, default=None, help='Augment data.\n')

    def init_from_args(selfself, args):
        # TODO: extend with more options
        pass

    def augment_batch(self, idx):
        input = np.array(self.input_dataset.flux[idx], copy=True, dtype=np.float)
        output = np.array(self.output_dataset.flux[idx], copy=True, dtype=np.float)

        if self.weight is not None:
            weight = np.array(self.dataset.params['weight'].iloc[idx], copy=True, dtype=np.float)
        else:
            weight = None

        return input, output, weight

    def get_output_mean(self):
        return np.mean(self.output_dataset.flux, axis=0)
