import numpy as np

from pfsspec.ml.dnn.keras.kerasdatagenerator import KerasDataGenerator

class AutoencodingDatasetAugmenter(KerasDataGenerator):
    def __init__(self, orig=None):
        super(AutoencodingDatasetAugmenter, self).__init__(orig=orig)

        if isinstance(orig, AutoencodingDatasetAugmenter):
            self.input_dataset = orig.input_dataset
            self.output_dataset = orig.output_dataset
            self.weight = orig.weight
        else:
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

    def init_from_args(self, args):
        # TODO: extend with more options
        pass

    def augment_batch(self, chunk_id, idx):
        # TODO: verify if chunks are read correctly from hdf5
        raise NotImplementedError()

        input = np.array(self.input_dataset.get_flux(idx, self.chunk_size, chunk_id), copy=True, dtype=np.float)
        output = np.array(self.output_dataset.get_flux(idx, self.chunk_size, chunk_id), copy=True, dtype=np.float)

        if self.weight is not None:
            weight = np.array(self.input_dataset.get_params(['weight'], idx, self.chunk_size, chunk_id), copy=True, dtype=np.float)[..., 0]
        else:
            weight = None

        return input, output, weight

