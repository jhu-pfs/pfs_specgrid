from pfsspec.data.dataset import Dataset
from pfsspec.stellarmod.modelspectrum import ModelSpectrum

class ModelDataset(Dataset):
    def __init__(self, orig=None, preload_arrays=False):
        super(ModelDataset, self).__init__(orig=orig, preload_arrays=preload_arrays)

    def create_spectrum(self):
        return ModelSpectrum()