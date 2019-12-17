from pfsspec.data.dataset import Dataset
from pfsspec.stellarmod.modelspectrum import ModelSpectrum

class ModelDataset(Dataset):
    def __init__(self, orig=None):
        super(ModelDataset, self).__init__(orig=orig)

    def create_spectrum(self):
        return ModelSpectrum()