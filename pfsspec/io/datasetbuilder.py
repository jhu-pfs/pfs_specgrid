from pfsspec.io.dataset import Dataset

class DatasetBuilder():
    def __init__(self, orig=None):
        if orig is not None:
            self.params = orig.params
            self.pipeline = orig.pipeline
        else:
            self.params = None
            self.pipeline = None

    def get_spectrum_count(self):
        raise NotImplementedError()

    def get_wave_count(self):
        raise NotImplementedError()

    def build(self):
        dataset = Dataset()
        dataset.params = self.params
        dataset.init_storage(self.get_wave_count(), self.get_spectrum_count())
        return dataset