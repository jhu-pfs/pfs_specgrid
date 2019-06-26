import sys
from pfsspec.parallel import prll_map

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

    def create_dataset(self):
        self.dataset = Dataset()
        self.dataset.params = self.params
        self.dataset.init_storage(self.get_wave_count(), self.get_spectrum_count())

    def process_item(self, i):
        raise NotImplementedError()

    def build(self):
        self.create_dataset()
        fluxes = prll_map(self.process_item, range(self.get_spectrum_count()), verbose=True)
        for i in range(len(fluxes)):
            self.dataset.flux[i, :] = fluxes[i]

        #with click.progressbar(range(self.get_spectrum_count()), file=sys.stderr) as bar:
         #   for i in bar:
         #       self.process_item(dataset, i)

        return self.dataset