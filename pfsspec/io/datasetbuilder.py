import sys
import click

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
        dataset = Dataset()
        dataset.params = self.params
        dataset.init_storage(self.get_wave_count(), self.get_spectrum_count())
        return dataset

    def process_item(self, i):
        raise NotImplementedError()

    def build(self):
        dataset = self.create_dataset()

        with click.progressbar(range(self.get_spectrum_count()), file=sys.stderr) as bar:
            for i in bar:
                self.process_item(dataset, i)

        return dataset