from pfsspec.train.trainingset import TrainingSet

class TrainingSetBuilder():
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
        ts = TrainingSet()
        ts.params = self.params
        ts.init_storage(self.get_wave_count(), self.get_spectrum_count())
        return ts