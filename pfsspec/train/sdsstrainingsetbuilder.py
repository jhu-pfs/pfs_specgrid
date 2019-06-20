from pfsspec.train.trainingsetbuilder import TrainingSetBuilder

class SdssTrainingSetBuilder(TrainingSetBuilder):
    def __init__(self, orig=None):
        super(SdssTrainingSetBuilder, self).__init__(orig)
        if orig is not None:
            self.dataset = orig.dataset
        else:
            self.dataset = None

    def get_spectrum_count(self):
        return len(self.dataset.spectra)

    def get_wave_count(self):
        return self.pipeline.rebin.shape[0]

    def build(self):
        ts = super(SdssTrainingSetBuilder, self).build()
        ts.wave[:] = self.pipeline.rebin

        for i in range(0, len(self.dataset.spectra)):
            spec = self.dataset.spectra[i]
            spec = self.pipeline.run(spec)
            ts.flux[i, :] = spec.flux

        return ts