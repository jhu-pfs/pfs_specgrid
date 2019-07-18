#!/usr/bin/env python

from pfsspec.scripts.train import Train
from pfsspec.surveys.sdssaugmenter import SdssAugmenter

class TrainSdss(Train):
    def __init__(self):
        super(TrainSdss, self).__init__()

    def create_generators(self):
        _, ts, vs = self.dataset.split(self.args['split'])
        self.training_generator = SdssAugmenter(ts, self.labels, self.coeffs, batch_size=self.args['batch'])
        self.training_generator.include_wave = self.args['wave']
        self.training_generator.multiplicative_bias = self.args['aug']
        self.training_generator.additive_bias = self.args['aug']

        self.validation_generator = SdssAugmenter(vs, self.labels, self.coeffs, batch_size=self.args['batch'])
        self.validation_generator.include_wave = self.args['wave']

        self.prediction_generator = SdssAugmenter(self.dataset, self.labels, self.coeffs, batch_size=self.args['batch'], shuffle=False)
        self.prediction_generator.include_wave = self.args['wave']

def main():
    script = TrainSdss()
    script.execute()

if __name__ == "__main__":
    main()