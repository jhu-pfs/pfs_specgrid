#!/usr/bin/env python

from pfsspec.scripts.train import Train
from pfsspec.stellarmod.kuruczregressionalaugmenter import KuruczRegressionalAugmenter

class TrainKurucz(Train):
    def __init__(self):
        super(TrainKurucz, self).__init__()

    def create_generators(self):
        self.training_generator = KuruczRegressionalAugmenter(self.dataset, self.labels, self.coeffs, batch_size=self.args.batch)
        self.training_generator.multiplicative_bias = self.args.aug
        self.training_generator.additive_bias = self.args.aug

        self.validation_generator = KuruczRegressionalAugmenter(self.dataset, self.labels, self.coeffs, batch_size=self.args.batch)
        self.validation_generator.multiplicative_bias = self.args.aug
        self.validation_generator.additive_bias = self.args.aug

        self.prediction_generator = KuruczRegressionalAugmenter(self.dataset, self.labels, self.coeffs, batch_size=self.args.batch, shuffle=False)

def __main__():
    script = TrainKurucz()
    script.execute()

__main__()