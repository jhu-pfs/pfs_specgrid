#!/usr/bin/env python

from pfsspec.scripts.train import Train
from pfsspec.stellarmod.kuruczregressionalaugmenter import KuruczRegressionalAugmenter

class TrainKurucz(Train):
    def __init__(self):
        super(TrainKurucz, self).__init__()

    def create_generators(self):
        if self.args['split'] != 0:
            _, ts, vs = self.dataset.split(self.args['split'])
            self.training_generator = KuruczRegressionalAugmenter(ts, self.labels, self.coeffs,
                                                                  batch_size=self.args['batch'])
            self.validation_generator = KuruczRegressionalAugmenter(vs, self.labels, self.coeffs,
                                                                    batch_size=self.args['batch'])
        else:
            self.training_generator = KuruczRegressionalAugmenter(self.dataset, self.labels, self.coeffs, batch_size=self.args['batch'])
            self.validation_generator = KuruczRegressionalAugmenter(self.dataset, self.labels, self.coeffs, batch_size=self.args['batch'])

        self.training_generator.multiplicative_bias = self.args['aug']
        self.training_generator.additive_bias = self.args['aug']

        self.validation_generator.multiplicative_bias = self.args['aug']
        self.validation_generator.additive_bias = self.args['aug']

def main():
    script = TrainKurucz()
    script.execute()

if __name__ == "__main__":
    main()