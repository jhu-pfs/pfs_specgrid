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

        self.initialize_generator(self.training_generator, self.args['noiz'])
        self.initialize_generator(self.validation_generator, 'full')

    def initialize_generator(self, g, noise):
        super(TrainKurucz, self).initialize_generator(g)

        g.multiplicative_bias = self.args['aug']
        g.additive_bias = self.args['aug']

        if noise is None or noise == 'no':
            g.noise = 0
        elif noise == 'full':
            g.noise = 1.0
        elif noise == 'prog':
            # progressively increasing noise
            g.noise_scheduler = 'linear'
        else:
            g.noise = float(noise)

def main():
    script = TrainKurucz()
    script.execute()

if __name__ == "__main__":
    main()