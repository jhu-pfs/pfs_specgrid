import os
import numpy as np

from pfsspec.data.regressionaldatasetaugmenter import RegressionalDatasetAugmenter

class KuruczRegressionalAugmenter(RegressionalDatasetAugmenter):
    def __init__(self):
        super(KuruczRegressionalAugmenter, self).__init__()
        self.multiplicative_bias = False
        self.additive_bias = False
        self.noise = None
        self.noise_scheduler = None
        self.normalize_weights = None
        self.normalize = None

    @classmethod
    def from_dataset(cls, dataset, labels, coeffs, weight=None, batch_size=1, shuffle=True, seed=None):
        d = super(KuruczRegressionalAugmenter, cls).from_dataset(dataset, labels, coeffs, weight,
                                  batch_size=batch_size, shuffle=shuffle, seed=seed)
        return d

    def add_args(self, parser):
        super(KuruczRegressionalAugmenter, self).add_args(parser)
        parser.add_argument('--noiz', type=str, help='Add noise.\n')
        parser.add_argument('--norm', type=str, default=None, help='Normalize with continuum.')

    def init_from_args(self, args, mode):
        super(KuruczRegressionalAugmenter, self).init_from_args(args, mode)

        if mode == 'train':
            if 'noiz' not in args or args['noiz'] is None or args['noiz'] == 'no':
                self.noise = 0
            elif args['noiz'] == 'full':
                self.noise = 1.0
            elif args['noiz'] == 'prog':
                # progressively increasing noise
                self.noise_scheduler = 'linear'
            else:
                self.noise = float(args['noiz'])
        elif mode == 'test' or mode == 'predict':
            if 'noiz' in args and args['noiz'] == 'no':
                self.noise = 0
            else:
                self.noise = 1.0
        else:
            raise NotImplementedError()

        if 'norm' in args and args['norm'] is not None:
            self.normalize = args['norm']
            self.normalize_weights = np.loadtxt(os.path.join(args['in'], 'weights.dat'))[:, 2].squeeze()

    def noise_scheduler_linear_onestep(self):
        break_point = int(0.5 * self.total_epochs)
        if self.current_epoch < break_point:
            return self.current_epoch / break_point
        else:
            return 1.0

    def noise_scheduler_linear_twostep(self):
        break_point_1 = int(0.2 * self.total_epochs)
        break_point_2 = int(0.5 * self.total_epochs)
        if self.current_epoch < break_point_1:
            return 0.0
        elif self.current_epoch < break_point_2:
            return (self.current_epoch - break_point_1) / (self.total_epochs - break_point_1 - break_point_2)
        else:
            return 1.0

    def augment_batch(self, batch_index):
        flux, labels, weight = super(KuruczRegressionalAugmenter, self).augment_batch(batch_index)
        error = np.array(self.dataset.error[batch_index], copy=True, dtype=np.float)

        if self.noise_scheduler == 'linear':
            self.noise = self.noise_scheduler_linear_onestep()

        if self.noise is not None and self.noise > 0.0:
            if error is not None:
                # If error vector is present, use as sigma
                err = self.noise * np.random.normal(size=flux.shape) * error
                flux = flux + err
            else:
                # Simple additive noise, one random number per bin
                err = np.random.uniform(0, self.noise, flux.shape)
                flux = flux + err

        # Fit continuum, if requested
        # TODO: figure out how to vectorize fitting
        if self.normalize == 'poly':
            for i in range(flux.shape[0]):
                poly = np.polyfit(self.dataset.wave, flux[i, :], 4, w=1/self.normalize_weights)
                cont = np.polyval(poly, self.dataset.wave)
                flux[i, :] = flux[i, :] / cont

        # Additive and multiplicative bias, two numbers per spectrum
        if self.multiplicative_bias:
            bias = np.random.uniform(0.95, 0.05, (flux.shape[0], 1))
            flux = flux * bias
        if self.additive_bias:
            bias = np.random.normal(0, 0.01, (flux.shape[0], 1))
            flux = flux + bias

        return flux, labels, weight