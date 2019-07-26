import numpy as np

from pfsspec.data.datasetaugmenter import DatasetAugmenter

class KuruczRegressionalAugmenter(DatasetAugmenter):
    def __init__(self, dataset, labels, coeffs, batch_size=1, shuffle=True, seed=None):
        input_shape = dataset.flux.shape
        output_shape = (len(labels),)
        super(KuruczRegressionalAugmenter, self).__init__(dataset, labels, coeffs,
                                                        input_shape, output_shape,
                                                        batch_size=batch_size, shuffle=shuffle, seed=seed)

        self.multiplicative_bias = False
        self.additive_bias = False
        self.noise = None
        self.noise_scheduler = None

    def copy(self):
        new = KuruczRegressionalAugmenter(self.dataset, self.labels, self.coeffs,
                                        self.batch_size, self.shuffle, self.seed)
        return new

    def scale_output(self, output):
        return output / self.coeffs

    def rescale_output(self, output):
        return output * self.coeffs

    def augment_batch(self, batch_index):
        flux = np.array(self.dataset.flux[batch_index], copy=True, dtype=np.float)
        error = np.array(self.dataset.error[batch_index], copy=True, dtype=np.float)
        labels = np.array(self.dataset.params[self.labels].iloc[batch_index], copy=True, dtype=np.float)

        if self.noise_scheduler == 'linear':
            break_point = int(0.2 * self.total_epochs)
            if self.current_epoch < break_point:
                self.noise = 0.0
            elif self.current_epoch < self.total_epochs - break_point:
                self.noise = (self.current_epoch - break_point) / (self.total_epochs - break_point - break_point)
            else:
                self.noise = 1.0

        if self.noise is not None and self.noise > 0.0:
            if error is not None:
                # If error vector is present, use as sigma
                err = self.noise * np.random.normal(size=flux.shape) * error
                flux = flux + err
            else:
                # Simple additive noise, one random number per bin
                err = np.random.uniform(0, self.noise, flux.shape)
                flux = flux + err

        # Additive and multiplicative bias, two numbers per spectrum
        if self.multiplicative_bias:
            bias = np.random.uniform(0.8, 1.2, (flux.shape[0], 1))
            flux = flux * bias
        if self.additive_bias:
            bias = np.random.normal(0, 1.0, (flux.shape[0], 1))
            flux = flux + bias

        return flux, labels