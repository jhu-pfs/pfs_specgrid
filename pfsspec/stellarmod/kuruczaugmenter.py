import os
import numpy as np

import pfsspec.util as util
from pfsspec.obsmod.spectrum import Spectrum

class KuruczAugmenter():
    def __init__(self):
        self.noise = None
        self.noise_schedule = None
        self.aug_offset = None
        self.aug_scale = None
        self.mask_data = False
        self.mask_snr = None
        self.mask_random = None
        self.mask_value = 0

        self.include_wave = False

    def add_args(self, parser):
        parser.add_argument('--noise', type=float, default=None, help='Add noise.\n')
        parser.add_argument('--noise-sch', type=str, choices=['constant', 'linear'], default='constant', help='Noise schedule.\n')
        parser.add_argument('--aug-offset', type=float, default=None, help='Augment by adding a random offset.\n')
        parser.add_argument('--aug-scale', type=float, default=None, help='Augment by multiplying with a random number.\n')
        parser.add_argument('--mask-data', action='store_true', help='Use mask from dataset.\n')
        parser.add_argument('--mask-snr', type=float, help='Mask noisy bins that are below.\n')
        parser.add_argument('--mask-random', type=float, nargs="*", help='Add random mask.\n')
        parser.add_argument('--mask-value', type=float, default=0, help='Use mask value.\n')

    def init_from_args(self, args, mode):
        self.noise = util.get_arg('noise', self.noise, args)
        if mode == 'train':
            self.noise_schedule = util.get_arg('noise_sch', self.noise_schedule, args)
        elif mode in ['test', 'predict']:
            self.noise_schedule = 'constant'
        else:
            raise NotImplementedError()
        self.aug_offset = util.get_arg('aug_offset', self.aug_offset, args)
        self.aug_scale = util.get_arg('aug_scale', self.aug_scale, args)
        self.mask_data = util.get_arg('mask_data', self.mask_data, args)
        self.mask_snr = util.get_arg('mask_snr', self.mask_snr, args)
        self.mask_random = util.get_arg('mask_random', self.mask_random, args)
        self.mask_value = util.get_arg('mask_value', self.mask_value, args)

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

    def augment_batch(self, dataset, idx, flux, labels, weight):
        if dataset.error is not None:
            error = np.array(dataset.error[idx], copy=True, dtype=np.float)
        else:
            error = None

        mask = np.full(flux.shape, False)

        if self.mask_data and dataset.mask is not None:
            # TODO: verify this with real survey data
            mask = mask | (dataset.mask[idx] != 0)
        
        # Mask out points where noise is too high
        if self.mask_snr is not None and self.mask_value is not None and error is not None:
            mask = mask | (np.abs(flux / error) < self.mask_snr)

        noise = self.noise
        if self.noise_schedule == 'constant':
            pass
        elif self.noise_schedule == 'linear':
            noise *= self.noise_scheduler_linear_onestep()
        elif self.noise_schedule is not None:
            raise NotImplementedError()

        if noise is not None and noise > 0.0:
            # Noise don't have to be reproducible during training, do not reseed.
            flux = Spectrum.generate_noise(flux, noise, error=error, random_seed=None)

        # Additive and multiplicative bias, two numbers per spectrum
        if self.aug_scale is not None:
            bias = np.random.normal(1, self.aug_scale, (flux.shape[0], 1))
            flux *= bias
        if self.aug_offset is not None:
            bias = np.random.normal(0, self.aug_offset, (flux.shape[0], 1))
            flux += bias

        # Mask out point based on data
        if self.mask_value is not None and mask is not None:
            flux[mask] = self.mask_value

        # Generate random mask
        if self.mask_random is not None and self.mask_value is not None:
            for k in range(flux.shape[0]):
                n = np.random.randint(0, self.mask_random[0] + 1)
                for i in range(n):
                    wl = dataset.wave[0] + np.random.rand() * (dataset.wave[-1] - dataset.wave[0])
                    ww = max(0.0, np.random.normal(self.mask_random[1], self.mask_random[2]))
                    mx = np.digitize([wl - ww / 2, wl + ww / 2], dataset.wave)
                    flux[k, mx[0]:mx[1]] = self.mask_value

        return flux, labels, weight