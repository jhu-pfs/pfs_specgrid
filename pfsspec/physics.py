import numpy as np

class Physics():
    h = 6.62607015e-34  # J s
    c = 299792458  # m/s
    k_B = 1.380649e-23  # J/K

    def planck(wave, T):
        a = 2.0 * Physics.h * Physics.c**2
        b = Physics.h * Physics.c / (wave * Physics.k_B * T)
        intensity = a / (wave**5 * (np.exp(b) - 1.0))
        return intensity        # J/m2/m