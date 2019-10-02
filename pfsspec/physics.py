import numpy as np

def planck(wave, T):
    h = 6.62607015e-34      # J s
    c = 299792458           # m/s
    k = 1.380649e-23        # J/K
    a = 2.0 * h * c**2
    b = h * c / (wave * k * T)
    intensity = a / (wave**5 * (np.exp(b) - 1.0))
    return intensity        # J/m2/m