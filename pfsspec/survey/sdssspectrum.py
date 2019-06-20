from astropy.io import fits
import numpy as np

from pfsspec.spectrum import Spectrum

class SdssSpectrum(Spectrum):
    def __init__(self, orig=None):
        super(SdssSpectrum, self).__init__(orig=orig)
