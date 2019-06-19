from astropy.io import fits
import numpy as np

from pfsspec.spectrum import Spectrum

class SdssSpectrum(Spectrum):
    def __init__(self):
        super(SdssSpectrum, self).__init__()
