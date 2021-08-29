import numpy as np
import pysynphot
import pysynphot.binning
import pysynphot.spectrum
import pysynphot.reddening
from scipy.integrate import simps

from pfsspec.stellarmod.kuruczspectrum import KuruczSpectrum

class BoszSpectrum(KuruczSpectrum):
    def __init__(self, orig=None):
        super(BoszSpectrum, self).__init__(orig=orig)

    def synthmag_bosz_carrie(self, filte,lum,temp):
        spec = pysynphot.spectrum.ArraySourceSpectrum(wave=self.wave, flux=self.flux, keepneg=True, fluxunits='flam')
        filt = pysynphot.spectrum.ArraySpectralElement(filte.wave, filte.thru, waveunits='angstrom')
        #normalising spectra
        #getting bounds of integral
        lam = spec.wave[(spec.wave<=filt.wave.max())&(spec.wave>=filt.wave.min())]
        T = np.interp(lam,filt.wave,filt.throughput)
        T = np.where(T<.001, 0, T)
        R = self.getrad(lum,temp) 
        #1/(3.08567758128*10**(19))**2 is just 1/10pc^2 in cm! (1/(3.086e19)**2)
        s = spec.flux[(spec.wave<=filt.wave.max())&(spec.wave>=filt.wave.min())]
        s = s*np.pi*(R/3.086e19)**2 #multiply by pi!!
        #doin classic integral to get flux in bandpass
        stzp = 3.631e-9

        a = -2.5*np.log10((simps(s*T*lam,lam)/(stzp*simps(T*lam,lam))))
        b = -2.5*np.log10((simps(T*lam,lam)/simps(T/lam,lam)))
        return a+b+18.6921