import numpy as np
import pysynphot
import pysynphot.binning
import pysynphot.spectrum
import pysynphot.reddening
from scipy.integrate import simps

from pfsspec.stellarmod.modelspectrum import ModelSpectrum

class PhoenixSpectrum(ModelSpectrum):
    def __init__(self, orig=None):
        super(PhoenixSpectrum, self).__init__(orig=orig)

    # def synthmag(self, filter, norm=1.0):
    #    flux = norm * self.synthflux(filter)
    #   return -2.5 * np.log10(flux) + 8.90
    ###### replacing old synthmag with my own - is right :) 
    def getrad(self, lum,temp):
        sb = 5.67e-5 #grams s^-3 kelvin^-4
        lsun = 3.8e33 #erg/s 
        l = lsun*(10**lum) #luminosity from isochrone is in log(L/lsun)
        t = np.round(10**temp) #teff from isochrone is in log(teff)
        radius = np.sqrt(l/(4*np.pi*sb*t**4))
        return radius
    
    #phoenix is slightly different than bosz
    def synthmag_phoenix(self, filte,lum,temp):
        spec = pysynphot.spectrum.ArraySourceSpectrum(wave=self.wave, flux=self.flux*1e-8, keepneg=True, fluxunits='flam')
        #remember - phoenix flux needs to be multiplied by *1e-8
        filt = pysynphot.spectrum.ArraySpectralElement(filte.wave, filte.thru, waveunits='angstrom')
        #normalising spectra
        #getting bounds of integral
        lam = spec.wave[(spec.wave<=filt.wave.max())&(spec.wave>=filt.wave.min())]
        T = np.interp(lam,filt.wave,filt.throughput)
        T = np.where(T<.001, 0, T)
        R = self.getrad(lum,temp)
        #1/(3.08567758128*10**(19))**2 is just 1/10pc^2 in cm! (1/(3.086e19)**2)
        s = spec.flux[(spec.wave<=filt.wave.max())&(spec.wave>=filt.wave.min())]
        s = s*(R/3.086e19)**2 #NOT multiplied by pi!
        #interpolating to get filter data on same scale as spectral data
        #doin classic integral to get flux in bandpass
        stzp = 3.631e-9

        a = -2.5*np.log10((simps(s*T*lam,lam)/(stzp*simps(T*lam,lam))))
        b = -2.5*np.log10((simps(T*lam,lam)/simps(T/lam,lam)))
        return a+b+18.6921

    def normalize_to_mag(self, filt, mag):
        try:
            m = self.synthmag_bosz(filt)
            if m <= -10:
                # Checking that not really negative number, which happens when flux is from
                # Phoenix but isn't properly re-scaled - i.e. flux is ~1e8 too big
                # this step probably isn't really catching everything - must look into better way
                m = self.synthmag_phoenix(filt)
        except Exception as ex:
            print('flux max', np.max(self.flux))
            print('mag', mag)
            raise ex
        DM = mag - m
        D = 10 ** (DM / 5)

        self.multiply(1 / D**2)
        self.mag = mag