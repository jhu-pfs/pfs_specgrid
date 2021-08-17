import numpy as np

class Physics():
    h = 6.62607015e-34  # J s
    c = 299792458  # m/s
    k_B = 1.380649e-23  # J/K

    HYDROGEN_LIMITS = [3646.0, 8203.6, 14584]                   # all numbers in air

    @staticmethod
    def planck(wave, T):
        a = 2.0 * Physics.h * Physics.c**2                      # J m+2 s-1
        b = Physics.h * Physics.c / (wave * Physics.k_B * T)    # 1
        intensity = a / (wave**5 * (np.exp(b) - 1.0))           # J m-2 m-1 s-1
        return intensity

    @staticmethod
    def angstrom_to_nm(wave):
        return 1e-1 * wave

    @staticmethod
    def fnu_to_flam(wave, fnu):
        # ergs/cm**2/s/Hz to erg/s/cm^2/A
        flam = fnu / (3.336e-19 * wave**2)
        return flam

    @staticmethod
    def flam_to_fnu(wave, flam):
        # erg/s/cm^2/A to ergs/cm**2/s/Hz
        fnu = flam * 3.336e-19 * wave**2
        return fnu

    @staticmethod
    def fnu_to_abmag(fnu):
        # erg/s/cm^2/Hz to mag_AB
        mags = -2.5 * np.log10(fnu) - 48.60
        return mags

    @staticmethod
    def abmag_to_fnu(abmag):
        # mag_AB to erg/s/cm^2/Hz
        fnu = 10**(-0.4 * (abmag + 48.60))
        return fnu

    @staticmethod
    def flam_to_abmag(wave, flam):
        # erg/s/cm^2/A to mag_AB
        fnu = Physics.flam_to_fnu(wave, flam)
        mags = Physics.fnu_to_abmag(fnu)
        return mags

    @staticmethod
    def air_to_vac(wave):
        """
        Implements the air to vacuum wavelength conversion described in eqn 65 of
        Griesen 2006
        """
        wlum = wave * 1e5
        return (1+1e-6*(287.6155+1.62887/wlum**2+0.01360/wlum**4)) * wave

    @staticmethod
    def air_to_vac_deriv(wave):
        """
        Eqn 66
        """
        wlum = wave * 1e5
        return (1+1e-6*(287.6155 - 1.62877/wlum**2 - 0.04080/wlum**4))

    @staticmethod
    def cm_to_pc(d):
        return d / 3.08567758128e19

    @staticmethod
    def pc_to_cm(d):
        return d * 3.08567758128e19

    @staticmethod
    def get_radius(log_L, T_eff):
        # Calculate the radius of a star with known luminosity and effective
        # temperature using the Stefan-Boltzmann law
        
        L_sun = 3.839e33        # erg/s
        sb = 5.6704e-5          # grams s^-3 kelvin^-4
        L = L_sun * 10**log_L   # luminosity from isochrone is in log(L/lsun)
        radius = np.sqrt(L / (4 * np.pi * sb * T_eff**4))
        return radius           # radius in cm