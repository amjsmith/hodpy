#! /usr/bin/env python
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import RegularGridInterpolator

# constants
Msun_g = 1.989e33 # solar mass in g
Mpc_cm = 3.086e24 # Mpc in cm

class Cosmology(object):
    """
    Class containing useful cosmology methods. Assumes flat LCDM Universe.

    Args:
        h0:     Hubble parameter at z=0, in units [100 km/s/Mpc]
        OmegaM: Omega matter at z=0
        OmegaL: Omega Lambda at z=0
    """
    def __init__(self, h0, OmegaM, OmegaL):

        self.h0     = h0
        self.OmegaM = OmegaM
        self.OmegaL = OmegaL

        # assumes cosmology is flat LCDM
        self.__cosmo = FlatLambdaCDM(H0=h0*100, Om0=OmegaM)

        self.__interpolator = self.__initialize_interpolator()


    def __initialize_interpolator(self):
        # create RegularGridInterpolator for converting comoving
        # distance to redshift
        z = np.arange(0, 3, 0.0001)
        rcom = self.comoving_distance(z)
        return RegularGridInterpolator((rcom,), z,
                                       bounds_error=False, fill_value=None)


    def critical_density(self, redshift):
        """
        Critical density of the Universe as a function of redshift

        Args:
            redshift: array of redshift
        Returns:
            array of critical density in units [Msun Mpc^-3 h^2]
        """
        rho_crit = self.__cosmo.critical_density(redshift).value # in g cm^-3

        # convert to Msun Mpc^-3 h^2
        rho_crit *= Mpc_cm**3 / Msun_g / self.h0**2

        return rho_crit


    def mean_density(self, redshift):
        """
        Mean matter density of the Universe as a function of redshift

        Args:
            redshift: array of redshift
        Returns:
            array of critical density in units [Msun Mpc^-3 h^2]
        """
        # mean density at z=0
        rho_mean0 = self.critical_density(0) * self.OmegaM

        # evolve to redshift z
        return  rho_mean0 * (1+redshift)**3


    def comoving_distance(self, redshift):
        """
        Comoving distance to redshift

        Args:
            redshift: array of redshift
        Returns:
            array of comoving distance in units [Mpc/h]
        """
        return self.__cosmo.comoving_distance(redshift).value*self.h0


    def redshift(self, distance):
        """
        Redshift to comoving distance

        Args:
            distance: comoving distance in units [Mpc/h]
        Returns:
            array of redshift
        """
        return self.__interpolator(distance)


if __name__ == "__main__":

    import parameters as par
    cos = Cosmology(par.h0, par.OmegaM, par.OmegaL)
    print(cos.critical_density(0.0))

    z = np.random.rand(100) * 2
    print(z)
    rcom = cos.comoving_distance(z)
    z2 = cos.redshift(rcom)
    print(z2)
    print(z-z2)
    print(np.max(np.absolute(z-z2)))
