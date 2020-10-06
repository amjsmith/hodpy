#! /usr/bin/env python
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import nbodykit.cosmology as cosmology_nbodykit


class Cosmology(object):
    """
    Class containing useful cosmology methods. Assumes flat LCDM Universe.

    Args:
        cosmo: nbodykit cosmology object
    """
    def __init__(self, cosmo):

        self.h0     = cosmo.h
        self.OmegaM = cosmo.Om0

        self.cosmo_nbodykit = cosmo

        self.__interpolator = self.__initialize_interpolator()


    def __initialize_interpolator(self):
        # create RegularGridInterpolator for converting comoving
        # distance to redshift
        z = np.arange(0, 3, 0.0001)
        rcom = self.comoving_distance(z)
        return RegularGridInterpolator((rcom,), z,
                                       bounds_error=False, fill_value=None)

    
    def H(self, redshift):
        return 100 * self.h0 * self.cosmo_nbodykit.efunc(redshift)
    

    def critical_density(self, redshift):
        """
        Critical density of the Universe as a function of redshift

        Args:
            redshift: array of redshift
        Returns:
            array of critical density in units [Msun Mpc^-3 h^2]
        """
        rho_crit = self.cosmo_nbodykit.rho_crit(redshift) * 1e10

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
        return self.cosmo_nbodykit.comoving_distance(redshift)


    def redshift(self, distance):
        """
        Redshift to comoving distance

        Args:
            distance: comoving distance in units [Mpc/h]
        Returns:
            array of redshift
        """
        return self.__interpolator(distance)
    
    
    def growth_factor(self, z):
        """
        Linear growth factor D(a), as a function of redshift

        Args:
            z: array of redshift
        Returns:
            Linear growth factor
        """
        return self.cosmo_nbodykit.scale_independent_growth_factor(z)
       
        
    def growth_rate(self, z):
        """
        Returns the growth rate, f = dln(D)/dln(a)

        Args:
            z: array of redshift
        Returns:
            Growth rate
        """
        return self.cosmo_nbodykit.scale_independent_growth_rate(z)

    

class CosmologyMXXL(Cosmology):
    
    def __init__(self):
        cosmo_nbody = cosmology_nbodykit.WMAP5
        cosmo_nbody = cosmo_nbody.clone(Omega0_b=0.045, Omega0_cdm=0.25-0.045, h=0.73, n_s=1)
        cosmo_nbody = cosmo_nbody.match(sigma8=0.9)
        super().__init__(cosmo_nbody)
        
        
    

