#! /usr/bin/env python
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from cosmoprimo.fiducial import AbacusSummit
from cosmoprimo import Cosmology as Cosmology_cosmoprimo

class Cosmology(object):
    """
    Class containing useful cosmology methods. Assumes flat LCDM Universe.
    
    For specific simulation cosmologies, use the classes provided, e.g.
    CosmologyAbacus for the AbacusSummit simulation cosmologies.

    Args:
        cosmo: cosmoprimo cosmology object
    """
    def __init__(self, cosmo):

        self.h0     = cosmo.h
        self.OmegaM = cosmo.Omega_m(0)

        self.cosmo_cosmoprimo = cosmo

        self.__interpolator = self.__initialize_interpolator()


    def __initialize_interpolator(self):
        # create RegularGridInterpolator for converting comoving
        # distance to redshift
        z = np.arange(0, 3, 0.0001)
        rcom = self.comoving_distance(z)
        return RegularGridInterpolator((rcom,), z,
                                       bounds_error=False, fill_value=None)

    
    def H(self, redshift):
        """
        Hubble parameter as a function of redshift

        Args:
            redshift: array of redshift
        Returns:
            array of H in units [km/s/Mpc]
        """
        return 100 * self.h0 * self.cosmo_cosmoprimo.efunc(redshift)
    

    def critical_density(self, redshift):
        """
        Critical density of the Universe as a function of redshift

        Args:
            redshift: array of redshift
        Returns:
            array of critical density in units [Msun Mpc^-3 h^2]
        """
        rho_crit = self.cosmo_cosmoprimo.rho_crit(redshift) * 1e10

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
        return self.cosmo_cosmoprimo.comoving_radial_distance(redshift)


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
        return self.cosmo_cosmoprimo.growth_factor(z)
       
        
    def growth_rate(self, z):
        """
        Returns the growth rate, f = dln(D)/dln(a)

        Args:
            z: array of redshift
        Returns:
            Growth rate
        """
        return self.cosmo_cosmoprimo.growth_rate(z)
    
    def dVdz(self, z):
        """
        Returns comoving volume element (multiplied by solid angle of full sky)
        
        Args:
            z: array of redshift
        Returns:
            Comoving volume element
        """
        c    = 3e5 # km/s
        H100 = 100 # km/s/Mpc
        return 4*np.pi*(c/H100) * self.comoving_distance(z)**2 / \
                                self.cosmo_cosmoprimo.efunc(z)
    

    def age(self, z):
        return self.cosmo_cosmoprimo.time(z)
    
    
    def get_xi_scaling_factor(self, cosmo_new, r_bins, pimax=120,
                               correlation_function="xi", scale=8, 
                               power_spectrum="lin", z=0.2):
        """
        Returns the cosmology rescaling factors for the correlation
        function

        Args:
            cosmo_new: new cosmology to rescale to
            r_bins:    array of r bins that xi is evaluated at, in Mpc/h
            [pimax]:   maximum value of pi in integral, for wp only, in Mpc/h.
                       Default is 120
            [correlation_function]: type of correlation function, "xi" or "wp".
                       Default is "xi"
            [scale]:   scale below which the scaling factor is set to 1, in Mpc/h.
                       Default is 8
            [power_spectrum]: "lin", "nl" or "zel"
            [z]:       redshift. Default is 0.2
        """

        from hodpy.power_spectrum import PowerSpectrum

        Pk1 = PowerSpectrum(self)
        Pk2 = PowerSpectrum(cosmo_new)
        
        if correlation_function=="xi":
            xi_c1 = Pk1.get_xi(r_bins, z=z, power_spectrum=power_spectrum)
            xi_c2 = Pk2.get_xi(r_bins, z=z, power_spectrum=power_spectrum)
            xi_c1_8 = Pk1.get_xi(np.array([scale,]), z=z, power_spectrum=power_spectrum)[0]
            xi_c2_8 = Pk2.get_xi(np.array([scale,]), z=z, power_spectrum=power_spectrum)[0]
            scaling_factor = xi_c2/xi_c1 * (xi_c1_8/xi_c2_8)
            
        elif correlation_function=="wp":
            wp_c1 = Pk1.get_wp(r_bins, z=z, pimax=pimax, power_spectrum=power_spectrum)
            wp_c2 = Pk2.get_wp(r_bins, z=z, pimax=pimax, power_spectrum=power_spectrum)
            wp_c1_8 = Pk1.get_wp(np.array([scale,]), z=z, pimax=pimax, power_spectrum=power_spectrum)[0]
            wp_c2_8 = Pk2.get_wp(np.array([scale,]), z=z, pimax=pimax, power_spectrum=power_spectrum)[0]
            scaling_factor = wp_c2/wp_c1 * (wp_c1_8/wp_c2_8)
            
        else: 
            raise ValueError("Invalid correlation function", correlation_function)
        
        # keep scaling_factor fixed to 1 below scale
        scaling_factor[r_bins<scale] = 1.0
    
        return scaling_factor
    

class CosmologyMXXL(Cosmology):
    """
    MXXL simulation cosmology
    """
    def __init__(self):
        
        cosmo_cosmoprimo = Cosmology_cosmoprimo(h=0.73, Omega_cdm=0.25-0.045, Omega_b=0.045,
                                                sigma8=0.9, n_s=1, engine='class')
        super().__init__(cosmo_cosmoprimo)
        
        
class CosmologyOR(Cosmology):
    """
    OuterRim simulation cosmology
    """
    def __init__(self):
        cosmo_cosmoprimo = Cosmology_cosmoprimo(h=0.71, omega_cdm=0.1109, omega_b=0.02258,
                                                sigma8=0.8, n_s=0.963, engine='class')
        super().__init__(cosmo_cosmoprimo)
        
        
class CosmologyUNIT(Cosmology):
    """
    UNIT simulation cosmology
    """
    def __init__(self):
        cosmo_cosmoprimo = Cosmology_cosmoprimo(h=0.6774, Omega_cdm=0.3089-0.04860, Omega_b=0.04860,
                                                sigma8=0.8147, n_s=0.9667, engine='class')
        super().__init__(cosmo_cosmoprimo)
        
        
class CosmologyUchuu(Cosmology):
    """
    Uchuu simulation cosmology
    """
    def __init__(self):
        cosmo_cosmoprimo = Cosmology_cosmoprimo(h=0.6774, Omega_cdm=0.3089-0.04860, Omega_b=0.04860,
                                                sigma8=0.8159, n_s=0.9667, engine='class')
        super().__init__(cosmo_cosmoprimo)

    

class CosmologyAbacus(Cosmology):
    """
    AbacusSummit simulation cosmology

    Args:
        cosmo: simulation cosmology number, e.g. 0 for `c000' Planck cosmology
    """
    def __init__(self, cosmo):
        
        cosmo_cosmoprimo = AbacusSummit(cosmo)
        
        super().__init__(cosmo_cosmoprimo)
        
        
    def __get_params(self, cosmo):
        
        cosmo_number = np.zeros(len(self.__param_array[:,0]), dtype="i")
        for i in range(len(self.__param_array[:,0])):
            cosmo_number[i] = int(self.__param_array[i,0][11:])
        
        idx = np.where(cosmo_number==cosmo)[0][0]
        
        return np.array(self.__param_array[idx,2:], dtype="f")
