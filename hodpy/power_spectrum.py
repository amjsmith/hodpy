#! /usr/bin/env python
import numpy as np
from scipy.integrate import simps, quad
from scipy.interpolate import splrep, splev
from scipy.optimize import minimize
from cosmoprimo import Fourier

from hodpy.cosmology import Cosmology

class PowerSpectrum(object):
    """
    Class containing the linear power spectrum and useful methods

    Args:
        cosmo: instance of Cosmology class
    """
    def __init__(self, cosmo):
        
        self.cosmo = cosmo   # this is my cosmology class
        
        # linear power spectrum
        fo_lin = Fourier(self.cosmo.cosmo_cosmoprimo, engine='class', non_linear=False)
        self.__p_lin = fo_lin.pk_interpolator()
        
        # non-linear power spectrum
        fo_nl = Fourier(self.cosmo.cosmo_cosmoprimo, engine='class', non_linear=True)
        self.__p_nl = fo_nl.pk_interpolator()
        
        # arrays needed to compute integral to get sigma(M)
        self.__k = 10**np.arange(-6,2,0.01)
        self.__P = self.P_lin(self.__k, z=0)
        self.__tck = self.__get_sigma_spline() #spline fit to sigma(M,z=0)

        
    def P_lin(self, k, z):
        """
        Returns the linear power spectrum at redshift z

        Args:
            k: array of k in units [h/Mpc]
            z: array of z
        Returns:
            array of linear power spectrum in units [Mpc/h]^-3
        """
        return self.__p_lin(k, z=0) * self.cosmo.growth_factor(z)**2

    
    def P_nl(self, k, z):
        """
        Returns the non-linear power spectrum at redshift z

        Args:
            k: array of k in units [h/Mpc]
            z: array of z
        Returns:
            array of linear power spectrum in units [Mpc/h]^-3
        """
        return self.__p_nl(k, z=0) * self.cosmo.growth_factor(z)**2
    
    
    def Delta2_lin(self, k, z):
        """
        Returns the dimensionless linear power spectrum at redshift z,
        defined as Delta^2(k) = 4pi * (k/2pi)^3 * P(k)

        Args:
            k: array of k in units [h/Mpc]
            z: array of z
        Returns:
            array of dimensionless linear power spectrum
        """
        return self.P_lin(k, z) * k**3 / (2*np.pi**2)

    
    def W(self, k, R):
        """
        Window function in k-space (Fourier transform of top hat window)

        Args:
            k: array of k in units [h/Mpc]
            z: array of R in units [Mpc/h]
        Returns:
            window function
        """
        return 3 * (np.sin(k*R) - k*R*np.cos(k*R)) / (k*R)**3

    
    def R_to_M(self, R):
        """
        Average mass enclosed by a sphere of comoving radius R

        Args:
            R: array of comoving radius in units [Mpc/h]
        Returns:
            array of mass in units [Msun/h]
        """
        return 4./3 * np.pi * R**3 * self.cosmo.mean_density(0)
    
    
    def M_to_R(self, M):
        """
        Comoving radius of a sphere which encloses on average mass M

        Args:
            M: array of mass in units [Msun/h]
        Returns:
            array of comoving radius in units [Mpc/h]
        """
        return (3*M / (4 * np.pi * self.cosmo.mean_density(0)))**(1./3)

    
    def __func(self, k, R):
        # function to integrate to get sigma(M)
        return self.__k**2 * self.__P * self.W(k,R)**2

    
    def __get_sigma_spline(self):
        # spline fit to sigma(R) at z=0
        logR = np.arange(-2,2,0.01)
        sigma = np.zeros(len(logR))
        R = 10**logR
        for i in range(len(R)):
            sigma[i] = simps(self.__func(self.__k, R[i]), self.__k)

        sigma = sigma / (2 * np.pi**2)
        sigma = np.sqrt(sigma)
        
        return splrep(logR, np.log10(sigma))

    
    def sigmaR_z0(self, R):
        """
        Returns sigma(R), the rms mass fluctuation in spheres of radius R,
        at redshift 0

        Args:
            R: array of comoving distance in units [Mpc/h]
        Returns:
            array of sigma
        """
        return 10**splev(np.log10(R), self.__tck)

    
    def sigmaR(self, R, z):
        """
        Returns sigma(R,z), the rms mass fluctuation in spheres of radius R,
        at redshift z

        Args:
            R: array of comoving distance in units [Mpc/h]
            z: array of redshift
        Returns:
            array of sigma
        """
        return self.sigmaR_z0(R) * self.delta_c(0) / self.delta_c(z)

    
    def sigma_z0(self, M):
        """
        Returns sigma(M), the rms mass fluctuation in spheres of mass M,
        at redshift 0

        Args:
            M: array of mass in units [Msun/h]
        Returns:
            array of sigma
        """
        R = self.M_to_R(M)
        return self.sigmaR_z0(R)

    
    def sigma(self, M, z):
        """
        Returns sigma(M), the rms mass fluctuation in spheres of mass M,
        at redshift z

        Args:
            M: array of mass in units [Msun/h]
            z: array of redshift
        Returns:
            array of sigma
        """
        return self.sigma_z0(M) * self.delta_c(0) / self.delta_c(z)
    
    
    def nu(self, M, z):
        """
        Returns nu = delta_c(z=0) / (sigma(M,z=0) * D(z))

        Args:
            M: array of mass in units [Msun/h]
            z: array of redshift
        Returns:
            array of nu
        """
        return self.delta_c(z) / self.sigma_z0(M)


    def delta_c(self, z):
        """
        Returns delta_c, the linear density threshold for collapse, 
        at redshift z

        Args:
            z: redshift
        Returns:
            delta_c
        """
        return 1.686 / self.cosmo.growth_factor(z)

    
    
    def get_xi(self, r, z, power_spectrum="lin"):
        """
        Returns the correlation function xi(r)
    
        Args:
            r:                array of separation r, in units Mpc/h
            z:                redshift to calculate xi
            [power_spectrum]: can be "lin", "nl" or "zel" (default is "lin")
        Returns:
            array of xi evaluated at r values
        """
        
        if power_spectrum=="lin":
            # linear power spectrum
            xi = self.__p_lin.to_xi()
            
        elif power_spectrum=="nl":
            # non-linear power spectrum
            xi = self.__p_nl.to_xi()
            
        elif power_spectrum=="zel":
            # zeldovich power spectrum
            # doesn't seem to be implemented in cosmoprimo
            raise NotImplementedError("Zeldovich power spectrum not implemented")
            
        else:
            raise ValueError("Invalid power spectrum", power_spectrum)

        xi = xi(r, z)
        return xi

    
    def get_wp(self, rp, z, pimax=120, power_spectrum="lin"):
        """
        Returns the projected correlation function wp(rp)
    
        Args:
            r:                array of separation r, in units Mpc/h
            z:                redshift to calculate xi
            [pi_max]:         maximum value of pi in integral, in Mpc/h, default is 120
            [power_spectrum]: can be "lin", "nl" or "zel" (default is "lin")
        Returns:
            array of wp evaluated at rp values
        """
        pi_bins = np.arange(0,pimax+0.01,0.1)
        rp_grid, pi_grid = np.meshgrid(rp, pi_bins)
        r_bins = (rp_grid**2 + pi_grid**2)**0.5
    
        # this is just in real space, and doesn't include the effect of RSD
        xi = self.get_xi(r=r_bins, z=z, power_spectrum=power_spectrum)
        wp = np.sum(xi,axis=0)
    
        return wp
        
