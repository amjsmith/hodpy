from __future__ import print_function
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from sigma import Sigma
import parameters as par
from cosmology import Cosmology

class MassFunctionMXXL(object):
    
    def __init__(self):
        
        self.sigma = Sigma() #sigma(M)
        self.cosmology = Cosmology(par.h0, par.OmegaM, par.OmegaL)
        
        # read in MXXL mass function fit parameters
        snap, redshift, A, a, p = \
                   np.loadtxt(par.mf_fits_file, skiprows=1, unpack=True)

        # interpolate parameters
        self._A = RegularGridInterpolator((redshift,), A, bounds_error=False, 
                                          fill_value=None)

        self._a = RegularGridInterpolator((redshift,), a, bounds_error=False, 
                                          fill_value=None)

        self._p = RegularGridInterpolator((redshift,), p, bounds_error=False, 
                                          fill_value=None)

    def A(self, redshift):
        return self._A(redshift)

    def a(self, redshift):
        return self._a(redshift)

    def p(self, redshift):
        return self._p(redshift)

    def mass_function(self, log_mass, redshift):
        '''
        Fit to the MXXL mass function with the form of a Sheth-Tormen
        mass function
        '''
        sigma = self.sigma.sigma(log_mass, redshift)

        dc=1
        A = self.A(redshift)
        a = self.a(redshift)
        p = self.p(redshift)
        
        mf = A * np.sqrt(2*a/np.pi)
        mf *= 1 + (sigma**2 / (a * dc**2))**p
        mf *= dc / sigma
        mf *= np.exp(-a * dc**2 / (2*sigma**2))

        return mf

    def number_density(self, log_mass, redshift):
        '''
        Number density of haloes
        '''        
        mf = self.mass_function(log_mass, redshift)

        return mf * self.cosmology.mean_density(0) / 10**log_mass
        
