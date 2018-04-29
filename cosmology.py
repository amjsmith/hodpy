import numpy as np
from astropy.cosmology import FlatLambdaCDM

# constants
Msun_g = 1.989e33 # solar mass in g
Mpc_cm = 3.086e24 # Mpc in cm

class Cosmology(object):
    
    def __init__(self, h0, OmegaM, OmegaL):

        # assumes cosmology is flat LCDM
        self.h0     = h0
        self.OmegaM = OmegaM
        self.OmegaL = OmegaL

        self._cosmo = FlatLambdaCDM(H0=h0*100, Om0=OmegaM)


    def critical_density(self, z):
        """
        Critical density at redshift z in Msun Mpc^-3 h^2
        """
        rho_crit = self._cosmo.critical_density(z).value # in g cm^-3

        # convert to Msun Mpc^-3 h^2
        rho_crit *= Mpc_cm**3 / Msun_g / self.h0**2

        return rho_crit


    def mean_density(self, z):
        """
        Mean matter density at redshift z in Msun Mpc^-3 h^2
        """
        # mean density at z=0
        rho_mean0 = self.critical_density(0) * self.OmegaM

        # evolve to redshift z
        return  rho_mean0 * (1+z)**3


    def comoving_distance(self, z):
        """
        Comoving distance to redshift z in Mpc/h
        """
        return self._cosmo.comoving_distance(z).value*self.h0


if __name__ == "__main__":

    cos = Cosmology()
    print(cos.critical_density(0.0))
