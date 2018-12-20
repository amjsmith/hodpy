from __future__ import print_function
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import parameters as par

class Sigma:

    def __init__(self):

        log_mass, sigma, alpha = np.loadtxt(par.sigma_file, unpack=True)
        redshift, delta_crit = \
            np.loadtxt(par.deltacrit_file, skiprows=1, unpack=True)

        self._sigma_interpolator = \
            RegularGridInterpolator((log_mass,), sigma, bounds_error=False, 
                                    fill_value=None)

        self._deltacrit_interpolator = \
           RegularGridInterpolator((redshift,), delta_crit, bounds_error=False, 
                                   fill_value=None)


    def sigma(self, log_mass, redshift):

        # sigma(M, z=0)
        sigma = self._sigma_interpolator(log_mass)

        # sigma(M, z)
        return sigma * \
               self.deltacrit(redshift) / self.deltacrit(np.array([0,]))[0]

    def deltacrit(self, redshift):
        
        return self._deltacrit_interpolator(redshift)

if __name__ == "__main__":

    s = Sigma()
    print(s.sigma(np.array([14,]), np.array([1,])))
