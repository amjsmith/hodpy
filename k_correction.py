#! /usr/bin/env python
from __future__ import print_function
import numpy as np
import parameters as par
from scipy.interpolate import RegularGridInterpolator
from cosmology import Cosmology


class KCorrection(object):

    def apparent_magnitude(self, absolute_magnitude, redshift):
        pass

    def absoulte_magnitude(self, apparent_magnitude, redshift):
        pass

    def magnitude_faint(self, redshift):
        pass



class GAMA_KCorrection(KCorrection):
    """
    Colour-dependent polynomial fit to the GAMA K-correction 
    (Fig. 13 of Smith+17), used to convert between SDSS r-band
    Petrosian apparent magnitudes, and rest frame absolute manigutues 
    at z_ref = 0.1
    """
    def __init__(self):
        
        # read file of parameters of polynomial fit to k-correction
        cmin, cmax, A, B, C, D, E, cmed = \
            np.loadtxt(par.k_corr_file, unpack=True)
    
        self.z0 = 0.1 # reference redshift

        # Polynomial fit parameters
        self.__A_interpolator = self.__initialize_parameter_interpolator(A,cmed)
        self.__B_interpolator = self.__initialize_parameter_interpolator(B,cmed)
        self.__C_interpolator = self.__initialize_parameter_interpolator(C,cmed)
        self.__D_interpolator = self.__initialize_parameter_interpolator(D,cmed)
        self.__E = E[0]

        self.colour_min = np.min(cmed)
        self.colour_max = np.max(cmed)
        self.colour_med = cmed

        self.cosmo = Cosmology(par.h0, par.OmegaM, par.OmegaL)

        # Linear extrapolation
        self.__X_interpolator = lambda x: None
        self.__Y_interpolator = lambda x: None
        self.__X_interpolator, self.__Y_interpolator = \
                                 self.__initialize_line_interpolators() 


    def __initialize_parameter_interpolator(self, parameter, median_colour):
        # interpolated polynomial coefficient as a function of colour
        return RegularGridInterpolator((median_colour,), parameter, 
                                       bounds_error=False, fill_value=None)

    def __initialize_line_interpolators(self):
        # linear coefficients for z>0.5
        X = np.zeros(7)
        Y = np.zeros(7)
        # find X, Y at each colour
        redshift = np.array([0.4,0.5])
        arr_ones = np.ones(len(redshift))
        for i in range(7):
            k = self.k(redshift, arr_ones*self.colour_med[i])
            X[i] = (k[1]-k[0]) / (redshift[1]-redshift[0])
            Y[i] = k[0] - X[i]*redshift[0]
            
        X_interpolator = RegularGridInterpolator((self.colour_med,), X, 
                                       bounds_error=False, fill_value=None)
        Y_interpolator = RegularGridInterpolator((self.colour_med,), Y, 
                                       bounds_error=False, fill_value=None)
        return X_interpolator, Y_interpolator

    def __A(self, colour):
        # coefficient of the z**4 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__A_interpolator(colour_clipped)

    def __B(self, colour):
        # coefficient of the z**3 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__B_interpolator(colour_clipped)

    def __C(self, colour):
        # coefficient of the z**2 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__C_interpolator(colour_clipped)

    def __D(self, colour):
        # coefficient of the z**1 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__D_interpolator(colour_clipped)

    def __X(self, colour):
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__X_interpolator(colour_clipped)

    def __Y(self, colour):
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__Y_interpolator(colour_clipped)


    def k(self, redshift, colour):
        """
        Polynomial fit to the GAMA K-correction for z<0.5
        The K-correction is extrapolated linearly for z>0.5

        Args:
            redshift: array of redshifts
            colour:   array of ^0.1(g-r) colour
        Returns:
            array of K-corrections
        """
        K = np.zeros(len(redshift))
        idx = redshift <= 0.5

        K[idx] = self.__A(colour[idx])*(redshift[idx]-self.z0)**4 + \
                 self.__B(colour[idx])*(redshift[idx]-self.z0)**3 + \
                 self.__C(colour[idx])*(redshift[idx]-self.z0)**2 + \
                 self.__D(colour[idx])*(redshift[idx]-self.z0) + self.__E

        idx = redshift > 0.5
        
        K[idx] = self.__X(colour[idx])*redshift[idx] + self.__Y(colour[idx])
        
        return K
        

    def apparent_magnitude(self, absolute_magnitude, redshift, colour):
        """
        Convert absolute magnitude to apparent magnitude

        Args:
            absolute_magnitude: array of absolute magnitudes (with h=1)
            redshift:           array of redshifts
            colour:             array of ^0.1(g-r) colour
        Returns:
            array of apparent magnitudes
        """
        # Luminosity distance
        D_L = (1.+redshift) * self.cosmo.comoving_distance(redshift) 

        return absolute_magnitude + 5*np.log10(D_L) + 25 + \
                                              self.k(redshift,colour)

    def absolute_magnitude(self, apparent_magnitude, redshift, colour):
        """
        Convert apparent magnitude to absolute magnitude

        Args:
            apparent_magnitude: array of apparent magnitudes
            redshift:           array of redshifts
            colour:             array of ^0.1(g-r) colour
        Returns:
            array of absolute magnitudes (with h=1)
        """
        # Luminosity distance
        D_L = (1.+redshift) * self.cosmo.comoving_distance(redshift) 

        return apparent_magnitude - 5*np.log10(D_L) - 25 - \
                                              self.k(redshift,colour)

    def magnitude_faint(self, redshift):
        """
        Convert faintest apparent magnitude (set in parameters.py)
        to faintest absolute magnitude

        Args:
            redshift: array of redshifts
        Returns:
            array of absolute magnitudes (with h=1)
        """
        # convert faint apparent magnitude to absolute magnitude
        # for bluest and reddest galaxies
        arr_ones = np.ones(len(redshift))
        abs_mag_blue = self.absolute_magnitude(arr_ones*par.mag_faint,
                                               redshift, arr_ones*0)
        abs_mag_red = self.absolute_magnitude(arr_ones*par.mag_faint,
                                               redshift, arr_ones*2)
        
        # find faintest absolute magnitude, add small amount to be safe
        mag_faint = np.maximum(abs_mag_red, abs_mag_blue) + 0.01

        # avoid infinity
        mag_faint = np.minimum(mag_faint, -10)
        return mag_faint


def test():
    import matplotlib.pyplot as plt
    kcorr = GAMA_KCorrection()

    redshifts = np.arange(0, 1, 0.01)
    arr_ones = np.ones(len(redshifts))
    for i in range(7):
        k = kcorr.k(redshifts, arr_ones*kcorr.colour_med[i])
        plt.plot(redshifts, k)
    plt.show()

    app_mag = np.ones(len(redshifts)) * 20.0
    colour = np.ones(len(app_mag))


    abs_mag = kcorr.absolute_magnitude(app_mag, redshifts, colour*2)
    plt.plot(redshifts, abs_mag, c="r", label="red")
    abs_mag = kcorr.absolute_magnitude(app_mag, redshifts, colour*0)
    plt.plot(redshifts, abs_mag, c="c", label="blue")

    mag_faint = kcorr.magnitude_faint(redshifts)
    plt.plot(redshifts, mag_faint, c="k", ls=":", label="mag faint")

    print(mag_faint)

    plt.legend(loc="upper right")

    plt.xlim(0,1)
    plt.ylim(-12, -24)

    plt.xlabel("redshift")
    plt.ylabel("Absolute magnitude")
    plt.title("r=20.0")

    plt.show()


if __name__ == "__main__":
    test()
