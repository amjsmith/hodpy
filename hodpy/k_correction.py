#! /usr/bin/env python
import numpy as np
from scipy.interpolate import RegularGridInterpolator, splrep, splev

from hodpy import lookup


class KCorrection(object):
    """
    K-correction base class
    """

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
    
    Args:
        k_corr_file: file of polynomial coefficients for each colour bin
        cosmology: object of type hodpy.Cosmology
        [z0]: reference redshift. Default value is z0=0.1
        [cubic_interpolation]: if set to True, will use cubic spline interpolation.
                               Default value is False (linear interpolation).
    """
    def __init__(self, cosmology, k_corr_file=lookup.kcorr_file, z0=0.1, cubic_interpolation=False):
        
        # read file of parameters of polynomial fit to k-correction
        cmin, cmax, A, B, C, D, E, cmed = \
            np.loadtxt(k_corr_file, unpack=True)
    
        self.z0 = 0.1 # reference redshift
        self.cubic = cubic_interpolation

        # Polynomial fit parameters
        if cubic_interpolation: 
            # cubic spline interpolation
            self.__A_interpolator = self.__initialize_parameter_interpolator_spline(A,cmed)
            self.__B_interpolator = self.__initialize_parameter_interpolator_spline(B,cmed)
            self.__C_interpolator = self.__initialize_parameter_interpolator_spline(C,cmed)
            self.__D_interpolator = self.__initialize_parameter_interpolator_spline(D,cmed)
        else:
            # linear interpolation
            self.__A_interpolator = self.__initialize_parameter_interpolator(A,cmed)
            self.__B_interpolator = self.__initialize_parameter_interpolator(B,cmed)
            self.__C_interpolator = self.__initialize_parameter_interpolator(C,cmed)
            self.__D_interpolator = self.__initialize_parameter_interpolator(D,cmed)
        self.__E = E[0]

        self.colour_min = np.min(cmed)
        self.colour_max = np.max(cmed)
        self.colour_med = cmed

        self.cosmo = cosmology

        # Linear extrapolation
        self.__X_interpolator = lambda x: None
        self.__Y_interpolator = lambda x: None
        self.__X_interpolator, self.__Y_interpolator = \
                                 self.__initialize_line_interpolators() 


    def __initialize_parameter_interpolator(self, parameter, median_colour):
        # interpolated polynomial coefficient as a function of colour
        return RegularGridInterpolator((median_colour,), parameter, 
                                       bounds_error=False, fill_value=None)

    
    def __initialize_parameter_interpolator_spline(self, parameter, median_colour):
        # interpolated polynomial coefficient as a function of colour
        tck = splrep(median_colour, parameter)
        return tck
    
    
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
    
    def __A_spline(self, colour):
        # coefficient of the z**4 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return splev(colour_clipped, self.__A_interpolator)

    def __B_spline(self, colour):
        # coefficient of the z**3 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return splev(colour_clipped, self.__B_interpolator)

    def __C_spline(self, colour):
        # coefficient of the z**2 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return splev(colour_clipped, self.__C_interpolator)

    def __D_spline(self, colour):
        # coefficient of the z**1 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return splev(colour_clipped, self.__D_interpolator)

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

        if self.cubic:
            K[idx] = self.__A_spline(colour[idx])*(redshift[idx]-self.z0)**4 + \
                     self.__B_spline(colour[idx])*(redshift[idx]-self.z0)**3 + \
                     self.__C_spline(colour[idx])*(redshift[idx]-self.z0)**2 + \
                     self.__D_spline(colour[idx])*(redshift[idx]-self.z0) + self.__E
        else:
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

    def magnitude_faint(self, redshift, mag_faint):
        """
        Convert faintest apparent magnitude to the faintest absolute magnitude

        Args:
            redshift: array of redshifts
            mag_faint: faintest apparent magnitude
        Returns:
            array of absolute magnitudes (with h=1)
        """
        # convert faint apparent magnitude to absolute magnitude
        # for bluest and reddest galaxies
        arr_ones = np.ones(len(redshift))
        abs_mag_blue = self.absolute_magnitude(arr_ones*mag_faint,
                                               redshift, arr_ones*0)
        abs_mag_red = self.absolute_magnitude(arr_ones*mag_faint,
                                               redshift, arr_ones*2)
        
        # find faintest absolute magnitude, add small amount to be safe
        mag_faint = np.maximum(abs_mag_red, abs_mag_blue) + 0.01

        # avoid infinity
        mag_faint = np.minimum(mag_faint, -10)
        return mag_faint
