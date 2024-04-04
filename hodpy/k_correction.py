#! /usr/bin/env python
import numpy as np
import h5py
from scipy import stats
from scipy.interpolate import RegularGridInterpolator, splrep, splev, interp1d

from hodpy import lookup


class KCorrection(object):
    """
    K-correction base class
    """

    def apparent_magnitude(self, absolute_magnitude, redshift):
        pass

    def absolute_magnitude(self, apparent_magnitude, redshift):
        pass

    def magnitude_faint(self, redshift):
        pass


############## BGS K-Corrections ################


class DESI_KCorrection(object):
    def __init__(self, band, photsys, cosmology=None, z0=0.1, kind="cubic"):
        """
        Colour-dependent polynomial fit to the FSF DESI K-corrections, 
        used to convert between SDSS r-band Petrosian apparent magnitudes, and rest 
        frame absolute magnitudes
        
        Args:
            band: band pass, e.g. 'r'
            photsys: photometric region, 'N' or 'S'
            [cosmology]: object of class Cosmology, needed for converting between apparent
                         and absolute magnitudes. Default is None.
            [z0]: reference redshift. Default value is z0=0.1
            [kind]: type of interpolation between colour bins,
                  e.g. "linear", "cubic". Default is "cubic"
        """
    
        k_corr_file = lookup.kcorr_file_bgs.format(photsys.upper(), band.lower())
                    
        # read file of parameters of polynomial fit to k-correction
        # polynomial k-correction is of the form
        # A*(z-z0)^6 + B*(z-z0)^5 + C*(z-z0)^4 + D*(z-z0)^3 + ... + G
        col_min, col_max, A, B, C, D, E, F, G, col_med = \
            np.loadtxt(k_corr_file, unpack=True)

        self.cosmo = cosmology
        
        self.z0 = z0             # reference redshift

        self.nbins = len(col_min) # number of colour bins in file
        self.colour_min = np.min(col_med)
        self.colour_max = np.max(col_med)
        self.colour_med = col_med

        # functions for interpolating polynomial coefficients in rest-frame color.
        self.__A_interpolator = self.__initialize_parameter_interpolator(A, col_med, kind=kind)
        self.__B_interpolator = self.__initialize_parameter_interpolator(B, col_med, kind=kind)
        self.__C_interpolator = self.__initialize_parameter_interpolator(C, col_med, kind=kind)
        self.__D_interpolator = self.__initialize_parameter_interpolator(D, col_med, kind=kind)
        self.__E_interpolator = self.__initialize_parameter_interpolator(E, col_med, kind=kind)
        self.__F_interpolator = self.__initialize_parameter_interpolator(F, col_med, kind=kind)
        self.__G_interpolator = self.__initialize_parameter_interpolator(G, col_med, kind=kind)

        # Linear extrapolation for z > 0.5
        self.__X_interpolator = lambda x: None
        self.__Y_interpolator = lambda x: None
        self.__X_interpolator, self.__Y_interpolator = self.__initialize_line_interpolators() 
   
    def __initialize_parameter_interpolator(self, parameter, median_colour, kind="linear"):
        # returns function for interpolating polynomial coefficients, as a function of colour
        return interp1d(median_colour, parameter, kind=kind, fill_value="extrapolate")
    
    def __initialize_line_interpolators(self):
        # linear coefficients for z>0.5
        X = np.zeros(self.nbins)
        Y = np.zeros(self.nbins)
        
        # find X, Y at each colour
        redshift = np.array([0.58,0.6])
        arr_ones = np.ones(len(redshift))
        for i in range(self.nbins):
            k = self.k(redshift, arr_ones*self.colour_med[i])
            X[i] = (k[1]-k[0]) / (redshift[1]-redshift[0])
            Y[i] = k[0] - X[i]*redshift[0]
        
        X_interpolator = interp1d(self.colour_med, X, kind='linear', fill_value="extrapolate")
        Y_interpolator = interp1d(self.colour_med, Y, kind='linear', fill_value="extrapolate")
        
        return X_interpolator, Y_interpolator

    def __A(self, colour):
        # coefficient of the z**6 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__A_interpolator(colour_clipped)

    def __B(self, colour):
        # coefficient of the z**5 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__B_interpolator(colour_clipped)

    def __C(self, colour):
        # coefficient of the z**4 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__C_interpolator(colour_clipped)

    def __D(self, colour):
        # coefficient of the z**3 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__D_interpolator(colour_clipped)
    
    def __E(self, colour):
        # coefficient of the z**2 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__E_interpolator(colour_clipped)
    
    def __F(self, colour):
        # coefficient of the z**1 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__F_interpolator(colour_clipped)
    
    def __G(self, colour):
        # coefficient of the z**0 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__G_interpolator(colour_clipped)

    def __X(self, colour):
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__X_interpolator(colour_clipped)

    def __Y(self, colour):
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__Y_interpolator(colour_clipped)

        
    def k(self, redshift, restframe_colour):
        """
        Polynomial fit to the DESI
        K-correction for z<0.6
        The K-correction is extrapolated linearly for z>0.6

        Args:
            redshift: array of redshifts
            restframe_colour:   array of ^0.1(g-r) colour
        Returns:
            array of K-corrections
        """
        K   = np.zeros(len(redshift))
        idx = redshift <= 0.6

        K[idx] = self.__A(restframe_colour[idx])*(redshift[idx]-self.z0)**6 + \
                 self.__B(restframe_colour[idx])*(redshift[idx]-self.z0)**5 + \
                 self.__C(restframe_colour[idx])*(redshift[idx]-self.z0)**4 + \
                 self.__D(restframe_colour[idx])*(redshift[idx]-self.z0)**3 + \
                 self.__E(restframe_colour[idx])*(redshift[idx]-self.z0)**2 + \
                 self.__F(restframe_colour[idx])*(redshift[idx]-self.z0)**1 + \
                 self.__G(restframe_colour[idx])

        idx = redshift > 0.6
        
        K[idx] = self.__X(restframe_colour[idx])*redshift[idx] + self.__Y(restframe_colour[idx])
        
        return  K    
    
    def k_nonnative_zref(self, refz, redshift, restframe_colour):
        """
        Returns the k-correction at any specified reference redshift

        Args:
            refz: reference redshift
            redshift: array of redshifts
            restframe_colour: array of ^0.1(g-r) colour
        Returns:
            array of K-corrections
        """
        refzs = refz * np.ones_like(redshift)
        
        return  self.k(redshift, restframe_colour) - self.k(refzs, restframe_colour) - 2.5 * np.log10(1. + refz)

    
    def apparent_magnitude(self, absolute_magnitude, redshift, colour, use_ecorr=True, Q=lookup.Q, zq=lookup.zq):
        """
        Convert absolute magnitude to apparent magnitude

        Args:
            absolute_magnitude: array of absolute magnitudes (with h=1)
            redshift:           array of redshifts
            colour:             array of ^0.1(g-r) colour
            [use_ecorr]:        if True, use an E-correction of the form Q*(redshift - zq). Default is True
            [Q]:                Q parameter if use_ecorr=True. Default is read from lookup.py
            [zq]:               zq parameter if use_ecorr=True. Default is read from lookup.py
        Returns:
            array of apparent magnitudes
        """
        # Luminosity distance
        D_L = (1.+redshift) * self.cosmo.comoving_distance(redshift)

        if use_ecorr:
            E = Q * (redshift - zq)
        else:
            E = 0
            
        return absolute_magnitude + 5*np.log10(D_L) + 25 + self.k(redshift,colour) - E

    
    def absolute_magnitude(self, apparent_magnitude, redshift, colour, use_ecorr=True, Q=lookup.Q, zq=lookup.zq):
        """
        Convert apparent magnitude to absolute magnitude

        Args:
            apparent_magnitude: array of apparent magnitudes
            redshift:           array of redshifts
            colour:             array of ^0.1(g-r) colour
            [use_ecorr]:        if True, use an E-correction of the form Q*(redshift - zq). Default is True
            [Q]:                Q parameter if use_ecorr=True. Default is read from lookup.py
            [zq]:               zq parameter if use_ecorr=True. Default is read from lookup.py
        Returns:
            array of absolute magnitudes (with h=1)
        """
        # Luminosity distance
        D_L = (1.+redshift) * self.cosmo.comoving_distance(redshift) 

        if use_ecorr:
            E = Q * (redshift - zq)
        else:
            E = 0
            
        return apparent_magnitude - 5*np.log10(D_L) - 25 - self.k(redshift,colour) + E

    
    def magnitude_faint(self, redshift, mag_faint, use_ecorr=True, Q=lookup.Q, zq=lookup.zq):
        """
        Convert faintest apparent magnitude to the faintest absolute magnitude

        Args:
            redshift: array of redshifts
            mag_faint: faintest apparent magnitude
            [use_ecorr]:        if True, use an E-correction of the form Q*(redshift - zq). Default is True
            [Q]:                Q parameter if use_ecorr=True. Default is read from lookup.py
            [zq]:               zq parameter if use_ecorr=True. Default is read from lookup.py
        Returns:
            array of absolute magnitudes (with h=1)
        """
        # convert faint apparent magnitude to absolute magnitude
        # for bluest and reddest galaxies
        arr_ones = np.ones(len(redshift))
        abs_mag_blue = self.absolute_magnitude(arr_ones*mag_faint, redshift, arr_ones*-1,
                                               use_ecorr=use_ecorr, Q=Q, zq=zq)
        abs_mag_red = self.absolute_magnitude(arr_ones*mag_faint, redshift, arr_ones*2,
                                              use_ecorr=use_ecorr, Q=Q, zq=zq)
        
        # find faintest absolute magnitude, add small amount to be safe
        mag_faint = np.maximum(abs_mag_red, abs_mag_blue) + 0.01

        # avoid infinity
        mag_faint = np.minimum(mag_faint, -10)
        
        return mag_faint


    
class DESI_KCorrection_color(object):
    def __init__(self, photsys):
        """
        Apply k-corrections for converting between the observed DESI g-r colours
        and the rest-frame colours in the SDSS bands at z=0.1.
        
        Args:
            photsys: photometric region, 'N' or 'S'
        """
        
        # lookup table for getting rest-frame colours
        kcorr_file_rest = lookup.kcorr_gmr_bgs.format(photsys.upper(), 'rest')
        self.__xedges_rest, self.__yedges_rest, self.__H_rest = self.__read_table(kcorr_file_rest)
        
        # lookup table for getting observer-frame colours
        kcorr_file_obs  = lookup.kcorr_gmr_bgs.format(photsys.upper(), 'obs')
        self.__xedges_obs, self.__yedges_obs, self.__H_obs = self.__read_table(kcorr_file_obs)
        
        # grids for interpolation
        self.__grid_rest = self.__get_grid(self.__xedges_rest, self.__yedges_rest, self.__H_rest)
        self.__grid_obs  = self.__get_grid(self.__xedges_obs,  self.__yedges_obs,  self.__H_obs)
        
        
    def __read_table(self, filename):
        # read the colour table file
        f = h5py.File(filename,'r')
        H = f['H'][...]
        xedges = f['z_bins'][...]
        yedges = f['gr_bins'][...]
        f.close()
        return xedges, yedges, H
    
    def __get_grid(self, xbins, ybins, H):
        # create the grid for interpolation
        xbin_cen = (xbins[1:] + xbins[:-1])/2.
        ybin_cen = (ybins[1:] + ybins[:-1])/2.
        return RegularGridInterpolator((xbin_cen, ybin_cen), np.asarray(H))

    
    def rest_frame_colour(self, redshift, colour_obs, interp=True):
        """
        Use a lookup table to get the rest-frame g-r colour (SDSS band at z0=0.1) of a 
        galaxy of a given redshift and observed DESI g-r colour
        
        Args:
            redshift: array of galaxy redshifts
            colour_obs: array of observer-frame colours
            [interp]: use interpolation. Default is True
        Returns:
            array of rest-frame colour
        """
        z = np.clip(redshift, 0.02, 0.59)
        col_obs = np.clip(colour_obs, -0.99, 5)
        
        if interp:
            return self.__grid_rest((z, col_obs))
        
        else:
            __, __, __, binnumber2 = stats.binned_statistic_2d(z, col_obs, values=z, 
                                statistic='median', bins = [self.__xedges_rest, self.__yedges_rest], 
                                expand_binnumbers=True)
        
            Harray=np.asarray(self.__H_rest)
            
            return Harray[binnumber2[0]-1,binnumber2[1]-1]

        
        
    def observer_frame_colour(self, redshift, colour_rest, interp=True):
        """
        Use a lookup table to get the DESI observer-frame g-r colour of a 
        galaxy of a given redshift and rest-frame g-r colour (SDSS band at z0=0.1) 
        
        Args:
            redshift: array of galaxy redshifts
            colour_rest: array of rest-frame colours
            [interp]: use interpolation. Default is True
        Returns:
            array of observer-frame colour
        """
        z = np.clip(redshift, 0.02, 0.59)
        col_rest = np.clip(colour_rest, -0.99, 5)

        if interp:
            return self.__grid_obs((z, col_rest))
            
        else:
            __, __, __, binnumber2 = stats.binned_statistic_2d(z, col_rest, values=z,
                                statistic='median', bins = [self.__xedges_obs, self.__yedges_obs], 
                                expand_binnumbers=True)

            Harray=np.asarray(self.__H_obs)
            
            return Harray[binnumber2[0]-1,binnumber2[1]-1]
        


############## GAMA K-Corrections ################

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
                                               redshift, arr_ones*-1)
        abs_mag_red = self.absolute_magnitude(arr_ones*mag_faint,
                                               redshift, arr_ones*2)
        
        # find faintest absolute magnitude, add small amount to be safe
        mag_faint = np.maximum(abs_mag_red, abs_mag_blue) + 0.01

        # avoid infinity
        mag_faint = np.minimum(mag_faint, -10)
        return mag_faint
