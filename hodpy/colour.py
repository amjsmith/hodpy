#! /usr/bin/env python
import numpy as np
from scipy.special import erfc
import h5py
from scipy.interpolate import RegularGridInterpolator, interp1d

from hodpy import lookup



######### Colour class for GAMA colours ############

class Colour(object):
    """
    Class containing methods for randomly assigning galaxies a g-r
    colour from the parametrisation of the GAMA colour magnitude diagram
    in Smith et al. 2017. r-band absolute magnitudes are k-corrected
    to z=0.1 and use h=1. g-r colours are also k-corrected to z=0.1
    """

    def red_mean(self, magnitude, redshift):
        """
        Mean of the red sequence as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """

        colour = 0.932 - 0.032 * (magnitude + 20)
        ind = redshift > 0.1
        colour[ind] -= 0.18 * (np.clip(redshift[ind], 0, 0.4)-0.1)

        return colour


    def red_rms(self, magnitude, redshift):
        """
        RMS of the red sequence as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        colour = 0.07 + 0.01 * (magnitude + 20)
        ind = redshift > 0.1
        colour[ind] += (0.05 + (redshift[ind]-0.1)*0.1) * (redshift[ind]-0.1)
        
        return colour


    def blue_mean(self, magnitude, redshift):
        """
        Mean of the blue sequence as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        colour_bright = 0.62 - 0.11 * (magnitude + 20)
        colour_faint = 0.4 - 0.0286*(magnitude + 16)
        colour = np.log10(1e9**colour_bright + 1e9**colour_faint)/9
        ind = redshift > 0.1
        colour[ind] -= 0.25 * (np.clip(redshift[ind],0,0.4) - 0.1)
                                                          
        return colour


    def blue_rms(self, magnitude, redshift):

        """
        RMS of the blue sequence as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        colour = np.clip(0.12 + 0.02 * (magnitude + 20), 0, 0.15)
        ind = redshift > 0.1
        colour[ind] += 0.2*(redshift[ind]-0.1)

        return colour


    def satellite_mean(self, magnitude, redshift):
        """
        Mean satellite colour as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        
        colour = 0.86 - 0.065 * (magnitude + 20)
        ind = redshift > 0.1
        colour[ind] -= 0.18 * (redshift[ind]-0.1) 

        return colour


    def fraction_blue(self, magnitude, redshift):
        """
        Fraction of blue galaxies as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of fraction of blue galaxies
        """
        frac_blue = 0.2*magnitude + \
            np.clip(4.4 + (1.2 + 0.5*(redshift-0.1))*(redshift-0.1), 4.45, 10)
        frac_blue_skibba = 0.46 + 0.07*(magnitude + 20)

        frac_blue = np.maximum(frac_blue, frac_blue_skibba)

        return np.clip(frac_blue, 0, 1)


    def fraction_central(self, magnitude, redshift):
        """
        Fraction of central galaxies as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of fraction of central galaxies
        """
        # number of satellites divided by number of centrals
        nsat_ncen = 0.35 * (2 - erfc(0.6*(magnitude+20.5)))
        return 1 / (1 + nsat_ncen)


    def probability_red_satellite(self, magnitude, redshift):
        """
        Probability a satellite is red as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of probabilities
        """
        
        sat_mean  = self.satellite_mean(magnitude, redshift)
        blue_mean = self.blue_mean(magnitude, redshift)
        red_mean  = self.red_mean(magnitude, redshift)

        p_red = np.clip(np.absolute(sat_mean-blue_mean) / \
                        np.absolute(red_mean-blue_mean), 0, 1)
        f_blue = self.fraction_blue(magnitude, redshift)
        f_cen = self.fraction_central(magnitude, redshift)

        return np.minimum(p_red, ((1-f_blue)/(1-f_cen)))


    def get_satellite_colour(self, magnitude, redshift):
        """
        Randomly assigns a satellite galaxy a g-r colour

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """

        num_galaxies = len(magnitude)

        # probability the satellite should be drawn from the red sequence
        prob_red = self.probability_red_satellite(magnitude, redshift)

        # random number for each galaxy 0 <= u < 1
        u = np.random.rand(num_galaxies)

        # if u <= p_red, draw from red sequence, else draw from blue sequence
        is_red = u <= prob_red
        is_blue = np.invert(is_red)
    
        mean = np.zeros(num_galaxies, dtype="f")
        mean[is_red]  = self.red_mean(magnitude[is_red],   redshift[is_red])
        mean[is_blue] = self.blue_mean(magnitude[is_blue], redshift[is_blue])

        stdev = np.zeros(num_galaxies, dtype="f")
        stdev[is_red]  = self.red_rms(magnitude[is_red],   redshift[is_red])
        stdev[is_blue] = self.blue_rms(magnitude[is_blue], redshift[is_blue])

        # randomly select colour from Gaussian
        colour = np.random.normal(loc=0.0, scale=1.0, size=num_galaxies)
        colour = colour * stdev + mean

        return colour


    def get_central_colour(self, magnitude, redshift):
        """
        Randomly assigns a central galaxy a g-r colour

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        num_galaxies = len(magnitude)

        # find probability the central should be drawn from the red sequence
        prob_red_sat  = self.probability_red_satellite(magnitude, redshift)
        prob_blue_sat = 1. - prob_red_sat

        frac_cent = self.fraction_central(magnitude, redshift)
        frac_blue = self.fraction_blue(magnitude, redshift)

        prob_blue = frac_blue/frac_cent - prob_blue_sat/frac_cent + \
                                                          prob_blue_sat
        prob_red = 1. - prob_blue

        # random number for each galaxy 0 <= u < 1
        u = np.random.rand(num_galaxies)

        # if u <= p_red, draw from red sequence, else draw from blue sequence
        is_red = u <= prob_red
        is_blue = np.invert(is_red)

        mean = np.zeros(num_galaxies, dtype="f")
        mean[is_red]  = self.red_mean(magnitude[is_red],   redshift[is_red])
        mean[is_blue] = self.blue_mean(magnitude[is_blue], redshift[is_blue])

        stdev = np.zeros(num_galaxies, dtype="f")
        stdev[is_red]  = self.red_rms(magnitude[is_red],   redshift[is_red])
        stdev[is_blue] = self.blue_rms(magnitude[is_blue], redshift[is_blue])

        # randomly select colour from gaussian
        colour = np.random.normal(loc=0.0, scale=1.0, size=num_galaxies)
        colour = colour * stdev + mean

        return colour




######### Colour class for GAMA colours ############
    
class ColourDESI(Colour):
    
    def __init__(self, photsys, hod=None):
        """
        Class containing methods for randomly assigning galaxies a rest-frame g-r 
        colour, using colour distributions fit to the DESI BGS Y1 data. The 
        colours are k-corrected to a reference redshift z=0.1. The HOD needs
        to be provided to calculate the fraction of central/satellite galaxies.

        Args:
            photsys: photometric region, 'N' or 'S'
            hod: object of class HOD
        """
        
        self.hod = hod

        colour_fits_file = lookup.colour_fits_bgs.format(photsys.upper())
        mag_bins, z_bins, mu_blues, sig_blues, mu_reds, sig_reds, f_blues = self.read_fits(colour_fits_file, Nbins=19)
        
        self.__blue_mean_interpolator = self.__get_interpolator(mag_bins, z_bins, mu_blues)
        self.__blue_rms_interpolator = self.__get_interpolator(mag_bins, z_bins, sig_blues)
        self.__red_mean_interpolator = self.__get_interpolator(mag_bins, z_bins, mu_reds)
        self.__red_rms_interpolator = self.__get_interpolator(mag_bins, z_bins, sig_reds)
        self.__fraction_blue_interpolator = self.__get_interpolator(mag_bins, z_bins, f_blues)
        
        #self.__central_fraction_interpolator = self.__initialize_central_fraction_interpolator(z=0.2)
            
            
    def __initialize_central_fraction_interpolator(self, z=0.2):

        # Fraction of central galaxies is calculated using HODs

        # TO DO: update this to create 2D array of fcen at different magnitudes
        # and redshifts
        
        magnitudes = np.arange(-23,-10,0.1)
        fcen = np.zeros(len(magnitudes))

        for i in range(len(magnitudes)):
            magnitude = np.array([magnitudes[i],])
            redshift = np.array([z,])
            try:
                logMmin = np.log10(self.hod.Mmin(magnitude))
                logM1 = np.log10(self.hod.M1(magnitude))
                logM0 = np.log10(self.hod.M0(magnitude))
                sigmalogM = self.hod.sigma_logM(magnitude)
                alpha = self.hod.alpha(magnitude)
            except:
                logMmin = np.log10(self.hod.Mmin(magnitude, redshift))
                logM1 = np.log10(self.hod.M1(magnitude, redshift))
                logM0 = np.log10(self.hod.M0(magnitude, redshift))
                sigmalogM = self.hod.sigma_logM(magnitude, redshift)
                alpha = self.hod.alpha(magnitude, redshift)

            try:
                n_all = self.hod.get_n_HOD(magnitude, redshift, logMmin, logM1, logM0, sigmalogM, alpha,
                                Mmin=10, Mmax=16, galaxies="all")

                n_cen = self.hod.get_n_HOD(magnitude, redshift, logMmin, logM1, logM0, sigmalogM, alpha,
                                Mmin=10, Mmax=16, galaxies="cen")
            
            except:
                n_all = self.hod.get_n_HOD2(magnitude, redshift, logMmin, logM1, logM0, sigmalogM, alpha,
                            Mmin=10, Mmax=16, galaxies="all")

                n_cen = self.hod.get_n_HOD2(magnitude, redshift, logMmin, logM1, logM0, sigmalogM, alpha,
                            Mmin=10, Mmax=16, galaxies="cen")

            fcen[i] = n_cen/n_all
            
        magnitudes2 = np.arange(-28,10,0.1)
        fcen2 = np.zeros(len(magnitudes2))
        fcen2[50:180] = fcen
        fcen2[:50] = fcen[0]
        fcen2[180:] = fcen[-1]
    
        return interp1d(magnitudes2, fcen2, kind='cubic')

        
        
    def read_fits(self, colour_fits, Nbins=19):
        
        f = h5py.File(colour_fits,'r')
        mag_bins = f['mag_binc'][...]
        z_bins = np.zeros(Nbins)
        for i in range(len(z_bins)):
            z_bins[i] = f['%i/zmed'%i][...][0]

        mu_blues  = np.zeros((len(z_bins), len(mag_bins)))
        sig_blues = np.zeros((len(z_bins), len(mag_bins)))
        mu_reds   = np.zeros((len(z_bins), len(mag_bins)))
        sig_reds   = np.zeros((len(z_bins), len(mag_bins)))
        f_blues   = np.zeros((len(z_bins), len(mag_bins)))

        for i in range(len(z_bins)):
            mu_blues[i,:]  = f['%i/mu_blue'%i][...]
            sig_blues[i,:] = f['%i/sig_blue'%i][...]
            mu_reds[i,:]   = f['%i/mu_red'%i][...]
            sig_reds[i,:]  = f['%i/sig_red'%i][...]
            f_blues[i,:]   = f['%i/f_blue'%i][...]

        f.close()
        
        return mag_bins, z_bins, mu_blues, sig_blues, mu_reds, sig_reds, f_blues
            
    
    def __get_interpolator(self, mag_bins, z_bins, array):
        
        func = RegularGridInterpolator((mag_bins, z_bins), array.transpose(),
                                       method='linear', bounds_error=False, fill_value=None)
        return func
        
        
    def blue_mean(self, magnitude, redshift):
        return self.__blue_mean_interpolator((magnitude, redshift))
    
    def blue_rms(self, magnitude, redshift):
        return np.clip(self.__blue_rms_interpolator((magnitude, redshift)), 0.02, 10)
    
    def red_mean(self, magnitude, redshift):
        return self.__red_mean_interpolator((magnitude, redshift))
    
    def red_rms(self, magnitude, redshift):
        return np.clip(self.__red_rms_interpolator((magnitude, redshift)), 0.02, 10)
        
    def fraction_blue(self, magnitude, redshift):
        frac_blue = np.clip(self.__fraction_blue_interpolator((magnitude, redshift)), 0, 1)
        
        # if at bright end blue_mean > red_mean, set all galaxies as being red
        b_m = self.blue_mean(magnitude, redshift)
        r_m = self.red_mean(magnitude, redshift)
        idx = np.logical_and(b_m > r_m, magnitude<-20)
        frac_blue[idx] = 0
        
        return frac_blue
    
    
    def fraction_central(self, magnitude, redshift):
        """
        Fraction of central galaxies as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of fraction of central galaxies
        """

        # To do: calculate this properly from the HODs
        
        print("WARNING: fraction_central set to 0.5 for testing")
        
        ###return self.__central_fraction_interpolator(magnitude)

        return np.ones(len(magnitude))*0.5
        
        
    def satellite_mean(self, magnitude, redshift):
        """
        Mean satellite colour as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
    
        blue_mean = self.blue_mean(magnitude, redshift)
        red_mean = self.red_mean(magnitude, redshift)
        
        colour = (0.8*red_mean + 0.2*blue_mean)
        
        return colour 
    
    
    def probability_red_satellite(self, magnitude, redshift):
        """
        Probability a satellite is red as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of probabilities
        """
        
        sat_mean  = self.satellite_mean(magnitude, redshift)
        blue_mean = self.blue_mean(magnitude, redshift)
        red_mean  = self.red_mean(magnitude, redshift)
    
        p_red = np.clip((sat_mean-blue_mean) / \
                        (red_mean-blue_mean), 0, 1)
        f_blue = self.fraction_blue(magnitude, redshift)
        
        idx = f_blue==0
        p_red[idx]=1
        idx = f_blue==1
        p_red[idx]=0
        
        f_cen = self.fraction_central(magnitude, redshift)

        return np.minimum(p_red, ((1-f_blue)/(1-f_cen)))
    
    
    def probability_red_central(self, magnitude, redshift):
        """
        Probability a central is red as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of probabilities
        """
        prob_red_sat  = self.probability_red_satellite(magnitude, redshift)
        prob_blue_sat = 1. - prob_red_sat

        frac_cent = self.fraction_central(magnitude, redshift)
        frac_blue = self.fraction_blue(magnitude, redshift)

        prob_blue = frac_blue/frac_cent - prob_blue_sat/frac_cent + prob_blue_sat
        
        return 1. - prob_blue
    
    

