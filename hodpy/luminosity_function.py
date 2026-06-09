#! /usr/bin/env python
from __future__ import print_function
import numpy as np
from scipy.special import gamma, gammaincc
from scipy.interpolate import RegularGridInterpolator, splrep, splev
from scipy.optimize import curve_fit


def mag2lum(magnitude):
    """
    Convert absolute magnitude to luminosity
    Args:
        magnitude: array of absolute magnitudes [M-5logh]
    Returns:
        array of luminosity [Lsun/h^2]
    """
    return 10**((4.76 - magnitude)/2.5)


def lum2mag(luminosity):
    """
    Convert luminosity to absolute magnitude
    Args:
        luminosity: array of luminsities [Lsun/h^2]
    Returns:
        array of absolute magnitude [M-5logh]
    """
    return 4.76 - 2.5*np.log10(luminosity)
    
    
class LuminosityFunction(object):
    """
    Luminsity function base class
    """
    def __init__(self):
        pass

    def __initialize_interpolator(self):
        #Initializes a RegularGridInterpolator for converting number densities
        #at a certain to redshift to the corresponding magnitude threshold
        
        # arrays of z and log_n, and empty 2d array magnitudes
        redshifts = np.arange(0, 1, 0.01)
        log_number_densities = np.arange(-12, -0.5, 0.01)
        magnitudes = np.zeros((len(redshifts),len(log_number_densities)))
            
        # fill in 2d array of magnitudes
        mags = np.arange(-25, 0, 0.001)
        for i in range(len(redshifts)):
            # find number density at each magnitude in mags
            log_ns = np.log10(self.Phi_cumulative(mags, redshifts[i]))
            
            # find this number density in the array log_number_densities
            idx = np.searchsorted(log_ns, log_number_densities)
            
            # interpolate to find magnitude at this number density
            f = (log_number_densities - log_ns[idx-1]) / \
                                     (log_ns[idx] - log_ns[idx-1])
            magnitudes[i,:] = mags[idx-1] + f*(mags[idx] - mags[idx-1])

        # create RegularGridInterpolator object
        return RegularGridInterpolator((redshifts,log_number_densities),
                        magnitudes, bounds_error=False, fill_value=None)

    def Phi(self, magnitude, redshift):
        """
        Luminosity function as a function of absoulte magnitude and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """
        magnitude01 = magnitude + self.Q * (redshift - 0.1)

        # find interpolated number density at z=0.1
        log_lf01 = self.__Phi_z01(magnitude01)

        # shift back to redshift
        log_lf = log_lf01 + 0.4 * self.P * (redshift - 0.1)
        
        return 10**log_lf

    def __Phi_z01(self, magnitude):
        # returns a spline fit to the LF at z=0.1 (using the cumulative LF)
        mags = np.arange(0, -25, -0.001)
        phi_cums = self.Phi_cumulative(mags, 0.1)
        phi = (phi_cums[:-1] - phi_cums[1:]) / 0.001
        tck = splrep((mags[1:]+0.0005)[::-1], np.log10(phi[::-1]))
        return splev(magnitude, tck)
        
    def Phi_cumulative(self, magnitude, redshift):
        raise NotImplementedError

    def mag2lum(self, magnitude):
        """
        Convert absolute magnitude to luminosity
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
        Returns:
            array of luminosity [Lsun/h^2]
        """
        return mag2lum(magnitude)

    def lum2mag(self, luminosity):
        """
        Convert luminosity to absolute magnitude
        Args:
            luminosity: array of luminsities [Lsun/h^2]
        Returns:
            array of absolute magnitude [M-5logh]
        """
        return lum2mag(luminosity)

    def magnitude(self, number_density, redshift):
        """
        Convert number density to absolute magnitude threshold
        Args:
            number_density: array of number densities [h^3/Mpc^3]
            redshift: array of redshift
        Returns:
            array of absolute magnitude [M-5logh]
        """
        points = np.array(list(zip(redshift, np.log10(number_density))))
        return self._interpolator(points)


class LuminosityFunctionSchechter(LuminosityFunction):
    """
    Schecter luminosity function with evolution
    Args:
        Phi_star: LF normalization [h^3/Mpc^3]
        M_star: characteristic absolute magnitude [M-5logh]
        alpha: faint end slope
        P: number density evolution parameter
        Q: magnitude evolution parameter
    """
    def __init__(self, Phi_star, M_star, alpha, P, Q):

        # Evolving Shechter luminosity function parameters
        self.Phi_star = Phi_star
        self.M_star = M_star
        self.alpha = alpha
        self.P = P
        self.Q = Q

    def Phi(self, magnitude, redshift):
        """
        Luminosity function as a function of absoulte magnitude and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """
    
        # evolve M_star and Phi_star to redshift
        M_star = self.M_star - self.Q * (redshift - 0.1)
        Phi_star = self.Phi_star * 10**(0.4*self.P*redshift)

        # calculate luminosity function
        lf = 0.4 * np.log(10) * Phi_star
        lf *= (10**(0.4*(M_star-magnitude)))**(self.alpha+1)
        lf *= np.exp(-10**(0.4*(M_star-magnitude)))
        
        return lf

    
    def Phi_cumulative(self, magnitude, redshift):
        """
        Cumulative luminosity function as a function of absoulte magnitude 
        and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """

        # evolve M_star and Phi_star to redshift
        M_star = self.M_star - self.Q * (redshift - 0.1)
        Phi_star = self.Phi_star * 10**(0.4*self.P*redshift)

        # calculate cumulative luminosity function
        t = 10**(0.4 * (M_star-magnitude))
        lf = Phi_star*(gammaincc(self.alpha+2, t)*gamma(self.alpha+2) - \
                           t**(self.alpha+1)*np.exp(-t)) / (self.alpha+1)

        return lf


class LuminosityFunctionTabulated(LuminosityFunction):
    """
    Luminosity function from tabulated file, with evolution
    Args:
        filename: path to ascii file containing tabulated values of cumulative
                  luminsity function
        P: number density evolution parameter
        Q: magnitude evolution parameter
    """
    def __init__(self, filename, P, Q):
        
        self.magnitude_bins, self.log_number_density = \
                              np.loadtxt(filename, unpack=True)
        self.P = P
        self.Q = Q

        self.__lf_interpolator = \
            RegularGridInterpolator((self.magnitude_bins,), self.log_number_density,
                                    bounds_error=False, fill_value=None)
        
        self._interpolator = \
                 self._LuminosityFunction__initialize_interpolator()

    def Phi_cumulative(self, magnitude, redshift):
        """
        Cumulative luminosity function as a function of absoulte magnitude 
        and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """

        # shift magnitudes to redshift z=0.1
        magnitude01 = magnitude + self.Q * (redshift - 0.1)

        # find interpolated number density at z=0.1
        log_lf01 = self.__lf_interpolator(magnitude01)

        # shift back to redshift
        log_lf = log_lf01 + 0.4 * self.P * (redshift - 0.1)
        
        return 10**log_lf
        

class LuminosityFunctionTarget(LuminosityFunction):
    """
    Target luminosity function. Transitions from tabulated file (z<0.15)
    to Schechter LF (z>0.15)
    Args:
        filename: path to ascii file containing tabulated values of cumulative
                  luminsity function
        Phi_star: LF normalization [h^3/Mpc^3]
        M_star: characteristic absolute magnitude [M-5logh]
        alpha: faint end slope
        P: number density evolution parameter
        Q: magnitude evolution parameter
    """
    
    def __init__(self, filename, Phi_star, M_star, alpha, P, Q):
        self.lf_sdss = LuminosityFunctionTabulated(filename, P, Q)
        self.lf_gama = \
               LuminosityFunctionSchechter(Phi_star, M_star, alpha, P, Q)
        self._interpolator = \
                 self._LuminosityFunction__initialize_interpolator()
        self.P = P
        self.Q = Q
        
    def transition(self, redshift):
        """
        Function which describes the transition between the SDSS LF
        at low z and the GAMA LF at high z
        """
        return 1. / (1. + np.exp(120*(redshift-0.15)))

    def Phi(self, magnitude, redshift):
        """
        Luminosity function as a function of absoulte magnitude and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """
        w = self.transition(redshift)
        
        lf_sdss = self.lf_sdss.Phi(magnitude, redshift)
        lf_gama = self.lf_gama.Phi(magnitude, redshift)

        return w*lf_sdss + (1-w)*lf_gama
        
    
    def Phi_cumulative(self, magnitude, redshift):
        """
        Cumulative luminosity function as a function of absoulte magnitude 
        and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """
        w = self.transition(redshift)

        lf_sdss = self.lf_sdss.Phi_cumulative(magnitude, redshift)
        lf_gama = self.lf_gama.Phi_cumulative(magnitude, redshift)
        
        return w*lf_sdss + (1-w)*lf_gama


class LuminosityFunctionTargetBGS(LuminosityFunctionTarget):
    """
    Class used to calculate the target luminosity function at z=0.1,
    used to create the BGS mock catalogue. This is the result of integrating the
    halo mass function multiplied by the HOD. The resulting LF smoothly
    transitions to the Blanton SDSS LF at the faint end, then is
    extrapolated as a power law.
    
    Args:
        lf_file: tabulated file of LF at z=0.1
        lf_param_file: file containing Schechter LF paramters at high z
    """
    
    def __init__(self, target_lf_file, sdss_lf_file, lf_param_file, hod_bgs_simple):

        self.Phi_star, self.M_star, self.alpha, self.P, self.Q = \
                        np.loadtxt(lf_param_file, skiprows=3, delimiter=",")
        
        try:
            self.lf_sdss = LuminosityFunctionTabulated(target_lf_file,self.P,self.Q)
        except IOError:
            self.lf_sdss = self.__initialize_target_lf(target_lf_file, sdss_lf_file)
            
        self.lf_gama = LuminosityFunctionSchechter(self.Phi_star, self.M_star,
                                                   self.alpha, self.P, self.Q)
        self._interpolator = \
                 self._LuminosityFunction__initialize_interpolator()
        
        self.hod_bgs_simple = hod_bgs_simple

        
    def __initialize_target_lf(self, target_lf_file, sdss_lf_file):
        # Create a file of the z=0.1 target LF

        print("Calculating target luminosity function")

        # array of magnitudes and corresponding number densities
        mags = np.arange(-16, -24, -0.1)
        ns = np.zeros(len(mags))

        # loop through each magnitude
        for i in range(len(mags)):
            f = np.array([1.0,]) # HOD 'slide factor', set to 1
            z = np.array([0.1,])
            mag = np.array([mags[i],])
            ns[i] = self.hod_bgs_simple.get_n_HOD(mag,z,f) #cumulative LF
            
        # convert to differential LF
        mag = (mags[1:] + mags[:-1]) / 2.
        n = (ns[:-1] - ns[1:]) / 0.1
        
        # do a spline fit to the differential LF
        mags = np.arange(-10, -25.0000001, -0.001)[::-1]
        zs = np.ones(len(mags)) * 0.1

        tck = splrep(mag[::-1], np.log10(n)[::-1])
        ns = 10**splev(mags, tck)
        
        # transition to blanton at the faint end
        lf_sdss = LuminosityFunctionTabulated(self.sdss_lf_file, self.P,
                                              self.Q)
        ns_sdss = lf_sdss.Phi(mags, zs)
        T =  1. / (1. + np.exp(5*(mags+19))) #transition around mag -19
        ns = ns*T + ns_sdss*(1-T)

        # power law fit to the faint end
        def line(x, a, b):
            return a*x + b
        keep = np.logical_and(mags<-16.3, mags>-19.5)
        popt, pcov = curve_fit(line, mags[keep], np.log10(ns[keep]))
        ns_line = 10**line(mags, *popt)
        T =  1. / (1. + np.exp(5*(mags+17))) #transition around mag -17
        ns = ns*T + ns_line*(1-T)

        # convert back to cumulative LF
        data = np.zeros((len(mags),2))
        data[:,0] = mags + 0.0005
        data[:,1] = np.log10(np.cumsum(ns*0.001))

        # save to file
        np.savetxt(self.target_lf_file, data)

        return LuminosityFunctionTabulated(self.target_lf_file, self.P,
                                           self.Q)
