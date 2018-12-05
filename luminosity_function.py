#! /usr/bin/env python
from __future__ import print_function
import numpy as np
from scipy.special import gamma, gammaincc
from scipy.interpolate import RegularGridInterpolator

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
        mags = np.arange(-25, -10, 0.001)
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
        from scipy.interpolate import splrep, splev
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
        return 10**((4.76 - magnitude)/2.5)

    def lum2mag(self, luminosity):
        """
        Convert luminosity to absolute magnitude
        Args:
            luminosity: array of luminsities [Lsun/h^2]
        Returns:
            array of absolute magnitude [M-5logh]
        """
        return 4.76 - 2.5*np.log10(luminosity)

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
        self.magnitude, self.log_number_density = \
                              np.loadtxt(filename, unpack=True)

        self.P = P
        self.Q = Q

        self.__lf_interpolator = \
            RegularGridInterpolator((self.magnitude,), self.log_number_density,
                                    bounds_error=False, fill_value=None)

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
        return 1. / (1. + np.exp(100*(redshift-0.15)))

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



def test():
    import matplotlib.pyplot as plt
    import parameters as par

    mags = np.arange(0,-25,-0.001)
    
    z = np.ones(len(mags))* 0.0005
    lf_targ = LuminosityFunctionTarget(par.lf_file, par.Phi_star, par.M_star, 
                                       par.alpha, par.P, par.Q)

    logn = np.log10(lf_targ.Phi_cumulative(mags, z))
    mag = lf_targ.magnitude(10**logn, z)
    
    plt.plot(mags, logn)
    plt.plot(mag, logn, ls="--")
    plt.show()
    
    lf_gama = LuminosityFunctionSchechter(par.Phi_star, par.M_star, par.alpha, 
                                          par.P, par.Q)
    lf_sdss = LuminosityFunctionTabulated(par.lf_file, par.P, par.Q)
    lf_targ = LuminosityFunctionTarget(par.lf_file, par.Phi_star, par.M_star, 
                                       par.alpha, par.P, par.Q)
    
    for z in np.arange(0, 0.26, 0.025):
        
        phi = lf_sdss.Phi_cumulative(mags, z)
        plt.plot(mags, phi, c="b")

        phi = lf_gama.Phi_cumulative(mags, z)
        plt.plot(mags, phi, c="g")

        phi = lf_targ.Phi_cumulative(mags, z)
        plt.plot(mags, phi, c="r", ls="--")
        
        plt.yscale("log")
        plt.title("z = %.2f"%z)
        plt.xlabel("mag")
        plt.ylabel("cumulative LF")
        plt.xlim(-18, -23)
        plt.ylim(1e-6, 3e-2)
        plt.show()
        
        
        phi = lf_sdss.Phi(mags, z)
        plt.plot(mags, phi, c="b")

        phi = lf_gama.Phi(mags, z)
        plt.plot(mags, phi, c="g")

        phi = lf_targ.Phi(mags, z)
        plt.plot(mags, phi, c="r", ls="--")

        plt.yscale("log")
        plt.title("z = %.2f"%z)
        plt.xlabel("mag")
        plt.ylabel("cumulative LF")
        plt.xlim(-18, -23)
        plt.ylim(1e-6, 3e-2)
        plt.show()
        


if __name__ == "__main__":
    test()
