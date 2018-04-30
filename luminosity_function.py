from __future__ import print_function
import numpy as np
from scipy.special import gamma, gammaincc
from scipy.interpolate import RegularGridInterpolator


class LuminosityFunction(object):
    def __init__(self):
        pass

    def _initialize_interpolator(self):
        """
        Initializes a RegularGridInterpolator for converting number densities
        at a certain to redshift to the corresponding magnitude threshold
        """
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
        Luminosisity function
        """
        pass

    def Phi_cumulative(self, magnitude, redshift):
        """
        Cumulative luminosity function
        """
        pass

    def mag2lum(self, magnitude):
        """
        Convert magnitude to luminosity
        """
        return 10**((4.76 - magnitude)/2.5)

    def lum2mag(self, luminosity):
        """
        Convert luminosity to magnitude
        """
        return 4.76 - 2.5*np.log10(luminosity)

    def magnitude(self, number_density, redshift):
        """
        Convert number density to magnitude
        """
        points = np.array(zip(redshift, np.log10(number_density)))
        return self._interpolator(points)


class LuminosityFunctionSchechter(LuminosityFunction):
    """
    Schecter luminosity function with evolution
    """
    def __init__(self, Phi_star, M_star, alpha, P, Q):

        # Evolving Shechter luminosity function parameters
        self.Phi_star = Phi_star
        self.M_star = M_star
        self.alpha = alpha
        self.P = P
        self.Q = Q

    def Phi(self, magnitude, redshift):

        # evolve M_star and Phi_star to redshift
        M_star = self.M_star - self.Q * (redshift - 0.1)
        Phi_star = self.Phi_star * 10**(0.4*self.P*redshift)

        # calculate luminosity function
        lf = 0.4 * np.log(10) * Phi_star
        lf *= (10**(0.4*(M_star-magnitude)))**(self.alpha+1)
        lf *= np.exp(-10**(0.4*(M_star-magnitude)))
        
        return lf

    def Phi_cumulative(self, magnitude, redshift):

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
    """
    def __init__(self, filename, P, Q):
        self.magnitude, self.log_number_density = \
                              np.loadtxt(filename, unpack=True)

        self.P = P
        self.Q = Q

        self._lf_interpolator = \
            RegularGridInterpolator((self.magnitude,), self.log_number_density,
                                    bounds_error=False, fill_value=None)

    def Phi(self, magnitude, redshift):
        pass

    def Phi_cumulative(self, magnitude, redshift):

        # shift magnitudes to redshift z=0.1
        magnitude01 = magnitude + self.Q * (redshift - 0.1)

        # find interpolated number density at z=0.1
        log_lf01 = self._lf_interpolator(magnitude01)

        # shift back to redshift
        log_lf = log_lf01 + 0.4 * self.P * (redshift - 0.1)
        
        return 10**log_lf
        

class LuminosityFunctionTarget(LuminosityFunction):

    def __init__(self, filename, Phi_star, M_star, alpha, P, Q):
        self.lf_sdss = LuminosityFunctionTabulated(filename, P, Q)
        self.lf_gama = \
               LuminosityFunctionSchechter(Phi_star, M_star, alpha, P, Q)
        self._interpolator = self._initialize_interpolator()

    def transition(self, redshift):
        """
        Function which describes the transition between the SDSS LF
        at low z and the GAMA LF at high z
        """
        return 1. / (1. + np.exp(-100*(redshift-0.15)))


    def Phi(self, magnitude, redshift):
        pass


    def Phi_cumulative(self, magnitude, redshift):
        w = self.transition(redshift)

        lf_sdss = self.lf_sdss.Phi_cumulative(magnitude, redshift)
        lf_gama = self.lf_gama.Phi_cumulative(magnitude, redshift)

        return w*lf_sdss + (1-w)*lf_gama


    def magnitude_faint(self, redshift):
        """
        Minimum absolute magnitude as a function of redshift
        """
        ### ADD CODE TO CONVERT par.mag_faint TO MINIMUM ABS MAGNITUDE ###

        # for now, just returns approx mag faint to r=20 from polynomial
        z=redshift
        a,b,c,d,e = -3045.98, 2187.512, -562.699, 66.359, 6.07331
        L = self.mag2lum(-(25./6)*z - 17) * 10**0.2
        ind = z < 0.2
        L[ind] = 10**(a*z[ind]**4 + b*z[ind]**3+c*z[ind]**2+d*z[ind] + e - 0.5)
        return self.lum2mag(L)


def test():
    import matplotlib.pyplot as plt
    import parameters as par

    mags = np.arange(0,-25,-0.1)
    z = np.ones(len(mags))* 0.0005
    lf_targ = LuminosityFunctionTarget(par.lf_file, par.Phi_star, par.M_star, 
                                       par.alpha, par.P, par.Q)

    #zs = np.arange(0, 1, 0.01)
    #mag = lf_targ.magnitude_faint(zs)
    #plt.plot(zs, mag)
    #plt.show()
    #exit()

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

    for z in np.arange(0, 0.26, 0.05):
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



if __name__ == "__main__":
    test()
