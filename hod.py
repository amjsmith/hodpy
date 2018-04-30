from __future__ import print_function
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from luminosity_function import LuminosityFunctionTarget
from mass_function import MassFunction
from cosmology import Cosmology
import parameters as par
import spline

class HOD(object):
    
    def __init__(self):
        pass

    def number_centrals_mean(self, log_mass, magnitude, redshift):
        pass

    def number_satellites_mean(self, log_mass, magnitude, redshift):
        pass

    def number_galaxies_mean(self, log_mass, magnitude, redshift):
        pass

    def get_number_satellites(self, log_mass, redshift):
        pass

    def get_magnitude_central(self, log_mass, redshift):
        pass

    def get_magnitude_satellites(self, log_mass, redshift, number_satellites):
        pass


class HOD_BGS(HOD):

    def __init__(self):
        self.mf = MassFunction()
        self.cosmo = Cosmology(par.h0, par.OmegaM, par.OmegaL)
        self.lf = LuminosityFunctionTarget(par.lf_file, par.Phi_star, 
                                           par.M_star, par.alpha, par.P, par.Q)

        self._slide_interpolator = self._initialize_slide_factor_interpolator()
        self._logMmin_interpolator = \
            self._initialize_mass_interpolator(par.Mmin_Ls, par.Mmin_Mt, 
                                               par.Mmin_am)
        self._logM1_interpolator = \
            self._initialize_mass_interpolator(par.M1_Ls, par.M1_Mt, par.M1_am)

        self._central_interpolator = self._initialize_central_interpolator()

        


    def _initialize_slide_factor_interpolator(self):
        # read file of slide factors
        factors = np.loadtxt(par.slide_file)[::-1]
        magnitudes = np.arange(-30, 0.01, 0.1)
        redshifts = np.arange(0, 0.91, 0.05)

        return RegularGridInterpolator((magnitudes, redshifts), factors,
                                       bounds_error=False, fill_value=None)

    def _initialize_mass_interpolator(self, L_s, M_t, a_m):
        
        log_mass = np.arange(10, 16, 0.001)[::-1]

        # same functional form as Eq. 11 from Zehavi 2011
        lum = L_s*(10**log_mass/M_t)**a_m * np.exp(1-(M_t/10**log_mass))
        magnitudes = self.lf.lum2mag(lum)

        return RegularGridInterpolator((magnitudes,), log_mass,
                                       bounds_error=False, fill_value=None)

    def _initialize_central_interpolator(self):

        ### ADD OPTION TO SAVE/READ FILE

        # arrays of mass, x, redshift, and 3d array of magnitudes
        # x is the scatter in the central luminosity from the mean
        log_masses = np.arange(10, 16, 0.1)
        redshifts = np.arange(0, 1, 0.1)
        xs = np.arange(-3.5, 3.501, 0.02)
        magnitudes = np.zeros((len(log_masses), len(redshifts), len(xs)))

        # fill in array of magnitudes
        mags = np.arange(-25, -10, 0.01)
        arr_ones = np.ones(len(mags), dtype="f")
        for i in range(len(log_masses)):
            for j in range(len(redshifts)):

                x = np.sqrt(2) * (log_masses[i]-np.log10(self.Mmin(mags, arr_ones*redshifts[j]))) / self.sigma_logM(mags, arr_ones*redshifts[j])

                if x[-1] < 3.5: continue

                # find this in the array xs
                idx = np.searchsorted(x, xs)

                # interpolate 
                f = (xs - x[idx-1]) / (x[idx] - x[idx-1])
                magnitudes[i,j,:] = mags[idx-1] + f*(mags[idx] - mags[idx-1])

        # create RegularGridInterpolator object
        return RegularGridInterpolator((log_masses, redshifts, xs),
                              magnitudes, bounds_error=False, fill_value=None)
    
    def _initialize_satellite_interpolator(self):
        pass


    def slide_factor(self, magnitude, redshift):
        """
        Factor by which the HOD for a fixed number density must
        slide along the mass axis in order to produce the number
        density as specified by the target luminosity function
        """
        points = np.array(zip(magnitude, redshift))
        return self._slide_interpolator(points)


    def Mmin(self, magnitude, redshift):
        """
        HOD parameter Mmin. Mass at which halo has a 50% probability
        of containing a central galaxy brighter than the magnitude threshold
        """
        # use target LF to convert magnitude to number density
        n = self.lf.Phi_cumulative(magnitude, redshift)

        # find magnitude at z0=0.1 which corresponds to the same number density
        magnitude_z0 = self.lf.magnitude(n, np.ones(len(n))*0.1)

        # find Mmin
        Mmin = 10**self._logMmin_interpolator(magnitude_z0)

        # use slide factor to evolve Mmin
        return Mmin * self.slide_factor(magnitude, redshift)


    def M1(self, magnitude, redshift):
        """
        HOD parameter M1. Mass at which the halo contains, on average,
        1 satellite galaxy brighter than the magnitude threshold
        """
        # use target LF to convert magnitude to number density
        n = self.lf.Phi_cumulative(magnitude, redshift)

        # find magnitude at z0=0.1 which corresponds to the same number density
        magnitude_z0 = self.lf.magnitude(n, np.ones(len(n))*0.1)

        # find M1
        M1 = 10**self._logM1_interpolator(magnitude_z0)

        # use slide factor to evolve M1
        return M1 * self.slide_factor(magnitude, redshift)


    def M0(self, magnitude, redshift):
        """
        HOD paramter M0. Cut-off mass scale for satellites
        """
        # use target LF to convert magnitude to number density
        n = self.lf.Phi_cumulative(magnitude, redshift)

        # find magnitude at z0=0.1 which corresponds to the same number density
        magnitude_z0 = self.lf.magnitude(n, np.ones(len(n))*0.1)
        log_lum_z0 = np.log10(self.lf.mag2lum(magnitude_z0))

        # find M0
        M0 = 10**(par.M0_A*log_lum_z0 + par.M0_B)
        
        # use slide factor to evolve M0
        return M0 * self.slide_factor(magnitude, redshift)


    def alpha(self, magnitude, redshift):
        """
        HOD paramter alpha. Sets the slope of the power law for satellites
        """
        # use target LF to convert magnitude to number density
        n = self.lf.Phi_cumulative(magnitude, redshift)

        # find magnitude at z0=0.1 which corresponds to the same number density
        magnitude_z0 = self.lf.magnitude(n, np.ones(len(n))*0.1)
        log_lum_z0 = np.log10(self.lf.mag2lum(magnitude_z0))

        # find alpha
        a = np.log10(par.alpha_C + (par.alpha_A*log_lum_z0)**par.alpha_B)
        
        # alpha is kept fixed with redshift
        return a


    def sigma_logM(self, magnitude, redshift):
        """
        HOD paramter sigma_logM. Sets the width of the step for central
        galaxies, ie the scatter in the magnitude of central galaxies
        """
        # use target LF to convert magnitude to number density
        n = self.lf.Phi_cumulative(magnitude, redshift)

        # find magnitude at z0=0.1 which corresponds to the same number density
        magnitude_z0 = self.lf.magnitude(n, np.ones(len(n))*0.1)

        # find sigma_logM
        sigma = par.sigma_A + (par.sigma_B-par.sigma_A) / \
            (1.+np.exp((magnitude_z0+par.sigma_C)*par.sigma_D))

        # sigma_logM is kept fixed with redshift
        return sigma

    
    def number_centrals_mean(self, log_mass, magnitude, redshift):
        """
        Returns the average number of central galaxies in haloes, 
        using the evolving HODs described in Smith et al 2017
        
        log_mass  : log10 of the mass of the halo, in Msun/h
        magnitude : r-band absolute magnitude threshold (at z=0.1)
        redshift  : redshift of the halo
        """

        # use pseudo gaussian spline kernel
        return spline.cumulative_spline_kernel(log_mass, 
                mean=np.log10(self.Mmin(magnitude, redshift)), 
                sig=self.sigma_logM(magnitude, redshift)/np.sqrt(2))


    def number_satellites_mean(self, log_mass, magnitude, redshift):
        """
        Returns the average number of central galaxies in haloes, 
        using the evolving HODs described in Smith et al 2017
        
        log_mass  : log10 of the mass of the halo, in Msun/h
        magnitude : r-band absolute magnitude threshold (at z=0.1)
        redshift  : redshift of the halo
        """
        num_cent = self.number_centrals_mean(log_mass, magnitude, redshift)

        num_sat = num_cent * ((10**log_mass - self.M0(magnitude, redshift)) / \
                 self.M1(magnitude, redshift))**self.alpha(magnitude, redshift)

        num_sat[np.where(np.isnan(num_sat))[0]] = 0

        return num_sat


    def number_galaxies_mean(self, log_mass, magnitude, redshift):
        """
        Returns the average total number of galaxies in haloes, 
        using the evolving HODs described in Smith et al 2017
        
        log_mass  : log10 of the mass of the halo, in Msun/h
        magnitude : r-band absolute magnitude threshold (at z=0.1)
        redshift  : redshift of the halo
        """
        return self.number_centrals_mean(log_mass, magnitude, redshift) + \
            self.number_satellites_mean(log_mass, magnitude, redshift)


    def get_number_satellites(self, log_mass, redshift):
        """
        Randomly draw the number of satellite galaxies for each halo
        from a Poisson distribution
        """
        # faint magnitude threshold at each redshift
        magnitude = lf.magnitude_faint(redshift)

        # mean number of satellites in each halo brighter than the
        # faint magnitude threshold
        number_mean = self.number_satellites_mean(log_mass, magnitude, redshift)

        # draw random number from Poisson distribution
        return np.random.poisson(number_mean)


    def get_magnitude_central(self, log_mass, redshift):
        """
        Use the HODs to draw a random magnitude for each central galaxy
        """
        # random number from spline kernel distribution
        x = spline.random(size=len(log_mass))

        # return corresponding central magnitude
        points = np.array(zip(log_mass, redshift, x))
        return self._central_interpolator(points)
    

    def get_magnitude_satellites(self, log_mass, redshift, number_satellites):
        """
        Use the HODs to draw a random magnitude for each satellite galaxy
        """
        pass



def test():
    # test plots
    import matplotlib.pyplot as plt
    hod = HOD_BGS()

    print("PLOTTING SLIDE FACTORS")

    mag = np.arange(-23, -17, 0.1)
    
    for z in np.arange(0, 0.6, 0.1):
        zs = np.ones(len(mag)) * z
        f = hod.slide_factor(mag, zs)
        plt.plot(mag, f, label="z = %.1f"%z)

    plt.legend(loc="upper left")
    plt.xlabel("mag")
    plt.ylabel("slide factor")
    plt.xlim(-17,-23)
    plt.ylim(0.7,1.2)
    plt.show()


    print("PLOTTING HODs at z=0.1")

    log_mass = np.arange(10,16,0.01)
    redshift = np.ones(len(log_mass)) * 0.1
    mags = np.arange(-18.5, -22.1, -0.5)

    for i in range(len(mags)):

        magnitude = np.ones(len(log_mass)) * mags[i]

        Ncen = hod.number_centrals_mean(log_mass, magnitude, redshift) 
        Nsat = hod.number_satellites_mean(log_mass, magnitude, redshift)

        plt.plot(log_mass, np.log10(Ncen), c="C%i"%i, ls=":")
        plt.plot(log_mass, np.log10(Nsat), c="C%i"%i, ls=":")
        plt.plot(log_mass, np.log10(Ncen+Nsat), c="C%i"%i, ls="-",
                 label="Mr<%.1f"%mags[i])
    plt.legend(loc="upper left")
    plt.xlabel("log(mass)")
    plt.ylabel("Ngals")
    plt.title("z = 0.1")
    plt.xlim(11,15.3)
    plt.ylim(-2,2.3)
    plt.show()

    print("PLOTTING HODs with Mr<-20")

    log_mass = np.arange(10,16,0.01)
    mag = np.ones(len(log_mass)) * -20
    zs = np.arange(0, 0.6, 0.1)

    for i in range(len(zs)):

        z = np.ones(len(log_mass)) * zs[i]

        Ncen = hod.number_centrals_mean(log_mass, mag, z) 
        Nsat = hod.number_satellites_mean(log_mass, mag, z)

        plt.plot(log_mass, np.log10(Ncen), c="C%i"%i, ls=":")
        plt.plot(log_mass, np.log10(Nsat), c="C%i"%i, ls=":")
        plt.plot(log_mass, np.log10(Ncen+Nsat), c="C%i"%i, ls="-",
                 label="z = %.1f" %zs[i])
    plt.legend(loc="upper left")
    plt.xlabel("log(mass)")
    plt.ylabel("Ngals")
    plt.title("Mr < -20")
    plt.xlim(11,15.3)
    plt.ylim(-2,2.3)
    plt.show()



    print('RANDOMLY GENERATING MAGNITUDES FOR CENTRAL GALAXIES')
    mag = -21
    log_mass = np.arange(10,16,0.01)
    z = np.ones(len(log_mass)) * 0.1
    magnitude = np.ones(len(log_mass)) * mag

    num = hod.number_centrals_mean(log_mass, magnitude, z)
    plt.plot(log_mass, num, label='mean')

    num_av = np.zeros(len(log_mass))
    for i in range(1000):
        mags = hod.get_magnitude_central(log_mass, z)
        idx = mags<mag
        num_av[idx] += 1./1000

    plt.plot(log_mass, num_av, label='random')
    plt.legend(loc='upper left')
    plt.xlabel('log(M)')
    plt.ylabel('N')
    plt.show()

if __name__ == "__main__":
    test()
