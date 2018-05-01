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

    def get_magnitude_centrals(self, log_mass, redshift):
        pass

    def get_magnitude_satellites(self, log_mass, redshift, number_satellites):
        pass


class HOD_BGS(HOD):
        """
        HODs used to create the mock catalogue described in Smith et al. 2017
        """

    def __init__(self):
        self.mf = MassFunction()
        self.cosmo = Cosmology(par.h0, par.OmegaM, par.OmegaL)
        self.lf = LuminosityFunctionTarget(par.lf_file, par.Phi_star, 
                                           par.M_star, par.alpha, par.P, par.Q)

        self.__slide_interpolator =self.__initialize_slide_factor_interpolator()
        self.__logMmin_interpolator = \
            self.__initialize_mass_interpolator(par.Mmin_Ls, par.Mmin_Mt, 
                                               par.Mmin_am)
        self.__logM1_interpolator = \
            self.__initialize_mass_interpolator(par.M1_Ls, par.M1_Mt, par.M1_am)

        print("Generating lookup table of central galaxy magnitudes")
        self.__central_interpolator = self.__initialize_central_interpolator()

        print("Generating lookup table of satellite galaxy magnitudes")
        self.__satellite_interpolator = \
                                  self.__initialize_satellite_interpolator()
        print("Done")


    def __initialize_slide_factor_interpolator(self):
        # creates a RegularGridInterpolator object used for finding 
        # the 'slide factor' as a function of mag and z

        ### ADD OPTION TO CALCULATE THESE FACTORS

        # read file of slide factors
        factors = np.loadtxt(par.slide_file)[::-1]
        magnitudes = np.arange(-30, 0.01, 0.1)
        redshifts = np.arange(0, 0.91, 0.05)

        return RegularGridInterpolator((magnitudes, redshifts), factors,
                                       bounds_error=False, fill_value=None)

    def __initialize_mass_interpolator(self, L_s, M_t, a_m):
        # creates a RegularGridInterpolator object used for finding 
        # the HOD parameters Mmin or M1 (at z=0.1) as a function of log_mass
        
        log_mass = np.arange(10, 16, 0.001)[::-1]

        # same functional form as Eq. 11 from Zehavi 2011
        lum = L_s*(10**log_mass/M_t)**a_m * np.exp(1-(M_t/10**log_mass))
        magnitudes = self.lf.lum2mag(lum)

        return RegularGridInterpolator((magnitudes,), log_mass,
                                       bounds_error=False, fill_value=None)

    def __initialize_central_interpolator(self):
        # creates a RegularGridInterpolator object used for finding 
        # the magnitude of central galaxies as a function of log_mass,
        # z, and random number x from spline kernel distribution

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
    

    def __initialize_satellite_interpolator(self):
        # creates a RegularGridInterpolator object used for finding 
        # the magnitude of satellite galaxies as a function of log_mass,
        # z, and random number log_x (x is uniform random between 0 and 1)
        log_masses = np.arange(10, 16, 0.1)
        redshifts = np.arange(0, 1, 0.1)
        log_xs = np.arange(-12, 0.01, 0.02)
        magnitudes = np.zeros((len(log_masses), len(redshifts), len(log_xs)))

        # fill in array of magnitudes
        mags = np.arange(-25, -8, 0.01)
        mag_faint = self.lf.magnitude_faint(redshifts)
        arr_ones = np.ones(len(mags))
        for i in range(len(log_masses)):
            for j in range(len(redshifts)):
                Nsat = self.number_satellites_mean(arr_ones*log_masses[i], mags,
                                                   arr_ones*redshifts[j])
                Nsat_faint = self.number_satellites_mean(arr_ones*log_masses[i],
                                   arr_ones*mag_faint[j],arr_ones*redshifts[j])

                log_x = np.log10(Nsat) - np.log10(Nsat_faint)

                if log_x[-1] == -np.inf: continue

                # find this in the array log_xs
                idx = np.searchsorted(log_x, log_xs)

                # interpolate 
                f = (log_xs - log_x[idx-1]) / (log_x[idx] - log_x[idx-1])
                magnitudes[i,j,:] = mags[idx-1] + f*(mags[idx] - mags[idx-1])

        # create RegularGridInterpolator object
        return RegularGridInterpolator((log_masses, redshifts, log_xs),
                              magnitudes, bounds_error=False, fill_value=None)


    def slide_factor(self, magnitude, redshift):
        """
        Factor by when the HOD mass parameters (ie Mmin, M0 and M1) must
        be multiplied by in order to produce the number density of
        galaxies as specified by the target luminosity function

        Args:
            magnitude: array of absolute magnitude threshold
            redshift:  array of halo redshifts
        Returns:
            array of slide factors
        """
        points = np.array(zip(magnitude, redshift))
        return self.__slide_interpolator(points)


    def Mmin(self, magnitude, redshift):
        """
        HOD parameter Mmin, which is the mass at which a halo has a 50%
        change of containing a central galaxy brighter than the magnitude 
        threshold

        Args:
            magnitude: array of absolute magnitude threshold
            redshift:  array of halo redshifts
        Returns:
            array of Mmin
        """
        # use target LF to convert magnitude to number density
        n = self.lf.Phi_cumulative(magnitude, redshift)

        # find magnitude at z0=0.1 which corresponds to the same number density
        magnitude_z0 = self.lf.magnitude(n, np.ones(len(n))*0.1)

        # find Mmin
        Mmin = 10**self.__logMmin_interpolator(magnitude_z0)

        # use slide factor to evolve Mmin
        return Mmin * self.slide_factor(magnitude, redshift)


    def M1(self, magnitude, redshift):
        """
        HOD parameter M1, which is the mass at which a halo contains an
        average of 1 satellite brighter than the magnitude threshold

        Args:
            magnitude: array of absolute magnitude threshold
            redshift:  array of halo redshifts
        Returns:
            array of M1
        """
        # use target LF to convert magnitude to number density
        n = self.lf.Phi_cumulative(magnitude, redshift)

        # find magnitude at z0=0.1 which corresponds to the same number density
        magnitude_z0 = self.lf.magnitude(n, np.ones(len(n))*0.1)

        # find M1
        M1 = 10**self.__logM1_interpolator(magnitude_z0)

        # use slide factor to evolve M1
        return M1 * self.slide_factor(magnitude, redshift)


    def M0(self, magnitude, redshift):
        """
        HOD parameter M0, which sets the cut-off mass scale for satellites
        satellites

        Args:
            magnitude: array of absolute magnitude threshold
            redshift:  array of halo redshifts
        Returns:
            array of M0
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
        HOD parameter alpha, which sets the slope of the power law for
        satellites

        Args:
            magnitude: array of absolute magnitude threshold
            redshift:  array of halo redshifts
        Returns:
            array of alpha
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
        HOD parameter sigma_logM, which sets the amount of scatter in 
        the luminosity of central galaxies

        Args:
            magnitude: array of absolute magnitude threshold
            redshift:  array of halo redshifts
        Returns:
            array of sigma_logM
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
        Average number of central galaxies in each halo brighter than
        some absolute magnitude threshold

        Args:
            log_mass:  array of the log10 of halo mass (Msun/h)
            magnitude: array of absolute magnitude threshold
            redshift:  array of halo redshifts
        Returns:
            array of mean number of central galaxies
        """

        # use pseudo gaussian spline kernel
        return spline.cumulative_spline_kernel(log_mass, 
                mean=np.log10(self.Mmin(magnitude, redshift)), 
                sig=self.sigma_logM(magnitude, redshift)/np.sqrt(2))


    def number_satellites_mean(self, log_mass, magnitude, redshift):
        """
        Average number of satellite galaxies in each halo brighter than
        some absolute magnitude threshold

        Args:
            log_mass:  array of the log10 of halo mass (Msun/h)
            magnitude: array of absolute magnitude threshold
            redshift:  array of halo redshifts
        Returns:
            array of mean number of satellite galaxies
        """
        num_cent = self.number_centrals_mean(log_mass, magnitude, redshift)

        num_sat = num_cent * ((10**log_mass - self.M0(magnitude, redshift)) / \
                 self.M1(magnitude, redshift))**self.alpha(magnitude, redshift)

        num_sat[np.where(np.isnan(num_sat))[0]] = 0

        return num_sat


    def number_galaxies_mean(self, log_mass, magnitude, redshift):
        """
        Average total number of galaxies in each halo brighter than
        some absolute magnitude threshold

        Args:
            log_mass:  array of the log10 of halo mass (Msun/h)
            magnitude: array of absolute magnitude threshold
            redshift:  array of halo redshifts
        Returns:
            array of mean number of galaxies
        """
        return self.number_centrals_mean(log_mass, magnitude, redshift) + \
            self.number_satellites_mean(log_mass, magnitude, redshift)


    def get_number_satellites(self, log_mass, redshift):
        """
        Randomly draw the number of satellite galaxies in each halo,
        brighter than mag_faint, from a Poisson distribution

        Args:
            log_mass: array of the log10 of halo mass (Msun/h)
            redshift: array of halo redshifts
        Returns:
            array of number of satellite galaxies
        """
        # faint magnitude threshold at each redshift
        magnitude = self.lf.magnitude_faint(redshift)

        # mean number of satellites in each halo brighter than the
        # faint magnitude threshold
        number_mean = self.number_satellites_mean(log_mass, magnitude, redshift)

        # draw random number from Poisson distribution
        return np.random.poisson(number_mean)


    def get_magnitude_centrals(self, log_mass, redshift):
        """
        Use the HODs to draw a random magnitude for each central galaxy

        Args:
            log_mass: array of the log10 of halo mass (Msun/h)
            redshift: array of halo redshifts
        Returns:
            array of central galaxy magnitudes
        """
        # random number from spline kernel distribution
        x = spline.random(size=len(log_mass))

        # return corresponding central magnitudes
        points = np.array(zip(log_mass, redshift, x))
        return self.__central_interpolator(points)
    

    def get_magnitude_satellites(self, log_mass, redshift, number_satellites):
        """
        Use the HODs to draw a random magnitude for each satellite galaxy

        Args:
            log_mass:          array of the log10 of halo mass (Msun/h)
            redshift:          array of halo redshifts
            number_satellites: array of number of sateillites in each halo
        Returns:
            array of the index of each galaxy's halo in the input arrays
            array of satellite galaxy magnitudes
        """
        # create arrays of log_mass, redshift and halo_index for galaxies
        halo_index = np.arange(len(log_mass))
        halo_index = np.repeat(halo_index, number_satellites)
        log_mass_satellite = np.repeat(log_mass, number_satellites)
        redshift_satellite = np.repeat(redshift, number_satellites)

        # uniform random number x
        # x=1 corresponds to mag_faint
        # mag -> -infinity as x -> 0
        log_x = np.log10(np.random.rand(len(log_mass_satellite)))

        # find corresponding satellite magnitudes
        points = np.array(zip(log_mass_satellite, redshift_satellite, log_x))
        return halo_index, self.__satellite_interpolator(points)
        



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

    
    print('RANDOMLY GENERATING MAGNITUDES FOR GALAXIES')
    mag = -21

    log_mass = np.arange(10,16,0.01)
    z = np.ones(len(log_mass)) * 0.1
    magnitude = np.ones(len(log_mass)) * mag

    num_cen = hod.number_centrals_mean(log_mass, magnitude, z)
    num_sat = hod.number_satellites_mean(log_mass, magnitude, z)
    plt.plot(log_mass, num_cen, c="b")
    plt.plot(log_mass, num_sat, c="b")
    plt.plot(log_mass, num_cen+num_sat, label='mean', c="b")

    num_av_cen = np.zeros(len(log_mass))
    num_av_sat = np.zeros(len(log_mass))
    N = 100
    for i in range(N):
        print("REALIZATION", i, "OF", N)
        mags = hod.get_magnitude_centrals(log_mass, z)
        idx = mags<mag
        num_av_cen[idx] += 1./N

        nsat = hod.get_number_satellites(log_mass, z)
        halo_idx, mags = hod.get_magnitude_satellites(log_mass, z, nsat)
        log_mass_sat = log_mass[halo_idx]
        for j in range(len(log_mass)):
            idx = np.where(log_mass_sat == log_mass[j])
            num_av_sat[j] += np.count_nonzero(mags[idx]<mag)/float(N)

    plt.plot(log_mass, num_av_cen, c="r")
    plt.plot(log_mass, num_av_sat, c="r")
    plt.plot(log_mass, num_av_cen+num_av_sat, c="r",label='random')
    plt.legend(loc='upper left')
    plt.yscale("log")
    plt.xlabel('log(M)')
    plt.ylabel('N')
    plt.show()


if __name__ == "__main__":
    test()
