#! /usr/bin/env python
from __future__ import print_function
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import quad
from scipy.optimize import minimize, root

from hodpy import luminosity_function
from hodpy.luminosity_function import LuminosityFunctionTabulated
from hodpy.mass_function import MassFunctionAbacus
from hodpy.cosmology import CosmologyAbacus
from hodpy.k_correction import DESI_KCorrection
from hodpy import spline
from hodpy.hod import HOD
from hodpy import lookup


def L_function(magnitude, A, B, C, D):
    return (A + 12) + B*(magnitude + 20) + C*(magnitude+20)**2 + D*(magnitude+20)**3


class HOD_BGS(HOD):
    """
    HOD class containing the HODs used to create the mock catalogue described in Smith et al. 2017

    args:
        [mass_function]: hodpy.MassFunction object, the mass function of the simulation (default is MassFunctionMXXL)
        [cosmology]:     hodpy.Cosmology object, the cosmology of the simulation (default is CosmologyMXXL)
        [mag_faint]:     faint apparent magnitude limit (default is 20.0)
        [kcorr]:         hodpy.KCorrection object, the k-correction (default is GAMA_KCorrection)
        [hod_param_file]: location of file which contains HOD parameters
        [slide_file]:    location of file which contains 'slide' factors for evolving HODs. Will be created
                         automatically if the file doesn't already exist
        [central_lookup_file]: location of lookup file of central magnitudes. Will be created if the file
                               doesn't already exist
        [satellite_lookup_file]: location of lookup file of satellite magnitudes. Will be created if the file
                                 doesn't already exist
        [target_lf_file]: location of file containing the target luminosity function. Will be created if the
                          file doesn't already exist
        [sdss_lf_file]:  location of file which contains the SDSS luminosity function
        [lf_param_file]: location of file which contains the parameters of the evolving GAMA luminosity function
        [replace_central_lookup]: if set to True, will replace central_lookup_file even if the file exists
        [replace_satellite_lookup]: if set to True, will replace satellite_lookup_file even if the file exists
    """

    def __init__(self, cosmo, photsys, mag_faint=20.0, hod_param_file=lookup.abacus_hod_parameters,
                 mass_function=None, cosmology=None, kcorr=None,
                 slide_file=lookup.abacus_hod_slide_factors, 
                 central_lookup_file=lookup.central_lookup_file, 
                 satellite_lookup_file=lookup.satellite_lookup_file, target_lf_file=lookup.bgs_lf_target,
                 replace_central_lookup=False, replace_satellite_lookup=False,
                 mag_faint_interp=None, z0=0.2):

        # TODO: include the option to set a faint absolute magnitude limit

        # TODO: check how HOD parameters extrapolate to faint magnitudes
        # this is important for Mmin and M1, which are cubic functions
        self.mag_faint_interp = mag_faint_interp # if set, the HOD parameters will be interpolated linearly fainter than this magnitude

        self.z0 = z0 # this is the redshift that the HODs were fitted at
        
        # use defaults for quantities not provided
        if mass_function is None:
            mass_function=MassFunctionAbacus(cosmo)
        if cosmology is None:
            cosmology=CosmologyAbacus(cosmo)
        if kcorr is None:
            kcorr=DESI_KCorrection(band='r', photsys=photsys)

        # read the HOD parameters
        print(hod_param_file)
        self.Mmin_A, self.Mmin_B, self.Mmin_C, self.Mmin_D, \
            self.sigma_A, self.sigma_B, self.sigma_C, self.sigma_D, \
            self.M0_A, self.M0_B, \
            self.M1_A, self.M1_B, self.M1_C, self.M1_D, \
            self.alpha_A, self.alpha_B, self.alpha_C = lookup.read_hod_param_file_abacus(hod_param_file)
        
        self.mf = mass_function
        self.cosmo = cosmology
        self.lf = LuminosityFunctionTabulated(target_lf_file, P=0, Q=0) # LF is E-corrected, so set P=Q=0
        self.kcorr = kcorr
        self.mag_faint = mag_faint

        # self.__slide_interpolator = \
        #     self.__initialize_slide_factor_interpolator(slide_file)
        
        self.__central_interpolator = \
            self.__initialize_central_interpolator(central_lookup_file, replace_central_lookup)
        
        self.__satellite_interpolator = \
            self.__initialize_satellite_interpolator(satellite_lookup_file, replace_satellite_lookup)

        
    def __integration_function(self, logM, mag, z, f):
        # function to integrate (number density of haloes * number of galaxies from HOD)
                
        # mean number of galaxies per halo
        N_gal = self.number_galaxies_mean(np.array([logM,]),mag,z,f)[0]
        # number density of haloes
        n_halo = self.mf.number_density(np.array([logM,]), z)
                
        return (N_gal * n_halo)[0]
 
                
    def get_n_HOD(self, magnitude, redshift, f):
        """
        Returns the number density of galaxies predicted by the HOD. This is evaluated from
        integrating the halo mass function multiplied by the HOD. The arguments must be
        arrays of length 1.
        Args:
            magnitude: absolute magnitude threshold
            redshift:  redshifts
            f:         evolution parameter
        Returns:
            number density
        """
        return quad(self.__integration_function, 10, 16, args=(magnitude, redshift, f))[0]


    def __root_function(self, f, mag, z):
        # function used in root finding procedure for calculating the slide factors
        mag = np.array([mag,])
        z = np.array([z,])
        log_n_targ = np.log10(self.lf.Phi_cumulative(mag, z))
        log_n_hod = np.log10(self.get_n_HOD(mag, z, f))
        return log_n_targ - log_n_hod
        
            
    def __initialize_slide_factor_interpolator(self, slide_file):
        # creates a RegularGridInterpolator object used for finding 
        # the 'slide factor' as a function of mag and z

        magnitudes = np.arange(-24, -9.99, 0.1)
        redshifts = np.arange(0, 0.81, 0.05)
        
        try:
            # try to read file
            factors = np.loadtxt(slide_file)
            
        except:
            # file doesn't exist - calculate factors
            # Note: this is very slow!!
            # The nested loop could be sped up by running the root finding in parallel

            print("Calculating evolution parameters")
            
            factors = np.zeros((len(magnitudes),len(redshifts)))

            # loop through magnitudes and redshifts. At each, find f required to get target LF
            for i in range(len(redshifts)):
                for j in range(len(magnitudes)):
                    print("z = %.2f, mag = %.2f"%(redshifts[i],magnitudes[j]))

                    f0 = 1.0 #initial guess
                    x = root(self.__root_function, f0,
                             args=(magnitudes[j], redshifts[i]))
                    # sometimes root finding fails with this initial guess
                    # make f0 smaller and try again
                    while x["x"][0]==f0 and f0>0:
                        f0 -= 0.1
                        print("Trying f0 = %.1f"%f0)
                        x = root(self.__root_function, f0,
                                 args=(magnitudes[j], redshifts[i]))
                    
                    factors[j,i] = x["x"][0]
                    print("f = %.6f"%x["x"][0])

            np.savetxt(slide_file, factors)
    
        return RegularGridInterpolator((magnitudes, redshifts), factors,
                                       bounds_error=False, fill_value=None)


    def __initialize_central_interpolator(self, central_lookup_file, replace_central_lookup=False):
        # creates a RegularGridInterpolator object used for finding 
        # the magnitude of central galaxies as a function of log_mass,
        # z, and random number x from spline kernel distribution

        # arrays of mass, x, redshift, and 3d array of magnitudes
        # x is the scatter in the central luminosity from the mean
        log_masses = np.arange(10, 16, 0.02)
        redshifts = np.arange(0, 1, 0.02)
        xs = np.arange(-3.5, 3.501, 0.02)
        magnitudes = np.zeros((len(log_masses), len(redshifts), len(xs)))

        try:
            if replace_central_lookup: raise IOError
                
            # try to read 3d array of magnitudes from file
            magnitudes = np.load(central_lookup_file)

            if magnitudes.shape != (len(log_masses), len(redshifts), len(xs)):
                raise ValueError("Central lookup table has unexpected shape")

        except IOError:
            # file doesn't exist - fill in array of magnitudes
            print("Generating lookup table of central galaxy magnitudes")
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
                    magnitudes[i,j,:] = mags[idx-1] + f*(mags[idx]-mags[idx-1])
            print("Saving lookup table to file")
            np.save(central_lookup_file, magnitudes)
            
        # create RegularGridInterpolator object
        return RegularGridInterpolator((log_masses, redshifts, xs),
                              magnitudes, bounds_error=False, fill_value=None)
    

    def __initialize_satellite_interpolator(self, satellite_lookup_file, replace_satellite_file=False):
        # creates a RegularGridInterpolator object used for finding 
        # the magnitude of satellite galaxies as a function of log_mass,
        # z, and random number log_x (x is uniform random between 0 and 1)

        # arrays of mass, x, redshift, and 3d array of magnitudes
        # x is the ratio of Nsat(mag,mass)/Nsat(mag_faint,mass)
        log_masses = np.arange(10, 16, 0.02)
        redshifts = np.arange(0, 1, 0.02)
        log_xs = np.arange(-12, 0.01, 0.05)
        magnitudes = np.zeros((len(log_masses), len(redshifts), len(log_xs)))

        try:
            if replace_satellite_file: raise IOError
            
            # try to read 3d array of magnitudes from file
            magnitudes = np.load(satellite_lookup_file)

            if magnitudes.shape!=(len(log_masses), len(redshifts), len(log_xs)):
                raise ValueError("Satellite lookup table has unexpected shape")
            
        except IOError:
            # file doesn't exist - fill in array of magnitudes
            print("Generating lookup table of satellite galaxy magnitudes")

            mags = np.arange(-25, -8, 0.01)
            abs_mag_faint = self.kcorr.magnitude_faint(redshifts, self.mag_faint)
            arr_ones = np.ones(len(mags))
            for i in range(len(log_masses)):
                for j in range(len(redshifts)):
                    Nsat = self.number_satellites_mean(arr_ones*log_masses[i], mags,
                                                   arr_ones*redshifts[j])
                    Nsat_faint = self.number_satellites_mean(arr_ones*log_masses[i],
                                   arr_ones*abs_mag_faint[j],arr_ones*redshifts[j])

                    log_x = np.log10(Nsat) - np.log10(Nsat_faint)

                    if log_x[-1] == -np.inf: continue

                    # find this in the array log_xs
                    idx = np.searchsorted(log_x, log_xs)

                    # interpolate 
                    f = (log_xs - log_x[idx-1]) / (log_x[idx] - log_x[idx-1])
                    magnitudes[i,j,:] = mags[idx-1] + f*(mags[idx]-mags[idx-1])

                    # Deal with NaN values
                    # if NaN for small x but not large x, replace all 
                    # NaN values with faintest mag
                    idx = np.isnan(magnitudes[i,j,:])
                    num_nan = np.count_nonzero(idx)
                    if num_nan < len(idx) and num_nan>0:
                        magnitudes[i,j,idx] = \
                            magnitudes[i,j,np.where(idx)[0][-1]+1]
                    # if previous mass bin contains all NaN, copy current
                    # mass bin
                    if i>0 and np.count_nonzero(np.isnan(magnitudes[i-1,j,:]))\
                                                ==len(magnitudes[i,j,:]):
                        magnitudes[i-1,j,:] = magnitudes[i,j,:]
                    # if all NaN and j>0, copy prev j to here
                    if j>0 and np.count_nonzero(np.isnan(magnitudes[i,j,:]))\
                                                ==len(magnitudes[i,j,:]):
                        magnitudes[i,j,:] = magnitudes[i,j-1,:]


            print("Saving lookup table to file")
            np.save(satellite_lookup_file, magnitudes)

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
        # points = np.array(list(zip(magnitude, redshift)))
        # return self.__slide_interpolator(points)

        return 1

    def __Mmin_z0(self, magnitude):
        # get Mmin as a function of magnitude at the redshift the HODs were fitted (z0=0.2)
        Mmin = 10**L_function(magnitude, self.Mmin_A, self.Mmin_B, self.Mmin_C, self.Mmin_D)

        # extrapolate as a power law at the faint end, fainter than mag_faint_interp
        if not self.mag_faint_interp is None:
            x = self.mag_faint_interp
            m_lin = self.Mmin_B + 2*(20+x)*self.Mmin_C + 3*(20+x)**2*self.Mmin_D
            c_lin = L_function(x, self.Mmin_A, self.Mmin_B, self.Mmin_C, self.Mmin_D) - (x*m_lin)
            Mmin_lin = 10**(m_lin * magnitude + c_lin)
        
            # linear extrapolation
            idx = magnitude > x
            Mmin[idx] = Mmin_lin[idx]
        
            # smoothly flatten curve at faint end
            Mmin_0 = m_lin * x + c_lin
            Mmin[idx] = 10**(np.exp((np.log10(Mmin[idx])-Mmin_0))+Mmin_0-1)
    
        return Mmin
    
    def Mmin(self, magnitude, redshift, f=None):
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

        # find magnitude at z0=0.2 which corresponds to the same number density
        magnitude_z0 = self.lf.magnitude(n, np.ones(len(n))*self.z0)

        # find Mmin
        Mmin = self.__Mmin_z0(magnitude_z0)
        # use slide factor to evolve Mmin
        if not f is None:
            return Mmin * f
        else:
            return Mmin * self.slide_factor(magnitude, redshift)


    def __M1_z0(self, magnitude):
        # get M1 as a function of magnitude at the redshift the HODs were fitted (z0=0.2)
        M1 = 10**L_function(magnitude, self.M1_A, self.M1_B, self.M1_C, self.M1_D)

        # extrapolate as a power law at the faint end, fainter than mag_faint_interp
        if not self.mag_faint_interp is None:
            x = self.mag_faint_interp
            m_lin = self.M1_B + 2*(20+x)*self.M1_C + 3*(20+x)**2*self.M1_D
            c_lin = L_function(x, self.M1_A, self.M1_B, self.M1_C, self.M1_D) - (x*m_lin)
            M1_lin = 10**(m_lin * magnitude + c_lin)
        
            # linear extrapolation
            idx = magnitude > x
            M1[idx] = M1_lin[idx]
        
            # smoothly flatten curve at faint end
            M1_0 = m_lin * x + c_lin
            M1[idx] = 10**(np.exp((np.log10(M1[idx])-M1_0))+M1_0-1)
    
        return M1
        
    def M1(self, magnitude, redshift, f=None):
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

        # find magnitude at z0=0.2 which corresponds to the same number density
        magnitude_z0 = self.lf.magnitude(n, np.ones(len(n))*self.z0)

        # find M1
        M1 = self.__M1_z0(magnitude_z0)

        # use slide factor to evolve M1
        if not f is None:
            return M1 * f
        else:
            return M1 * self.slide_factor(magnitude, redshift)

    def __M0_z0(self, magnitude):
        # get M0 as a function of magnitude at the redshift the HODs were fitted (z0=0.2)
        logM0 = self.M0_A*(magnitude+20) + (self.M0_B+11)
        logM0[logM0 <= 1] = 1
        return 10**logM0
    
    def M0(self, magnitude, redshift, f=None):
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

        # find magnitude at z0=0.2 which corresponds to the same number density
        magnitude_z0 = self.lf.magnitude(n, np.ones(len(n))*self.z0)
        log_lum_z0 = np.log10(self.lf.mag2lum(magnitude_z0))

        # find M0
        M0 = self.__M0_z0(magnitude)
        
        # use slide factor to evolve M0
        if not f is None:
            return M0 * f
        else:
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

        # find magnitude at z0=0.2 which corresponds to the same number density
        magnitude_z0 = self.lf.magnitude(n, np.ones(len(n))*self.z0)
        log_lum_z0 = np.log10(self.lf.mag2lum(magnitude_z0))

        # find alpha
        a = self.alpha_A + self.alpha_B**(self.alpha_C - magnitude_z0 - 20)
        
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

        # find magnitude at z0=0.2 which corresponds to the same number density
        magnitude_z0 = self.lf.magnitude(n, np.ones(len(n))*self.z0)

        # find sigma_logM
        sigma = self.sigma_A + (self.sigma_B-self.sigma_A) / (1.+np.exp((magnitude_z0+self.sigma_C)*self.sigma_D))

        # sigma_logM is kept fixed with redshift
        return sigma

    
    def number_centrals_mean(self, log_mass, magnitude, redshift, f=None):
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
                mean=np.log10(self.Mmin(magnitude, redshift, f)), 
                sig=self.sigma_logM(magnitude, redshift)/np.sqrt(2))


    def number_satellites_mean(self, log_mass, magnitude, redshift, f=None):
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
        num_cent = self.number_centrals_mean(log_mass, magnitude, redshift, f)

        num_sat = num_cent * ((10**log_mass - self.M0(magnitude, redshift, f))/\
            self.M1(magnitude, redshift, f))**self.alpha(magnitude, redshift)

        num_sat[np.where(np.isnan(num_sat))[0]] = 0

        return num_sat


    def number_galaxies_mean(self, log_mass, magnitude, redshift, f=None):
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
        return self.number_centrals_mean(log_mass, magnitude, redshift, f) + \
            self.number_satellites_mean(log_mass, magnitude, redshift, f)


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
        magnitude = self.kcorr.magnitude_faint(redshift, self.mag_faint)

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
        points = np.array(list(zip(log_mass, redshift, x)))
        return self.__central_interpolator(points)
    

    def get_magnitude_satellites(self, log_mass, number_satellites, redshift):
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
        points = np.array(list(zip(log_mass_satellite, redshift_satellite, 
                                   log_x)))
        return halo_index, self.__satellite_interpolator(points)


