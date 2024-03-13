#! /usr/bin/env python
from __future__ import print_function
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import quad
from scipy.optimize import minimize, root

from hodpy import luminosity_function
from hodpy import spline
from hodpy.hod import HOD
from hodpy import lookup


def L_function(magnitude, A, B, C, D):
    return (A + 12) + B*(magnitude + 20) + C*(magnitude+20)**2 + D*(magnitude+20)**3



class HOD_BGS(HOD):
    """
    HOD class for the HODs used to create the AbacusSummit 2ndGen BGS mocks

    args:
        cosmology:     hodpy.Cosmology object, the cosmology of the simulation (default is CosmologyMXXL)
        abs_mag_faint:     faint apparent magnitude limit (default is 20.0)
        [kcorr]:         hodpy.KCorrection object, the k-correction (default is GAMA_KCorrection)
        [hod_param_file]: location of file which contains HOD parameters
        [slide_file]:    location of file which contains 'slide' factors for evolving HODs. Will be created
                         automatically if the file doesn't already exist
        [central_lookup_file]: location of lookup file of central magnitudes. Will be created if the file
                               doesn't already exist
        [satellite_lookup_file]: location of lookup file of satellite magnitudes. Will be created if the file
                                 doesn't already exist
        [target_lf_file]: location of file containing the target luminosity function
        [replace_central_lookup]: if set to True, will replace central_lookup_file even if the file exists
        [replace_satellite_lookup]: if set to True, will replace satellite_lookup_file even if the file exists
        [mass_function]
    """

    def __init__(self, cosmology, abs_mag_faint, hod_param_file, 
                 central_lookup_file, satellite_lookup_file,
                 replace_central_lookup=False, replace_satellite_lookup=False,
                 mass_function=None):
        
        
        print("HOD params:", hod_param_file)
        
        self.Mmin_A, self.Mmin_B, self.Mmin_C, self.Mmin_D, \
            self.sigma_A, self.sigma_B, self.sigma_C, self.sigma_D, \
            self.M0_A, self.M0_B, \
            self.M1_A, self.M1_B, self.M1_C, self.M1_D, \
            self.alpha_A, self.alpha_B, self.alpha_C = lookup.read_hod_param_file_abacus(hod_param_file)
        
        #self.cosmo = cosmology
        self.mf = mass_function
        
        self.abs_mag_faint = abs_mag_faint

        self.__central_interpolator = \
            self.__initialize_central_interpolator(central_lookup_file, replace_central_lookup)
        
        self.__satellite_interpolator = \
            self.__initialize_satellite_interpolator(satellite_lookup_file, replace_satellite_lookup)



    
    def __initialize_central_interpolator(self, central_lookup_file, replace_central_lookup=False):
        # creates a RegularGridInterpolator object used for finding 
        # the magnitude of central galaxies as a function of log_mass,
        # z, and random number x from spline kernel distribution

        # arrays of mass, x, redshift, and 3d array of magnitudes
        # x is the scatter in the central luminosity from the mean
        log_masses = np.arange(10, 16, 0.02)
        xs = np.arange(-3.5, 3.501, 0.02)
        magnitudes = np.zeros((len(log_masses), len(xs)))

        try:
            if replace_central_lookup: raise IOError
                
            # try to read 3d array of magnitudes from file
            magnitudes = np.load(central_lookup_file)

            if magnitudes.shape != (len(log_masses), len(xs)):
                raise ValueError("Central lookup table has unexpected shape")

        except IOError:
            # file doesn't exist - fill in array of magnitudes
            print("Generating lookup table of central galaxy magnitudes")
            mags = np.arange(-25, -10, 0.01)
            arr_ones = np.ones(len(mags), dtype="f")
            for i in range(len(log_masses)):

                x = np.sqrt(2) * (log_masses[i]-np.log10(self.Mmin(mags))) / self.sigma_logM(mags)

                if x[-1] < 3.5: continue

                # find this in the array xs
                idx = np.searchsorted(x, xs)

                # interpolate 
                f = (xs - x[idx-1]) / (x[idx] - x[idx-1])
                magnitudes[i,:] = mags[idx-1] + f*(mags[idx]-mags[idx-1])
            print("Saving lookup table to file")
            np.save(central_lookup_file, magnitudes)
            
        # create RegularGridInterpolator object
        return RegularGridInterpolator((log_masses, xs),
                              magnitudes, bounds_error=False, fill_value=None)
    

    def __initialize_satellite_interpolator(self, satellite_lookup_file, replace_satellite_file=False):
        # creates a RegularGridInterpolator object used for finding 
        # the magnitude of satellite galaxies as a function of log_mass,
        # z, and random number log_x (x is uniform random between 0 and 1)

        # arrays of mass, x, redshift, and 3d array of magnitudes
        # x is the ratio of Nsat(mag,mass)/Nsat(mag_faint,mass)
        log_masses = np.arange(10, 16, 0.02)
        log_xs = np.arange(-12, 0.01, 0.02) #0.05
        magnitudes = np.zeros((len(log_masses), len(log_xs)))

        try:
            if replace_satellite_file: raise IOError
            
            # try to read 3d array of magnitudes from file
            magnitudes = np.load(satellite_lookup_file)

            if magnitudes.shape!=(len(log_masses), len(log_xs)):
                raise ValueError("Satellite lookup table has unexpected shape")
            
        except IOError:
            # file doesn't exist - fill in array of magnitudes
            print("Generating lookup table of satellite galaxy magnitudes")

            #mags = np.arange(-25, -8, 0.01)
            mags = np.arange(-25, -8, 0.01)
            abs_mag_faint = self.abs_mag_faint
            arr_ones = np.ones(len(mags))
            
            for i in range(len(log_masses)):
                Nsat = self.number_satellites_mean(arr_ones*log_masses[i], mags)
                Nsat_faint = self.number_satellites_mean(arr_ones*log_masses[i],
                                       arr_ones*abs_mag_faint)
                
                log_x = np.log10(Nsat) - np.log10(Nsat_faint)

                if log_x[-1] == -np.inf: continue

                # find this in the array log_xs
                idx = np.searchsorted(log_x, log_xs)
            
                # interpolate 
                f = (log_xs - log_x[idx-1]) / (log_x[idx] - log_x[idx-1])
                magnitudes[i,:] = mags[idx-1] + f*(mags[idx]-mags[idx-1])
                
                # Deal with NaN values
                # if NaN for small x but not large x, replace all 
                # NaN values with faintest mag
                idx = np.isnan(magnitudes[i,:])
                num_nan = np.count_nonzero(idx)
                if num_nan < len(idx) and num_nan>0:
                    magnitudes[i,idx] = magnitudes[i,np.where(idx)[0][-1]+1]
                    
                # if previous mass bin contains all NaN, copy current mass bin
                if i>0 and np.count_nonzero(np.isnan(magnitudes[i-1,:]))==len(magnitudes[i,:]):
                    magnitudes[i-1,:] = magnitudes[i,:]
            
            print("Saving lookup table to file")
            np.save(satellite_lookup_file, magnitudes)

        # create RegularGridInterpolator object
        return RegularGridInterpolator((log_masses, log_xs),
                              magnitudes, bounds_error=False, fill_value=None)




    def Mmin(self, magnitude):
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
        x = -17
        Mmin = 10**L_function(magnitude, self.Mmin_A, self.Mmin_B, self.Mmin_C, self.Mmin_D)
        
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


    def M1(self, magnitude):
        """
        HOD parameter M1, which is the mass at which a halo contains an
        average of 1 satellite brighter than the magnitude threshold
        Args:
            magnitude: array of absolute magnitude threshold
            redshift:  array of halo redshifts
        Returns:
            array of M1
        """
        
        x = -17
        M1 = 10**L_function(magnitude, self.M1_A, self.M1_B, self.M1_C, self.M1_D)

        #m_lin = self.M1_B + 4*self.M1_C + 12*self.M1_D
        m_lin = self.M1_B + 2*(20+x)*self.M1_C + 3*(20+x)**2*self.M1_D
        
        c_lin = L_function(x, self.M1_A, self.M1_B, self.M1_C, self.M1_D) - (x*m_lin)
        M1_lin = 10**(m_lin * magnitude + c_lin)
        
        # linear extrapolation
        idx = magnitude > x
        M1[idx] = M1_lin[idx]
        
        # keep it fixed
        # idx = magnitude > x
        # M1[idx] = 10**L_function(np.ones(np.count_nonzero(idx))*x, 
        #                          self.M1_A, self.M1_B, self.M1_C, self.M1_D)
        #print(M1[idx])
        
        # smoothly flatten curve at faint end
        M1_0 = m_lin * x + c_lin
        M1[idx] = 10**(np.exp((np.log10(M1[idx])-M1_0))+M1_0-1)
    
        return M1


    def M0(self, magnitude):
        """
        HOD parameter M0, which sets the cut-off mass scale for satellites
        satellites
        Args:
            magnitude: array of absolute magnitude threshold
            redshift:  array of halo redshifts
        Returns:
            array of M0
        """
        
        logM0 = self.M0_A*(magnitude+20) + (self.M0_B+11)
        logM0[logM0 <= 1] = 1
        return 10**logM0 


    def alpha(self, magnitude):
        """
        HOD parameter alpha, which sets the slope of the power law for
        satellites
        Args:
            magnitude: array of absolute magnitude threshold
            redshift:  array of halo redshifts
        Returns:
            array of alpha
        """
        a = self.alpha_A + self.alpha_B**(self.alpha_C - magnitude - 20)
        
        return a


    def sigma_logM(self, magnitude):
        """
        HOD parameter sigma_logM, which sets the amount of scatter in 
        the luminosity of central galaxies
        Args:
            magnitude: array of absolute magnitude threshold
            redshift:  array of halo redshifts
        Returns:
            array of sigma_logM
        """
        sigma = self.sigma_A + (self.sigma_B-self.sigma_A) / (1.+np.exp((magnitude+self.sigma_C)*self.sigma_D))
        
        return sigma

    
    def number_centrals_mean(self, log_mass, magnitude, redshift=None):
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
        return spline.cumulative_spline_kernel(log_mass, mean=np.log10(self.Mmin(magnitude)), 
                sig=self.sigma_logM(magnitude)/np.sqrt(2))


    def number_satellites_mean(self, log_mass, magnitude, redshift=None):
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
        num_cent = self.number_centrals_mean(log_mass, magnitude)
        
        num_sat = num_cent * ((10**log_mass - self.M0(magnitude))/self.M1(magnitude))**self.alpha(magnitude)

        num_sat[np.where(np.isnan(num_sat))[0]] = 0

        return num_sat


    def number_galaxies_mean(self, log_mass, magnitude, redshift=None):
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
        return self.number_centrals_mean(log_mass, magnitude) + \
            self.number_satellites_mean(log_mass, magnitude)


    def get_number_satellites(self, log_mass, redshift=None):
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
        magnitude = self.abs_mag_faint

        # mean number of satellites in each halo brighter than the
        # faint magnitude threshold
        number_mean = self.number_satellites_mean(log_mass, np.ones(len(log_mass))*magnitude)
        
        # draw random number from Poisson distribution
        return np.random.poisson(number_mean)


    def get_magnitude_centrals(self, log_mass, redshift=None):
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
        points = np.array(list(zip(log_mass, x)))
        return self.__central_interpolator(points)
    

    def get_magnitude_satellites(self, log_mass, number_satellites, redshift=None):
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

        # uniform random number x
        # x=1 corresponds to mag_faint
        # mag -> -infinity as x -> 0
        log_x = np.log10(np.random.rand(len(log_mass_satellite)))

        # find corresponding satellite magnitudes
        points = np.array(list(zip(log_mass_satellite, log_x)))
        return halo_index, self.__satellite_interpolator(points)

    
    def __integration_function(self, logM, mag, z, logMmin, logM1, logM0, sigmalogM, alpha,
                              galaxies):
        # function to integrate (number density of haloes * number of galaxies from HOD)
                
        # mean number of galaxies per halo
        #N_gal = self.number_galaxies_mean(np.array([logM,]),mag,z,f)[0]
        
        Ncen = spline.cumulative_spline_kernel(logM, mean=logMmin, sig=sigmalogM/np.sqrt(2))
        Nsat = Ncen * ((10**logM - 10**logM0)/10**logM1)**alpha
        Nsat[np.where(np.isnan(Nsat))[0]] = 0
        
        if galaxies=="all":
            N_gal = Ncen + Nsat
        elif galaxies=="cen":
            N_gal = Ncen
        elif galaxies=="sat":
            N_gal = Nsat
        
        # number density of haloes
        n_halo = self.mf.number_density(np.array([logM,]))
                
        return (N_gal * n_halo)[0]
    
    
    def get_n_HOD(self, magnitude, redshift, logMmin, logM1, logM0, sigmalogM, alpha,
                    Mmin=10, Mmax=16, galaxies="all"):
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
        return quad(self.__integration_function, Mmin, Mmax, 
                    args=(magnitude, redshift, logMmin, logM1, logM0, sigmalogM, alpha, galaxies))[0]

