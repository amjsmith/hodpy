import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
sys.path.append('..')

from hodpy.hod_bgs_abacus import HOD_BGS
from hodpy.hod_bgs_abacus import HOD_BGS
from hodpy.colour import ColourDESI
from hodpy import lookup

def get_luminosity_function(cosmo):
    '''
    Save a file of the 'target' luminosity function predicted by the 
    best-fitting HODs. By default, the HOD parameters are read from
    lookup/abacus/hod_fits_c[cosmo]_ph000.txt
    with mass function fits from
    lookup/abacus/abacus_mass_functions_c[cosmo]_ph000.hdf5
    and will create
    lookup/bgs/bgs_samples_cumulative_lf_c[cosmo]_ph000.dat
    
    Args:
        cosmo: AbacusSummit cosmology number
    '''
    # Initialize the HOD at z=0.2, with redshift_evolution set to False
    hod = HOD_BGS(photsys='S', cosmo=cosmo, redshift_evolution=False,
                  replace_central_lookup=True, replace_satellite_lookup=True)

    # use the HOD (and halo mass function) to get number densities
    magnitudes = np.arange(-23.5,-14,0.1)
    n = np.zeros(len(magnitudes))
    for i in range(len(magnitudes)):
        magnitude = np.array([magnitudes[i]])
        redshift = np.array([0.2])
        n[i] = hod.get_n_HOD(magnitude, redshift)

    # interpolate between measured n values (and extrapolate linearly in log(n))
    func_interpolate = interp1d(magnitudes, np.log10(n), kind='cubic', 
                                bounds_error=False, fill_value='extrapolate')
    func_extrapolate = interp1d(magnitudes, np.log10(n), kind='linear', 
                                bounds_error=False, fill_value='extrapolate')

    # save file in fine magnitude bine
    magnitudes_fine = np.arange(-28,-10,0.001)
    n_fine = func_extrapolate(magnitudes_fine)
    idx = np.logical_and(magnitudes_fine>-23.5, magnitudes_fine<-14)
    n_fine[idx] = func_interpolate(magnitudes_fine[idx])

    #output_file = '../lookup/bgs/bgs_samples_cumulative_lf_c%03d_ph000.dat'%cosmo
    output_file = lookup.bgs_lf_target.format(cosmo,0)

    np.savetxt(output_file, np.array([magnitudes_fine, n_fine]).transpose())
    
    
if __name__ == '__main__':
    
    # This script will regenerate the necessary lookup files when changing the HOD
    
    # First, update the file set by
    # > abacus_hod_parameters 
    # in lookup.py with the new HOD parameter fits
    
    # The following files will need to be remade for the new HODs
    # > bgs_lf_target [the target LF]
    # > abacus_hod_slide_factors [factors needed to evolve the HODs to reproduce target LF]
    # > central_fraction_file [what fraction of galaxies are centrals, as defined by the HOD]
    
    # The following files also need to be remade, but they depend on the exact magnitude
    # cuts chosen, so it is safer to remake them each time the HOD code is run.
    # These files are made fairly quickly.
    # > central_lookup_file [lookup table of central galaxy magnitudes]
    # > satellite_lookup_file [lookup table of satellite galaxy magnitudes]
    
    # Create a new target LF file, which comes from integrating the HOD * halo mass function
    # This is used as the target LF so that the HODs don't change from their best-fitting values
    cosmo = 0 #AbacusSummit cosmology number
    get_luminosity_function(cosmo=cosmo)

    # Calculate the 'slide factors' used to evolve the HODs with redshift
    # The HOD mass parameters are scaled to reproduce the target LF at each redshift
    # Here, it doesn't matter what values we set photsys or mag_faint to
    # Note that this is very slow! Could be parallelized to speed it up
    hod = HOD_BGS(cosmo=cosmo, photsys='S', mag_faint_type='absolute', 
              mag_faint=-18, redshift_evolution=True, replace_slide_file=True,
                 replace_central_lookup=True, replace_satellite_lookup=True)
   
    # Use the HOD to calculate the fraction of galaxies that are centrals
    # Again, it doesn't matter here what photsys is set to
    col = ColourDESI(photsys='S', hod=hod, replace_central_fraction_lookup_file=True)
