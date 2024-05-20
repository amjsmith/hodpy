import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
sys.path.append('..')

from hodpy.hod_bgs_abacus import HOD_BGS
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

    output_file = '../lookup/bgs/bgs_samples_cumulative_lf_c%03d_ph000.dat'%cosmo

    np.savetxt(output_file, np.array([magnitudes_fine, n_fine]).transpose())
    
    
if __name__ == '__main__':
    get_luminosity_function(cosmo=0)
