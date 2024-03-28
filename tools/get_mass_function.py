#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import interp1d
import h5py

import sys
sys.path.append('..')
from hodpy.mass_function import MassFunction
from hodpy.cosmology import CosmologyAbacus
from hodpy.power_spectrum import PowerSpectrum

from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from abacusnbody.data.read_abacus import read_asdf


def measure_mass_function_box(cosmo, redshift, phase=0, simulation="base",
                              Lbox=2000., Nfiles=34, bin_size=0.01):
    """
    Measure the halo mass function of the cubic box
    Args:
    Returns:
        Array of halo mass bin centres, in log10(mass)
        Array of log(n)
    """
    # initialize AbacusSummit cosmology
    cosmology = CosmologyAbacus(cosmo)
    
    # input path of the simulation
    abacus_path = "/global/cfs/cdirs/desi/cosmosim/Abacus/"
    mock = "AbacusSummit_%s_c%03d_ph%03d"%(simulation, cosmo, phase)
    input_file = abacus_path+mock+"/halos/z%.3f/halo_info/halo_info_%03d.asdf"

    log_mass = [None]*Nfiles
    
    for file_number in range(Nfiles):
        print("Reading files ({}%)".format(int(100*file_number/Nfiles)), end='\r')
        
        halo_cat = CompaSOHaloCatalog(input_file%(redshift, file_number), 
                                      cleaned=True, fields=['N'])
        m_par = halo_cat.header["ParticleMassHMsun"]
        halo_mass = np.array(halo_cat.halos["N"])*m_par
        log_mass[file_number] = np.log10(halo_mass[halo_mass>0])

    print("Reading files (100%)")
    print("Getting mass function")
    
    log_mass = np.concatenate(log_mass)

    # get number densities in mass bins  
    mass_bins = np.arange(10,16,bin_size)
    mass_binc = mass_bins[:-1]+bin_size/2.
    hist, bins = np.histogram(log_mass, bins=mass_bins)
    n_halo = hist/bin_size/Lbox**3

    # remove bins with zero haloes
    keep = n_halo > 0
    return mass_binc[keep], n_halo[keep]


def get_mass_functions(cosmo, mass_function_file, simulation="base", phase=0, Lbox=2000., 
                       Nfiles=34, snapshot_redshifts=None):
    """
    Measure the mass function at each snapshot, and smooth to remove noise
    """
    # if no redshifts provided, use all the snapshots with z < 1.0
    if snapshot_redshifts is None:
        snapshot_redshifts = 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, \
                             0.450, 0.500, 0.575, 0.650, 0.725, 0.800, 0.875, 0.950
    
    #for snapshot_redshift in redshifts:
    for i in range(len(snapshot_redshifts)):
        snapshot_redshift = snapshot_redshifts[i]

        print("z = %.3f"%snapshot_redshift)
        logM, n = measure_mass_function_box(cosmo, snapshot_redshift, phase=phase, 
                    simulation=simulation, Lbox=Lbox, Nfiles=Nfiles, bin_size=0.01)

        cosmology = CosmologyAbacus(cosmo)

        mf = MassFunction(cosmology=cosmology, redshift=snapshot_redshift, 
                          measured_mass_function=[logM, n])

        # get fit to mass function
        fit_params = mf.get_fit()

        # take ratio of measured mass function to the fit, and smooth
        # smooth the ratio since it covers a smaller dynamic range
        ratio = np.log10(n/mf.number_density(logM))

        kernel = norm.pdf(np.arange(-25,26), loc=0, scale=8)
        kernel /= np.sum(kernel)
        ratio_convolved = np.convolve(ratio, kernel, mode='same')

        start, stop = 20, -20
        logM = logM[start:stop]
        logn = np.log10(10**ratio_convolved[start:stop] * mf.number_density(logM))

        logM_bins = np.arange(10,16,0.01)
        f_interp = interp1d(logM[30:-1], logn[30:-1], bounds_error=False, fill_value='extrapolate', kind='linear')
        logn_interp = f_interp(logM_bins)

        f = h5py.File(mass_function_file,'a')
        f.create_dataset('%i/z'%i, data=np.array([snapshot_redshift,]))
        f.create_dataset('%i/log_mass'%i, data=logM_bins, compression='gzip')
        f.create_dataset('%i/log_n'%i, data=logn_interp, compression='gzip')
        f.close()
        
        
if __name__ == '__main__':
    
    cosmo=0 # AbacusSummit cosmology number
    phase=0 
    
    # Output file to save the mass functions
    mass_function_file = 'abacus_mass_functions_c%03d_ph%03d.hdf5'%(cosmo,phase)

    # Measure the mass functions. By default for all snapshots z<1.0
    get_mass_functions(cosmo, mass_function_file, phase=phase)
