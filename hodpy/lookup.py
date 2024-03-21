#! /usr/bin/env python
import numpy as np
import os
import hodpy

def get_lookup_dir():
    """
    Returns the directory containing the lookup files
    """
    path = os.path.abspath(hodpy.__file__)
    path = path.split("/")[:-1]
    path[-1] = "lookup"
    return "/".join(path)


def read_hod_param_file_mxxl(hod_param_file):
    """
    Read the MXXL HOD parameter file
    """
    params = np.loadtxt(hod_param_file, skiprows=8, delimiter=",")
    Mmin_Ls, Mmin_Mt, Mmin_am          = params[0,:3]
    M1_Ls,   M1_Mt,   M1_am            = params[1,:3]
    M0_A,    M0_B                      = params[2,:2]
    alpha_A, alpha_B, alpha_C          = params[3,:3]
    sigma_A, sigma_B, sigma_C, sigma_D = params[4,:4]
    return Mmin_Ls, Mmin_Mt, Mmin_am, M1_Ls, M1_Mt, M1_am, M0_A, M0_B, \
            alpha_A, alpha_B, alpha_C, sigma_A, sigma_B, sigma_C, sigma_D


def read_hod_param_file_abacus(hod_param_file):
    """
    Read the HOD parameter file
    """
    params = np.loadtxt(hod_param_file)
    
    Mmin_A, Mmin_B, Mmin_C, Mmin_D, \
    sigma_A, sigma_B, sigma_C, sigma_D, \
    M0_A, M0_B, \
    M1_A, M1_B, M1_C, M1_D, \
    alpha_A, alpha_B, alpha_C = params
    
    return Mmin_A, Mmin_B, Mmin_C, Mmin_D, sigma_A, sigma_B, sigma_C, sigma_D, \
           M0_A, M0_B, M1_A, M1_B, M1_C, M1_D, alpha_A, alpha_B, alpha_C

path = get_lookup_dir()

######## Parameter values to set ##########

# E-correction of the form Q*(z-zp)
Q = 0.67
zq = 0.1

######### File locations for AbacusSummit lightcone ##########

# AbacusSummit simulation
abacus_mass_function = path+"/abacus/mass_function_c{:03d}_ph{:03d}.txt"

# HOD parameters for BGS mock
abacus_hod_parameters    = path+'/abacus/hod_fits_c000_ph000.txt'
abacus_hod_slide_factors = path+'/abacus/slide_factors.dat' # will be created if doesn't exist

# lookup files for central/satellite magnitudes
central_lookup_file   = path+"/abacus/central_magnitudes.npy"   # will be created if doesn't exist
satellite_lookup_file = path+"/abacus/satellite_magnitudes.npy" # will be created if doesn't exist

# BGS k-corrections
kcorr_file_bgs = path+'/bgs/jmext_kcorr_{}_{}band_z01.dat' # for magnitudes
kcorr_gmr_bgs = path+'/bgs/gmr_lookup_{}_{}.hdf5'          # for g-r colours

# BGS cumulative luminosity function
# bgs_lf_target = path+'/bgs/bgs_N_cumulative_lf.dat' # measurement in the North from Sam Moore
# bgs_lf_target = path+'/bgs/bgs_S_cumulative_lf.dat' # measurement in the South from Sam Moore
# bgs_lf_target = path+'/bgs/bgs_target_cumulative_lf.dat' # measurements from volume limited samples (used to fit HODs)
bgs_lf_target = path+'/bgs/bgs_samples_cumulative_lf.dat' # LF predicted from best-fitting HODs - use this one to avoid changing the clustering at z=0.2

# BGS g-r colour distribution fits
colour_fits_bgs = path+'/bgs/gmr_colour_fits_{}.hdf5'



######### File locations for MXXL lightcone ##########

# MXXL simulation
mxxl_mass_function = path+"/mxxl/mf_fits.dat"
mxxl_snapshots     = path+"/mxxl/mxxl_snapshots.dat"

# HOD parameters for BGS mock
bgs_hod_parameters    = path+"/mxxl/hod_params.dat"
bgs_hod_slide_factors = path+"/mxxl/slide_factors.dat" # will be created if doesn't exist

# lookup files for central/satellite magnitudes
# central_lookup_file   = path+"/mxxl/central_magnitudes.npy"   # will be created if doesn't exist
# satellite_lookup_file = path+"/mxxl/satellite_magnitudes.npy" # will be created if doesn't exist

# k-corrections
kcorr_file = path+"/gama/k_corr_rband_z01.dat"

# SDSS/GAMA luminosity functions
sdss_lf_tabulated = path+"/sdss/sdss_cumulative_lf.dat" # low z SDSS LF
gama_lf_fits      = path+"/gama/lf_params.dat" # high z GAMA LF
target_lf         = path+"/mxxl/target_lf.dat" # will be created if doesn't exist
