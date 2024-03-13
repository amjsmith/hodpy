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


def read_hod_param_file(hod_param_file):
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


def read_hod_param_file(hod_param_file):
    """
    Read the AbacusSummit HOD parameter file
    """
    return

path = get_lookup_dir()


######### File locations for AbacusSummit lightcone ##########


# k-corrections
kcorr_file_bgs = path+'bgs/jmext_kcorr_{}_{}band_z01.dat' # for magnitudes
kcorr_gmr_bgs = path+'bgs/gr_lookup_{}_{}.hdf5'           # for g-r colours





######### File locations for MXXL lightcone ##########

# MXXL simulation
mxxl_mass_function = path+"/mxxl/mf_fits.dat"
mxxl_snapshots     = path+"/mxxl/mxxl_snapshots.dat"

# HOD parameters for BGS mock
bgs_hod_parameters    = path+"/mxxl/hod_params.dat"
bgs_hod_slide_factors = path+"/mxxl/slide_factors.dat" # will be created if doesn't exist

# lookup files for central/satellite magnitudes
central_lookup_file   = path+"/mxxl/central_magnitudes.npy"   # will be created if doesn't exist
satellite_lookup_file = path+"/mxxl/satellite_magnitudes.npy" # will be created if doesn't exist

# k-corrections
kcorr_file = path+"/gama/k_corr_rband_z01.dat"

# SDSS/GAMA luminosity functions
sdss_lf_tabulated = path+"/sdss/sdss_cumulative_lf.dat" # low z SDSS LF
gama_lf_fits      = path+"/gama/lf_params.dat" # high z GAMA LF
target_lf         = path+"/mxxl/target_lf.dat" # will be created if doesn't exist
